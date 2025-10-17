import os
import gc
import re
import warnings
import pickle as pkl
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import print_df
from ryp import r, to_r, to_py
from matplotlib.cm import ScalarMappable
from single_cell import SingleCell
from matplotlib.colors import Normalize, ListedColormap
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram, fcluster

warnings.filterwarnings('ignore')

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

working_dir = '/home/karbabi/spatial-pregnancy-postpart'
cell_type_col = 'subclass'

#region Load data ##############################################################

file = f'{working_dir}/output/data/exp_gene_dict.pkl'
if not os.path.exists(file):
    adata = sc.read_h5ad('single-cell/ABC/anndata/combined_10Xv3.h5ad')
    gene_dict = {
        s: adata.var_names[np.ravel(
            (adata[adata.obs[cell_type_col] == s].X > 0).sum(axis=0) >=
            (adata.obs[cell_type_col] == s).sum() * 0.10
        )].tolist()
        for s in adata.obs[cell_type_col].unique()
    }
    pkl.dump(gene_dict, open(file, 'wb'))
    del adata; gc.collect()
else:
    gene_dict = pkl.load(open(file, 'rb'))

gene_dict = {
    (m.group(2) if (m := re.match(r'^(\d+)\s+(.*)', str(k))) else str(k)): v
    for k, v in gene_dict.items()
}

adata_curio = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_curio_final.h5ad')
adata_merfish = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')
adata_merfish.var.index = adata_merfish.var['gene_symbol']

for adata in [adata_curio, adata_merfish]:
    adata.var_names_make_unique()
    for col in ['class', 'subclass']:
        adata.obs[col] = adata.obs[col].astype(str).str.extract(
            r'^(\d+)\s+(.*)', expand=False)[1]
    adata.obs['type'] = adata.obs[cell_type_col]\
        .astype(str).str.extract(r'(\w+)$', expand=False)
    adata.obs['type'] = adata.obs['type'].replace({'IMN': 'Gaba'})
    adata.obs['type'] = adata.obs['type'].replace({'Chol': 'Gaba'})
    adata.obs = adata.obs[[
        'sample', 'condition', 'source', 'x', 'y', 'x_ffd', 'y_ffd',
        'type', 'class', 'subclass', 'subclass_keep']]
    adata.var = adata.var[[
        col for col in ['gene_symbol', 'protein_coding', 'mt', 'ribo']
        if col in adata.var.columns]]
    adata.var.index.name = None
    g = adata.var_names
    adata.var['mt'] = g.str.match(r'^(mt-|MT-)')
    adata.var['ribo'] = g.str.match(r'^(Rps|Rpl)')
    for key in ('uns', 'varm', 'obsp', 'obsm'):
        if hasattr(adata, key):
            try:
                delattr(adata, key)
            except:
                pass

common_cell_types = (
    set(adata_curio.obs[
        adata_curio.obs[f'{cell_type_col}_keep']][cell_type_col])
    & set(adata_merfish.obs[
        adata_merfish.obs[f'{cell_type_col}_keep']][cell_type_col]))

adata_curio = adata_curio[
    adata_curio.obs[cell_type_col].isin(common_cell_types)].copy()
adata_merfish = adata_merfish[
    adata_merfish.obs[cell_type_col].isin(common_cell_types)].copy()

pb_curio = SingleCell(adata_curio)\
    .qc(allow_float=True)\
    .filter_var(
        pl.col('protein_coding') & 
        pl.col('ribo').not_() & 
        pl.col('mt').not_())\
    .pseudobulk('sample', 'subclass')\
    .qc('condition',
        min_samples=2,
        min_cells=10,
        max_standard_deviations=None,
        min_nonzero_fraction=0,
        verbose=False)\
    .library_size(allow_float=True, num_threads=1)

pb_preg_ctrl = pb_curio.filter_obs(
    pl.col('condition').is_in(['PREG', 'CTRL'])
)
pb_postpart_preg = pb_curio.filter_obs(
    pl.col('condition').is_in(['POSTPART', 'PREG'])
)

def populate_r_list(pb_object, r_list_name, gene_dict):
    r(f'{r_list_name} <- list()')
    for cell_type, (X, obs, var) in pb_object.items():
        genes_to_keep = gene_dict.get(cell_type) 
        if not genes_to_keep:
            print(f'no genes filtered for {cell_type}')
            genes_to_keep = var['gene_symbol'].to_list()
        var_mask = var['gene_symbol'].is_in(genes_to_keep)
        var_filtered = var.filter(var_mask)
        gene_indices = var.with_row_index().filter(var_mask)['index']
        X_filtered = X[:, gene_indices]
        to_r(obs, 'obs')
        to_r(cell_type, 'cell_type')
        to_r(X_filtered, 'X', colnames=var_filtered['gene_symbol'])
        r(f'''
        counts <- t(X)
        element <- list(counts = counts, obs = obs)
        {r_list_name}[[cell_type]] <- element
        ''')

populate_r_list(pb_preg_ctrl, 'pseudobulks_preg_ctrl', gene_dict)
populate_r_list(pb_postpart_preg, 'pseudobulks_postpart_preg', gene_dict)

#endregion 

#region DEG analysis ###########################################################

r('''
suppressPackageStartupMessages({
    library(edgeR)
    library(dplyr)
    library(tibble)
    library(purrr)
})

run_edgeR_LRT <- function(pseudobulks, ref_level) {
    imap(pseudobulks, function(element, cell_type_name) {
        tryCatch({
            targets <- element$obs
            
            all_levels <- unique(as.character(targets$condition))
            other_level <- all_levels[all_levels != ref_level]
            targets$group <- factor(
                targets$condition, levels = c(ref_level, other_level)
            )
            if (n_distinct(targets$group) < 2) return(NULL)
            
            design <- model.matrix(~ group, data = targets)
            
            y <- DGEList(counts = element$counts, samples = targets) %>%
                calcNormFactors(method = 'TMM') %>%
                estimateDisp(design)
            
            fit <- glmFit(y, design = design)
            test <- glmLRT(fit, coef = 2)
            
            tt <-topTags(test, n = Inf) %>%
                as.data.frame() %>%
                rownames_to_column('gene')

            return(tt)
        }, error = function(e) {
            warning(paste("Error in", cell_type_name, ":", e$message))
            return(NULL)
        })
    }) %>% 
    bind_rows(.id = 'cell_type')
}

de_preg_ctrl <- run_edgeR_LRT(
    pseudobulks_preg_ctrl, ref_level = "CTRL"
)
if (nrow(de_preg_ctrl) > 0) {
    de_preg_ctrl$contrast <- "PREG_vs_CTRL"
}
de_postpart_preg <- run_edgeR_LRT(
    pseudobulks_postpart_preg, ref_level = "PREG"
)
if (nrow(de_postpart_preg) > 0) {
    de_postpart_preg$contrast <- "POSTPART_vs_PREG"
}
de_results <- bind_rows(de_preg_ctrl, de_postpart_preg)

if (nrow(de_results) > 0) {
    deg_summary <- de_results %>%
        group_by(contrast, cell_type) %>%
        summarise(DEGs = sum(FDR < 0.10), .groups = 'drop') %>%
        arrange(desc(DEGs))
    print(n = Inf, deg_summary)
}
''')

de_results = to_py('de_results')

de_results.write_csv(f'{working_dir}/output/data/de_results.csv')
de_results\
    .filter(pl.col('FDR') < 0.10)\
    .write_csv(f'{working_dir}/output/data/de_results_sig.csv')

for cell_type in de_results.select('cell_type').unique().to_series():
    print(f'\n{cell_type} - PREG_vs_CTRL - de_results:')
    print(de_results
          .filter((pl.col('cell_type').eq(cell_type)) & 
                  (pl.col('contrast').eq('PREG_vs_CTRL')))
          .sort('PValue').head(15))
    print(f'\n{cell_type} - POSTPART_vs_PREG - de_results:')
    print(de_results
          .filter((pl.col('cell_type').eq(cell_type)) & 
                  (pl.col('contrast').eq('POSTPART_vs_PREG')))
          .sort('PValue').head(15))

#endregion #####################################################################

#region DEG landscape ##########################################################

FDR_CUTOFF = 0.1
GENES_TO_LABEL = {
    # Contrast: Pregnant vs. Nulliparous
    ('PREG_vs_CTRL', 'Oligo NN'): ['Sgk1', 'Hif3a', 'Fkbp5', 'Hccs'],
    ('PREG_vs_CTRL', 'Microglia NN'): ['Ccnd3', 'Fkbp5', 'Ndufa5'],
    ('PREG_vs_CTRL', 'Astro-TE NN'): ['Mfsd2a', 'Fkbp5', 'Cnr1'],
    ('PREG_vs_CTRL', 'Endo NN'): ['Kdr', 'Ccn2', 'Igfbp3', 'Zbtb16'],
    ('PREG_vs_CTRL', 'Ependymal NN'): ['Pcdh15'],
    ('PREG_vs_CTRL', 'OPC NN'): ['Cox6c'],
    ('PREG_vs_CTRL', 'L2/3 IT CTX Glut'): ['Zbtb18', 'Ckb', 'Cox6c', 'Hspa9'],
    ('PREG_vs_CTRL', 'L4/5 IT CTX Glut'): ['Hmgcs1'],
    ('PREG_vs_CTRL', 'L5 IT CTX Glut'): ['Tshz2', 'Glul'],
    ('PREG_vs_CTRL', 'L6 CT CTX Glut'): ['Sdk1'],
    ('PREG_vs_CTRL', 'L6 IT CTX Glut'): ['Lsamp'],
    ('PREG_vs_CTRL', 'CLA-EPd-CTX Car3 Glut'): ['Ckb', 'Atp1a2', 'Cox6c', 'Glul'],
    ('PREG_vs_CTRL', 'Pvalb Gaba'): ['Aldoc'],
    ('PREG_vs_CTRL', 'Sst Gaba'): ['Hmgcs1'],
    ('PREG_vs_CTRL', 'Vip Gaba'): ['Cnr1'],
    ('PREG_vs_CTRL', 'LSX Nkx2-1 Gaba'): ['Prkca', 'Slc1a2', 'Slc4a4'],
    ('PREG_vs_CTRL', 'NDB-SI-MA-STRv Lhx8 Gaba'): ['Slc6a11', 'Slc1a2'],
    ('PREG_vs_CTRL', 'Sst Chodl Gaba'): ['Sdk1', 'Gabrg3'],
    ('PREG_vs_CTRL', 'STR D1 Gaba'): ['Cnr1', 'Ckb'],
    ('PREG_vs_CTRL', 'STR D2 Gaba'): ['Cnr1'],

    # Contrast: Postpartum vs. Pregnant
    ('POSTPART_vs_PREG', 'Oligo NN'): ['Man1a', 'Slc38a2'],
    ('POSTPART_vs_PREG', 'Microglia NN'): ['Atp5b'],
    ('POSTPART_vs_PREG', 'Astro-TE NN'): ['Vegfa'],
    ('POSTPART_vs_PREG', 'Endo NN'): ['Ankrd37', 'Edn1', 'Dbp'],
    ('POSTPART_vs_PREG', 'Ependymal NN'): ['Pcdh15', 'Galntl6'],
    ('POSTPART_vs_PREG', 'L2/3 IT CTX Glut'): ['Ckb', 'Ubc', 'Tshz2'],
    ('POSTPART_vs_PREG', 'L4/5 IT CTX Glut'): ['Insig1', 'Hmgcs1'],
    ('POSTPART_vs_PREG', 'L5 IT CTX Glut'): ['Tshz2', 'Lsamp'],
    ('POSTPART_vs_PREG', 'L6b CTX Glut'): ['Tshz2', 'Il1rapl2'],
    ('POSTPART_vs_PREG', 'L6b EPd Glut'): ['Tafa1'],
    ('POSTPART_vs_PREG', 'Sst Gaba'): ['Ckb'],
    ('POSTPART_vs_PREG', 'Sst Chodl Gaba'): ['Rnf220', 'Gpc5'],
    ('POSTPART_vs_PREG', 'Lamp5 Gaba'): ['Schip1', 'Npas3'],
    ('POSTPART_vs_PREG', 'LSX Nkx2-1 Gaba'): ['Kcnip4', 'Kcnc2'],
    ('POSTPART_vs_PREG', 'STR D1 Gaba'): ['Stard5'],
    ('POSTPART_vs_PREG', 'STR D2 Gaba'): ['Slit2', 'Nr4a3'],
}

de_results = pl.read_csv(f'{working_dir}/output/data/de_results.csv')

annot_df = pl.DataFrame(adata_curio.obs)\
    .select(cell_type_col, 'type').unique()\
    .rename({cell_type_col: 'cell_type'})

df_sig = de_results\
    .filter(pl.col('FDR') < FDR_CUTOFF)\
    .with_columns(pl.col('FDR').add(1e-300).log10().neg().alias('log10_fdr'))

deg_counts = df_sig\
    .group_by(['cell_type', 'contrast'])\
    .agg((pl.col('logFC').gt(0)).sum().alias('up'),
         (pl.col('logFC').lt(0)).sum().alias('down'))

type_order_map = {'Glut': 0, 'Gaba': 1, 'NN': 2}
cell_type_order_df = deg_counts\
    .group_by('cell_type')\
    .agg((pl.sum('up').add(pl.sum('down'))).alias('total_degs'))\
    .join(annot_df, on='cell_type')\
    .with_columns(pl.col('type').replace(type_order_map).alias('type_order'))\
    .sort(['type_order', 'total_degs'], descending=[False, True])

cell_type_order = cell_type_order_df['cell_type'].to_list()
df_plot = df_sig\
    .with_columns(pl.col('cell_type').cast(pl.Categorical(cell_type_order)))\
    .sort('cell_type')

groups = cell_type_order_df.group_by('type', maintain_order=True).all()
height_ratios = groups['cell_type'].list.len().to_list()
major_types = groups['type'].to_list()

fig = plt.figure(figsize=(9, 13))
outer_gs = gridspec.GridSpec(
    len(major_types), 1, figure=fig, height_ratios=height_ratios, hspace=0.1
)
vmin, vmax = (df_plot['log10_fdr'].quantile(0.05), 
              df_plot['log10_fdr'].quantile(0.98))
norm = Normalize(vmin=vmin, vmax=vmax)
original_cmap = plt.get_cmap('GnBu')
n_colors = 256
colors = original_cmap(np.linspace(0.3, 1.0, n_colors))
cmap = ListedColormap(colors)
seismic_cmap = plt.get_cmap('seismic')
up_color, down_color = seismic_cmap(0.9), seismic_cmap(0.1)

all_axes = []
for i, group_type in enumerate(major_types):
    group_cell_types = groups.filter(pl.col('type') == group_type)\
        ['cell_type'].explode().to_list()
    
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer_gs[i],
        width_ratios=[10, 2.5, 10, 2.5], wspace=0.05
    )
    ax1 = fig.add_subplot(inner_gs[0])
    ax1_bar = fig.add_subplot(inner_gs[1])
    ax2 = fig.add_subplot(inner_gs[2])
    ax2_bar = fig.add_subplot(inner_gs[3])
    all_axes.append((ax1, ax1_bar, ax2, ax2_bar))

    axes_config = [
        (ax1, ax1_bar, 'PREG_vs_CTRL', 'Pregnant vs Nulliparous'),
        (ax2, ax2_bar, 'POSTPART_vs_PREG', 'Postpartum vs Pregnant')
    ]

    for j, (ax_main, ax_bar, contrast_name, title) in enumerate(axes_config):
        df_group = df_plot.filter(pl.col('cell_type').is_in(group_cell_types))
        contrast_df = df_group.filter(pl.col('contrast').eq(contrast_name))
        counts_data = deg_counts.filter(pl.col('contrast').eq(contrast_name))\
            .select(['cell_type', 'up', 'down'])\
            .to_dicts()
        
        counts_dict = {row['cell_type']: {'up': row['up'], 'down': row['down']} 
                       for row in counts_data}
        
        up_counts = [counts_dict.get(ct, {'up': 0})['up'] 
                    for ct in group_cell_types]
        down_counts = [counts_dict.get(ct, {'down': 0})['down'] 
                    for ct in group_cell_types]
        
        y_pos = {ct: i for i, ct in enumerate(group_cell_types)}
        contrast_y = [y_pos[ct] for ct in contrast_df['cell_type']]

        facecolors = cmap(norm(contrast_df['log10_fdr']))
        scatter = ax_main.scatter(
            x=contrast_df['logFC'], y=contrast_y, s=15,
            facecolors=facecolors, edgecolors='gray', linewidth=0.5, zorder=10
        )
        
        ax_bar.barh(range(len(group_cell_types)), up_counts, left=0,
                    color=up_color, alpha=0.8)
        ax_bar.barh(range(len(group_cell_types)), down_counts,
                    left=[-x for x in down_counts], color=down_color, alpha=0.8)
        ax_bar.axvline(0, color='grey', linestyle='-', linewidth=0.5)
        
        max_fc = (contrast_df['logFC'].abs().max() * 1.1 
                  if not contrast_df.is_empty() else 1)
        
        label_offset = 0.18
        
        for ct_idx, ct in enumerate(group_cell_types):
            genes = GENES_TO_LABEL.get((contrast_name, ct), [])
            if genes:
                ct_df = contrast_df.filter(
                    (pl.col('cell_type') == ct) & 
                    pl.col('gene').is_in(genes)
                )
                if ct_df.height > 0:
                    ct_y = y_pos[ct]
                    label_y = ct_y - label_offset
                    ct_data = ct_df.to_dicts()
                    ct_data.sort(key=lambda x: x['logFC'])
                    gene_width = 0.12 * (max_fc * 2)
                    x_positions = [r['logFC'] for r in ct_data]
                    adjusted_x = []
                    if len(x_positions) == 1:
                        adjusted_x = x_positions
                    else:
                        adjusted_x = [x_positions[0]]
                        for i in range(1, len(x_positions)):
                            ideal_x = x_positions[i]
                            min_x = adjusted_x[-1] + gene_width
                            adjusted_x.append(max(ideal_x, min_x))
                    
                    for i, record in enumerate(ct_data):
                        ax_main.text(adjusted_x[i], label_y, record['gene'], 
                                    style='italic', fontsize=7,
                                    ha='center', va='bottom', zorder=150)
                        
                        ax_main.plot([record['logFC'], adjusted_x[i]], 
                                    [ct_y, label_y],
                                    'k-', lw=0.6, alpha=0.8, zorder=90)

        if contrast_name == 'PREG_vs_CTRL':
            bar_xlim = 300
        else:
            bar_xlim = 30
        
        ax_main.grid(True, 'major', 'y', ls='-', lw=0.5, c='lightgray', 
                     zorder=0)
        ax_main.axvline(0, color='lightgray', linestyle='-', linewidth=0.5, 
                        zorder=0)
        ax_main.set_xlim(-max_fc, max_fc)
        # ax_bar.set_xscale('symlog', linthresh=1)
        ax_bar.set_xlim(-bar_xlim, bar_xlim)
        
        for idx, ct in enumerate(group_cell_types):
            if ct in counts_dict:
                total_degs = int(counts_dict[ct]['up'] + counts_dict[ct]['down'])
                if total_degs > 0:
                    ax_bar.text(bar_xlim * 0.95, idx, str(total_degs), 
                               ha='right', va='center', fontsize=7,
                               color='black', zorder=100)
        
        y_ticks = range(len(group_cell_types))
        ax_main.set_yticks(y_ticks)
        ax_main.set_yticklabels(group_cell_types)
        ax_bar.set_yticks(y_ticks)
        ax_bar.set_yticklabels(group_cell_types)
        ax_main.set_ylim(len(group_cell_types)-0.5, -0.5)
        ax_bar.set_ylim(len(group_cell_types)-0.5, -0.5)
            
        for ax in [ax_main, ax_bar]: ax.tick_params(length=0)
            
for i, (ax1, ax1_bar, ax2, ax2_bar) in enumerate(all_axes):
    plt.setp(ax1_bar.get_yticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax2_bar.get_yticklabels(), visible=False)
    ax1.set_ylabel(None)
    
    if i == 0:
        ax1.set_title('Pregnant vs Nulliparous', fontsize=14)
        ax2.set_title('Postpartum vs Pregnant', fontsize=14)
    
    if i == len(all_axes) - 1:
        ax1.set_xlabel('log2(Fold Change)', fontsize=12)
        ax1_bar.set_xlabel('# DEGs', fontsize=12)
        ax2.set_xlabel('log2(Fold Change)', fontsize=12)
        ax2_bar.set_xlabel('# DEGs', fontsize=12)
    else:
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1_bar.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2_bar.get_xticklabels(), visible=False)

cbar_ax = fig.add_axes([1.0, 0.65, 0.015, 0.2])
cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cbar.set_label('-log10(FDR)', size=12)

fig.tight_layout(rect=[0, 0, 0.98, 1])
os.makedirs(f'{working_dir}/figures', exist_ok=True)
plt.savefig(
    f'{working_dir}/figures/deg_landscape.png', dpi=300, bbox_inches='tight'
)
plt.savefig(
    f'{working_dir}/figures/deg_landscape.svg', dpi=300, bbox_inches='tight'
)
plt.close()

#endregion 

#region DEG exemplars ##########################################################




#endregion

#region Pathway analysis #######################################################

de_results = pl.read_csv(f'{working_dir}/output/data/de_results.csv')
to_r(de_results, 'de_results_r')
to_r(working_dir, 'working_dir')

r('''
suppressPackageStartupMessages({
    library(fgsea)
    library(msigdbr)
    library(dplyr)
    library(tibble)
})

cache_file <- paste0(working_dir, "/output/data/m_df_themed.rds")
if (!file.exists(cache_file)) {
    m_df <- msigdbr(
        species = "Mus musculus", category = "C5", subcategory = "GO:BP")
    theme_keywords <- list(
        'Neuronal' = c(
            'NEURO', 'SYNAP', 'AXON', 'DENDRITE', 'GLUTAMATE', 'GABA',
            'CHOLINERGIC', 'DOPAMINERGIC', 'SEROTONERGIC', 'ADRENERGIC',
            'ACTION_POTENTIAL', 'REGULATION_NEUROTRANSMITTER_LEVELS',
            'REGULATION_SYNAPTIC_PLASTICITY', 'EXCITATORY_POSTSYNAPTIC',
            'INHIBITORY_POSTSYNAPTIC'
        ),
        'Glial' = c(
            'GLIA', 'MYELIN', 'ASTROCYTE', 'OLIGODENDROCYTE', 'MICROGLIA',
            'REACTIVE_GLIOSIS', 'REGULATION_NEURON_GLIAL_COMMUNICATION',
            'CNS_MYELIN_MAINTENANCE', 'GLIAL_CELL_MIGRATION', 'GLIOGENESIS'
        ),
        'Immune' = c(
            'IMMUNE', 'INFLAMMATORY', 'CYTOKINE', 'INTERFERON',
            'NEUROINFLAMMATORY', 'MICROGLIAL_CELL_MIGRATION',
            'REGULATION_CYTOKINE_MEDIATED_SIGNALING', 'INNATE_IMMUNE'
        ),
        'Hormonal' = c(
            'HORMONE', 'STEROID', 'ESTROGEN', 'PROGESTERONE', 'OXYTOCIN',
            'CORTISOL', 'GLUCOCORTICOID', 'MINERALOCORTICOID',
            'STEROID_HORMONE_BIOSYNTHETIC', 'REGULATION_HORMONE_SECRETION',
            'CELLULAR_RESPONSE_HORMONE_STIMULUS'
        ),
        'Metabolic' = c(
            'METABOLIC', 'LIPID', 'CHOLESTEROL', 'GLUCOSE_METABOLIC',
            'ATP_METABOLIC', 'CELLULAR_RESPIRATION'
        ),
        'Plasticity_Dev' = c(
            'NEUROGENESIS', 'PROLIFERATION', 'DENDRITIC_SPINE',
            'DIFFERENTIATION', 'MIGRATION', 'APOPTOSIS',
            'ADULT_NEUROGENESIS', 'AXON_GUIDANCE', 'SYNAPSE_ORGANIZATION',
            'NEURON_PROJECTION_DEVELOPMENT'
        ),
        'Reproductive' = c(
            'PROLACTIN', 'GALANIN', 'VASOPRESSIN', 'RELAXIN', 'KISSPEPTIN',
            'GONADOTROPIN', 'LUTEINIZING', 'FOLLICLE_STIMULATING',
            'GONADOTROPIN_RELEASING_HORMONE', 'REGULATION_OVULATION_CYCLE',
            'PARTURITION', 'LACTATION'
        ),
        'Maternal_Social' = c(
            'MATERNAL', 'PARENTAL', 'SOCIAL', 'OXYTOCIN_SIGNALING',
            'SOCIAL_RECOGNITION', 'MATERNAL_AGGRESSIVE',
            'RESPONSE_PHEROMONE'
        ),
        'Stress_Adaptation' = c(
            'STRESS', 'CORTICOSTERONE', 'ADRENERGIC',
            'REGULATION_HPA_AXIS', 'CELLULAR_RESPONSE_STRESS',
            'GENERAL_ADAPTATION_SYNDROME'
        ),
        'Neurotransmitter' = c(
            'DOPAMINE', 'SEROTONIN', 'ACETYLCHOLINE', 'NOREPINEPHRINE',
            'NEUROTRANSMITTER_SECRETION', 'REGULATION_SYNAPTIC_VESICLE',
            'NEUROTRANSMITTER_UPTAKE'
        ),
        'Growth_Factors' = c(
            'GROWTH_FACTOR', 'NEUROTROPHIC', 'BDNF', 'NGF', 'IGF',
            'INSULIN_LIKE', 'EPIDERMAL_GROWTH', 'FIBROBLAST_GROWTH',
            'REGULATION_CELL_GROWTH', 'POSITIVE_REGULATION_NEURON_PROJECTION',
            'CELLULAR_RESPONSE_GROWTH_FACTOR'
        ),
        'Cell_Cycle' = c(
            'CELL_CYCLE', 'MITOTIC', 'DNA_REPLICATION', 'CHROMOSOME',
            'SPINDLE', 'CYTOKINESIS', 'CELL_CYCLE_CHECKPOINT',
            'REGULATION_CELL_CYCLE_PHASE_TRANSITION', 'DNA_REPAIR'
        ),
        'Vascular' = c(
            'VASCULAR', 'VASCULATURE', 'ANGIOGENESIS', 'ENDOTHELIAL', 
            'BLOOD_BRAIN_BARRIER', 'CAPILLARY', 'BLOOD_VESSEL', 
            'VASOCONSTRICTION', 'ENDOTHELIAL_CELL_MIGRATION', 
            'BRAIN_ANGIOGENESIS'
        ),
        'Circadian_Sleep' = c(
            'CIRCADIAN', 'SLEEP', 'WAKE', 'MELATONIN', 'CLOCK',
            'ENTRAINMENT_CIRCADIAN_CLOCK', 'REGULATION_SLEEP_CYCLE',
            'RHYTHMIC_PROCESS'
        ),
        'Hypothalamic_Pituitary' = c(
            'RELEASING_HORMONE', 'NEUROSECRETORY_SYSTEM_DEVELOPMENT',
            'REGULATION_HORMONE_SECRETION', 'NEUROPEPTIDE_SIGNALING'
        ),
        'Ion_Transport' = c(
            'CALCIUM', 'SODIUM', 'POTASSIUM', 'TRANSPORT',
            'MEMBRANE_POTENTIAL', 'SYNAPTIC_VESICLE_EXOCYTOSIS',
            'CALCIUM_MEDIATED_SIGNALING', 'ION_HOMEOSTASIS'
        ),
        'Epigenetic' = c(
            'EPIGENETIC', 'METHYLATION', 'HISTONE', 'CHROMATIN',
            'TRANSCRIPTIONAL', 'CHROMATIN_REMODELING',
            'HISTONE_ACETYLATION', 'REGULATION_GENE_EXPRESSION_EPIGENETIC'
        ),
        'Protein_Dynamics' = c(
            'PROTEIN_SYNTHESIS', 'TRANSLATION', 'RIBOSOMAL', 'PROTEASOME',
            'UBIQUITIN', 'AUTOPHAGY', 'PROTEIN_FOLDING',
            'REGULATION_TRANSLATION', 'ER_ASSOCIATED_PROTEIN_CATABOLIC'
        ),
        'Structural_ECM' = c(
            'ADHESION', 'EXTRACELLULAR'
        )
    )
    
    all_keywords <- unlist(theme_keywords)
    regex_pattern <- paste(all_keywords, collapse = "|")
    
    get_theme <- function(gs_name, themes) {
        for (theme_name in names(themes)) {
            if (any(sapply(themes[[theme_name]], grepl, gs_name, 
                          ignore.case=TRUE))) {
                return(theme_name)
            }
        }
        return(NA_character_)
    }
    
    m_df_themed <- m_df %>%
        filter(grepl(regex_pattern, gs_name, ignore.case = TRUE)) %>%
        rowwise() %>%
        mutate(theme = get_theme(gs_name, theme_keywords)) %>%
        ungroup() %>%
        filter(!is.na(theme))
    
    saveRDS(m_df_themed, cache_file)
} else {
    m_df_themed <- readRDS(cache_file)
}

filtered_pathways <- m_df_themed %>% 
    split(x = .$gene_symbol, f = .$gs_name)

pathway_theme_lookup <- m_df_themed %>%
    select(gs_name, theme) %>% 
    distinct() %>% 
    rename(pathway = gs_name)

fgsea_results <- de_results_r %>%
    group_by(cell_type, contrast) %>%
    group_map(~ {
        ranks <- .x %>%
            mutate(rank = -log10(PValue) * sign(logFC)) %>%
            arrange(desc(rank)) %>%
            select(gene, rank) %>%
            tibble::deframe()
        
        res <- fgsea(pathways = filtered_pathways, stats = ranks, 
                    minSize=15)
        
        if (nrow(res) > 0) {
            res %>% 
                as_tibble() %>%
                mutate(cell_type = .y$cell_type, contrast = .y$contrast) %>%
                left_join(pathway_theme_lookup, by = "pathway")
        } else {
            tibble()
        }
    }) %>%
    bind_rows()
''')

pathway_de_results_gsea = to_py('fgsea_results')

pathway_de_results_gsea\
    .write_parquet(f'{working_dir}/output/data/pathway_results_gsea.parquet')

available_genes = de_results\
    .filter(pl.col('FDR') < 0.10)\
    .group_by(['cell_type', 'contrast'])\
    .agg(pl.col('gene').unique().alias('available_genes'))

pathway_de_results_gsea_sig = pathway_de_results_gsea\
    .filter(pl.col('padj') < 0.01)\
    .join(available_genes, on=['cell_type', 'contrast'], how='left')\
    .with_columns(
        pl.struct(['leadingEdge', 'available_genes']).map_elements(
            lambda x: [gene for gene in x['leadingEdge'] 
                       if x['available_genes'] is not None 
                       and gene in x['available_genes']] 
                       if x['available_genes'] is not None else [],
            return_dtype=pl.List(pl.Utf8)
        ).alias('leadingEdge_filtered')
    )\
    .with_columns(
        pl.col('leadingEdge_filtered').list.join(", ")
        .alias('leadingEdge_genes'))\
    .drop(['leadingEdge', 'leadingEdge_filtered', 'available_genes'])

pathway_de_results_gsea_sig.write_csv(
    f'{working_dir}/output/data/pathway_results_gsea_sig.csv')

#endregion 

#region Pathway heatmaps #######################################################

pathway_de_results_gsea = pl.read_parquet(
    f'{working_dir}/output/data/pathway_results_gsea.parquet')

datasets = [
    {
        'name': 'glut',
        'cell_types': ['L2/3 IT CTX Glut', 'L4/5 IT CTX Glut', 'L5 IT CTX Glut', 
                       'L6 IT CTX Glut', 'L6 CT CTX Glut'],
        'pathways': ['GOBP_SYNAPSE_ORGANIZATION',
                     'GOBP_REGULATION_OF_TRANS_SYNAPTIC_SIGNALING',
                     'GOBP_CHAPERONE_MEDIATED_PROTEIN_FOLDING',
                     'GOBP_STEROID_METABOLIC_PROCESS',
                     'GOBP_CELLULAR_RESPIRATION'],
        'labels': ['Synapse Organization', 
                   'Regulation of Trans-Synaptic Signaling',
                   'Chaperone Mediated Protein Folding',
                   'Steroid Metabolic Process',
                   'Cellular Respiration']
    },
    {
        'name': 'gaba',
        'cell_types': ['Sst Gaba', 'Pvalb Gaba', 'Vip Gaba', 
                       'STR D1 Gaba', 'LSX Nkx2-1 Gaba'],
        'pathways': ['GOBP_CELLULAR_RESPIRATION',
                     'GOBP_STEROID_METABOLIC_PROCESS',
                     'GOBP_REGULATION_OF_SYNAPTIC_PLASTICITY',
                     'GOBP_CELL_CELL_ADHESION_VIA_PLASMA_MEMBRANE_ADHESION_MOLECULES',
                     'GOBP_PROTON_TRANSMEMBRANE_TRANSPORT',
                     'GOBP_NEURON_PROJECTION_ORGANIZATION'],
        'labels': ['Cellular Respiration',
                   'Steroid Metabolic Process',
                   'Regulation of Synaptic Plasticity',
                   'Cell-Cell Adhesion via PM Molecules',
                   'Proton Transmembrane Transport',
                   'Neuron Projection Organization']
    },
    {
        'name': 'nn',
        'cell_types': ['Endo NN', 'Astro-TE NN', 'Microglia NN', 
                       'OPC NN', 'VLMC NN', 'Oligo NN'],
        'pathways': ['GOBP_VASCULATURE_DEVELOPMENT',
                     'GOBP_CELL_ADHESION',
                     'GOBP_RESPONSE_TO_HORMONE',
                     'GOBP_ATP_SYNTHESIS_COUPLED_ELECTRON_TRANSPORT',
                     'GOBP_ENDOTHELIAL_CELL_MIGRATION'],
        'labels': ['Vasculature Development', 'Cell Adhesion', 
                   'Response to Hormone', 'ATP Synthesis Electron Transport',
                   'Endothelial Cell Migration']
    }
]

fig = plt.figure(figsize=(7, 13))
gs = fig.add_gridspec(3, 2, hspace=0.60, wspace=0.15, 
                      left=0.05, right=0.85, top=0.96, bottom=0.16)

from matplotlib.colors import LinearSegmentedColormap
seismic = plt.cm.get_cmap('seismic')
n_colors = 256
colors = seismic(np.linspace(0, 1, n_colors))
white_range = 0.50
center = n_colors // 2
spread = int(n_colors * white_range)
for i in range(center - spread, center + spread):
    weight = 1 - abs(i - center) / spread
    colors[i] = (1-weight) * colors[i] + weight * np.array([1, 1, 1, 1])
custom_seismic = LinearSegmentedColormap.from_list('custom_seismic', colors)

from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

all_matrices = []
vmin_global = np.inf
vmax_global = -np.inf

for row_idx, ds in enumerate(datasets):
    cell_types = ds['cell_types']
    pathways = ds['pathways']
    labels = ds['labels']
    
    es_matrix1 = np.full((len(pathways), len(cell_types)), np.nan)
    sig_matrix1 = np.full((len(pathways), len(cell_types)), False)
    es_matrix2 = np.full((len(pathways), len(cell_types)), np.nan)
    sig_matrix2 = np.full((len(pathways), len(cell_types)), False)
    
    for contrast_idx, (contrast, es_mat, sig_mat) in enumerate([
        ('PREG_vs_CTRL', es_matrix1, sig_matrix1),
        ('POSTPART_vs_PREG', es_matrix2, sig_matrix2)
    ]):
        df = pathway_de_results_gsea.filter(
            pl.col('cell_type').is_in(cell_types) & 
            pl.col('pathway').is_in(pathways) &
            pl.col('contrast').eq(contrast)
        )
        
        for i, pathway in enumerate(pathways):
            for j, cell_type in enumerate(cell_types):
                data = df.filter(
                    (pl.col('pathway') == pathway) & 
                    (pl.col('cell_type') == cell_type)
                )
                if data.height > 0:
                    row = data.row(0, named=True)
                    es_mat[i, j] = row['NES']
                    sig_mat[i, j] = row['padj'] < 0.01
    
    vmin_local = np.nanmin([np.nanmin(es_matrix1), np.nanmin(es_matrix2)])
    vmax_local = np.nanmax([np.nanmax(es_matrix1), np.nanmax(es_matrix2)])
    if not np.isnan(vmin_local):
        vmin_global = min(vmin_global, vmin_local)
    if not np.isnan(vmax_local):
        vmax_global = max(vmax_global, vmax_local)
    
    avg_matrix = np.nanmean([es_matrix1, es_matrix2], axis=0)
    avg_filled = np.where(~np.isnan(avg_matrix), avg_matrix, 0)
    
    row_order = list(range(avg_filled.shape[0]))
    col_order = list(range(avg_filled.shape[1]))
    
    if avg_filled.shape[0] > 1:
        row_linkage = hierarchy.linkage(pdist(avg_filled), method='average')
        row_order = hierarchy.leaves_list(row_linkage)
    if avg_filled.shape[1] > 1:
        col_linkage = hierarchy.linkage(pdist(avg_filled.T), method='average')
        col_order = hierarchy.leaves_list(col_linkage)
    
    es_matrix1 = es_matrix1[row_order, :][:, col_order]
    es_matrix2 = es_matrix2[row_order, :][:, col_order]
    sig_matrix1 = sig_matrix1[row_order, :][:, col_order]
    sig_matrix2 = sig_matrix2[row_order, :][:, col_order]
    
    cell_types_ordered = [cell_types[i] for i in col_order]
    labels_ordered = [labels[i] for i in row_order]
    
    ax1 = fig.add_subplot(gs[row_idx, 0])
    ax2 = fig.add_subplot(gs[row_idx, 1])
    
    im1 = ax1.imshow(es_matrix1, cmap=custom_seismic, aspect='auto')
    im2 = ax2.imshow(es_matrix2, cmap=custom_seismic, aspect='auto')
    
    for i in range(es_matrix1.shape[0]):
        for j in range(es_matrix1.shape[1]):
            if sig_matrix1[i, j]:
                ax1.text(j, i, '*', ha='center', va='center', 
                        fontsize=16, color='white', weight='bold')
            if sig_matrix2[i, j]:
                ax2.text(j, i, '*', ha='center', va='center', 
                        fontsize=16, color='white', weight='bold')
    
    ax1.set_xticks(range(len(cell_types_ordered)))
    ax1.set_xticklabels(cell_types_ordered, rotation=45, ha='right', fontsize=12)
    ax2.set_xticks(range(len(cell_types_ordered)))
    ax2.set_xticklabels(cell_types_ordered, rotation=45, ha='right', fontsize=12)
    
    ax1.set_yticks(range(len(labels_ordered)))
    ax2.set_yticks(range(len(labels_ordered)))
    
    if row_idx == 0:
        ax1.set_title('Pregnant vs Nulliparous', fontsize=14, pad=10)
        ax2.set_title('Postpartum vs Pregnant', fontsize=14, pad=10)
    
    ax1.set_yticklabels([])
    ax1.tick_params(axis='y', left=False)
    ax1.tick_params(axis='x', labelrotation=45, pad=5)
    
    ax2.set_yticklabels(labels_ordered, fontsize=12)
    ax2.tick_params(axis='y', labelright=True, labelleft=False, 
                    right=True, left=False)
    ax2.tick_params(axis='x', labelrotation=45, pad=5)
    
    all_matrices.append((im1, im2))

vmax_global = max(abs(vmin_global), abs(vmax_global)) if vmax_global != -np.inf else 1

for (im1, im2) in all_matrices:
    im1.set_clim(-vmax_global, vmax_global)
    im2.set_clim(-vmax_global, vmax_global)

cax = fig.add_axes([0.25, 0.04, 0.4, 0.02])
cbar = fig.colorbar(im2, cax=cax, orientation='horizontal')
cbar.set_label('Normalized Enrichment Score', fontsize=12)
cbar.ax.tick_params(labelsize=9)

plt.savefig(f'{working_dir}/figures/go_heatmaps.png', 
            dpi=300, bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/go_heatmaps.svg',
            bbox_inches='tight')
plt.close()

#endregion

#region Key gene expression patterns ###########################################

genes_cells = [
    # Glutamatergic Neurons (4)
    ('Tshz2', 'L5 IT CTX Glut'),
    ('Ckb', 'L2/3 IT CTX Glut'),
    ('Rxfp1', 'CLA-EPd-CTX Car3 Glut'),
    ('Fkbp5', 'L2/3 IT CTX Glut'),

    # GABAergic Neurons (4)
    ('Prlr', 'LSX Nkx2-1 Gaba'),
    ('Cnr1', 'STR D1 Gaba'),
    ('Gabrg3', 'Sst Chodl Gaba'),
    ('Npy', 'LSX Nkx2-1 Gaba'),

    # Non-Neuronal Cells (4)
    ('Sgk1', 'Oligo NN'),
    ('Pcdh15', 'Ependymal NN'),
    ('Ptgds', 'Astro-TE NN'),
    ('Ccnd3', 'Microglia NN')
]

condition_colors = {
    'CTRL': '#7209b7',
    'PREG': '#b5179e',
    'POSTPART': '#f72585'
}

adata_norm = adata_curio.copy()
sc.pp.normalize_total(adata_norm, target_sum=1e4)
sc.pp.log1p(adata_norm)

fig, axes = plt.subplots(6, 2, figsize=(4.5, 13))
axes = axes.flatten()

for idx, (gene, cell_type) in enumerate(genes_cells):
    ax = axes[idx]
    
    cell_mask = adata_curio.obs['subclass'] == cell_type
    raw_subset = adata_curio[cell_mask, gene]
    norm_subset = adata_norm[cell_mask, gene]
    
    expr_data = []
    positions = []
    colors = []
    
    for pos, cond in enumerate(['CTRL', 'PREG', 'POSTPART']):
        cond_mask = raw_subset.obs['condition'] == cond
        raw_expr = raw_subset[cond_mask].X.toarray().flatten()
        norm_expr = norm_subset[cond_mask].X.toarray().flatten()
        
        nonzero_mask = raw_expr > 0
        expr_filtered = norm_expr[nonzero_mask]
        
        expr_data.append(expr_filtered)
        positions.append(pos)
        colors.append(condition_colors[cond])
    
    parts = ax.violinplot(expr_data, positions=positions, 
                         widths=0.5, showmeans=False, showextrema=False)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.8)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    for i, expr in enumerate(expr_data):
        ax.plot(i, np.median(expr), 'o', color='white', 
               markersize=6, markeredgecolor='black', markeredgewidth=1.5)
    
    ax.set_title(f'{gene}\n{cell_type}', fontsize=10, pad=8)
    
    if idx >= 10:
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Nulliparous', 'Pregnant', 'Postpartum'], 
                          fontsize=12, rotation=45, ha='right')
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    
    # ax.set_yscale('log')
    
    if idx % 2 == 0:
        ax.set_ylabel('')
    else:
        ax.set_ylabel('')
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

fig.text(0.02, 0.5, 'Normalized expression', va='center', rotation='vertical', 
         fontsize=12)

plt.tight_layout(rect=[0.03, 0, 1, 1])
plt.savefig(f'{working_dir}/figures/key_gene_patterns.png', 
            dpi=300, bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/key_gene_patterns.svg', 
            bbox_inches='tight')
plt.close()

#endregion

#region scratchpad #############################################################

to_r(working_dir, 'working_dir')
r('''
suppressPackageStartupMessages({
    library(msigdbr)
    library(GSVA)
    library(limma)
    library(edgeR)
    library(dplyr)
    library(purrr)
    library(tibble)
})

cache_file <- paste0(working_dir, "/output/data/m_df_themed.rds")
m_df_themed <- readRDS(cache_file)
gobp_gene_sets <- m_df_themed %>% 
    split(x = .$gene_symbol, f = .$gs_name)

pathway_theme_lookup <- m_df_themed %>%
    select(gs_name, theme) %>% distinct() %>% rename(pathway = gs_name)

run_differential_pathways <- function(
    pseudobulks, gene_sets, ref_level, contrast_str
) {
    imap_dfr(pseudobulks, function(element, cell_type_name) {
        tryCatch({
            y <- DGEList(counts = element$counts)
            y <- calcNormFactors(y, method = 'TMM')
            log_cpm <- cpm(y, log = TRUE, prior.count = 1)
            
            gsvapar <- gsvaParam(log_cpm, gene_sets, minSize=10)
            es <- gsva(gsvapar, verbose=FALSE)
  
            targets <- element$obs
            all_levels <- unique(as.character(targets$condition))
            other_level <- all_levels[all_levels != ref_level]
            targets$group <- factor(
                targets$condition, levels = c(ref_level, other_level))
            if (n_distinct(targets$group) < 2) return(NULL)
  
            design <- model.matrix(~ group, data = targets)
  
            fit <- lmFit(es, design)
            fit <- eBayes(fit)
            topTable(fit, coef=2, n=Inf, sort.by="p") %>%
                as.data.frame() %>%
                rownames_to_column("pathway") %>%
                mutate(contrast = contrast_str) %>%
                left_join(pathway_theme_lookup, by = "pathway")
        }, error = function(e) { NULL })
    }, .id = "cell_type")
}

pathway_de_preg_ctrl <- run_differential_pathways(
    pseudobulks_preg_ctrl, gobp_gene_sets, "CTRL", "PREG_vs_CTRL")
pathway_de_postpart_preg <- run_differential_pathways(
    pseudobulks_postpart_preg, gobp_gene_sets, "PREG", "POSTPART_vs_PREG")

pathway_de_results <- bind_rows(
    pathway_de_preg_ctrl, pathway_de_postpart_preg) %>%
    select(pathway, theme, cell_type, contrast, logFC, P.Value, adj.P.Val)
''')

pathway_de_results = to_py('pathway_de_results')

pathway_de_results\
    .write_csv(f'{working_dir}/output/data/pathway_results_gsva.csv')



cells_joined = pd.read_csv(
    'projects/rrg-wainberg/single-cell/ABC/metadata/MERFISH-C57BL6J-638850/'
    '20231215/views/cells_joined.csv')
color_mappings = {
    'class': dict(zip(
        cells_joined['class'].str.replace('/', '_'), 
        cells_joined['class_color'])),
    'subclass': {k.replace('_', '/'): v for k,v in dict(zip(
        cells_joined['subclass'].str.replace('/', '_'), 
        cells_joined['subclass_color'])).items()}
}

de_results = pl.read_csv(f'{working_dir}/output/data/de_results.csv')

preg_genes = de_results\
    .filter((pl.col('contrast') == 'PREG_vs_CTRL') & 
            (pl.col('PValue') < 0.05))\
    .select(['gene', 'cell_type', 'logFC'])\
    .rename({'logFC': 'preg_logFC'})

post_genes = de_results\
    .filter((pl.col('contrast') == 'POSTPART_vs_PREG') & 
            (pl.col('PValue') < 0.05))\
    .select(['gene', 'cell_type', 'logFC'])\
    .rename({'logFC': 'post_logFC'})

gene_patterns = preg_genes\
    .join(post_genes, on=['gene', 'cell_type'], how='full')\
    .filter(pl.col('cell_type').is_not_null())\
    .with_columns([
        pl.when(pl.col('preg_logFC') > 0.25).then(pl.lit('U'))
          .when(pl.col('preg_logFC') < -0.25).then(pl.lit('D'))
          .otherwise(pl.lit('='))
          .alias('preg_dir'),
        pl.when(pl.col('post_logFC') > 0.25).then(pl.lit('U'))
          .when(pl.col('post_logFC') < -0.25).then(pl.lit('D'))
          .otherwise(pl.lit('-'))
          .alias('post_dir')
    ])\
    .with_columns(
        (pl.col('preg_dir') + pl.col('post_dir')).alias('pattern')
    )\
    .filter(pl.col('pattern').is_in(['UD', 'DU', '=U', '=D', 'U-', 'D-']))

pattern_counts = gene_patterns\
    .group_by(['cell_type', 'pattern'])\
    .agg(pl.len().alias('n_genes'))

print(f"Final pattern distribution after filtering:")
print(gene_patterns['pattern'].value_counts().sort('pattern'))

sankey_data = pattern_counts\
    .with_columns([
        pl.col('pattern').alias('source'),
        pl.col('cell_type').alias('target'),
        pl.col('n_genes').alias('value'),
        pl.when(pl.col('cell_type').str.contains('Glut'))
          .then(pl.lit('Glutamatergic neurons'))
          .when(pl.col('cell_type').str.contains('Gaba'))
          .then(pl.lit('GABAergic neurons'))
          .when(pl.col('cell_type').str.contains('NN'))
          .then(pl.lit('Non-neuronal cells'))
          .otherwise(pl.lit('Other'))
          .alias('panel')
    ])

to_r(sankey_data, 'sankey_data')
to_r(working_dir, 'working_dir')
to_r(color_mappings['subclass'], 'subclass_colors')

r('''
suppressPackageStartupMessages({
    library(ggplot2)
    library(dplyr)
    library(ggsankey)
    library(patchwork)
})

sankey_data <- sankey_data %>%
    mutate(
        target_label = gsub(' Glut| Gaba| NN', '', target),
        # Strip numbers and cell type suffixes to match color mapping keys
        target_base = gsub(' Glut$| Gaba$| NN$', '', gsub('^[0-9]+ ', '', target))
    )

pattern_colors <- c(
    'UD' = '#E74C3C', 'DU' = '#3498DB', '=U' = '#E67E22',
    '=D' = '#9B59B6', 'U-' = '#F39C12', 'D-' = '#2ECC71'
)

pattern_order <- c('UD', 'DU', '=U', '=D', 'U-', 'D-')
sankey_data$source <- factor(sankey_data$source, levels = pattern_order)

make_panel_plot <- function(panel_name) {
    panel_data <- sankey_data %>% 
        filter(panel == panel_name)
    
    if (nrow(panel_data) == 0) return(NULL)
    

    
    # Get cell type colors
    cell_type_colors <- character()
    for (i in 1:nrow(panel_data)) {
        ct <- panel_data$target_base[i]
        label <- panel_data$target_label[i]
        if (ct %in% names(subclass_colors)) {
            cell_type_colors[label] <- subclass_colors[[ct]]
        } else {
            cell_type_colors[label] <- '#CCCCCC'
        }
    }
    
    # Use ggalluvial instead of ggsankey for proper value weighting
    library(ggalluvial)
    
    # All colors combined
    all_colors <- c(pattern_colors, cell_type_colors)
    
    p <- ggplot(panel_data, aes(axis1 = source, axis2 = target_label, y = value)) +
        geom_alluvium(aes(fill = source), alpha = 0.5, width = 1/12) +
        geom_stratum(aes(fill = after_stat(stratum)), width = 1/12, color = "black") +
        geom_text(stat = "stratum", aes(label = after_stat(stratum)), size = 3) +
        scale_fill_manual(values = all_colors) +
        scale_x_discrete(limits = c("Pattern", "Cell Type"), expand = c(0.15, 0.05)) +
        theme_void() +
        theme(
            legend.position = "none",
            plot.title = element_text(size = 14, face = 'bold', hjust = 0.5)
        ) +
        ggtitle(panel_name)
    
    return(p)
}

p1 <- make_panel_plot('Glutamatergic neurons')
p2 <- make_panel_plot('GABAergic neurons')
p3 <- make_panel_plot('Non-neuronal cells')

combined_plot <- p1 / p2 / p3 + 
    plot_layout(heights = c(1, 1, 1))

ggsave(paste0(working_dir, '/figures/pregnancy_transitions_sankey.png'),
       plot = combined_plot, width = 12, height = 15, dpi = 300)
ggsave(paste0(working_dir, '/figures/pregnancy_transitions_sankey.svg'),
       plot = combined_plot, width = 12, height = 15)
''')


for cell_type in pathway_de_results.select('cell_type').unique().to_series():
    print(f'\n{cell_type} - PREG_vs_CTRL - pathway_de_results:')
    filtered_df = pathway_de_results\
        .filter((pl.col('cell_type').eq(cell_type)) & 
                (pl.col('contrast').eq('PREG_vs_CTRL')))\
        .sort('P.Value').head(15)
    with pl.Config(fmt_str_lengths=1000):
        print(filtered_df)
    
    print(f'\n{cell_type} - POSTPART_vs_PREG - pathway_de_results:')
    filtered_df = pathway_de_results\
        .filter((pl.col('cell_type').eq(cell_type)) & 
                (pl.col('contrast').eq('POSTPART_vs_PREG')))\
        .sort('P.Value').head(15)
    with pl.Config(fmt_str_lengths=1000):
        print(filtered_df)

pb_curio = SingleCell(adata_curio)\
    .qc(allow_float=True)\
    .filter_var(
        pl.col('protein_coding') & 
        pl.col('ribo').not_() &
        pl.col('mt').not_())\
    .with_columns_obs(
        pl.when(pl.col('condition').eq('CTRL')).then(0)
            .when(pl.col('condition').eq('PREG')).then(1)
            .otherwise(None)
            .alias('PREG_vs_CTRL'),
        pl.when(pl.col('condition').eq('PREG')).then(0)
            .when(pl.col('condition').eq('POSTPART')).then(1)
            .otherwise(None)
            .alias('POSTPART_vs_PREG'))\
    .pseudobulk('sample', 'subclass')

de_table = []
for contrast in ['PREG_vs_CTRL', 'POSTPART_vs_PREG']:
    pb_filt = pb_curio\
        .qc('condition',
            custom_filter=pl.col(contrast).is_not_null(),
            min_samples=2,
            min_cells=10,
            max_standard_deviations=None,
            min_nonzero_fraction=0,
            verbose=False)\
    .library_size(allow_float=True)
    formula = f'~ {contrast} + log2(num_cells)'
    excluded_cell_types = ['Ependymal NN'] if contrast == 'PREG_vs_CTRL' \
        else ['L6b EPd Glut', 'Ependymal NN', 'OT D3 Folh1 Gaba', 'Peri NN']
    de = pb_filt\
        .DE(formula,
            coefficient=contrast,
            group=contrast,
            excluded_cell_types=excluded_cell_types,
            allow_float=True,
            verbose=False,
            num_threads=1)
    de_table.append(
        de.table.with_columns(
        pl.lit(contrast).alias('contrast')))
    print_df(de.get_num_hits(threshold = 0.1)
             .sort('num_hits', descending=True))

de_table = pl.concat(de_table)

for cell_type in de_table.select('cell_type').unique().to_series():
    print(f"\n{cell_type} - de_table:")
    print(de_table.filter((pl.col('cell_type') == cell_type) & (pl.col('contrast') == 'PREG_vs_CTRL')).sort('p').head(10))

#endregion



