import os
import gc
import re
import warnings
import requests
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
from collections import Counter
from goatools.obo_parser import GODag
from matplotlib.cm import ScalarMappable
from single_cell import SingleCell
from matplotlib.colors import Normalize, ListedColormap
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram, fcluster

warnings.filterwarnings('ignore')

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'
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
            continue
        
        var_mask = var['gene_symbol'].is_in(genes_to_keep)
        var_filtered = var.filter(var_mask)
        if var_filtered.height == 0:
            continue
        
        gene_indices = var.with_row_index().filter(var_mask)['index']
        X_filtered = X[:, gene_indices]
        
        to_r(cell_type, 'cell_type')
        to_r(X_filtered, 'X', colnames=var_filtered['gene_symbol'])
        to_r(obs, 'obs')
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
    #--- PREG_vs_CTRL: Coordinated response to pregnancy ---
    ('PREG_vs_CTRL', 'Oligo NN'): ['Sgk1', 'Hif3a', 'Zbtb16'],
    ('PREG_vs_CTRL', 'Microglia NN'): ['Ccnd3', 'Fkbp5'],
    ('PREG_vs_CTRL', 'Astro-TE NN'): ['Phyhd1', 'Fkbp5', 'Cnr1'],
    ('PREG_vs_CTRL', 'L2/3 IT CTX Glut'): ['Zbtb18', 'Glul', 'Ckb', 'Cox6c'],
    ('PREG_vs_CTRL', 'L4/5 IT CTX Glut'): ['Glul', 'Aldoc', 'Cox6c', 'Dbi'],
    ('PREG_vs_CTRL', 'L5 IT CTX Glut'): ['Tshz2'],
    ('PREG_vs_CTRL', 'L6 CT CTX Glut'): ['Sdk1'],
    ('PREG_vs_CTRL', 'L6b CTX Glut'): ['Tshz2'],
    ('PREG_vs_CTRL', 'L6b EPd Glut'): ['Rnf121'],
    ('PREG_vs_CTRL', 'LSX Nkx2-1 Gaba'): ['Prkca', 'Pld5', 'Dach2', 'Nfia'],
    ('PREG_vs_CTRL', 'LSX Prdm12 Zeb2 Gaba'): ['Cpa6', 'Hs6st3'],
    ('PREG_vs_CTRL', 'STR D1 Gaba'): ['Cnr1', 'Rgs4', 'Drd3'],
    ('PREG_vs_CTRL', 'STR D2 Gaba'): ['Cnr1', 'Rgs4'],
    ('PREG_vs_CTRL', 'STR Prox1 Lhx6 Gaba'): ['Adarb2'],
    ('PREG_vs_CTRL', 'Sst Gaba'): ['Dbi', 'Camk2n1'],
    ('PREG_vs_CTRL', 'Sst Chodl Gaba'): ['Sdk1', 'Sst'],
    ('PREG_vs_CTRL', 'Pvalb Gaba'): ['Zbtb18'],
    ('PREG_vs_CTRL', 'Vip Gaba'): ['Col25a1'],
    ('PREG_vs_CTRL', 'Ependymal NN'): ['Pcdh15'],

    #--- POSTPART_vs_PREG: Rebound and adaptation after birth ---
    ('POSTPART_vs_PREG', 'Oligo NN'): ['Man1a', 'Slc38a2'],
    ('POSTPART_vs_PREG', 'Endo NN'): ['Dbp', 'Nr1d1'],
    ('POSTPART_vs_PREG', 'L2/3 IT CTX Glut'): ['Gpc5', 'Dhcr24'],
    ('POSTPART_vs_PREG', 'L4/5 IT CTX Glut'): ['Insig1', 'Adamtsl1'],
    ('POSTPART_vs_PREG', 'L5 IT CTX Glut'): ['Tshz2', 'Nrp1'],
    ('POSTPART_vs_PREG', 'L6b EPd Glut'): ['Tafa1', 'Car10'],
    ('POSTPART_vs_PREG', 'LSX Nkx2-1 Gaba'): ['Kcnip4', 'Kcnc2'],
    ('POSTPART_vs_PREG', 'Sst Gaba'): ['Cartpt'],
    ('POSTPART_vs_PREG', 'Sst Chodl Gaba'): ['Vgf', 'Rnf220'],
    ('POSTPART_vs_PREG', 'STR D2 Gaba'): ['Slit2', 'Nr4a3'],
    ('POSTPART_vs_PREG', 'Lamp5 Gaba'): ['Npas3', 'Schip1'],
    ('POSTPART_vs_PREG', 'Ependymal NN'): ['Pcdh15']
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
        counts = deg_counts.filter(pl.col('contrast').eq(contrast_name))\
            .to_pandas().set_index('cell_type')\
            .reindex(group_cell_types).fillna(0)
            
        y_pos = {ct: i for i, ct in enumerate(group_cell_types)}
        contrast_y = [y_pos[ct] for ct in contrast_df['cell_type']]

        facecolors = cmap(norm(contrast_df['log10_fdr']))
        scatter = ax_main.scatter(
            x=contrast_df['logFC'], y=contrast_y, s=15,
            facecolors=facecolors, edgecolors='gray', linewidth=0.5, zorder=10
        )
        ax_bar.barh(range(len(group_cell_types)), counts['up'], left=0,
                    color=up_color, alpha=0.8)
        ax_bar.barh(range(len(group_cell_types)), counts['down'],
                    left=-counts['down'], color=down_color, alpha=0.8)
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
        ax_bar.set_xlim(-bar_xlim, bar_xlim)
        
        for idx, ct in enumerate(group_cell_types):
            if ct in counts.index:
                total_degs = int(counts.loc[ct, 'up'] + counts.loc[ct, 'down'])
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
    m_df <- msigdbr(species = "Mus musculus", category = "C5")
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
pathway_de_results_gsea\
    .drop('leadingEdge')\
    .filter(pl.col('padj') < 0.01)\
    .write_csv(f'{working_dir}/output/data/pathway_results_gsea_sig.csv')


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
                       'Lamp5 Gaba', 'LSX Nkx2-1 Gaba'],
        'pathways': ['GOBP_CELLULAR_RESPIRATION',
                     'GOBP_GLIOGENESIS',
                     'GOBP_REGULATION_OF_SYNAPTIC_PLASTICITY',
                     'GOBP_CELL_CELL_ADHESION_VIA_PLASMA_MEMBRANE_ADHESION_MOLECULES',
                     'GOBP_PROTON_TRANSMEMBRANE_TRANSPORT',
                     'GOBP_NEURON_PROJECTION_ORGANIZATION'],
        'labels': ['Cellular Respiration',
                   'Gliogenesis',
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

fig = plt.figure(figsize=(7, 14))
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
                    es_mat[i, j] = row['ES']
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
cbar.set_label('Enrichment Score', fontsize=10)
cbar.ax.tick_params(labelsize=9)

plt.savefig(f'{working_dir}/figures/dotplot_hybrid.png', 
            dpi=300, bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/dotplot_hybrid.svg',
            bbox_inches='tight')
plt.close()

#endregion






































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
        rowwise() %>%
        mutate(theme = get_theme(gs_name, theme_keywords)) %>%
        ungroup() %>%
        filter(!is.na(theme))
    
    saveRDS(m_df_themed, cache_file)
} else {
    m_df_themed <- readRDS(cache_file)
}

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




for cell_type, (X, obs, var) in pb_curio.items():
    to_r(cell_type, 'cell_type')
    to_r(X, 'X', colnames=var['gene_symbol'], rownames=obs['sample'])
    to_r(obs, 'obs', rownames=obs['sample'])
    to_r(var, 'var')

    r('''
    print(cell_type)
    head(obs, 10)
    head(var, 10)
    head(X, 10)
    ''')

de_table = pl.read_csv('DE_results_EdgeR.csv')


