import os
import gc
import re
import pickle as pkl
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from single_cell import SingleCell, Pseudobulk
from utils import print_df
from ryp import r, to_r, to_py

working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'
cell_type_col = 'subclass'

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

import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
from adjustText import adjust_text
import matplotlib.gridspec as gridspec

FDR_CUTOFF = 0.1

GENES_TO_LABEL = {
    #--- PREG_vs_CTRL: Coordinated response to pregnancy ---
    ('PREG_vs_CTRL', 'Oligo NN'): ['Sgk1', 'Fkbp5', 'Hif3a', 'Ptgds'],
    ('PREG_vs_CTRL', 'Microglia NN'): ['Ncf1', 'Abi3', 'Fkbp5'],
    ('PREG_vs_CTRL', 'Astro-TE NN'): ['Cnr1', 'Fabp7'],
    ('PREG_vs_CTRL', 'L2/3 IT CTX Glut'): ['Zbtb18', 'Fkbp5'],
    ('PREG_vs_CTRL', 'L4/5 IT CTX Glut'): ['Hmgcs1', 'Dbi'],
    ('PREG_vs_CTRL', 'L5 IT CTX Glut'): ['Glul', 'Slc6a11'],
    ('PREG_vs_CTRL', 'LSX Nkx2-1 Gaba'): ['Prkca', 'Nfia'],
    ('PREG_vs_CTRL', 'LSX Prdm12 Zeb2 Gaba'): ['Npas4', 'Gfra1'],
    ('PREG_vs_CTRL', 'STR D1 Gaba'): ['Cnr1', 'Drd3'],
    ('PREG_vs_CTRL', 'Sst Chodl Gaba'): ['Nos1'],
    ('PREG_vs_CTRL', 'Pvalb Gaba'): ['Zbtb18'],

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
    ('POSTPART_vs_PREG', 'Lamp5 Gaba'): ['Npas3', 'Schip1']
}

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
        width_ratios=[10, 2, 10, 2], wspace=0.05
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

    for ax_main, ax_bar, contrast_name, title in axes_config:
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
        
        label_offset = 0.25
        
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
                    
                    for i, r in enumerate(ct_data):
                        ax_main.text(adjusted_x[i], label_y, r['gene'], 
                                    style='italic', fontsize=6,
                                    ha='center', va='bottom', zorder=150)
                        
                        ax_main.plot([r['logFC'], adjusted_x[i]], 
                                    [ct_y, label_y],
                                    'k-', lw=0.6, alpha=0.8, zorder=90)

        if contrast_name == 'PREG_vs_CTRL':
            bar_xlim = 250
        else:
            bar_xlim = 25
        
        ax_main.grid(True, 'major', 'y', ls='-', lw=0.5, c='lightgray', 
                     zorder=0)
        ax_main.axvline(0, color='lightgray', linestyle='-', linewidth=0.5, 
                        zorder=0)
        ax_main.set_xlim(-max_fc, max_fc)
        ax_bar.set_xlim(-bar_xlim, bar_xlim)
        
        y_ticks = range(len(group_cell_types))
        ax_main.set_yticks(y_ticks)
        ax_main.set_yticklabels(group_cell_types)
        ax_bar.set_yticks(y_ticks)
        ax_bar.set_yticklabels(group_cell_types)
        ax_main.set_ylim(len(group_cell_types)-0.5, -0.5)
        ax_bar.set_ylim(len(group_cell_types)-0.5, -0.5)
        
        if i == 0: ax_main.set_title(title, fontsize=14)
        if i < len(major_types) - 1:
            plt.setp(ax_main.get_xticklabels(), visible=False)
            plt.setp(ax_bar.get_xticklabels(), visible=False)
        else:
            ax_main.set_xlabel('log2(Fold Change)', fontsize=12)
            ax_bar.set_xlabel('# DEGs', fontsize=12)
            
        for ax in [ax_main, ax_bar]: ax.tick_params(length=0)
            
for i, (ax1, ax1_bar, ax2, ax2_bar) in enumerate(all_axes):
    plt.setp(ax1_bar.get_yticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax2_bar.get_yticklabels(), visible=False)
    ax1.set_ylabel(None)

cbar_ax = fig.add_axes([1.0, 0.65, 0.015, 0.2])
cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cbar.set_label('-log10(FDR)', size=12)

fig.tight_layout(rect=[0, 0, 0.98, 1])
os.makedirs(f'{working_dir}/figures', exist_ok=True)
plt.savefig(
    f'{working_dir}/figures/deg_landscape.png', dpi=300, bbox_inches='tight'
)
plt.close()







r('''
suppressPackageStartupMessages({
    library(msigdbr)
    library(GSVA)
    library(limma)
    library(dplyr)
    library(purrr)
    library(tibble)
})
m_df <- msigdbr(
    species = "Mus musculus", 
    collection = "C5", 
    subcategory = "GO:BP"
)
gobp_gene_sets <- m_df %>% 
    split(x = .$gene_symbol, f = .$gs_name)

run_differential_pathways <- function(
    pseudobulks, gene_sets, ref_level, contrast_str
) {
    imap_dfr(pseudobulks, function(element, cell_type_name) {
        tryCatch({
            gsvapar <- gsvaParam(
                element$counts, 
                gene_sets, 
                minSize = 10,
            )
            es <- gsva(gsvapar, verbose = FALSE)
            
            targets <- element$obs
            all_levels <- unique(as.character(targets$condition))
            other_level <- all_levels[all_levels != ref_level]
            targets$group <- factor(
                targets$condition, levels = c(ref_level, other_level)
            )
            if (n_distinct(targets$group) < 2) return(NULL)

            design <- model.matrix(~ group, data = targets)
            fit <- lmFit(es, design)
            fit <- eBayes(fit)
            
            topTable(fit, coef = 2, number = Inf, sort.by = "p") %>%
                as.data.frame() %>%
                rownames_to_column("pathway") %>%
                mutate(contrast = contrast_str)
        }, error = function(e) {
            warning(paste("Error in", cell_type_name, ":", e$message))
            return(NULL)
        })
    }, .id = "cell_type")
}

pathway_de_preg_ctrl <- run_differential_pathways(
    pseudobulks_preg_ctrl, gobp_gene_sets, "CTRL", "PREG_vs_CTRL"
)
pathway_de_postpart_preg <- run_differential_pathways(
    pseudobulks_postpart_preg, gobp_gene_sets, "PREG", "POSTPART_vs_PREG"
)
pathway_de_results <- bind_rows(
    pathway_de_preg_ctrl, pathway_de_postpart_preg) %>%
    select(pathway, cell_type, contrast, logFC, P.Value, adj.P.Val)
''')

pathway_de_results = to_py('pathway_de_results')

import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
from scipy.cluster.hierarchy import linkage, leaves_list
import pandas as pd
import numpy as np

df_filtered = pathway_de_results.clone()\
    .with_columns(
        pl.col('pathway').str.replace('GOBP_', '', literal=True),
        pl.col('adj.P.Val').add(1e-300).log10()
        .mul(pl.col('logFC').sign()).neg().alias('score'))\
    .filter(
        pl.col('adj.P.Val').lt(0.05).any().over('pathway'))

pivot_preg_ctrl = df_filtered\
    .filter(pl.col('contrast') == 'PREG_vs_CTRL')\
    .pivot(index='cell_type', on='pathway', values='score')\
    .to_pandas().set_index('cell_type').fillna(0)

pivot_postpart_preg = df_filtered\
    .filter(pl.col('contrast') == 'POSTPART_vs_PREG')\
    .pivot(index='cell_type', on='pathway', values='score')\
    .to_pandas().set_index('cell_type').fillna(0)

annot_df = adata.obs[[cell_type_col, 'type']]\
    .drop_duplicates()\
    .rename(columns={cell_type_col: 'cell_type', 'type': 'Major Type'})\
    .set_index('cell_type')

type_order = ['Glut', 'Gaba', 'NN']
final_row_order = []
for t in type_order:
    all_types_in_group = annot_df[annot_df['Major Type'] == t].index
    existing_types = pivot_preg_ctrl.index.intersection(all_types_in_group)
    if len(existing_types) == 0: continue
    
    group_data = pivot_preg_ctrl.loc[existing_types]
    if len(existing_types) > 1:
        link = linkage(group_data.fillna(0), method='average')
        order = leaves_list(link)
        final_row_order.extend(group_data.index[order])
    else:
        final_row_order.extend(existing_types)

combined_pivot = pivot_preg_ctrl.add(pivot_postpart_preg, fill_value=0)
col_link = linkage(combined_pivot.T.fillna(0), method='average')
final_col_order = combined_pivot.columns[leaves_list(col_link)]

mat1 = pivot_preg_ctrl.reindex(index=final_row_order, columns=final_col_order)
mat2 = pivot_postpart_preg.reindex(index=final_row_order, columns=final_col_order)

type_boundaries = annot_df.loc[final_row_order, 'Major Type']\
    .ne(annot_df.loc[final_row_order, 'Major Type'].shift()).cumsum()
gap_indices = np.where(type_boundaries.diff() > 0)[0]

def insert_gaps(df, indices):
    for i in sorted(indices, reverse=True):
        gap_row = [[np.nan] * df.shape[1]]
        df = pd.concat([
            df.iloc[:i],
            pd.DataFrame(gap_row, columns=df.columns, index=['']),
            df.iloc[i:]
        ])
    return df

mat1_gapped = insert_gaps(mat1, gap_indices)
mat2_gapped = insert_gaps(mat2, gap_indices)

fig, axes = plt.subplots(
    1, 2, figsize=(8, 12), sharey=True,
    gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.05}
)
max_val = 2

sns.heatmap(
    mat1_gapped, ax=axes[0], cmap='seismic', center=0,
    vmin=-max_val, vmax=max_val, xticklabels=False, cbar=False
)
axes[0].tick_params(left=False, right=False)
axes[0].set_title('PREG vs CTRL', fontsize=16)
axes[0].set_ylabel('Cell Type', fontsize=14)

sns.heatmap(
    mat2_gapped, ax=axes[1], cmap='seismic', center=0,
    vmin=-max_val, vmax=max_val, xticklabels=False, cbar=False
)
axes[1].tick_params(left=False, right=False)
axes[1].set_title('POSTPART vs PREG', fontsize=16)
axes[1].set_ylabel('')

norm = plt.Normalize(vmin=-max_val, vmax=max_val)
sm = plt.cm.ScalarMappable(cmap="seismic", norm=norm)
sm.set_array([])

cbar_ax = fig.add_axes([0.92, 0.6, 0.02, 0.2])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Significance Score', size=12)

os.makedirs(f'{working_dir}/figures', exist_ok=True)
plt.savefig(
    f'{working_dir}/figures/tmp.png', dpi=300, bbox_inches='tight')

plt.close()
























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