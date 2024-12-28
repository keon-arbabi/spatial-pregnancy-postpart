import sys
import scipy
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('project/utils')

from single_cell import SingleCell, options
options(num_threads=-1, seed=42)

working_dir = 'project/spatial-pregnancy-postpart'




import mygene 
mg = mygene.MyGeneInfo()

sc_query = SingleCell(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')

mapping = {
    r['query']: (r['ensembl']['gene'] if isinstance(r['ensembl'], dict) 
    else r['ensembl'][0]['gene']) 
    for r in mg.querymany(
        sc_query.var['_index'].to_list(), 
        scopes='symbol',
        fields='ensembl.gene',
        species='mouse')
    if 'ensembl' in r
}
mapping.update({
   'Il1f6': 'ENSMUSG00000026984',
   'Tdgf1': 'ENSMUSG00000032494', 
   'Il1f5': 'ENSMUSG00000026983',
   'Fcrls': 'ENSMUSG00000015852'
})

sc_query.var = sc_query.var\
    .with_columns(pl.col('_index').map_elements(
        lambda x: mapping.get(x, x), return_dtype=pl.String))
# temp file 
sc_query.save(
    f'{working_dir}/output/data/adata_query_merfish_tmp.h5ad')

df_cast = sc_query.obs
df_mmc = pl.read_csv(
    f'{working_dir}/output/data/mapper_output.csv',
    skip_rows=4)









import json
import numpy as np
from scipy.stats import median_abs_deviation

with open(f'{working_dir}/output/data/mapper_output.json') as f:
   data = json.load(f)

cells = [dict(m, cell_id=c['cell_id'], level=l) 
        for c in data['results'] 
        for l,m in c.items() if isinstance(m, dict)]

df = pd.DataFrame(cells)
subclass = df[df['level'].str.contains('SUBC')]

def plot_violin_by_sample(df, columns):
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    df['sample'] = df['cell_id'].str.split('_').str[0]
    fig, axes = plt.subplots(len(columns), 1, figsize=(8, 4*len(columns)))
    if len(columns) == 1:
        axes = [axes]
    for i, col in enumerate(columns):
        sns.violinplot(data=df, x='sample', y=col, ax=axes[i])
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
        axes[i].set_title(col)
    
    plt.tight_layout()
    return fig

plot_violin_by_sample(
    subclass,
    columns=['bootstrapping_probability', 'avg_correlation', 
            'aggregate_probability'])
plt.savefig(
    f'{working_dir}/figures/merfish/subclass_metrics_by_sample.png',
    dpi=300, bbox_inches='tight')


good_cells = subclass[
    (subclass['directly_assigned']) &
    (subclass['bootstrapping_probability'] > 0.9) & 
    (subclass['avg_correlation'] > 0.4)][['cell_id', 'assignment']].copy()

name_mapper = data['taxonomy_tree']['name_mapper']['CCN20230722_SUBC']
good_cells['subclass'] = good_cells['assignment'].map(
    lambda x: name_mapper[x]['name'])

import anndata as ad
import os

ref_obs = ad.read_h5ad(
    f'{working_dir}/output/data/adata_ref_final.h5ad').obs
query_obs = ad.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad').obs
query_obs_filt = query_obs[query_obs['cosine_knn_cdist'] < 0.3]

good_cells = good_cells.set_index('cell_id').reindex(query_obs.index)

unique_categories = ref_obs['subclass'].unique()
unique_colors = ref_obs['subclass_color'].unique() 
color_map = dict(zip(unique_categories, unique_colors))

good_cells['subclass_color'] = good_cells['subclass'].map(color_map)
good_cells['sample'] = query_obs['sample']
good_cells['x_final'] = query_obs['x_final']
good_cells['y_final'] = query_obs['y_final']
good_cells = good_cells.dropna(subset=['subclass'])

print(good_cells.shape[0])
print(query_obs_filt.shape[0])


def create_multi_sample_plot(ref_obs, query_obs, col, cell_type, output_dir):
    ref_samples = ref_obs['sample'].unique()
    query_samples = query_obs['sample'].unique()
    n_cols = 4
    n_rows = 1 + -(-len(query_samples) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.flatten()
    
    for i, (sample, obs, coord_cols) in enumerate(
        [(s, ref_obs, ['x', 'y']) for s in ref_samples] +
        [(s, query_obs, ['x_final', 'y_final']) for s in query_samples]):
        if i >= len(axes):
            break
        ax = axes[i]
        plot_df = obs[obs['sample'] == sample]
        mask = plot_df[col] == cell_type
        if mask.sum() > 0:
            # Plot background cells
            ax.scatter(
                plot_df[~mask][coord_cols[0]], plot_df[~mask][coord_cols[1]], 
                c='grey', s=1 if obs is ref_obs else 2, alpha=0.1)
            # Plot cells of interest with individual colors
            ax.scatter(
                plot_df[mask][coord_cols[0]], plot_df[mask][coord_cols[1]], 
                c=plot_df[mask][f'{col}_color'], s=1 if obs is ref_obs else 6)
        else:
            ax.text(0.5, 0.5, 'no cells of this type', ha='center', 
                    va='center', transform=ax.transAxes)
        ax.set_title(f'{sample}\n{col}: {cell_type}')
        ax.axis('off')
        ax.set_aspect('equal')
    
    for ax in axes[i+1:]:
        fig.delaxes(ax)
    plt.tight_layout()
    safe_filename = cell_type.replace('/', '_').replace(' ', '_')
    plt.savefig(f'{output_dir}/{safe_filename}.png', dpi=100, 
                bbox_inches='tight')
    plt.close(fig)

output_dir = f'{working_dir}/figures/merfish/spatial_cell_types_subclass_mmc_filt'
os.makedirs(output_dir, exist_ok=True)

cell_types = pd.concat([ref_obs['subclass'], good_cells['subclass']]).unique()
for cell_type in cell_types:
   if ref_obs['subclass'].value_counts().get(cell_type, 0) > 50:
       create_multi_sample_plot(
           ref_obs, good_cells, 'subclass', cell_type, output_dir)









plot_distributions(
    df_mmc_full[df_mmc_full['level'].str.contains('SUBC')], 
    columns=['bootstrapping_probability', 'avg_correlation',
             'aggregate_probability'])
plt.savefig(
    f'{working_dir}/figures/merfish/mmc_subclass_distributions.png', 
    dpi=300, bbox_inches='tight')


def confusion_matrix_plot(df_true, df_pred, true_label_column,
                          pred_label_column, file):
    if isinstance(df_true, pl.DataFrame):
        df_true = df_true.to_pandas()
    if isinstance(df_pred, pl.DataFrame):
        df_pred = df_pred.to_pandas()
    df_true[true_label_column] = df_true[true_label_column].astype(str)
    df_pred[pred_label_column] = df_pred[pred_label_column].astype(str)
    true_labels = df_true[true_label_column].values
    pred_labels = df_pred[pred_label_column].values
    unique_labels = np.union1d(true_labels, pred_labels)
    confusion_matrix = pd.DataFrame(
        0, index=unique_labels, columns=unique_labels)
    
    for true, pred in zip(true_labels, pred_labels):
        confusion_matrix.loc[true, pred] += 1
    confusion_matrix = confusion_matrix.loc[
        (confusion_matrix.sum(axis=1) != 0),
        (confusion_matrix.sum(axis=0) != 0)
    ]
    confusion_matrix = confusion_matrix.div(
        confusion_matrix.sum(axis=1), axis=0)
    
    plt.figure(figsize=(15, 12))
    ax = sns.heatmap(
        confusion_matrix, xticklabels=1, yticklabels=1,
        rasterized=True, square=True, linewidths=0.5,
        cmap='rocket_r', cbar_kws=dict(pad=0.01), vmin=0, vmax=1)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel(f'predicted ({pred_label_column})')
    plt.ylabel(f'true ({true_label_column})')
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    plt.tight_layout()
    plt.savefig(file, dpi=300, bbox_inches='tight')
    plt.close()

confusion_matrix_plot(
    df_true, df_pred, 'class', 'class_name',
    f'{working_dir}/figures/merfish/confusion_matrix_class.png')
confusion_matrix_plot(
    query_obs, query_obs_mmc, 'subclass', 'subclass',
    f'{working_dir}/figures/curio/confusion_matrix_subclass.png')





















sc_ref = SingleCell(
    'project/single-cell/ABC/anndata/zeng_combined_10Xv3.h5ad')\
    .make_var_names_unique()\
    .qc(custom_filter=pl.col('class').is_not_null(),
        allow_float=True)

'''
Starting with 2,349,544 cells.
Filtering to cells with ≤5.0% mitochondrial counts...
1,863,931 cells remain after filtering to cells with ≤5.0% mitochondrial counts.
Filtering to cells with ≥100 genes detected (with non-zero count)...
1,863,931 cells remain after filtering to cells with ≥100 genes detected.
Filtering to cells with non-zero MALAT1 expression...
1,863,912 cells remain after filtering to cells with non-zero MALAT1 expression.
'''

sc_query = SingleCell(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')\
    .qc(allow_float=True,
        max_mito_fraction=None,
        min_genes=10,
        nonzero_MALAT1=False)
sc_query_X = sc_query.X

'''
Starting with 132,183 cells.
Filtering to cells with ≤5.0% mitochondrial counts...
130,796 cells remain after filtering to cells with ≤5.0% mitochondrial counts.
Filtering to cells with ≥100 genes detected (with non-zero count)...
130,796 cells remain after filtering to cells with ≥100 genes detected.
Filtering to cells with non-zero MALAT1 expression...
130,627 cells remain after filtering to cells with non-zero MALAT1 expression.
'''

sc_query, sc_ref = sc_query.hvg(sc_ref, allow_float=True)
sc_query = sc_query.normalize(allow_float=True)
sc_ref = sc_ref.normalize(allow_float=True)
sc_query, sc_ref = sc_query.PCA(sc_ref)
sc_query, sc_ref = sc_query.harmonize(sc_ref)

sc_query = sc_query\
    .label_transfer_from(
        sc_ref, 
        original_cell_type_column='class',
        cell_type_column='class_dissoc',
        confidence_column='class_dissoc_confidence')

sc_query = sc_query\
    .hvg(allow_float=True, overwrite=True)\
    .normalize(allow_float=True)\
    .PCA()\
    .neighbors()\
    .embed()

sc_query.plot_embedding(
    color_column='class', 
    filename=f'{working_dir}/figures/curio/umap_class.png',
    cells_to_plot_column=None,
    legend_kwargs={
        'fontsize': 7,
        'loc': 'center left'},
    savefig_kwargs={'dpi': 600})

sc_query.save(f'{working_dir}/output/data/adata_query_curio_final_filt.h5ad',
              overwrite=True)
















mmc = pl.read_csv(
    f'{working_dir}/output/curio/data/curio_mmc_corr_annotations.csv')
sc_ref = SingleCell(
    f'{working_dir}/output/data/adata_ref_zeng_imputed.h5ad')\
    .with_uns(QCed=True)\
    .with_uns(normalized=True)\
    .filter_obs(pl.col.brain_section_label_x.is_in([
        'C57BL6J-638850.49', 'C57BL6J-638850.48', 
        'C57BL6J-638850.47', 'C57BL6J-638850.46']))



sc_query = SingleCell(
    f'{working_dir}/output/data/adata_query_curio_positioned_final.h5ad')\
    .with_uns(QCed=True)\
    .with_uns(normalized=False)\
    .filter_var(pl.col.gene.is_in(sc_ref.var['gene_symbol']))\
    .join_obs(mmc, left_on='_index', right_on='cell_id')


mdr = 0.05
mlfc = 1.2

query_cast_markers = sc_query.find_markers(
    cell_type_column='subclass',
    all_genes=True)\
    .cast({'cell_type': pl.String})\
    .filter((pl.col.detection_rate > mdr) & (pl.col.fold_change > mlfc))

query_mmc_markers = sc_query.find_markers(
    cell_type_column='subclass_right',
    all_genes=True)\
    .cast({'cell_type': pl.String})\
    .filter((pl.col.detection_rate > mdr) & (pl.col.fold_change > mlfc))

ref_markers = sc_ref.find_markers(
    cell_type_column='subclass',
    all_genes=True)\
    .cast({'cell_type': pl.String})\
    .filter((pl.col.detection_rate > mdr) & (pl.col.fold_change > mlfc))

matched_markers = ref_markers.join(
    query_cast_markers, 
    on=['cell_type', 'gene'], 
    how='inner')

print('corr ref vs query cast')
print(matched_markers.shape[0])
print(scipy.stats.spearmanr(
    matched_markers['fold_change'], 
    matched_markers['fold_change_right']))

# corr ref vs query cast
# 13879
# SignificanceResult(statistic=0.2965941996495267, pvalue=7.723594458794918e-280)

matched_markers = ref_markers.join(
    query_mmc_markers, 
    on=['cell_type', 'gene'], 
    how='inner')

print('corr ref vs query mmc')
print(matched_markers.shape[0])
print(scipy.stats.spearmanr(
    matched_markers['fold_change'], 
    matched_markers['fold_change_right']))

# corr ref vs query mmc
# 15600
# SignificanceResult(statistic=0.3865836709477447, pvalue=0.0)