import sys, gc, polars as pl, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import anndata as ad

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import SingleCell
from utils import Timer, savefig, print_df, debug

debug(third_party=True)

working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart/'

def confusion_matrix_plot(sc, original_labels_column,
                          transferred_labels_column, file):
    confusion_matrix = sc.obs\
        .select(original_labels_column, transferred_labels_column)\
        .to_pandas()\
        .groupby([original_labels_column, transferred_labels_column],
                 observed=True)\
        .size()\
        .unstack(fill_value=0)\
        .sort_index(axis=1)\
        .assign(broad_cell_type=lambda df: df.index.str.split('.').str[0],
                cell_type_cluster=lambda df: df.index.str.split('.').str[1]
                .astype('Int64').fillna(0))\
        .sort_values(['broad_cell_type', 'cell_type_cluster'])\
        .drop(['broad_cell_type', 'cell_type_cluster'], axis=1)
    print(confusion_matrix)
    ax = sns.heatmap(
        confusion_matrix.T.div(confusion_matrix.T.sum()),
        xticklabels=1, yticklabels=1, rasterized=True,
        square=True, linewidths=0.5, cmap='rocket_r',
        cbar_kws=dict(pad=0.01), vmin=0, vmax=1)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    w, h = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(3.5 * w, h)
    savefig(file)

levels = ['class', 'subclass', 'supertype', 'cluster']

with Timer('Loading and cleaning CURIO single cell'):
    adata_comb = ad.read_h5ad(
        f'{working_dir}/output/CURIO/data/adata_comb_project.h5ad')

    obs = adata_comb.obs
    ref_cell_types = {
        level: [str(ct) for ct in obs[
            obs['sample'] == "Zhuang-ABCA-1.060"][level]\
            .value_counts()[lambda x: x >= 10].index
        if str(ct) != 'Unknown']
        for level in levels
    }
    adata_comb = adata_comb[adata_comb.obs['source'] == "CURIO"]

    sc_query = ad.read_h5ad(
        f'{working_dir}/output/CURIO/data/adata_query.h5ad')
    sc_query.obs = adata_comb.obs 
    sc_query.obsm['CAST_MARK_embed'] = adata_comb.obsm['CAST_MARK_embed']
    sc_query = SingleCell(sc_query)\
        .qc(cell_type_confidence_column=None,
            doublet_column=None, allow_float=True)\
        .rename_obs({
            'class': 'class_cast', 'subclass': 'subclass_cast',
            'supertype': 'supertype_cast', 'cluster': 'cluster_cast',
            'class_color': 'class_color_cast', 
            'subclass_color': 'subclass_color_cast',
            'supertype_color': 'supertype_color_cast', 
            'cluster_color': 'cluster_color_cast'})
    del adata_comb; gc.collect()

'''
Starting with 50,560 cells.
Filtering to cells with ≤10.0% mitochondrial counts...
50,557 cells remain after filtering to cells with ≤10.0% mitochondrial counts.
Filtering to cells with ≥100 genes detected (with nonzero count)...
50,557 cells remain after filtering to cells with ≥100 genes detected.
'''

with Timer('Loading reference single cell'):
    sc_ref = SingleCell(
        'projects/def-wainberg/single-cell/Zeng/combined_10Xv3.h5ad',
        num_threads=None)\
        .set_var_names('gene_symbol')\
        .filter_var(pl.col.gene_symbol.is_unique())\
        .cast_var({'gene_symbol': pl.String})\
        .qc(cell_type_confidence_column=None,
            doublet_column=None, allow_float=True)
    for level in levels:   
        sc_ref = sc_ref\
        .with_columns_obs(pl.col.passed_QC & pl.col(level).is_not_null())\
        .with_columns_obs(pl.col.passed_QC & pl.col(level).len().over(
            pl.when('passed_QC').then(level)).ge(50))

'''
Starting with 2,349,544 cells.
Filtering to cells with ≤10.0% mitochondrial counts...
2,307,104 cells remain after filtering to cells with ≤10.0% mitochondrial counts.
Filtering to cells with ≥100 genes detected (with nonzero count)...
2,307,104 cells remain after filtering to cells with ≥100 genes detected.
'''

# cell type label transfer pipeline 
with Timer('Highly-variable genes'):
    sc_query, sc_ref = sc_query.hvg(sc_ref, allow_float=True, num_threads=None)

with Timer('Normalize'):
    sc_query = sc_query.normalize(allow_float=True, num_threads=None)
    sc_ref = sc_ref.normalize(allow_float=True, num_threads=None)

with Timer('PCA'):
    sc_query, sc_ref = sc_query.PCA(sc_ref, verbose=True, num_threads=None)

with Timer('Harmony'):
    sc_query, sc_ref = sc_query.harmonize(
        sc_ref, pytorch=True, num_threads=None)

with Timer('Label transfer'):
    for level in levels:   
        sc_query = sc_query\
            .label_transfer_from(
                sc_ref.filter_obs(pl.col(level).is_in(ref_cell_types[level])), 
                cell_type_column=level,
                cell_type_confidence_column=f'{level}_dissoc_confidence')\
            .filter_obs(pl.col(level).is_not_null())\
            .rename_obs({level: f'{level}_dissoc'})

with Timer('Confusion matrices'):  
    for level in levels:   
        confusion_matrix_plot(
            sc_query, 
            f'{level}_cast', f'{level}_dissoc', 
            f'{working_dir}/figures/CURIO/conf_matrix_{level}.png')    
        
with Timer('Add color columns'):  
    for level in levels:   
        sc_query = sc_query.join_obs(
            sc_ref.obs\
                .rename({'class_color': 'class_color_dissoc', 
                        'subclass_color': 'subclass_color_dissoc',
                        'supertype_color': 'supertype_color_dissoc', 
                        'cluster_color': 'cluster_color_dissoc'})\
                .select(pl.col(level).alias('tmp'),
                        pl.col(f'{level}_color_dissoc')).unique(),
            left_on=f'{level}_dissoc',
            right_on='tmp')

with Timer('Plot embeddings'):
    sc_query = sc_query.embed(num_threads=24)
    for level in levels:
        for type in ['cast', 'dissoc']:
            palette = dict(zip(sc_query.obs[f'{level}_{type}'], 
                               sc_query.obs[f'{level}_color_{type}']))
            sc_query.plot_embedding(
                color_column=f'{level}_{type}',
                filename=f'{working_dir}/figures/CURIO/'
                f'pacmap_{level}_{type}.png',
                palette=palette,
                legend=level == 'class',
                legend_kwargs={'fontsize': 'x-small'} 
                if level == 'class' else None)
        
sc_query.save(f'{working_dir}/output/CURIO/data/adata_query_labelled.h5ad', 
              overwrite=True)

sc_query = SingleCell(
    f'{working_dir}/output/CURIO/data/adata_query_labelled.h5ad')

level = 'parcellation_substructure'
type = 'dissoc'
df = sc_query.cast_obs({f'{level}': pl.String}).obs\
    .filter(sample='Preg1_1_L')
palette = dict(zip(df[f'{level}'], df[f'{level}_color']))

plt.clf()
plt.figure(figsize=((8, 8)))
ax  = sns.scatterplot(data=df, x='x_final', y='y_final', linewidth=0,
                hue=f'{level}', palette=palette, s=20, legend=False)
ax.set(xlabel=None, ylabel=None)
sns.despine(bottom = True, left = True)
plt.legend(fontsize=9, markerscale=3)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
savefig(f'{working_dir}/figures/CURIO/spat_{level}.png', dpi=200)

from scipy import sparse

sc_comb = ad.read_h5ad(
    f'{working_dir}/output/CURIO/data/adata_comb_project.h5ad')
sc_comb.X = sparse.csr_matrix(sc_comb.X)
sc_comb = SingleCell(sc_comb)

df = sc_comb.cast_obs({level: pl.String}).obs\
    .filter(sample='Zhuang-ABCA-1.060')
palette = dict(zip(df[level], df[f'{level}_color']))

plt.clf()
plt.figure(figsize=((8, 8)))
ax  = sns.scatterplot(data=df, x='x', y='y', linewidth=0,
                       hue=level, palette=palette, s=20, legend=True)
ax.set(xlabel=None, ylabel=None)
sns.despine(bottom = True, left = True)
plt.legend(fontsize=9, markerscale=3)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
savefig(f'{working_dir}/figures/CURIO/spat_{level}_ref.png', dpi=200)