import os
import scvi
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as sp

os.environ['SCIPY_ARRAY_API'] = '1'
import scarches as sca

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

condition_colors = {
    'CTRL': '#7209b7',
    'PREG': '#b5179e',
    'POSTPART': '#f72585'
}
modality_colors = {
    'merfish': '#4361ee',
    'curio': '#4cc9f0'
}

#region load data #############################################################

working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'
cell_type_col = 'subclass'

adata_curio = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_curio_final.h5ad')

adata_merfish = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')
adata_merfish.var.index = adata_merfish.var['gene_symbol']

for adata in [adata_curio, adata_merfish]:
    for col in ['class', 'subclass']:
        adata.obs[col] = adata.obs[col].astype(str)\
            .str.extract(r'^(\d+)\s+(.*)', expand=False)[1]

adata_curio.obs = adata_curio.obs[
    ['sample', 'condition', 'source', 'x_ffd', 'y_ffd', 'class', 'subclass']]
adata_curio.var = adata_curio.var[['gene_symbol']]
adata_curio.var.index.name = None
del adata_curio.uns, adata_curio.varm, adata_curio.obsp

adata_merfish.var.index = adata_merfish.var['gene_symbol']
adata_merfish.obs = adata_merfish.obs[
    ['sample', 'condition', 'source', 'x_ffd', 'y_ffd', 'class', 'subclass']]
adata_merfish.var = adata_merfish.var[['gene_symbol']]
adata_merfish.var.index.name = None
del adata_merfish.uns, adata_merfish.varm, adata_merfish.obsp

ad_ref = adata_curio[adata_curio.obs['subclass'].isin([
    'Endo NN', 'Endo N'])
].copy()
ad_query = adata_merfish[
    adata_merfish.obs['subclass'].isin(['Endo NN', 'Endo N'])
].copy()

#endregion

#region endo analyses #########################################################

model_dir = 'spatial-pregnancy-postpart/output/curio'
ref_model_dir = f'{model_dir}/scvi/ref'
resolutions = [0.1, 0.2, 0.3]
os.makedirs(ref_model_dir, exist_ok=True)

ad_ref = adata_curio[
    adata_curio.obs['subclass'] == 'Endo NN'
].copy()
ad_query = adata_merfish[
    adata_merfish.obs['subclass'] == 'Endo NN'
].copy()

ad_ref.layers['counts'] = ad_ref.X.copy()
sc.pp.normalize_total(ad_ref, target_sum=1e4)
sc.pp.log1p(ad_ref)

if not os.path.exists(os.path.join(ref_model_dir, 'model.pt')):
    scvi.model.SCVI.setup_anndata(ad_ref, layer='counts')
    ref_model = scvi.model.SCVI(ad_ref)
    ref_model.train()
    ref_model.save(ref_model_dir, overwrite=True)
else:
    ref_model = scvi.model.SCVI.load(ref_model_dir, adata=ad_ref)

ad_ref.obsm['X_scvi'] = ref_model.get_latent_representation()
sc.pp.neighbors(ad_ref, use_rep='X_scvi')
sc.tl.umap(ad_ref)

for res in resolutions:
    sc.tl.leiden(ad_ref, resolution=res, key_added=f'leiden_res_{res}')
    sc.tl.rank_genes_groups(ad_ref, f'leiden_res_{res}', use_raw=False)

for res in resolutions:
    res_key = f'leiden_res_{res}'
    score_cols = []
    markers_df = sc.get.rank_genes_groups_df(ad_ref, group=None)

    for cluster_id in ad_ref.obs[res_key].cat.categories:
        score_col = f'score_{res_key}_c{cluster_id}'
        score_cols.append(score_col)
        
        ref_markers = markers_df[
            markers_df['group'] == cluster_id
        ]['names'].tolist()
        
        query_markers = list(set(ref_markers) & set(ad_query.var_names))
        
        if len(query_markers) > 0:
            sc.tl.score_genes(ad_query, gene_list=query_markers, score_name=score_col)
        else:
            ad_query.obs[score_col] = 0

    ad_query.obs[res_key] = ad_query.obs[score_cols].idxmax(axis=1)\
        .str.replace(f'score_{res_key}_c', '')

cluster_keys = [f'leiden_res_{res}' for res in resolutions]

sc.pl.umap(
    ad_ref,
    color=['condition'] + cluster_keys,
    ncols=3
)
plt.savefig(
    f'{working_dir}/figures/endo_umap_reference.png', bbox_inches='tight')
plt.close()

sc.pp.neighbors(ad_query)
sc.tl.umap(ad_query)
sc.pl.umap(
    ad_query,
    color=['condition'] + cluster_keys,
    ncols=3
)
plt.savefig(
    f'{working_dir}/figures/endo_umap_projected_signature.png', bbox_inches='tight')
plt.close()


del ad_ref.uns
sc_ref = SingleCell(ad_ref)

print_df(sc_ref.with_uns(QCed = True).find_markers('leiden_res_0.2'))






























model_dir = 'spatial-pregnancy-postpart/output/curio'
ref_model_dir = f'{model_dir}/scvi/ref'
query_model_dir = f'{model_dir}/scvi/query'
resolutions = [0.1, 0.2, 0.3, 0.5, 1.0]
os.makedirs(ref_model_dir, exist_ok=True)
os.makedirs(query_model_dir, exist_ok=True)

ad_ref = adata_curio[
    adata_curio.obs['subclass'] == 'Endo NN'
].copy()
ad_query = adata_merfish[
    adata_merfish.obs['subclass'] == 'Endo NN'
].copy()

if not os.path.exists(os.path.join(ref_model_dir, 'model.pt')):
    scvi.model.SCVI.setup_anndata(ad_ref)
    ref_model = scvi.model.SCVI(ad_ref)
    ref_model.train()
    ref_model.save(ref_model_dir, overwrite=True)
else:
    ref_model = scvi.model.SCVI.load(ref_model_dir, adata=ad_ref)

scvi.model.SCVI.prepare_query_anndata(ad_query, ref_model_dir)

if not os.path.exists(os.path.join(query_model_dir, 'model.pt')):
    query_model = sca.models.SCVI.load_query_data(
        ad_query, ref_model_dir)
    query_model.train(max_epochs=200)
    query_model.save(query_model_dir, overwrite=True)
else:
    query_model = sca.models.SCVI.load(
        query_model_dir, adata=ad_query)

ad_query.obsm['X_scvi_projected'] = query_model.get_latent_representation()

for res in resolutions:
    knn = KNeighborsClassifier()
    knn.fit(
        ad_ref.obsm['X_scvi'],
        ad_ref.obs[f'leiden_res_{res}']
    )
    ad_query.obs[f'leiden_res_{res}'] = knn.predict(
        ad_query.obsm['X_scvi_projected']
    )

cluster_keys = [f'leiden_res_{res}' for res in resolutions]

sc.pl.umap(
    ad_ref,
    color=['source'] + cluster_keys,
    ncols=3
)
plt.savefig(
    f'{working_dir}/figures/endo_umap_reference.svg', bbox_inches='tight')
plt.close()

sc.tl.umap(ad_query, min_dist=0.1)
sc.pl.umap(
    ad_query,
    color=['source'] + cluster_keys,
    ncols=3
)
plt.savefig(
    f'{working_dir}/figures/endo_umap_projected.svg', bbox_inches='tight')
plt.close()























adata_full = ad_ref.concatenate(ad_query)
adata_full.obsm['X_scvi'] = query_model.get_latent_representation(adata_full)

sc.pp.neighbors(adata_full, use_rep='X_scvi')
sc.tl.umap(adata_full)

for res in resolutions:
    sc.tl.leiden(adata_full, resolution=res, key_added=f'leiden_res_{res}')

cluster_keys = ['batch'] + [f'leiden_res_{res}' for res in resolutions]

sc.pl.umap(
    adata_full,
    color=cluster_keys,
    ncols=3
)
plt.savefig(
    f'{working_dir}/figures/endo_umap.svg', bbox_inches='tight')
plt.close()

















common_genes = adata_curio.var_names.intersection(adata_merfish.var_names)
adata_c = adata_curio[:, common_genes].copy()
adata_m = adata_merfish[:, common_genes].copy()

adata = adata_c.concatenate(adata_m)

adata.layers['counts'] = adata.X.copy()
scvi.model.SCVI.setup_anndata(adata, layer='counts', batch_key='source')

model = scvi.model.SCVI(adata, n_latent=30)
model.train()

# 3. Find and visualize shared subclusters
adata.obsm['X_scvi'] = model.get_latent_representation()
sc.pp.neighbors(adata, use_rep='X_scvi')
sc.tl.leiden(adata, resolution=1.5, key_added='subclusters')
sc.tl.umap(adata)

sc.pl.umap(
    adata,
    color=['source', 'condition', 'subclusters'],
    title=['Technology', 'Condition', 'Shared Subclusters']
)

# 4. Quantify cluster composition
tech_dist = pd.crosstab(adata.obs.subclusters, adata.obs.source)
print('Technology distribution in clusters:')
print(tech_dist)

condition_dist = pd.crosstab(adata.obs.subclusters, adata.obs.condition)
print('\nCondition distribution in clusters:')
print(condition_dist)

# 5. Find marker genes for new subclusters
de_df = model.differential_expression(groupby='subclusters')




















common_genes = adata_curio.var_names.intersection(adata_merfish.var_names)
adata_curio_subset = adata_curio[:, common_genes].copy()
adata_merfish_subset = adata_merfish[:, common_genes].copy()

adata_combined = adata_curio_subset.concatenate(
    adata_merfish_subset, batch_key='source')

sc.pp.normalize_total(adata_combined, target_sum=1e4)
sc.pp.log1p(adata_combined)
sc.pp.pca(adata_combined)

sc.external.pp.bbknn(adata_combined, batch_key='source')

sc.tl.leiden(adata_combined, resolution=1.0)
sc.tl.umap(adata_combined)
sc.pl.umap(adata_combined, color=['source', 'leiden'])
plt.savefig(
    f'{working_dir}/figures/endo_umap.svg', bbox_inches='tight')
plt.close()


#endregion
