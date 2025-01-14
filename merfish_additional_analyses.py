import sys
import anndata as ad
import scanpy as sc
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('project/utils')

from single_cell import SingleCell, options
options(num_threads=-1, seed=42)

working_dir = 'project/spatial-pregnancy-postpart'

# load query data
adata_query = ad.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')

# load reference data (imputed)
adata_ref = ad.read_h5ad(
    f'{working_dir}/output/data/adata_ref_zeng_imputed.h5ad')
adata_ref.var.index = adata_ref.var['gene_identifier']

# Find common genes
var_names = adata_ref.var_names.intersection(adata_query.var_names)
adata_ref = adata_ref[:, var_names]
adata_query = adata_query[:, var_names]

# Generate reference UMAP
sc.pp.pca(adata_ref)
sc.pp.neighbors(adata_ref)
sc.tl.umap(adata_ref)

# Project query onto reference
sc.tl.ingest(adata_query, adata_ref, embedding_method='umap')

# Plot UMAPs colored by subclass
color_dict_ref = dict(zip(
    adata_ref.obs['subclass'], adata_ref.obs['subclass_color']))
color_dict_query = dict(zip(
    adata_query.obs['subclass'], adata_query.obs['subclass_color']))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sc.pl.umap(adata_ref, 
           color='subclass',
           palette=color_dict_ref,
           ax=ax1, 
           title='Reference',
           show=False)
sc.pl.umap(adata_query, 
           color='subclass',
           palette=color_dict_query,
           ax=ax2, 
           title='Query',
           show=False)
plt.savefig(f'{working_dir}/figures/merfish/umap_subclass.png', dpi=300)



sc_ref = SingleCell(
    f'{working_dir}/output/data/adata_ref_zeng_imputed.h5ad')

sc_query = SingleCell(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad',
    X_key='layers/volume_log1p')











sc_query, sc_ref = scanorama.correct_scanpy([
    sc_query.to_anndata(), sc_ref.to_anndata()])
sc_query = SingleCell(sc_query); sc_ref = SingleCell(sc_ref)

sc_query, sc_ref = sc_query.PCA(
    sc_ref, allow_float=True, hvg_column=None)






