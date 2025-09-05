import gc
import polars as pl
import scanpy as sc
import pandas as pd
from single_cell import SingleCell
from utils import print_df

workdir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'

adata_curio = sc.read_h5ad(
    f'{workdir}/output/data/adata_query_curio_final.h5ad')
adata_merfish = sc.read_h5ad(
    f'{workdir}/output/data/adata_query_merfish_final.h5ad')
for adata in [adata_curio, adata_merfish]:
    for col in ['class', 'subclass']:
        adata.obs[col] = adata.obs[col].astype(str)\
            .str.extract(r'^(\d+)\s+(.*)', expand=False)[1]
common_cell_types = (
    set(adata_curio.obs[
        adata_curio.obs['subclass_keep']]['subclass'])
    & set(adata_merfish.obs[
        adata_merfish.obs['subclass_keep']]['subclass']))
adata_merfish = adata_merfish[
    adata_merfish.obs['subclass'].isin(common_cell_types)]

with pd.option_context('display.max_rows', None):
    print(adata_merfish.obs['subclass'].value_counts())
    print(adata_merfish.var[['gene_id', 'gene_symbol']])

del adata_curio, adata_merfish; gc.collect()

ref_disocc = SingleCell('single-cell/ABC/anndata/combined_10Xv3.h5ad')
query = SingleCell('single-cell/ABC/anndata/combined_10Xv3.h5ad')