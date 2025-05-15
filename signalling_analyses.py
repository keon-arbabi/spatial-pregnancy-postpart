import re
import gc
import pandas as pd
import scanpy as sc
import polars as pl
from single_cell import SingleCell
from ryp import r, to_r, to_py

#region load data ##############################################################

working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'

cell_type_col = 'subclass'

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
for level in color_mappings:
    color_mappings[level] = {
        k.split(' ', 1)[1]: v for k, v in color_mappings[level].items()
}

adata_curio = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_curio_final.h5ad')

adata_merfish = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')
adata_merfish.var.index = adata_merfish.var['gene_symbol']

adata_curio.X = adata_curio.layers['log1p'].copy()
adata_merfish.X = adata_merfish.layers['volume_log1p'].copy()

common_subclasses_numbered = (
    set(adata_curio.obs[adata_curio.obs['subclass_keep']]['subclass'])
    & set(adata_merfish.obs[adata_merfish.obs['subclass_keep']]['subclass']))

subclass_map = {}
for subclass in common_subclasses_numbered:
    if isinstance(subclass, str) and re.match(r'^\d+\s+', subclass):
        clean_name = re.sub(r'^\d+\s+', '', subclass)
        subclass_map[clean_name] = subclass

for adata in [adata_curio, adata_merfish]:
    for col in ['class', 'subclass']:
        adata.obs[col] = adata.obs[col].astype(str)\
            .str.extract(r'^(\d+)\s+(.*)', expand=False)[1]

common_subclasses = (
    set(adata_curio.obs[adata_curio.obs['subclass_keep']]['subclass'])
    & set(adata_merfish.obs[adata_merfish.obs['subclass_keep']]['subclass']))

common_cell_types = (
    set(adata_curio.obs[
        adata_curio.obs[f'{cell_type_col}_keep']][cell_type_col])
    & set(adata_merfish.obs[
        adata_merfish.obs[f'{cell_type_col}_keep']][cell_type_col]))

adata_curio = adata_curio[
    adata_curio.obs[cell_type_col].isin(common_cell_types)].copy()
adata_merfish = adata_merfish[
    adata_merfish.obs[cell_type_col].isin(common_cell_types)].copy()

#endregion 

del adata_curio.uns, adata_curio.obsm, adata_curio.varm, \
    adata_curio.obsp, adata_curio.layers

sample = 'CTRL_1'
SingleCell(adata_curio)\
    .filter_obs(pl.col('sample').eq(sample))\
    .to_seurat('sobj', v3=True)

r('''
  library(liana)
  library(Seurat)
  
  liana_test <- liana_wrap(sobj, idents_col = 'subclass')

''')

res = to_py('liana_test')
