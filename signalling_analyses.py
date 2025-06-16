import os
import re
import pandas as pd
import scanpy as sc
import squidpy as sq
import warnings
import logging
import omnipath as op

warnings.filterwarnings('ignore') 

cache_directory = os.path.expanduser('~/.omnipath_cache') 
if not os.path.exists(cache_directory):
    os.makedirs(cache_directory)
op.options.cache_dir = cache_directory
op.options.convert_dtypes = True 
op.interactions.OmniPath().get()

logging.info(f"OmniPath cache directory: {op.options.cache_dir}")

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

#region L-R ####################################################################

adata_curio.obs = adata_curio.obs[[
    'sample', 'condition', cell_type_col, f'{cell_type_col}_color']]
adata_curio.obs[cell_type_col] = pd.Categorical(adata_curio.obs[cell_type_col])
adata_curio.var = adata_curio.var[['gene_symbol']]

del adata_curio.uns, adata_curio.obsm, adata_curio.varm, \
    adata_curio.obsp, adata_curio.layers

sc.pp.normalize_total(adata_curio)

conditions = adata_curio.obs['condition'].unique()
lr_results = {}

condition = conditions[0]

for condition in conditions:
    lr_results[condition] = sq.gr.ligrec(
        adata_curio[adata_curio.obs['condition'] == condition].copy(),
        cluster_key=cell_type_col,
        transmitter_params={'categories': 'ligand'},
        receiver_params={'categories': 'receptor'},
        n_perms=1000,
        threshold=0.1,
        use_raw=False,
        copy=True)

#endregion 