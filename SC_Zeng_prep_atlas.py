import os, pandas as pd, anndata as ad, scanpy as sc
from pathlib import Path
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

download_base = Path('projects/def-wainberg/single-cell/Zeng/abc-project-cache')
abc_cache = AbcProjectCache.from_s3_cache(download_base)

abc_cache.list_metadata_files('WMB-10X')
abc_cache.list_metadata_files('WMB-taxonomy')
abc_cache.get_directory_metadata('WMB-taxonomy')
abc_cache.list_data_files('WMB-10Xv3')

# cell meta data ###############################################################

cell = abc_cache.get_metadata_dataframe(
    directory='WMB-10X',
    file_name='cell_metadata',
    dtype={'cell_label': str}
)
cell.set_index('cell_label', inplace=True)
print("Number of cells = ", len(cell))

cluster_details = abc_cache.get_metadata_dataframe(
    directory='WMB-taxonomy',
    file_name='cluster_to_cluster_annotation_membership_pivoted',
    keep_default_na=False
)
cluster_details.set_index('cluster_alias', inplace=True)

cluster_colors = abc_cache.get_metadata_dataframe(
    directory='WMB-taxonomy',
    file_name='cluster_to_cluster_annotation_membership_color'
)
cluster_colors.set_index('cluster_alias', inplace=True)

roi = abc_cache.get_metadata_dataframe(
    directory='WMB-10X', file_name='region_of_interest_metadata')
roi.set_index('acronym', inplace=True)
roi.rename(columns={'order': 'region_of_interest_order',
                    'color_hex_triplet': 'region_of_interest_color'},
           inplace=True)

cell_extended = cell.join(cluster_details, on='cluster_alias')
cell_extended = cell_extended.join(cluster_colors, on='cluster_alias')
cell_extended = cell_extended.join(
    roi[['region_of_interest_order', 'region_of_interest_color']], 
    on='region_of_interest_acronym')

cell_extended.to_csv(
    'projects/def-wainberg/single-cell/Zeng/cell_extended.csv')

gene = abc_cache.get_metadata_dataframe(directory='WMB-10X', file_name='gene')
gene.to_csv('projects/def-wainberg/single-cell/Zeng/gene.csv', index=False)

# cell expr data ###############################################################

download_base = 'projects/def-wainberg/single-cell/Zeng/abc-project-cache'
abc_cache = AbcProjectCache.from_s3_cache(download_base)

# download anndata objects (run once); must be online
files = abc_cache.list_data_files('WMB-10Xv3')
files = [f for f in files if 'raw' in f]
for f in files:
    abc_cache.get_data_path(directory='WMB-10Xv3', file_name=f)

# load combine into single anndata 
data_dir = 'projects/def-wainberg/single-cell/Zeng/' \
    'abc-project-cache/expression_matrices/WMB-10Xv3/20230630'
files = os.listdir(data_dir)
adatas = [ad.read_h5ad(f'{data_dir}/{f}') for f in files]
adata = sc.concat(adatas)

# load extended metadata and genes
cell_extended = pd.read_csv(
    'projects/def-wainberg/single-cell/Zeng/cell_extended.csv')\
    .set_index('cell_label')\
    .drop(['cell_barcode', 'library_label'], axis=1)
gene = pd.read_csv('projects/def-wainberg/single-cell/Zeng/gene.csv')

adata.obs = adata.obs.join(cell_extended, on='cell_label', how='left')
adata.var = pd.DataFrame({'gene_identifier': adata.var.index})
adata.var = adata.var.join(gene, on='gene_identifier', how='left')
adata.var['gene_identifier'] = adata.var['gene_identifier'].astype(str)
gene['gene_identifier'] = gene['gene_identifier'].astype(str)
adata.var = adata.var.merge(gene, on='gene_identifier', how='left')

# save
adata.write('projects/def-wainberg/single-cell/Zeng/combined_10Xv3.h5ad')


