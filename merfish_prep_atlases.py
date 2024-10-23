import os, pandas as pd, numpy as np, anndata as ad, scanpy as sc
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import warnings
warnings.filterwarnings('ignore')

working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/data', exist_ok=True)

########################################################################

# load zeng reference, output from `merfish_zeng_prep_atlas.py`
# the X matrix is log CPM
ref_dir = 'projects/def-wainberg/single-cell/ABC'
cell_joined = pd.read_csv(f'{ref_dir}/Zeng/cells_joined.csv')

sections = ['C57BL6J-638850.49', 'C57BL6J-638850.48', 
            'C57BL6J-638850.47', 'C57BL6J-638850.46']
z_threshold = 5.5

data_types = [
    ('imputed', 
     'MERFISH-C57BL6J-638850-imputed/20240831/C57BL6J-638850-imputed-log2.h5ad', 
     'adata_ref_zeng_imputed.h5ad'),
    ('raw', 
     'MERFISH-C57BL6J-638850/20230830/C57BL6J-638850-raw.h5ad', 
     'adata_ref_zeng_raw.h5ad')
]
for data_type, input_path, output_filename in data_types:
    print(f'Processing {data_type} data...')
    adata_input = ad.read_h5ad(f'{ref_dir}/expression_matrices/{input_path}')
    
    adatas_processed = []
    for section in sections:
        adata = adata_input[adata_input.obs['brain_section_label'] == section]
        adata.obs = adata.obs.reset_index()
        adata.obs = pd.merge(adata.obs, cell_joined, on='cell_label', 
                             how='left')
        adata.obs = adata.obs.set_index('cell_label', drop=True)
        exclude = ['unassigned', 'brain-unassigned', 'fiber tracts-unassigned']
        adata = adata[~adata.obs['parcellation_division'].isin(exclude)]
        adata = adata[adata.obs['x_ccf'].notna()]
        adata.var = adata.var.reset_index()
        
        subset = adata[adata.obs['z_ccf'] > z_threshold]
        subset.obs['z_ccf'] *= -1
        subset.obs['y_ccf'] *= -1
        
        subset.obs['x'] = subset.obs['z_ccf']
        subset.obs['y'] = subset.obs['y_ccf']
        subset.obs['sample'] = f'{section}_R'
        subset.obs['source'] = 'Zeng-ABCA-Reference'
        print(f'[{section}] {adata.shape[0]} cells')
        adatas_processed.append(subset)

    adata_combined = ad.concat(adatas_processed, axis=0, merge='same')
    adata_combined.var = adata_input.var.reset_index().set_index('gene_symbol')
    adata_combined.var['gene_symbol'] = adata_combined.var.index
    adata_combined.var = adata_combined.var.rename_axis(None)
    adata_combined = adata_combined[:, 
                     ~adata_combined.var.index.duplicated(keep='first')]

    # save as sparse matrix
    adata_combined = adata_combined.copy()
    adata_combined.X = sparse.csr_matrix(adata_combined.X.astype(np.float32))
    adata_combined.write(f'{working_dir}/output/data/{output_filename}')
    print(f'Saved {data_type} data to {output_filename}')

# plot each slice 
for selection in [
    'class_color', 'subclass_color', 'supertype_color',
    'parcellation_division_color', 'parcellation_structure_color',
    'parcellation_substructure']:
    fig, axes = plt.subplots(1, 4, figsize=(25, 7))
    fig.suptitle('Zeng ABCA Reference', fontsize=16)
    for ax, (sample, data) in zip(axes, adata_combined.obs.groupby('sample')):
        ax.scatter(data['x'], data['y'], s=0.8, c=data[selection])
        ax.set_title(sample)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(
        f'{working_dir}/figures/reference/zeng_reference_{selection}.png',
        dpi=200, bbox_inches='tight', pad_inches=0)

########################################################

# load reference, output from `merfish_zhuang_prep_atlas.py`
# the X matrix is raw counts
ref_dir = 'projects/def-wainberg/spatial/Zhuang/direct-downloads'
datasets_ref = ['Zhuang-ABCA-1', 'Zhuang-ABCA-2']
samples_ref = [
    'Zhuang-ABCA-1.057', 'Zhuang-ABCA-1.058', 'Zhuang-ABCA-1.059',
    'Zhuang-ABCA-1.060', 'Zhuang-ABCA-1.061', 'Zhuang-ABCA-1.062',
    'Zhuang-ABCA-1.063', 'Zhuang-ABCA-1.064', 
    'Zhuang-ABCA-2.026', 'Zhuang-ABCA-2.027', 'Zhuang-ABCA-2.028',
    'Zhuang-ABCA-2.030']

# load and munge each dataset 
adatas_ref = []
for data in datasets_ref:
    metadata = pd.read_csv(
        f'{ref_dir}/{data}-metadata.csv', index_col='cell_label')
    adata = ad.read_h5ad(f'{ref_dir}/{data}-raw.h5ad')
    adata = adata[adata.obs.index.isin(metadata.index)]
    adata = adata[adata.obs['brain_section_label'].isin(samples_ref)]
    adata.obs['sample'] = adata.obs['brain_section_label']
    adata.obs['source'] = 'Zhuang-ABCA-Reference'
    adata.obs = adata.obs.join(
        metadata, on='cell_label', lsuffix='_l', rsuffix='')
    adata.obs['y'] = -adata.obs['y']
    adata.var.reset_index()
    adata.var.index = adata.var['gene_symbol']
    print(f'[{data}] {adata.shape[0]} cells')
    adatas_ref.append(adata)

# concat and store raw counts 
adata_ref = ad.concat(adatas_ref, axis=0, merge='same')
adata_ref.layers['counts'] = adata_ref.X.copy()
adata_ref.var['gene_symbol'] = adata_ref.var.index

# plot each slice 
for selection in [
    'class_color', 'subclass_color', 'supertype_color',
    'parcellation_division_color', 'parcellation_structure_color',
    'parcellation_substructure_color']:
    fig, axes = plt.subplots(2, 6, figsize=(38, 14))
    fig.suptitle('Zhuang ABCA Reference', fontsize=16)
    axes = axes.flatten()
    for ax, (sample, data) in zip(axes, adata_ref.obs.groupby('sample')):
        ax.scatter(data['x'], data['y'], s=1, c=data[selection])
        ax.set_title(sample)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(
        f'{working_dir}/figures/reference/zhuang_reference_{selection}.png',
        dpi=200, bbox_inches='tight', pad_inches=0)

# normalize 
adata_ref = adata_ref.copy()
sc.pp.normalize_total(adata_ref)
sc.pp.log1p(adata_ref, base=2)
adata_ref.X = adata_ref.X.astype(np.float32)  
adata_ref.X = sparse.csr_matrix(adata_ref.X)
# save
adata_ref.write(f'{working_dir}/output/data/adata_ref_zhuang.h5ad')