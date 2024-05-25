import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import os, torch, CAST, warnings
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore")

# Prep raw images ####################################################################

data_dir = '../../spatial/Kalish/pregnancy-postpart'
work_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
os.chdir(work_dir)
os.makedirs('output', exist_ok=True)
os.makedirs('figures', exist_ok=True)
os.makedirs(f'{data_dir}/rotate-split-raw', exist_ok=True)

def rotate_and_crop(points, angle, x_min=None, x_max=None, mirror_y=False):
    theta = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = np.dot(points.to_numpy(), rotation_matrix)
    df = pd.DataFrame(rotated, index=points.index, columns=['x', 'y'])
    mask = (df['x'] > x_min if x_min is not None else df['x'] == df['x']) & \
           (df['x'] < x_max if x_max is not None else df['x'] == df['x'])
    cropped_df = df[mask]
    if mirror_y:
        cropped_df['x'] = -cropped_df['x']
    return cropped_df

sample_names = [
    'Ctrl1', 'Ctrl2', 'Ctrl3', 'Preg1', 'Preg2', 'Preg3',
    'PostPart1', 'PostPart2', 'PostPart3'] 
params = {
    'Ctrl1': {'L': (72, 6000), 'R': (70, 5200)}, 
    'Ctrl2': {'L': (110, 3200), 'R': (110, 2600)},
    'Ctrl3': {'L': (-33, 2200), 'R': (-33, 1800)}, 
    'Preg1': {'L': (3, 5800), 'R': (3, 5000)},
    'Preg2': {'L': (-98, -4900), 'R': (-98, -5400)},
    'Preg3': {'L': (-138, -5700), 'R': (-138, -6100)},
    'PostPart1': {'L': (75, 5800), 'R': (75, 5000)},
    'PostPart2': {'L': (115, 2600), 'R': (115, 1900)},
    'PostPart3': {'L': (-65, -1800), 'R': (-65, -2200)}}
plot_index = 1
plt.figure(figsize=(3 * 5, 6 * 4))

for sample in sample_names:
    adata = ad.read_h5ad(f'{data_dir}/raw/{sample}.hdf5')
    coords = adata.obs[['center_x', 'center_y']]
    for hemi in ['L', 'R']:
        plt.subplot(6, 3, plot_index)
        angle, value = params[sample][hemi]
        coords_hemi = rotate_and_crop(
            coords, angle=angle,
            x_max=value if hemi == 'L' else None,
            x_min=None if hemi == 'L' else value,
            mirror_y=(hemi == 'R'))
        adata_hemi = adata[coords_hemi.index]
        adata_hemi.obs[['x', 'y']] = coords_hemi
        adata_hemi.write(f'{data_dir}/rotate-split-raw/{sample}_{hemi}.h5ad')

        spines = plt.gca().spines
        spines['top'].set_visible(False)
        spines['right'].set_visible(False)
        sns.scatterplot(data=coords_hemi, x='x', y='y', color='black', s=0.2)
        plt.title(f'{sample} - {hemi}')
        plot_index += 1  
        
plt.tight_layout() 
plt.savefig('figures/crop_and_rotate_all_small.png', 
            dpi=200, bbox_inches='tight', pad_inches=0)

# CAST_MARK ####################################################################

work_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
output_dir = 'output/CAST-MARK'
os.chdir(work_dir)
os.makedirs(output_dir, exist_ok=True)

ref_dir = '../../spatial/Zhuang/direct-downloads'  
datasets_ref = ['Zhuang-ABCA-1', 'Zhuang-ABCA-2']
samples_ref = [
    'Zhuang-ABCA-1.057', 'Zhuang-ABCA-1.058', 'Zhuang-ABCA-1.059',
    'Zhuang-ABCA-1.060', 'Zhuang-ABCA-1.061', 'Zhuang-ABCA-1.062',
    'Zhuang-ABCA-2.026', 'Zhuang-ABCA-2.027', 'Zhuang-ABCA-2.028',
    'Zhuang-ABCA-2.030']  

adatas_ref = []
for data in datasets_ref:  
    metadata = pd.read_csv(
        f'{ref_dir}/{data}-metadata.csv', index_col='cell_label')
    adata = ad.read_h5ad(f'{ref_dir}/{data}-raw.h5ad')
    adata = adata[adata.obs.index.isin(metadata.index)]
    adata = adata[adata.obs['brain_section_label'].isin(samples_ref)]
    adata.obs['sample'] = adata.obs['brain_section_label']
    adata.obs = adata.obs.join(
        metadata, on='cell_label', lsuffix='_l', rsuffix='')
    adata.var.reset_index()
    adata.var.index = adata.var['gene_symbol']
    adatas_ref.append(adata)
adata_ref = ad.concat(adatas_ref)
adata_ref.var['gene_symbol'] = adata_ref.var.index

query_dir = '../../spatial/Kalish/pregnancy-postpart/rotate-split-raw'
samples_query = [file.replace('.h5ad', '') for file in os.listdir(query_dir)]

adatas_query = []
for sample in samples_query:
    adata = ad.read_h5ad(f'{query_dir}/{sample}.h5ad')
    adata.obs['sample'] = sample
    adatas_query.append(adata)
adata_query = sc.concat(adatas_query)
adata_query.var['gene_symbol'] = adata_query.var.index

adata_comb = ad.concat([adata_ref, adata_query])
all_var = [a.var for a in [adata_ref, adata_query]]
all_var = pd.concat(all_var, join='inner')
all_var = all_var[~all_var.duplicated()]
adata_comb.var = all_var.loc[adata_comb.var_names]

adata_comb.layers['norm'] = sc.pp.normalize_total(
    adata_comb, target_sum=1e4, inplace=False)['X']

sample_names = adata_comb.obs['sample'].unique()
coords_raw = {sample: np.array(adata_comb.obs[['x','y']])
              [adata_comb.obs['sample'] == sample] for sample in sample_names}
exp_dict = {sample: adata_comb[adata_comb.obs['sample'] == sample]
    .layers['norm'] for sample in sample_names}

embed_dict = CAST.CAST_MARK(coords_raw, exp_dict, output_dir)
embed_dict = torch.load(f'{output_dir}/demo_embed_dict.pt')

cluster_label = CAST.kmeans_plot_multiple(
    embed_dict, sample_names, coords_raw, 'all_samples', output_dir,
    k=30, dot_size = 1, minibatch=False)

# identify nuisance cluster surrounding slices as the cluster label
# commonly closest to the origin (0,0) across samples, and filter out 
idx = 0; label_dict = {}
for sample, tensor in embed_dict.items():
    count = tensor.shape[0]
    label_dict[sample] = cluster_label[idx:idx + count]
    idx += count

nuisance_label = [
    label_dict[sample][
        np.argmin((coords[:, 0] - 0)**2 + (coords[:, 1] - 0)**2)]
    for sample, coords in coords_raw.items()]
nuisance_label = max(set(nuisance_label), key=nuisance_label.count)

embed_dict = {
    sample: tensor[label_dict[sample] != nuisance_label]
    for sample, tensor in embed_dict.items()}
coords_raw = {
    sample: coords[label_dict[sample] != nuisance_label]
    for sample, coords in coords_raw.items()}
exp_dict = {
    sample: exp[label_dict[sample] != nuisance_label]
    for sample, exp in exp_dict.items()}

# CAST_MARK ####################################################################

adata_ref = ad.read_h5ad(
    '../../spatial/Zhuang/WB_MERFISH_animal1_coronal.h5ad')

adata_ref.obs['brain_section_label'].unique().tolist()


len(set(adata.var.index) & set(adata_ref.var['gene_name']))

import matplotlib.pyplot as plt, seaborn as sns 
import seaborn as sns

cell_types = adata_ref.obs['cell_type'].cat.categories
palette = sns.color_palette("tab20", len(cell_types))
color_dict = dict(zip(cell_types, palette))
colors = adata_ref.obs['cell_type'].map(color_dict)

sns.scatterplot(
    x=adata_ref.obsm['X_spatial'][:, 0], 
    y=adata_ref.obsm['X_spatial'][:, 1],w
    hue=adata_ref.obs['cell_type'], s=0.1)
plt.savefig('tmp.png')