import os, warnings
import numpy as np, pandas as pd, anndata as ad
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings('ignore')

# Prep raw images ##############################################################

# set paths 
data_dir = 'projects/def-wainberg/spatial/Kalish/pregnancy-postpart/curio'
working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
os.makedirs(f'{data_dir}/rotate-split-raw', exist_ok=True)

# function for rotating, cropping, and mirroring 
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

# rotation and cropping parameters for each sample
sample_names = [
    'PP_1_1', 'PP_1_2', 'PP_2_1', 'PP_2_2', 
    'Preg_1_1', 'Preg_1_2', 'Preg_2_1', 'Preg_2_2',
    'Virg_1_1', 'Virg_1_2', 'Virg_2_1', 'Virg_3_1', 'Virg_3_2'] 
params = {
    'PP_1_1': {'L': (-110, -7500), 'R': (-110, -8300)},
    'PP_1_2': {'L': (-110, -7500), 'R': (-110, -8300)},
    'PP_2_1': {'L': (90, 6600), 'R': (90, 5500)},
    'PP_2_2': {'L': (90, 6600), 'R': (90, 5500)},
    'Preg_1_1': {'L': (180, -6200), 'R': (180, -7200)}, 
    'Preg_1_2': {'L': (180, -6200), 'R': (180, -7200)},
    'Preg_2_1': {'L': (5, 7400), 'R': (5, 6500)},
    'Preg_2_2': {'L': (5, 7400), 'R': (10, 6500)},
    'Virg_1_1': {'L': (-55, -100), 'R': (-55, -1000)},
    'Virg_1_2': {'L': (-55, -100), 'R': (-55, -1000)},
    'Virg_2_1': {'L': (140, -800), 'R': (140, -1600)},
    'Virg_3_1': {'L': (10, 7600), 'R': (10, 6900)},
    'Virg_3_2': {'L': (10, 7600), 'R': (10, 6900)}}

# rotate, crop, and mirror
# plot outliers and cells
# save rotated and cropped anndata objects 
fig1, axes1 = plt.subplots(8, 4, figsize=(3*5, 6*4)) 
fig2, axes2 = plt.subplots(8, 4, figsize=(3*5, 6*4))
axes1 = axes1.flatten()
axes2 = axes2.flatten()
ax_index = 0

for sample in sample_names:
    from sklearn.cluster import DBSCAN
    adata = ad.read_h5ad(f'{data_dir}/raw-h5ad/{sample}.h5ad')
    coords = adata.obs[['SPATIAL_1', 'SPATIAL_2']]
    if sample in ['Virg_1_1', 'Virg_1_2']:
        outliers = DBSCAN(eps=500, min_samples=150).fit(coords) 
    else:
        outliers = DBSCAN(eps=500, min_samples=90).fit(coords) 

    for hemi in ['L', 'R']:
        angle, value = params[sample][hemi]
        coords_rotated = rotate_and_crop(
            coords, angle=angle, x_max=value if hemi == 'L' else None,
            x_min=None if hemi == 'L' else value, mirror_y=(hemi == 'R'))
        mask = coords.index.isin(coords_rotated.index)
        outliers_labels_subset = outliers.labels_[mask]

        for ax in [axes1[ax_index], axes2[ax_index]]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title(f'{sample} - {hemi}')
            if ax.figure == fig1:
                sns.scatterplot(
                    data=coords_rotated[outliers_labels_subset == -1],
                    x='x', y='y', color='blue', s=1, ax=ax)
            sns.scatterplot(
                data=coords_rotated[outliers_labels_subset != -1],
                x='x', y='y', color='black', s=1, ax=ax)
        ax_index += 1
            
        adata_hemi = adata[mask][outliers_labels_subset != -1]
        adata_hemi.obs[['x', 'y']] = coords_rotated[
            outliers_labels_subset != -1]
        print(f'[{sample}] Removed {sum(outliers_labels_subset == -1)} points.')
        print(f'[{sample}] {adata_hemi.shape[0]} cells')
        adata_hemi.write(f'{data_dir}/rotate-split-raw/{sample}_{hemi}.h5ad')

fig1.tight_layout()
fig2.tight_layout()
fig1.savefig(f'{working_dir}/figures/curio/crop_and_rotate_outliers.png',
             dpi=200, bbox_inches='tight', pad_inches=0)
fig2.savefig(f'{working_dir}/figures/curio/crop_and_rotate.png',
             dpi=200, bbox_inches='tight', pad_inches=0)

# CAST_MARK ####################################################################

import sys, os, torch, CAST, warnings
from ryp import r, to_py
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import matplotlib.pyplot as plt, seaborn as sns
from scipy import sparse
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import SingleCell

# set paths
data_dir = 'projects/def-wainberg/spatial'
working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/curio/CAST-MARK', exist_ok=True)
os.makedirs(f'{working_dir}/output/curio/data', exist_ok=True)

# load rotated and cropped query anndata objects 
query_dir = f'{data_dir}/Kalish/pregnancy-postpart/curio/rotate-split-raw'
samples_query = [file.replace('.h5ad', '') for file in os.listdir(query_dir)]
samples_query = sorted(samples_query)

# munge each sample, adding placeholders for metadata columns to be added 
adatas_query = []
for sample in samples_query:
    adata = ad.read_h5ad(f'{query_dir}/{sample}.h5ad')
    adata.obs['sample'] = sample
    adata.obs['source'] = 'curio'
    adata.obs[[
        'class', 'class_color', 'subclass', 'subclass_color',
        'supertype', 'supertype_color', 'cluster', 'cluster_color',
        'parcellation_division', 'parcellation_division_color',
        'parcellation_structure', 'parcellation_structure_color',
        'parcellation_substructure', 
        'parcellation_substructure_color']] = 'Unknown'
    adata.obs = adata.obs.drop(columns=[
        'Row.names', 'cells', 'nCount_RNA', 'nFeature_RNA', 'number_clusters',
        'percent.mt', 'log10_nCount_RNA', 'log10_nFeature_RNA', 'nCount_SCT',
        'nFeature_SCT'])
    adata.obs.index = adata.obs.index.astype(str) + '_' + \
        adata.obs['sample'].astype(str)
    print(f'[{sample}] {adata.shape[0]} cells')
    adatas_query.append(adata)

# concat and store raw counts 
adata_query = sc.concat(adatas_query, axis=0, merge='same')
adata_query.layers['counts'] = adata_query.X.copy()
adata_query.var['gene_symbol'] = adata_query.var.index

# get qc metrics 
sc.pp.calculate_qc_metrics(adata_query, inplace=True)
# plot qc metrics 
cols = ['n_genes_by_counts', 'total_counts']
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for ax, col in zip(axs, cols):
    data = adata_query.obs[col].values
    x = np.sort(data)
    ax.plot(x, np.arange(1, len(data) + 1) / len(data))
    p5, p95 = np.percentile(data, [5, 95])
    for p in [p5, p95]:
        ax.axvline(p, color='r', linestyle='--')
        ax.text(p, 0.5, f'{p:.2f}', rotation=90, va='center', ha='left')
    ax.set(title=col, xlabel='Value', ylabel='Cumulative Probability')
plt.tight_layout()
plt.savefig(f'{working_dir}/figures/curio/qc_counts_genes.png')

# detect doublets 
# https://github.com/plger/scDblFinder
file = f'{working_dir}/output/curio/data/coldata.csv'
if os.path.exists(file):
    # add doublet metrics 
    coldata = pd.read_csv(f'{working_dir}/output/curio/data/coldata.csv')
    adata_query.obs = coldata.set_index('index')
else:
    SingleCell(adata_query)\
        .save(f'{working_dir}/output/curio/data/adata_query.rds', 
              sce=True, overwrite=True)
    r('''
    library(scDblFinder)
    library(BiocParallel)
    path = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart/'
    sce = readRDS(paste0(path, 'output/curio/data/adata_query.rds'))
    sce = scDblFinder(sce, samples='sample', BPPARAM=MulticoreParam())
    table(sce$scDblFinder.class)
    coldata = as.data.frame(colData(sce))
    ''')
    coldata = to_py('coldata', format='pandas')
    adata_query.obs = coldata
    coldata.to_csv(file)

# plot doublet scores 
cols = ['scDblFinder.score', 'scDblFinder.weighted', 'scDblFinder.cxds_score']
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for ax, col in zip(axs, cols):
    data = adata_query.obs[col].values
    x = np.sort(data)
    ax.plot(x, np.arange(1, len(data) + 1) / len(data))
    p5, p95 = np.percentile(data, [5, 95])
    for p in [p5, p95]:
        ax.axvline(p, color='r', linestyle='--')
        ax.text(p, 0.5, f'{p:.2f}', rotation=90, va='center', ha='left')
    ax.set(title=col, xlabel='Value', ylabel='Cumulative Probability')
plt.tight_layout()
plt.savefig(f'{working_dir}/figures/curio/qc_doublets.png')

# normalize 
# sc.pp.normalize_total(adata_query, target_sum=1e4)
# save
adata_query.write(f'{working_dir}/output/curio/data/adata_query.h5ad')

######################################

# load reference, output from `merfish_zhuang_prep_atlas.py`
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

# plot reference data 
fig, axes = plt.subplots(2, 6, figsize=(30, 14))
fig.suptitle('Zhuang ABCA Reference', fontsize=16)
axes = axes.flatten()
for ax, (sample, data) in zip(axes, adata_ref.obs.groupby('sample')):
    ax.scatter(
        data['x'], data['y'],
        s=1, c=data['parcellation_division_color'])
    ax.set_title(sample)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig(f'{working_dir}/figures/curio/zhuang_reference.png',
            dpi=200, bbox_inches='tight', pad_inches=0)

# normalize 
# sc.pp.normalize_total(adata_ref, target_sum=1e4)
adata_ref.X = adata_ref.X.astype(np.float32)  
adata_ref.X = sparse.csr_matrix(adata_ref.X)
# save
adata_ref.write(f'{working_dir}/output/curio/data/adata_ref_zhuang.h5ad')

######################################

# function for plotting 
def plot_slices(sample_names, coords_raw, cell_label, cluster_pl, n_clust):
    num_plot = len(sample_names)
    plot_row = int(np.ceil(num_plot / 5))
    plt.figure(figsize=(30, 3.5 * plot_row))

    cell_start_idx = 0
    for j, sample in enumerate(sample_names):
        plt.subplot(plot_row, 5, j+1)
        coords = coords_raw[sample]
        n_cells = coords.shape[0]
        cell_type = cell_label[cell_start_idx:cell_start_idx + n_cells]
        cell_start_idx += n_cells
        size = np.log(1e4 / n_cells) + 3
        plt.scatter(
            coords[:, 0], coords[:, 1],
            c=cell_type, cmap=plt.cm.colors.ListedColormap(cluster_pl),
            s=size, edgecolors='none')
        plt.title(f'{sample} (KMeans, k = {n_clust})', fontsize=20)
        plt.axis('equal')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    return plt.gcf()

# load preprocessed anndata objects 
adata_query = ad.read_h5ad(
    f'{working_dir}/output/curio/data/adata_query.h5ad')
# load reference created in `merfish_cast_analyses.py`
adata_ref = ad.read_h5ad(
    f'{working_dir}/output/curio/data/adata_ref_zhuang.h5ad')

# combine, keepingly only the same metadat columns and genes
sample_names = sorted(set(adata_query.obs['sample'].unique()) |
                      set(adata_ref.obs['sample'].unique()))
# CRUCIAL, must maintain cell and sample order  
adata_comb = ad.concat([adata_query, adata_ref], axis=0, merge='same')
adata_comb.obs['sample'] = pd.Categorical(
    adata_comb.obs['sample'], categories=sample_names, ordered=True)
adata_comb = adata_comb[adata_comb.obs.sort_values('sample').index]
adata_comb = adata_comb.copy()

# normalize 
sc.pp.normalize_total(adata_comb, target_sum=1e4)

# get coordinates and expression data for each sample
coords_raw = {
    s: np.array(adata_comb.obs[['x', 'y']])[adata_comb.obs['sample']==s]
    for s in sample_names}
exp_dict = {
    s: adata_comb[adata_comb.obs['sample']==s].X.toarray() 
    for s in sample_names}

# run cast mark
from CAST.models.model_GCNII import Args
embed_dict = CAST.CAST_MARK(
    coords_raw, exp_dict, 
    f'{working_dir}/output/curio/CAST-MARK',
    graph_strategy='delaunay', 
    args = Args(
        dataname='curio', # name of the dataset, used to save the log file
        gpu = 0, # gpu id, set to zero for single-GPU nodes
        epochs=400, # number of epochs for training
        lr1=1e-3, # learning rate
        wd1=0, # weight decay
        lambd=1e-3, # lambda in the loss function, refer to online methods
        n_layers=9, # number of GCNII layers, more layers mean a deeper model,
                    # larger reception field, at cost of VRAM usage and time
        der=0.5, # edge dropout rate in CCA-SSG
        dfr=0.3, # feature dropout rate in CCA-SSG
        use_encoder=True, # perform single-layer dimension reduction before 
                          # GNNs, helps save VRAM and time if gene panel large
        encoder_dim=512 # encoder dimension, ignore if use_encoder is False
    )
)

# detach from gpu and stack 
embed_dict = {
    k: v.cpu().detach() for k, v in embed_dict.items()}
embed_stacked = np.vstack([
    embed_dict[name].numpy() for name in sample_names])

n_clust = 30

# k-means cluster on cast embeddings
# and plot 
for n_clust in list(range(6, 20 + 1, 2)) + [30, 40, 50]:
    print(f'Clustering with k={n_clust}')
    kmeans = KMeans(n_clusters=n_clust, random_state=0)\
        .fit(embed_stacked)
    cell_label = kmeans.labels_
    cluster_pl = sns.color_palette('Set3', n_clust)
    
    fig = plot_slices(
        sample_names, coords_raw, cell_label, cluster_pl, n_clust)
    fig.savefig(f'{working_dir}/figures/curio/all_samples_k{str(n_clust)}.png')
    plt.close(fig)
    
    adata_comb.obs[f'k{n_clust}_cluster'] = cell_label
    color_map = {k: color for k, color in enumerate(cluster_pl.as_hex())}
    adata_comb.obs[f'k{n_clust}_cluster_colors'] = \
        pd.Series(cell_label).map(color_map).tolist()

# save
torch.save(coords_raw, f'{working_dir}/output/curio/data/coords_raw.pt')
torch.save(exp_dict, f'{working_dir}/output/curio/data/exp_dict.pt')
torch.save(embed_dict, f'{working_dir}/output/curio/data/embed_dict.pt')
adata_comb.write(f'{working_dir}/output/curio/data/adata_comb_cast_mark.h5ad')




















def subsample_reference(exp_dict, coords_raw, target_cells=3000, 
n_replicates=4):
    new_exp, new_coords = {}, {}
    for s in exp_dict:
        if not s.startswith('C57BL6J'):
            new_exp[s], new_coords[s] = exp_dict[s], coords_raw[s]
        else:
            cells = exp_dict[s].shape[0]
            cells_per_rep = min(target_cells, cells // n_replicates)
            idx = np.random.permutation(cells)
            for i in range(n_replicates):
                name = f'{s}_rep{i+1}'
                start = i * cells_per_rep
                end = start + cells_per_rep
                sel = idx[start:end]
                new_exp[name] = exp_dict[s][sel]
                new_coords[name] = coords_raw[s][sel]
    return new_exp, new_coords

def collapse_replicates(exp_dict, coords_raw, embed_dict):
    new_exp, new_coords, new_embed = {}, {}, {}
    for s in list(exp_dict.keys()):
        if '_rep' not in s:
            new_exp[s] = exp_dict[s]
            new_coords[s] = coords_raw[s]
            new_embed[s] = embed_dict[s]
        else:
            base = s.rsplit('_rep', 1)[0]
            if base not in new_exp:
                reps = [r for r in exp_dict if r.startswith(f"{base}_rep")]
                new_exp[base] = np.vstack([exp_dict[r] for r in reps])
                new_coords[base] = np.vstack([coords_raw[r] for r in reps])
                new_embed[base] = torch.cat([embed_dict[r] for r in reps], dim=0)
    return new_exp, new_coords, new_embed

def plot_slices(sample_names, coords_raw, cell_label, cluster_pl, n_clust):
    num_plot = len(sample_names)
    plot_row = int(np.ceil(num_plot / 5))
    plt.figure(figsize=(30, 3.5 * plot_row))

    cell_start_idx = 0
    for j, sample in enumerate(sample_names):
        plt.subplot(plot_row, 5, j+1)
        coords = coords_raw[sample]
        n_cells = coords.shape[0]
        cell_type = cell_label[cell_start_idx:cell_start_idx + n_cells]
        cell_start_idx += n_cells
        size = np.log(1e4 / n_cells) + 3
        plt.scatter(
            coords[:, 0], coords[:, 1],
            c=cell_type, cmap=plt.cm.colors.ListedColormap(cluster_pl),
            s=size, edgecolors='none')
        plt.title(f'{sample} (KMeans, k = {n_clust})', fontsize=20)
        plt.axis('equal')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    return plt.gcf()

# load preprocessed anndata objects 
adata_query = ad.read_h5ad(
    f'{working_dir}/output/curio/data/adata_query.h5ad')
# load reference created in `merfish_cast_analyses.py`
adata_ref = ad.read_h5ad(
    f'{working_dir}/output/merfish/data/adata_ref.h5ad')

# combine, keepingly only the same metadat columns and genes
sample_names = sorted(set(adata_query.obs['sample'].unique()) |
                      set(adata_ref.obs['sample'].unique()))
# CRUCIAL
adata_comb = ad.concat([adata_query, adata_ref], axis=0, merge='same')
adata_comb.obs['sample'] = pd.Categorical(
    adata_comb.obs['sample'], categories=sample_names, ordered=True)
adata_comb = adata_comb[adata_comb.obs.sort_values('sample').index]
adata_comb = adata_comb.copy()





# remove batch effect of technology
sc.pp.combat(adata_comb, key='source')

sc.pp.pca(adata_comb, n_comps=50)
sc.external.pp.harmony_integrate(adata_comb, key='source')


# get the coordinates and expression data for each sample
coords_raw = {
    s: np.array(adata_comb.obs[['x', 'y']])[adata_comb.obs['sample']==s]
    for s in sample_names}
exp_dict = {
    s: adata_comb[adata_comb.obs['sample']==s].X.toarray() 
    for s in sample_names}

exp_dict = {
    s: adata_comb[adata_comb.obs['sample']==s].obsm['X_pca_harmony'].toarray()
    for s in sample_names
}

# subsample reference to avoid imbalance in cell number
exp_dict_subsampled, coords_raw_subsampled = \
    subsample_reference(exp_dict, coords_raw)

# run cast mark
from CAST.models.model_GCNII import Args
embed_dict_subsampled = CAST.CAST_MARK(
    coords_raw_subsampled, exp_dict_subsampled, 
    f'{working_dir}/output/curio/CAST-MARK',
    graph_strategy='delaunay', 
    args = Args(
        dataname='curio', # name of the dataset, used to save the log file
        gpu = 0, # gpu id, set to zero for single-GPU nodes
        epochs=400, # number of epochs for training
        lr1=1e-3, # learning rate
        wd1=0, # weight decay
        lambd=1e-3, # lambda in the loss function, refer to online methods
        n_layers=9, # number of GCNII layers, more layers mean a deeper model,
                    # larger reception field, at cost of VRAM usage and time
        der=0.5, # edge dropout rate in CCA-SSG
        dfr=0.3, # feature dropout rate in CCA-SSG
        use_encoder=False, # perform single-layer dimension reduction before 
                          # GNNs, helps save VRAM and time if gene panel large
        encoder_dim=512 # encoder dimension, ignore if use_encoder is False
    )
)

coords_raw_collapsed, exp_dict_collapsed, embed_dict_collapsed = \
    collapse_replicates(coords_raw_subsampled, exp_dict_subsampled, 
                        embed_dict_subsampled)

# detach from gpu and stack 
embed_dict_collapsed = {
    k: v.cpu().detach() for k, v in embed_dict_collapsed.items()}
embed_stacked = np.vstack([
    embed_dict_collapsed[name].numpy() for name in sample_names])

n_clust = 10
print(f'Clustering with k={n_clust}')
kmeans = KMeans(n_clusters=n_clust, random_state=0)\
    .fit(embed_stacked)
cell_label = kmeans.labels_
cluster_pl = sns.color_palette('Set3', n_clust)

fig = plot_slices(
    sample_names, coords_raw_collapsed, cell_label, cluster_pl, n_clust)
fig.savefig(f'{working_dir}/figures/curio/test4.png')
plt.close(fig)

# k-means cluster on cast embeddings
# and plot 
for n_clust in list(range(6, 20 + 1, 2)) + [30, 40, 50]:
    print(f'Clustering with k={n_clust}')
    kmeans = KMeans(n_clusters=n_clust, random_state=0)\
        .fit(embed_stacked)
    cell_label = kmeans.labels_
    cluster_pl = sns.color_palette('Set3', n_clust)
    
    fig = plot_slices(
        sample_names, coords_raw_collapsed, cell_label, cluster_pl, n_clust)
    fig.savefig(f'{working_dir}/figures/curio/all_samples_k{str(n_clust)}.png')
    plt.close(fig)
    
    adata_comb.obs[f'k{n_clust}_cluster'] = cell_label
    color_map = {k: color for k, color in enumerate(cluster_pl.as_hex())}
    adata_comb.obs[f'k{n_clust}_cluster_colors'] = \
        pd.Series(cell_label).map(color_map).tolist()

# save
torch.save(coords_raw, f'{working_dir}/output/curio/data/coords_raw.pt')
torch.save(exp_dict, f'{working_dir}/output/curio/data/exp_dict.pt')
torch.save(embed_dict, f'{working_dir}/output/curio/data/embed_dict.pt')
adata_comb.write(f'{working_dir}/output/curio/data/adata_comb_cast_mark.h5ad')

# CAST_STACK ###################################################################

import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import sys, os, torch, CAST, warnings
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings('ignore')

# set paths 
working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/curio/CAST-STACK', exist_ok=True)

def plot_slices(adata_comb, coords, n_clust):
    color_col = f'k{n_clust}_cluster_colors'
    samples = coords.keys()
    num_samples = len(samples)
    rows = int(np.ceil(num_samples / 5))
    fig, axes = plt.subplots(rows, 5, figsize=(25, 5*rows))
    axes = axes.flatten()
    for ax, sample in zip(axes, samples):
        sample_data = adata_comb[adata_comb.obs['sample'] == sample]
        coords_i = coords[sample]
        colors = sample_data.obs[color_col]
        ax.scatter(coords_i[:, 0], coords_i[:, 1], 
                   c=colors, s=1, edgecolor='none')
        ax.set_title(sample)
        ax.axis('off')
    for ax in axes[num_samples:]:
        ax.axis('off')
    plt.tight_layout()
    return plt.gcf()

# load data 
adata_comb = ad.read_h5ad(f'{working_dir}/output/curio/data/'
                          'adata_comb_cast_mark.h5ad')
coords_raw = torch.load(
    f'{working_dir}/output/curio/data/coords_raw.pt')
embed_dict = torch.load(
    f'{working_dir}/output/curio/data/embed_dict.pt')

query_reference_list = {}
for key in coords_raw:
    if not key.startswith('C57BL6J-638850'):
        query_reference_list[key] = [key, 'C57BL6J-638850.47_R']

# run cast-stack, parameters modified for default are commented 
coords_final = {}
for sample in sorted(query_reference_list.keys()):
    params_dist = CAST.reg_params(
        dataname = query_reference_list[sample],
        gpu = 0 if torch.cuda.is_available() else -1, 
        diff_step = 5,
        #### Affine parameters
        iterations=200, # 500
        dist_penalty1=0,
        bleeding=500,
        d_list=[3,2,1,1/2,1/3],
        attention_params=[None,3,1,0],
        #### FFD parameters                                    
        dist_penalty2=[0],
        alpha_basis_bs=[100], # 500
        meshsize=[8],
        iterations_bs=[100], # 400
        attention_params_bs=[[None,3,1,0]],
        mesh_weight = [None])
    params_dist.alpha_basis = torch.Tensor(
        [1/1000, 1/1000, 1/50, 5, 5]).reshape(5, 1).to(params_dist.device)

    coords_final[sample] = CAST.CAST_STACK(
        coords_raw, embed_dict, f'{working_dir}/output/curio/CAST-STACK',
        query_reference_list[sample], params_dist, rescale=True)
    print(coords_final[sample])

# save
torch.save(coords_final, f'{working_dir}/output/curio/data/coords_final.pt')

# add final coords to anndata object 
sample_names = list(coords_final.keys())
sample_names = sorted(sample_names)
cell_index = adata_comb.obs.index[adata_comb.obs['source'] == 'curio']
coords_stack = np.vstack([
    coords_final[sample] for sample in sample_names])
coords_df = pd.DataFrame(
    coords_stack, columns=['x_final', 'y_final'], index=cell_index)
adata_comb.obs = adata_comb.obs.join(coords_df)

plot_slices(adata_comb, coords_final, n_clust=16)
plt.savefig(f'{working_dir}/figures/curio/all_samples_k16_final.png',
            dpi=300)

adata_comb.write(
    f'{working_dir}/output/curio/data/adata_comb_cast_stack.h5ad')

# CAST_PROJECT #################################################################

import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import sys, os, torch, pickle, warnings
import matplotlib.pyplot as plt, seaborn as sns

sys.path.insert(0, 'projects/def-wainberg/karbabi/CAST')
import CAST
print(CAST.__file__)

warnings.filterwarnings('ignore')

# set paths 
working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/merfish/CAST-PROJECT', exist_ok=True)

# transfer region labels by nearest neighbors and majority vote
def transfer_region_labels(coords_target, coords_source, source_region_labels, 
                           k=10):
    from sklearn.neighbors import NearestNeighbors

    coords_target = np.array(coords_target, dtype=np.float64)
    coords_source = np.array(coords_source, dtype=np.float64)
    source_region_labels = np.array(source_region_labels)

    nn_target = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn_target.fit(coords_target)
    _, knn_indices = nn_target.kneighbors(coords_target)

    nn_source = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn_source.fit(coords_source)
    _, closest_source_indices = nn_source.kneighbors(coords_target)
    closest_source_indices = closest_source_indices.flatten()

    neighbor_labels = source_region_labels[closest_source_indices[knn_indices]]
    unique_labels = np.unique(source_region_labels)

    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    neighbor_labels_int = np.vectorize(label_to_int.get)(neighbor_labels)

    label_counts = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=len(unique_labels)), 
        axis=1, arr=neighbor_labels_int)
    transferred_labels = unique_labels[np.argmax(label_counts, axis=1)]

    return transferred_labels

# load data
coords_final = torch.load(
    f'{working_dir}/output/merfish/data/coords_final.pt', map_location='cpu')
adata_comb = ad.read_h5ad(
    f'{working_dir}/output/merfish/data/adata_comb_cast_stack.h5ad')
# scale data separately for each source 
sc.pp.scale(adata_comb, zero_center = True, 
            mask_obs=(adata_comb.obs['source'] == 'Zeng-ABCA-Reference'))
sc.pp.scale(adata_comb, zero_center = True, 
            mask_obs=(adata_comb.obs['source'] == 'merfish'))
adata_comb.layers['log1p_norm_scaled'] = adata_comb.X.copy()

source_target_list = {
    key: ['C57BL6J-638850.47_R', key] for key in coords_final.keys()
}
batch_key = 'sample'
level = 'class'

color_dict = (
    adata_comb.obs
    .drop_duplicates()
    .set_index(level)[f'{level}_color']
    .to_dict()
)
color_dict['Unknown'] = '#A9A9A9'

list_ts = {}
for sample in source_target_list.keys():
    print(f'Processing {sample}')
    source_sample, target_sample = source_target_list[sample]
    output_dir_t = f'{working_dir}/output/merfish/CAST-PROJECT/' \
        f'{source_sample}_to_{target_sample}'
    os.makedirs(output_dir_t, exist_ok=True)
    
    # check if precomputed Harmony exists
    harmony_file = f'{output_dir_t}/' \
        f'X_harmony_{source_sample}_to_{target_sample}.h5ad'
    if os.path.exists(harmony_file):
        print(f'Loading precomputed harmony from {harmony_file}')
        adata_subset = ad.read_h5ad(harmony_file)
    else:
        print('Computing harmony')
        # subset the data for the current source-target pair
        adata_subset = adata_comb[
            np.isin(adata_comb.obs[batch_key], [source_sample, target_sample])]
        
        # use the correct harmony function from cast
        adata_subset = CAST.Harmony_integration(
            sdata_inte=adata_subset,
            scaled_layer='log1p_norm_scaled',
            use_highly_variable_t=False,
            batch_key=batch_key,
            n_components=50,
            umap_n_neighbors=15,
            umap_n_pcs=30,
            min_dist=0.1,
            spread_t=1.0,
            source_sample_ctype_col=level,
            output_path=output_dir_t,
            ifplot=False,
            ifcombat=True)
        # save harmony-adjusted data to disk
        adata_subset.write_h5ad(harmony_file)
    
    # run CAST_PROJECT
    print(f'Running CAST_PROJECT for {sample}')
    _, list_ts[sample] = CAST.CAST_PROJECT(
        sdata_inte=adata_subset,
        source_sample=source_sample,
        target_sample=target_sample,
        coords_source=np.array(
            adata_subset[adata_subset.obs[batch_key] == source_sample,:]
                .obs.loc[:,['x','y']]),
        coords_target=np.array(
            adata_subset[adata_subset.obs[batch_key] == target_sample,:]
                .obs.loc[:,['x_final','y_final']]),
        k2=1,
        scaled_layer='log1p_norm_scaled',
        raw_layer='log1p_norm_scaled',
        batch_key=batch_key,
        use_highly_variable_t=False,
        ifplot=False,
        source_sample_ctype_col=level,
        output_path=output_dir_t,
        umap_feature='X_umap',
        pc_feature='X_pca_harmony',
        integration_strategy=None, 
        ave_dist_fold=20,
        alignment_shift_adjustment=0,
        color_dict=color_dict,
        adjust_shift=False,
        metric_t='cosine',
        working_memory_t=1000
    )
    print(list_ts[sample])

# with open(f'{working_dir}/output/merfish/data/list_ts.pickle', 'wb') as f:
#     pickle.dump(list_ts, f)

with open(f'{working_dir}/output/merfish/data/list_ts.pickle', 'rb') as f:
    list_ts = pickle.load(f)

# transfer cell type and region labels
new_obs = []
for sample, (source_sample, target_sample) in source_target_list.items():
    project_ind = list_ts[sample][0][:, 0].flatten()
    source_obs = adata_comb.obs[
        adata_comb.obs['sample'] == source_sample].copy()
    target_obs = adata_comb.obs[
        adata_comb.obs['sample'] == target_sample].copy()
    target_index = target_obs.index
    target_obs = target_obs.reset_index(drop=True)

    for col in ['class', 'subclass', 'supertype', 'cluster']:
        target_obs[col] = source_obs[col].iloc[project_ind].values
        target_obs[f'{col}_color'] = source_obs[f'{col}_color']\
            .iloc[project_ind].values

    for level in ['division', 'structure', 'substructure']:
        region_labels = transfer_region_labels(
            coords_target=target_obs[['x_final', 'y_final']].values,
            coords_source=source_obs[['x', 'y']].values,
            source_region_labels=source_obs[f'parcellation_{level}'].values,
            k=10)
        target_obs[f'parcellation_{level}'] = region_labels
        color_mapping = dict(zip(
            source_obs[f'parcellation_{level}'],
            source_obs[f'parcellation_{level}_color']))
        target_obs[f'parcellation_{level}_color'] = \
            pd.Series(region_labels).map(color_mapping).values

    target_obs['ref_cell_id'] = source_obs.index[project_ind]
    target_obs['cosine_knn_weight'] = list_ts[sample][1][:, 0]
    target_obs['cosine_knn_cdist'] = list_ts[sample][2][:, 0]
    target_obs['cosine_knn_physical_dist'] = list_ts[sample][3][:, 0]
    new_obs.append(target_obs.set_index(target_index))

new_obs = pd.concat(new_obs)
adata_comb.obs[new_obs.columns.difference(adata_comb.obs.columns)] = np.nan
adata_comb.obs.loc[new_obs.index] = new_obs

# save
adata_comb.write(
    f'{working_dir}/output/merfish/data/adata_comb_cast_project.h5ad')

# final query data
adata_query = ad.read_h5ad(
    f'{working_dir}/output/merfish/data/adata_query.h5ad')
adata_query_i = adata_comb[adata_comb.obs['source'] == 'merfish'].copy()
adata_query = adata_query[adata_query_i.obs_names]

cols_to_keep = ['n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts',
                'log1p_total_counts', 'pct_counts_in_top_10_genes',
                'pct_counts_in_top_20_genes', 'pct_counts_in_top_50_genes',
                'pct_counts_in_top_100_genes', 'scDblFinder.sample', 
                'scDblFinder.class', 'scDblFinder.score', 
                'scDblFinder.weighted', 'scDblFinder.cxds_score']
adata_query.obs = pd.concat([adata_query.obs[cols_to_keep], 
                             adata_query_i.obs], axis=1)
adata_query.X = adata_query.layers['counts']
del adata_query.layers['counts']; del adata_query.uns
adata_query.write(
    f'{working_dir}/output/merfish/data/adata_query_final.h5ad')

# subset reference data to projection sample 
adata_ref = ad.read_h5ad(
    f'{working_dir}/output/merfish/data/adata_ref.h5ad')
adata_ref = adata_ref[adata_ref.obs['sample'] == 'C57BL6J-638850.47_R']
adata_ref.write(
    f'{working_dir}/output/merfish/data/adata_ref_final.h5ad')

# plotting #####################################################################

adata_comb = ad.read_h5ad(
    f'{working_dir}/output/merfish/data/adata_comb_cast_project.h5ad')

def create_comparison_plot(adata_comb, col, query_sample,
                           ref_sample, random_state=42):
    query_data = adata_comb.obs[adata_comb.obs['sample'] == query_sample]
    query_cell = query_data.sample(1, random_state=random_state)
    ref_data = adata_comb.obs[adata_comb.obs['sample'] == ref_sample]
    selected_value = query_cell[col].values[0]
    color_map = {selected_value: query_cell[f'{col}_color'].values[0]}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    for ax, data, x, y in [(ax1, query_data, 'x_final', 'y_final'),
                           (ax2, ref_data, 'x', 'y')]:
        mask = data[col] == selected_value
        ax.scatter(data[~mask][x], data[~mask][y], c='grey', s=1, alpha=0.1)
        ax.scatter(data[mask][x], data[mask][y], 
                   c=data[mask][col].map(color_map), s=1)
        ax.set_title(f'{data["sample"].iloc[0]} (Cells: {len(data)})\n'
                     f'{col}: {selected_value}')
        ax.axis('off')
        ax.set_aspect('equal')
    
    ax1.scatter(query_cell['x_final'], query_cell['y_final'], c='none', s=50,
                edgecolors='black', linewidths=2)
    ax2.scatter(ref_data.loc[query_cell['ref_cell_id'].values[0], 'x'],
                ref_data.loc[query_cell['ref_cell_id'].values[0], 'y'],
                c='none', s=50, edgecolors='black', linewidths=2)
    print(selected_value)
    plt.tight_layout()
    save_dir = f'{working_dir}/figures/merfish/comparison/'
    os.makedirs(save_dir, exist_ok=True)
    safe_filename = selected_value.replace('/', '_')
    plt.savefig(f'{save_dir}{safe_filename}.png', dpi=200)
    plt.close(fig)

create_comparison_plot(
    adata_comb, col='subclass', 
    query_sample='CTRL3_R', ref_sample='C57BL6J-638850.47_R',
    random_state=None)

col = 'parcellation_division'

# Plot for CTRL3_R (merfish sample)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
plot_df = adata_comb.obs[adata_comb.obs['sample'] == 'CTRL3_R']
plot_df[col] = plot_df[col].astype('category').cat.remove_unused_categories()
color_map = plot_df.drop_duplicates(col).set_index(col)[f'{col}_color'].to_dict()

sns.scatterplot(
    data=plot_df, x='x_final', y='y_final', linewidth=0,
    hue=col, palette=color_map, s=10, ax=ax1, legend=False)
ax1.set(xlabel=None, ylabel=None, title='CTRL3_R')
sns.despine(ax=ax1, bottom=True, left=True)
ax1.axis('equal')
ax1.set_xticks([])
ax1.set_yticks([])

# Plot for C57BL6J-638850.47_R
plot_df = adata_comb.obs[adata_comb.obs['sample'] == 'C57BL6J-638850.47_R']
plot_df[col] = plot_df[col].cat.remove_unused_categories()
color_map = plot_df.drop_duplicates(col).set_index(col)[f'{col}_color'].to_dict()

sns.scatterplot(
    data=plot_df, x='x', y='y', linewidth=0,
    hue=col, palette=color_map, s=15, ax=ax2, legend=False)
ax2.set(xlabel=None, ylabel=None, title='C57BL6J-638850.47_R')
sns.despine(ax=ax2, bottom=True, left=True)
ax2.axis('equal')
ax2.set_xticks([])
ax2.set_yticks([])

plt.tight_layout()
plt.savefig(f'{working_dir}/figures/merfish/exemplar_{col}.png', 
            dpi=200, bbox_inches='tight')

# Plot all merfish samples
merfish_samples = adata_comb.obs[
    adata_comb.obs['source'] == 'merfish']['sample'].unique()
n_samples = len(merfish_samples)
n_cols = 6
n_rows = (n_samples + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
axes = axes.flatten()

for i, sample in enumerate(merfish_samples):
    plot_df = adata_comb.obs[adata_comb.obs['sample'] == sample]
    plot_df[col] = plot_df[col].astype('category').cat.remove_unused_categories()
    color_map = plot_df.drop_duplicates(col).set_index(col)[f'{col}_color'].to_dict()

    sns.scatterplot(
        data=plot_df, x='x_final', y='y_final', linewidth=0,
        hue=col, palette=color_map, s=2, ax=axes[i], legend=False)
    axes[i].set(xlabel=None, ylabel=None, title=sample)
    sns.despine(ax=axes[i], bottom=True, left=True)
    axes[i].axis('equal')
    axes[i].set_xticks([])
    axes[i].set_yticks([])

for i in range(n_samples, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig(f'{working_dir}/figures/merfish/all_samples_{col}.png', 
            dpi=200, bbox_inches='tight')

# DELTA_ANALYSES ###############################################################

from CAST import delta_cell_cal
from scipy.spatial import KDTree

adata_comb = ad.read_h5ad('output/CURIO/data/adata_comb_project.h5ad')

tgt_indices = adata_comb.obs.query(
    'source == "CURIO" and sample.str.contains("Preg")').index
tgt_indices = tgt_indices[np.random.choice(
    len(tgt_indices), size=len(tgt_indices)//3, replace=False)]
ref_indices = adata_comb.obs.query(
    'source == "CURIO" and sample.str.contains("Virg")').index

coords_tgt = adata_comb.obs.loc[tgt_indices, ['x_final', 'y_final']].to_numpy()
coords_ref = adata_comb.obs.loc[ref_indices, ['x_final', 'y_final']].to_numpy()
ctype_tgt = adata_comb.obs.loc[tgt_indices, 'supertype'].to_list()
ctype_ref = adata_comb.obs.loc[ref_indices, 'supertype'].to_list()

niche_cols = adata_comb.obs.loc[ref_indices, 'k15_cluster_colors'].to_list()

coords_all = np.vstack((coords_tgt, coords_ref))
tree = KDTree(coords_all)
median_rad = np.median(tree.query(coords_all, k=2)[0][:, 1])

df_delta_cell_tgt, df_delta_cell_ref, df_delta_cell = \
    delta_cell_cal(coords_tgt, coords_ref, ctype_tgt, ctype_ref, 
                   radius_px= 30 * median_rad)

df_delta_cell = df_delta_cell.applymap(
    lambda x: np.log1p(x) if x > 0 else -np.log1p(-x))

plt.clf()
g = sns.clustermap(df_delta_cell.T,
               figsize=(10, 24),
               xticklabels=False,
               yticklabels=1,
               col_colors=niche_cols,
               cmap='coolwarm', center=0) 
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)
plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=5)
plt.tight_layout()
plt.savefig('figures/CURIO/delta_cell_all.png', dpi=300)