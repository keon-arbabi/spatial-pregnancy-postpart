# Prep raw images ##############################################################

import os
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# set paths 
data_dir = 'project/single-cell/Kalish/pregnancy-postpart/merfish'
working_dir = 'project/spatial-pregnancy-postpart'
os.makedirs(f'{data_dir}/rotate-split-raw', exist_ok=True)

# function for rotating and cropping 
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

# rotation and crop at the midline for each sample, mirror right hemisphere 
# plot, and save as separate anndata files 
sample_names = [
    'CTRL1', 'CTRL2', 'CTRL3', 'PREG1', 'PREG2', 'PREG3',
    'POSTPART1', 'POSTPART2', 'POSTPART3']
params = {
    'CTRL1': {'L': (72, 5800), 'R': (70, 5400)}, 
    'CTRL2': {'L': (110, 3100), 'R': (110, 2800)},  
    'CTRL3': {'L': (-33, 2000), 'R': (-33, 1600)}, 
    'PREG1': {'L': (3, 5600), 'R': (3, 5200)},  
    'PREG2': {'L': (-98, -5100), 'R': (-98, -5400)}, 
    'PREG3': {'L': (-138, -5900), 'R': (-138, -6200)},  
    'POSTPART1': {'L': (75, 5400), 'R': (75, 5200)}, 
    'POSTPART2': {'L': (115, 2400), 'R': (115, 2400)},  
    'POSTPART3': {'L': (-65, -2000), 'R': (-65, -1800)}  
}
plot_index = 1
plt.figure(figsize=(3 * 5, 6 * 4))

for sample in sample_names:
    print(sample)
    adata = ad.read_h5ad(f'{data_dir}/raw-h5ad/{sample}.h5ad')
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
plt.savefig(f'{working_dir}/figures/merfish/crop_and_rotate_all.png', 
            dpi=200, bbox_inches='tight', pad_inches=0)

# Preprocess query ##############################################################

import sys
import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from ryp import r, to_py
import warnings

warnings.filterwarnings('ignore')

sys.path.append('project/utils')
from single_cell import SingleCell
from ryp import r, to_r, to_py

# set paths
working_dir = 'project/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/merfish/CAST-MARK', exist_ok=True)

# load rotated and cropped query 
query_dir = 'project/single-cell/Kalish/pregnancy-postpart/merfish/rotate-split-raw'
samples_query = [file.replace('.h5ad', '') for file in os.listdir(query_dir)]
# sorting samples is important for maintaining cell order for all steps
samples_query = sorted(samples_query)

# munge each sample, adding placeholders for metadata columns to be added 
adatas_query = []
for sample in samples_query:
    adata = ad.read_h5ad(f'{query_dir}/{sample}.h5ad')
    adata.obs['sample'] = sample
    adata.obs['source'] = 'merfish'
    adata.obs[[
        'class', 'class_color', 'subclass', 'subclass_color',
        'supertype', 'supertype_color', 'cluster', 'cluster_color',
        'parcellation_division', 'parcellation_division_color',
        'parcellation_structure', 'parcellation_structure_color',
        'parcellation_substructure', 
        'parcellation_substructure_color']] = 'Unknown'
    adata.obs = adata.obs.drop(columns=[
        'nCount_Vizgen', 'nFeature_Vizgen', 'nCount_SCT', 'nFeature_SCT'])
    adata.obs.index = adata.obs.index.astype(str) + '_' + \
        adata.obs['sample'].astype(str)
    del adata.layers['orig_norm']
    print(f'[{sample}] {adata.shape[0]} cells')
    adatas_query.append(adata)

# concat and store raw counts 
adata_query = sc.concat(adatas_query, axis=0, merge='same')
adata_query.layers['counts'] = adata_query.X.copy()
adata_query.var['gene_symbol'] = adata_query.var.index

# detect doublets 
# https://github.com/plger/scDblFinder
file = f'{working_dir}/output/merfish/data/coldata.csv'
if os.path.exists(file):
    # add doublet metrics 
    coldata = pd.read_csv(f'{working_dir}/output/merfish/data/coldata.csv')
    adata_query.obs = coldata.set_index('index')
else:
    SingleCell(adata_query)\
        .save(f'{working_dir}/output/merfish/data/adata_query.rds', 
              sce=True, overwrite=True)
    to_r(working_dir, 'working_dir')
    r('''
    library(scDblFinder)
    library(BiocParallel)
    set.seed(123)
    sce = readRDS(paste0(working_dir, '/output/merfish/data/adata_query.rds'))
    sce = scDblFinder(sce, samples='sample', BPPARAM=MulticoreParam())
    table(sce$scDblFinder.class)
    # singlet doublet 
    #  996672  483021 
    coldata = as.data.frame(colData(sce))
    ''')
    coldata = to_py('coldata', format='pandas')
    adata_query.obs = coldata
    coldata.to_csv(file)

# get qc metrics 
sc.pp.calculate_qc_metrics(
    adata_query, percent_top=[20], log1p=True, inplace=True)
# save
adata_query.write(f'{working_dir}/output/data/adata_query_merfish_raw.h5ad')

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
plt.savefig(f'{working_dir}/figures/merfish/qc_doublets.png')

# plot
sc.pl.scatter(
    adata_query, 'total_counts', 'n_genes_by_counts', size=1) 
plt.savefig(f'{working_dir}/figures/merfish/qc_counts_genes.png', dpi=200)

# threshold outliers 
def is_outlier(adata, metric: str, nmads: int):
    from scipy.stats import median_abs_deviation
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M)
    return outlier

adata_query.obs['outlier'] = (
    is_outlier(adata_query, 'log1p_total_counts', 5) | 
    is_outlier(adata_query, 'log1p_n_genes_by_counts', 5) |
    is_outlier(adata_query, 'pct_counts_in_top_20_genes', 5))
adata_query.obs.outlier.value_counts()
# False    1478936
# True         757

adata_query.obs['doublet_outlier'] = is_outlier(
    adata_query, 'scDblFinder.score', 5)
adata_query.obs.doublet_outlier.value_counts()
# False    1477136
# True        2557

# filter to thresholds
print(f'total number of cells: {adata_query.n_obs}')
adata_query = adata_query[
    (~adata_query.obs.outlier) & 
    (~adata_query.obs.doublet_outlier)].copy()
print(f'number of cells after filtering of low quality cells: '
      f'{adata_query.n_obs}')
# total number of cells: 136923
# number of cells after filtering of low quality cells: 127739

# plot after filtering
sc.pl.scatter(
    adata_query, 'total_counts', 'n_genes_by_counts', size=1) 
plt.savefig(f'{working_dir}/figures/merfish/qc_counts_genes_filt.png', dpi=200)

# normalize 
sc.pp.normalize_total(adata_query)
sc.pp.log1p(adata_query, base=2)
# save
adata_query.write(f'{working_dir}/output/data/adata_query_merfish.h5ad')

# CAST-MARK ####################################################################

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scanorama
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import CAST
from CAST.models.model_GCNII import Args

warnings.filterwarnings('ignore')

working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/merfish/CAST-MARK', exist_ok=True)

# load query data
adata_query = ad.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish.h5ad')
# load reference data (imputed)
adata_ref = ad.read_h5ad(
    f'{working_dir}/output/data/adata_ref_zeng_imputed.h5ad')

# batch correction
adata_query_s, adata_ref_s = scanorama.correct_scanpy([adata_query, adata_ref])

# combine data for CAST-MARK input
adata_comb = ad.concat([adata_query, adata_ref], axis=0, merge='same')
adata_comb = adata_comb[:, adata_comb.var_names.sort_values()]
adata_comb.layers['X_scanorama'] = ad.concat(
    [adata_query_s, adata_ref_s], axis=0, merge='same').X.copy()

# crucially, order by sample names
sample_names = sorted(adata_comb.obs['sample'].unique())
adata_comb.obs['sample'] = pd.Categorical(
    adata_comb.obs['sample'], categories=sample_names, ordered=True)
adata_comb = adata_comb[adata_comb.obs.sort_values('sample').index].copy()

# extract coords_raw and exp_dict for CAST-MARK
coords_raw = {
    s: np.array(adata_comb.obs[['x', 'y']])[adata_comb.obs['sample']==s]
    for s in sample_names}
exp_dict = {
    s: adata_comb.layers['X_scanorama'][adata_comb.obs['sample']==s].toarray() 
    for s in sample_names}

# duplicate reference samples so that the embeddings are
# more balanced between query and reference
def duplicate_dict(d, n):
    return {f'{k}_dup{i + 1}' if k in adata_ref.obs['sample'].unique()
        else k: v for k, v in d.items() for i in range(n or 1)}

coords_raw_dup = duplicate_dict(coords_raw, 3)
exp_dict_dup = duplicate_dict(exp_dict, 3)

# run cast mark
from CAST.models.model_GCNII import Args
embed_dict_dup = CAST.CAST_MARK(
    coords_raw, exp_dict, 
    f'{working_dir}/output/merfish/CAST-MARK',
    graph_strategy='delaunay', 
    args = Args(
        dataname='merfish', # name of the dataset, used to save the log file
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
# detach, remove duplicated embeddings, and stack 
embed_dict = {k.split('_dup')[0]: v.cpu().detach() 
              for k, v in embed_dict_dup.items()}
embed_stack = np.vstack([embed_dict[name].numpy() for name in sample_names])

# kmeans clustering and plotting
plot_dir = f'{working_dir}/figures/merfish/k_clusters'
os.makedirs(plot_dir, exist_ok=True)

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

for n_clust in list(range(4, 20 + 1, 2)) + [30, 40, 50]:
    print(f'Clustering with k={n_clust}')
    kmeans = KMeans(n_clusters=n_clust, random_state=0).fit(embed_stack)
    cell_label = kmeans.labels_
    cluster_pl = sns.color_palette('Set3', n_clust)
    
    fig = plot_slices(sample_names, coords_raw, cell_label, cluster_pl, n_clust)
    fig.savefig(f'{plot_dir}/all_samples_k{str(n_clust)}.png', dpi=300)
    plt.close(fig)
    
    adata_comb.obs[f'k{n_clust}_cluster'] = cell_label
    color_map = {k: color for k, color in enumerate(cluster_pl.as_hex())}
    adata_comb.obs[f'k{n_clust}_cluster_colors'] = \
        pd.Series(cell_label).map(color_map).tolist()

# Save results
torch.save(coords_raw, f'{working_dir}/output/merfish/data/coords_raw.pt')
torch.save(exp_dict, f'{working_dir}/output/merfish/data/exp_dict.pt')
torch.save(embed_dict, f'{working_dir}/output/merfish/data/embed_dict.pt')
adata_comb.write_h5ad(f'{working_dir}/output/merfish/data/adata_comb_cast_mark.h5ad')

# CAST_STACK ###################################################################

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import sys
import os
import torch
import CAST
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# set paths 
working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/merfish/CAST-STACK', exist_ok=True)

# split the data randomly, while remembering the original cell order
# this is because a single image requires too much GPU mem
def split_dicts(coords_raw, embed_dict, n_split, seed=42):
    torch.manual_seed(seed)
    indices_dict = {}
    new_coords = {}; new_embeds = {}; query_reference_list = {}
    for key in coords_raw:
        if not key.startswith('C57BL6J-638850'):
            indices = torch.randperm(coords_raw[key].shape[0])
            indices_dict[key] = indices  
            splits = torch.tensor_split(indices, n_split)
            for i, split in enumerate(splits, 1):
                new_key = f'{key}_{i}'
                new_coords[new_key] = coords_raw[key][split]
                new_embeds[new_key] = embed_dict[key][split]
                query_reference_list[new_key] = [new_key, 'C57BL6J-638850.47_R']
        else:
            new_coords[key] = coords_raw[key]
            new_embeds[key] = embed_dict[key]
    return new_coords, new_embeds, indices_dict, query_reference_list

# after the final coordinates are determined, collapse back at the sample level 
def collapse_dicts(coords_final, indices_dict):
    collapsed = {}
    for base_key, indices in indices_dict.items():
        if base_key.startswith('C57BL6J-638850'):
            collapsed[base_key] = coords_final[base_key][base_key]
        else:
            full_array = torch.zeros((len(indices), 2), dtype=torch.float32)
            start_idx = 0
            for i in range(1, len(coords_final) + 1):
                key = f'{base_key}_{i}'
                if key in coords_final:
                    split_data = coords_final[key][key]
                    end_idx = start_idx + len(split_data)
                    split_indices = indices[start_idx:end_idx]
                    full_array[split_indices] = split_data
                    start_idx = end_idx
            collapsed[base_key] = full_array
    return collapsed

# load data 
adata_comb = ad.read_h5ad(
    f'{working_dir}/output/merfish/data/adata_comb_cast_mark.h5ad')
coords_raw = torch.load(
    f'{working_dir}/output/merfish/data/coords_raw.pt')
embed_dict = torch.load(
    f'{working_dir}/output/merfish/data/embed_dict.pt')

# split data 
coords_raw, embed_dict, indices_dict, query_reference_list  = \
      split_dicts(coords_raw, embed_dict, n_split=3)

# run cast-stack, parameters modified for default are commented 
coords_final_split = {}
for sample in sorted(query_reference_list.keys()):
    params_dist = CAST.reg_params(
        dataname = query_reference_list[sample],
        gpu = 0 if torch.cuda.is_available() else -1, 
        diff_step = 5,
        #### Affine parameters
        iterations=100, # 500
        dist_penalty1=0.4, # 0
        bleeding=500,
        d_list=[3,2,1,1/2,1/3],
        attention_params=[None,3,1,0],
        #### FFD parameters                                    
        dist_penalty2=[0.8], # 0
        alpha_basis_bs=[100], # 500
        meshsize=[8],
        iterations_bs=[100], # 400
        attention_params_bs=[[None,3,1,0]],
        mesh_weight = [None])
    params_dist.alpha_basis = torch.Tensor(
        [1/1000, 1/1000, 1/50, 5, 5]).reshape(5, 1).to(params_dist.device)

    coords_final_split[sample] = CAST.CAST_STACK(
        coords_raw, embed_dict, f'{working_dir}/output/merfish/CAST-STACK',
        query_reference_list[sample], params_dist, rescale=True)
    print(coords_final_split[sample])

# collapse back, save
coords_final = collapse_dicts(coords_final_split, indices_dict)

# add final coords to anndata object 
sample_names = sorted(list(coords_final.keys()))
cell_index = adata_comb.obs.index[adata_comb.obs['source'] == 'merfish']
coords_stack = np.vstack([coords_final[s] for s in sample_names])
coords_df = pd.DataFrame(
    coords_stack, columns=['x_final', 'y_final'], index=cell_index)
adata_comb.obs = adata_comb.obs.join(coords_df)

# plot final coords for all samples
n_cols = 5
n_rows = (len(sample_names) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 5*n_rows))
axes = axes.flatten()
for ax, sample in zip(axes, sample_names):
    plot_df = adata_comb.obs[adata_comb.obs['sample'] == sample]
    x, y = ('x', 'y') if plot_df['source'].iloc[0] == 'Zeng-ABCA-Reference' \
        else ('x_final', 'y_final')
    ax.scatter(plot_df[x], plot_df[y], c=plot_df['k10_cluster_colors'], s=2)
    ax.set_title(sample)
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
for ax in axes[len(sample_names):]:
    fig.delaxes(ax)
plt.tight_layout()
plt.savefig(f'{working_dir}/figures/merfish/all_samples_final.png', dpi=300)

# save
torch.save(coords_final, f'{working_dir}/output/merfish/data/coords_final.pt')
adata_comb.write(f'{working_dir}/output/merfish/data/adata_comb_cast_stack.h5ad')

# CAST_PROJECT #################################################################

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import sys
import os
import torch
import pickle
import warnings
import gc
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# modified CAST_Projection.py 
sys.path.insert(0, 'projects/def-wainberg/karbabi/CAST')
import CAST
print(CAST.__file__)

# set paths 
working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/merfish/CAST-PROJECT', exist_ok=True)

# load data
adata_comb = ad.read_h5ad(
    f'{working_dir}/output/merfish/data/adata_comb_cast_stack.h5ad')

# add batch, we will process all reference samples together with each query 
adata_comb.obs['batch'] = adata_comb.obs['sample'].astype(str)
adata_comb.obs.loc[adata_comb.obs['source'] == 
    'Zeng-ABCA-Reference', 'batch'] = 'Zeng-ABCA-Reference'
adata_comb.obs['batch'] = adata_comb.obs['batch'].astype('category')

# set parameters 
batch_key = 'batch'
level = 'class'
source_target_list = {
    key: ['Zeng-ABCA-Reference', key] 
    for key in adata_comb.obs.loc[
        adata_comb.obs['source'] == 'merfish', 'sample'].unique()
}
color_dict = (
    adata_comb.obs
    .drop_duplicates()
    .set_index(level)[f'{level}_color']
    .to_dict()
)
color_dict['Unknown'] = '#A9A9A9'

list_ts = {}
for _, (source_sample, target_sample) in source_target_list.items():
    print(f'Processing {target_sample}')
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
            (adata_comb.obs[batch_key] == target_sample) |
            (adata_comb.obs[batch_key] == source_sample)]

        # use the correct harmony function from cast
        adata_subset = CAST.Harmony_integration(
            sdata_inte=adata_subset,
            scaled_layer='X_scanorama',
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
            ifcombat=False)
        # save harmony-adjusted data to disk
        adata_subset.write_h5ad(harmony_file)
    
    # run CAST_PROJECT
    print(f'Running CAST_PROJECT for {target_sample}')
    _, list_ts[target_sample] = CAST.CAST_PROJECT(
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
        scaled_layer='X_scanorama',
        raw_layer='X_scanorama',
        batch_key=batch_key,
        use_highly_variable_t=False,
        ifplot=False,
        source_sample_ctype_col=level,
        output_path=output_dir_t,
        umap_feature='X_umap',
        pc_feature='X_pca_harmony',
        integration_strategy=None, 
        ave_dist_fold=10,
        alignment_shift_adjustment=0,
        color_dict=color_dict,
        adjust_shift=False,
        metric_t='cosine',
        working_memory_t=1000
    )
    print(list_ts[target_sample])
    del adata_subset; gc.collect()

# with open(f'{working_dir}/output/merfish/data/list_ts.pickle', 'wb') as f:
#     pickle.dump(list_ts, f)
    
with open(f'{working_dir}/output/merfish/data/list_ts.pickle', 'rb') as f:
    list_ts = pickle.load(f)

# transfer cell type and region labels
new_obs_list = []
for sample, (source_sample, target_sample) in source_target_list.items():
    project_ind = list_ts[sample][0][:, 0].flatten()
    source_obs = adata_comb.obs[
        adata_comb.obs[batch_key] == source_sample].copy()
    target_obs = adata_comb.obs[
        adata_comb.obs[batch_key] == target_sample].copy()
    target_index = target_obs.index
    target_obs = target_obs.reset_index(drop=True)

    for col in ['class', 'subclass', 'supertype', 'cluster']:
        target_obs[col] = source_obs[col].iloc[project_ind].values
        color_mapping = dict(zip(source_obs[col], source_obs[f'{col}_color']))
        target_obs[f'{col}_color'] = target_obs[col].map(color_mapping)

    for level in ['division', 'structure', 'substructure']:
        target_obs[f'parcellation_{level}'] = target_obs[f'parcellation_{level}']        
        color_mapping = dict(zip(
            target_obs[f'parcellation_{level}'],
            target_obs[f'parcellation_{level}_color']))
        target_obs[f'parcellation_{level}_color'] = \
            target_obs[f'parcellation_{level}'].map(color_mapping)

    target_obs['ref_cell_id'] = source_obs.index[project_ind]
    target_obs['cosine_knn_weight'] = list_ts[sample][1][:, 0]
    target_obs['cosine_knn_cdist'] = list_ts[sample][2][:, 0]
    target_obs['cosine_knn_physical_dist'] = list_ts[sample][3][:, 0]
    new_obs_list.append(target_obs.set_index(target_index))

new_obs = pd.concat(new_obs_list)
adata_comb.obs[new_obs.columns.difference(adata_comb.obs.columns)] = np.nan
adata_comb.obs.loc[new_obs.index] = new_obs
# save
adata_comb.write(
    f'{working_dir}/output/merfish/data/adata_comb_cast_project.h5ad')

######################################

# final query data
adata_comb = ad.read_h5ad(
    f'{working_dir}/output/merfish/data/adata_comb_cast_project.h5ad')
adata_query = ad.read_h5ad(f'{working_dir}/output/data/adata_query_merfish.h5ad')
adata_query_i = adata_comb[adata_comb.obs['source'] == 'merfish'].copy()
adata_query = adata_query[adata_query_i.obs_names]

cols_to_keep = [
    'volume', 'center_x', 'center_y', 'BarcodeCountOrig', 'BarcodeCountNor',
    'scDblFinder.sample', 'scDblFinder.class', 'scDblFinder.score', 
    'scDblFinder.weighted', 'scDblFinder.cxds_score', 'n_genes_by_counts', 
    'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 
    'pct_counts_in_top_20_genes', 'outlier', 'doublet_outlier'
]
adata_query.obs = pd.concat([
    adata_query_i.obs, adata_query.obs[cols_to_keep]], axis=1)
adata_query.X = adata_query.layers['counts']

# save
del adata_query.layers['counts']; del adata_query.uns
adata_query.write(f'{working_dir}/output/data/adata_query_merfish_final.h5ad')

# save the reference obs
adata_ref = adata_comb[adata_comb.obs['source'] == 'Zeng-ABCA-Reference']
adata_ref_obs = adata_ref.obs
adata_ref_obs.to_csv(
    f'{working_dir}/output/data/adata_ref_final_merfish_obs.csv')

# plotting #####################################################################

import os
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
working_dir = f'{Path.home()}/projects/def-wainberg/karbabi/' \
 'spatial-pregnancy-postpart'

# load data 
ref_obs = ad.read_h5ad(
    f'{working_dir}/output/data/adata_ref_final.h5ad').obs
query_obs = ad.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad').obs

def create_multi_sample_plot(ref_obs, query_obs, col, cell_type, output_dir):
    ref_samples = ref_obs['sample'].unique()
    query_samples = query_obs['sample'].unique() 
    n_cols = 4
    n_rows = 1 + -(-len(query_samples) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.flatten()
    
    for i, (sample, obs, coord_cols) in enumerate(
            [(s, ref_obs, ['x', 'y']) for s in ref_samples] +
            [(s, query_obs, ['x_final', 'y_final']) for s in query_samples]):
        if i >= len(axes):
            break
        ax = axes[i]
        plot_df = obs[obs['sample'] == sample]
        mask = plot_df[col] == cell_type
        if mask.sum() > 0:
            ax.scatter(plot_df[~mask][coord_cols[0]], 
                      plot_df[~mask][coord_cols[1]], 
                      c='grey', s=0.1, alpha=0.1)
            ax.scatter(plot_df[mask][coord_cols[0]], 
                      plot_df[mask][coord_cols[1]], 
                      c=plot_df[mask][f'{col}_color'], s=0.8)
        else:
            ax.text(0.5, 0.5, 'no cells of this type', 
                   ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{sample}\n{col}: {cell_type}')
        ax.axis('off')
        ax.set_aspect('equal')
    
    for ax in axes[i+1:]:
        fig.delaxes(ax)
    plt.tight_layout()
    safe_filename = cell_type.replace('/', '_').replace(' ', '_')
    plt.savefig(f'{output_dir}/{safe_filename}.png', dpi=300, 
                bbox_inches='tight')
    plt.close(fig)

col = 'subclass'
output_dir = f'{working_dir}/figures/merfish/spatial_cell_types_{col}_merfish'
os.makedirs(output_dir, exist_ok=True)
cell_types = pd.concat([ref_obs[col], query_obs[col]]).unique()

for cell_type in cell_types:
    if (ref_obs[col].value_counts().get(cell_type, 0) > 0 or 
        query_obs[col].value_counts().get(cell_type, 0) > 0):
        create_multi_sample_plot(ref_obs, query_obs, col, cell_type, output_dir)