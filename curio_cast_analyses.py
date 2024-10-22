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
fig1.savefig(f'{working_dir}/figures/curio/crop_and_rotate_outliers.png',
             dpi=200, bbox_inches='tight', pad_inches=0)
fig2.tight_layout()
fig2.savefig(f'{working_dir}/figures/curio/crop_and_rotate.png',
             dpi=200, bbox_inches='tight', pad_inches=0)

# CAST_MARK ####################################################################

import sys, os, torch, CAST, scanorama, warnings
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import matplotlib.pyplot as plt, seaborn as sns
from ryp import r, to_py
warnings.filterwarnings('ignore')

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import SingleCell

# set paths
data_dir = 'projects/def-wainberg/spatial'
working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/curio/CAST-MARK', exist_ok=True)
os.makedirs(f'{working_dir}/output/curio/data', exist_ok=True)

######################################

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
    set.seed(123)
    path = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart/'
    sce = readRDS(paste0(path, 'output/curio/data/adata_query.rds'))
    sce = scDblFinder(sce, samples='sample', BPPARAM=MulticoreParam())
    table(sce$scDblFinder.class)
    # singlet doublet 
    # 125175   11748 
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

# get qc metrics 
adata_query.var['mt'] = adata_query.var_names.str.startswith('mt-')
sc.pp.calculate_qc_metrics(
    adata_query, qc_vars=['mt'], percent_top=[20], log1p=True, inplace=True)
# plot
sc.pl.scatter(
    adata_query, 'total_counts', 'n_genes_by_counts', color='pct_counts_mt') 
plt.savefig(f'{working_dir}/figures/curio/qc_counts_genes.png', dpi=200)

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
# False    136346
# True        577

adata_query.obs['mt_outlier'] = (
    is_outlier(adata_query, 'pct_counts_mt', 3) | 
    (adata_query.obs['pct_counts_mt'] > 10))
adata_query.obs.mt_outlier.value_counts()
# False    129880
# True       7043

adata_query.obs['doublet_outlier'] = is_outlier(
    adata_query, 'scDblFinder.score', 5)
adata_query.obs.doublet_outlier.value_counts()
# False    135077
# True       1846

# filter to thresholds
print(f'total number of cells: {adata_query.n_obs}')
adata_query = adata_query[
    (~adata_query.obs.outlier) & 
    (~adata_query.obs.mt_outlier) & 
    (~adata_query.obs.doublet_outlier)].copy()
print(f'number of cells after filtering of low quality cells: '
      f'{adata_query.n_obs}')
# total number of cells: 136923
# number of cells after filtering of low quality cells: 127739

# plot after filtering
sc.pl.scatter(
    adata_query, 'total_counts', 'n_genes_by_counts', color='pct_counts_mt') 
plt.savefig(f'{working_dir}/figures/curio/qc_counts_genes_filt.png', dpi=200)

# normalize 
sc.pp.normalize_total(adata_query)
sc.pp.log1p(adata_query, base=2)
# save
adata_query.write(f'{working_dir}/output/data/adata_query_curio.h5ad')

######################################

# load preprocessed anndata objects 
adata_query = ad.read_h5ad(
    f'{working_dir}/output/data/adata_query_curio.h5ad')
# load reference created in `merfish_prep_atlases.py`
adata_ref = ad.read_h5ad(
    f'{working_dir}/output/data/adata_ref_zeng.h5ad')

# run scanorama batch correction 
adata_query_s, adata_ref_s = scanorama.correct_scanpy([adata_query, adata_ref])

# combine, keeping only the same metadata columns and genes
adata_comb = ad.concat([adata_query, adata_ref], axis=0, merge='same')
adata_comb = adata_comb[:, adata_comb.var_names.sort_values()]
# keep scanorama corrected expression separate 
adata_comb.layers['X_scanorama'] = \
    ad.concat([adata_query_s, adata_ref_s], axis=0, merge='same').X.copy()

# maintain cell, gene, and sample order  
sample_names = sorted(
    set(adata_query.obs['sample'].unique()) |
    set(adata_ref.obs['sample'].unique()))
adata_comb.obs['sample'] = pd.Categorical(
    adata_comb.obs['sample'], categories=sample_names, ordered=True)
adata_comb = adata_comb[adata_comb.obs.sort_values('sample').index]
adata_comb = adata_comb.copy()

# get raw coordinates and corrected expression data for each sample
coords_raw = {
    s: np.array(adata_comb.obs[['x', 'y']])[adata_comb.obs['sample']==s]
    for s in sample_names}
exp_dict = {
    s: adata_comb.layers['X_scanorama'][adata_comb.obs['sample']==s].toarray() 
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
        n_layers=12, # number of GCNII layers, more layers mean a deeper model,
                    # larger reception field, at cost of VRAM usage and time
        der=0.5, # edge dropout rate in CCA-SSG
        dfr=0.3, # feature dropout rate in CCA-SSG
        use_encoder=True, # perform single-layer dimension reduction before 
                          # GNNs, helps save VRAM and time if gene panel large
        encoder_dim=512 # encoder dimension, ignore if use_encoder is False
    )
)
# detach from gpu 
embed_dict = {
    k: v.cpu().detach() for k, v in embed_dict.items()}

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

# load data 
adata_comb = ad.read_h5ad(
    f'{working_dir}/output/curio/data/adata_comb_cast_mark.h5ad')
coords_raw = torch.load(f'{working_dir}/output/curio/data/coords_raw.pt')
embed_dict = torch.load(f'{working_dir}/output/curio/data/embed_dict.pt')

query_reference_list = {}
for key in coords_raw:
    if not key.startswith('C57BL6J-638850'):
        query_reference_list[key] = [key, 'C57BL6J-638850.47_R']

selected_samples = ['PP_1_1_L', 'PP_1_2_L', 'Preg_1_1_L', 'Preg_2_1_R']
query_reference_list = {
    k: v for k, v in query_reference_list.items() if k in selected_samples
}
coords_final = torch.load(f'{working_dir}/output/curio/data/coords_final.pt')

# run cast-stack, parameters modified for default are commented 
coords_final = {}
for sample in sorted(query_reference_list.keys()):
    params_dist = CAST.reg_params(
        dataname = query_reference_list[sample],
        gpu = 0 if torch.cuda.is_available() else -1, 
        diff_step = 5,
        #### Affine parameters
        iterations=20, # 500
        dist_penalty1=0.3, # 0
        bleeding=300, # 500
        d_list=[3,2,1,1/2,1/3],
        attention_params=[None,3,1,0],
        #### FFD parameters                                    
        dist_penalty2=[0.3],
        alpha_basis_bs=[500],
        meshsize=[8],
        iterations_bs=[80], # 400 # 10
        attention_params_bs=[[None,3,1,0]],
        mesh_weight = [None])
    params_dist.alpha_basis = torch.Tensor(
        [1/1000, 1/1000, 1/50, 5, 5]).reshape(5, 1).to(params_dist.device)

    coords_final[sample] = CAST.CAST_STACK(
        coords_raw, embed_dict,  
        f'{working_dir}/output/curio/CAST-STACK',
        query_reference_list[sample], params_dist, rescale=True)
    print(coords_final[sample])

# add final coords to anndata object 
sample_names = sorted(list(coords_final.keys()))
cell_index = adata_comb.obs.index[adata_comb.obs['source'] == 'curio']
coords_stack = np.vstack([coords_final[s][s] for s in sample_names])
coords_df = pd.DataFrame(
    coords_stack, columns=['x_final', 'y_final'], index=cell_index)
adata_comb.obs = adata_comb.obs.join(coords_df)

# plot all samples
n_cols = 6; n_rows = -(-len(sample_names) // n_cols)  
fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 5*n_rows))
axes = axes.flatten()
for ax, sample in zip(axes, sample_names):
    plot_df = adata_comb.obs[adata_comb.obs['sample'] == sample]
    x, y = ('x', 'y') if plot_df['source'].iloc[0] == 'Zeng-ABCA-Reference' \
        else ('x_final', 'y_final')
    ax.scatter(plot_df[x], plot_df[y], c='black', s=1)
    ax.set_title(sample)
    ax.axis('off')
for ax in axes[len(sample_names):]:
    fig.delaxes(ax)
plt.tight_layout()
plt.savefig(f'{working_dir}/figures/curio/all_samples_final.png', dpi=300)

# save
torch.save(coords_final, f'{working_dir}/output/curio/data/coords_final.pt')
adata_comb.write(f'{working_dir}/output/curio/data/adata_comb_cast_stack.h5ad')

# CAST_PROJECT #################################################################

import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import sys, os, torch, pickle, warnings, gc
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings('ignore')

# modified CAST_Projection.py 
sys.path.insert(0, 'projects/def-wainberg/karbabi/CAST')
import CAST
print(CAST.__file__)

# set paths 
working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
# os.makedirs(f'{working_dir}/output/curio/CAST-PROJECT', exist_ok=True)

# load data
adata_comb = ad.read_h5ad(
    f'{working_dir}/output/curio/data/adata_comb_cast_stack.h5ad')
# scale data separately for each source 
sc.pp.scale(adata_comb, zero_center = True, 
            mask_obs=(adata_comb.obs['source'] == 'Zeng-ABCA-Reference'))
sc.pp.scale(adata_comb, zero_center = True, 
            mask_obs=(adata_comb.obs['source'] == 'curio'))
adata_comb.layers['log1p_norm_scaled'] = adata_comb.X.copy()

# add batch, we will process all reference samples together with each query 
adata_comb.obs['batch'] = adata_comb.obs['sample'].astype(str)
adata_comb.obs.loc[adata_comb.obs['source'] == 'Zeng-ABCA-Reference', 
                   'batch'] = 'Zeng-ABCA-Reference'
adata_comb.obs['batch'] = adata_comb.obs['batch'].astype('category')

# set parameters 
batch_key = 'batch'
level = 'class'
source_target_list = {
    key: ['Zeng-ABCA-Reference', key] 
    for key in adata_comb.obs.loc[
        adata_comb.obs['source'] == 'curio', 'sample'].unique()
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
    output_dir_t = f'{working_dir}/output/curio/CAST-PROJECT/' \
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
        ave_dist_fold=2,
        alignment_shift_adjustment=0,
        color_dict=color_dict,
        adjust_shift=False,
        metric_t='cosine',
        working_memory_t=1000
    )
    print(list_ts[target_sample])
    del adata_subset; gc.collect()

# with open(f'{working_dir}/output/curio/data/list_ts.pickle', 'wb') as f:
#     pickle.dump(list_ts, f)
    
with open(f'{working_dir}/output/curio/data/list_ts.pickle', 'rb') as f:
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
    f'{working_dir}/output/curio/data/adata_comb_cast_project.h5ad')

######################################

# final query data
adata_comb = ad.read_h5ad(
    f'{working_dir}/output/curio/data/adata_comb_cast_project.h5ad')
adata_query = ad.read_h5ad(f'{working_dir}/output/data/adata_query_curio.h5ad')
adata_query_i = adata_comb[adata_comb.obs['source'] == 'curio'].copy()
adata_query = adata_query[adata_query_i.obs_names]

cols_to_keep = [
    'SPATIAL_1', 'SPATIAL_2', 'n_genes_by_counts', 
    'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 
    'pct_counts_in_top_10_genes', 'pct_counts_in_top_20_genes', 
    'pct_counts_in_top_50_genes', 'pct_counts_in_top_100_genes', 
    'scDblFinder.sample', 'scDblFinder.class', 'scDblFinder.score', 
    'scDblFinder.weighted','scDblFinder.cxds_score', 'total_counts_mt', 
    'log1p_total_counts_mt', 'pct_counts_mt'
]
adata_query.obs = pd.concat([
    adata_query_i.obs, adata_query.obs[cols_to_keep]], axis=1)
adata_query.X = adata_query.layers['counts']
del adata_query.layers['counts']; del adata_query.uns
# save
adata_query.write(f'{working_dir}/output/data/adata_query_curio_final.h5ad')

# save the reference obs
adata_ref_obs= adata_comb[
    adata_comb.obs['source'] == 'Zeng-ABCA-Reference'].obs
adata_ref_obs.to_csv(
    f'{working_dir}/output/data/adata_ref_obs_final.csv')

# plotting #####################################################################

import os, numpy as np, pandas as pd, anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns

working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'

# load data 
ref_obs = pd.read_csv(
    f'{working_dir}/output/data/adata_ref_obs_final.csv', index_col=0)
query_obs = ad.read_h5ad(
    f'{working_dir}/output/data/adata_query_curio_final.h5ad').obs
query_obs_mmc = pd.read_csv(
    f'{working_dir}/output/curio/data/curio_mmc_corr_annotations.csv',
    index_col=0)

categories = ['class', 'subclass', 'supertype', 'cluster']
color_maps = {}
for cat in categories:
    unique_categories = ref_obs[cat].unique()
    unique_colors = ref_obs[f'{cat}_color'].unique()
    color_maps[cat] = dict(zip(unique_categories, unique_colors))
for cat in categories:
    query_obs_mmc[f'{cat}_color'] = query_obs_mmc[cat].map(color_maps[cat])
query_obs_mmc['sample'] = query_obs['sample']
query_obs_mmc['x_final'] = query_obs['x_final']
query_obs_mmc['y_final'] = query_obs['y_final']

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
            # Plot background cells
            ax.scatter(
                plot_df[~mask][coord_cols[0]], plot_df[~mask][coord_cols[1]], 
                c='grey', s=1 if obs is ref_obs else 2, alpha=0.1)
            # Plot cells of interest with individual colors
            ax.scatter(
                plot_df[mask][coord_cols[0]], plot_df[mask][coord_cols[1]], 
                c=plot_df[mask][f'{col}_color'], s=1 if obs is ref_obs else 6)
        else:
            ax.text(0.5, 0.5, 'no cells of this type', ha='center', 
                    va='center', transform=ax.transAxes)
        ax.set_title(f'{sample}\n{col}: {cell_type}')
        ax.axis('off')
        ax.set_aspect('equal')
    
    for ax in axes[i+1:]:
        fig.delaxes(ax)
    plt.tight_layout()
    safe_filename = cell_type.replace('/', '_').replace(' ', '_')
    plt.savefig(f'{output_dir}/{safe_filename}.png', dpi=200, 
                bbox_inches='tight')
    plt.close(fig)

col = 'subclass'
output_dir = f'{working_dir}/figures/curio/spatial_cell_types_{col}_mmc'
os.makedirs(output_dir, exist_ok=True)
cell_types = pd.concat([ref_obs[col], query_obs_mmc[col]]).unique()

for cell_type in cell_types:
    if (ref_obs[col].value_counts().get(cell_type, 0) > 0 or 
        query_obs_mmc[col].value_counts().get(cell_type, 0) > 0):
        create_multi_sample_plot(
            ref_obs, query_obs_mmc, col, cell_type, output_dir)


def confusion_matrix_plot(df_true, df_pred, true_label_column,
                          pred_label_column, file):
    df_true[true_label_column] = df_true[true_label_column].astype(str)
    df_pred[pred_label_column] = df_pred[pred_label_column].astype(str)
    true_labels = df_true[true_label_column].values
    pred_labels = df_pred[pred_label_column].values
    unique_labels = np.union1d(true_labels, pred_labels)
    confusion_matrix = pd.DataFrame(
        0, index=unique_labels, columns=unique_labels)
    
    for true, pred in zip(true_labels, pred_labels):
        confusion_matrix.loc[true, pred] += 1
    confusion_matrix = confusion_matrix.loc[
        (confusion_matrix.sum(axis=1) != 0),
        (confusion_matrix.sum(axis=0) != 0)
    ]
    confusion_matrix = confusion_matrix.div(
        confusion_matrix.sum(axis=1), axis=0)
    
    plt.figure(figsize=(15, 12))
    ax = sns.heatmap(
        confusion_matrix, xticklabels=1, yticklabels=1,
        rasterized=True, square=True, linewidths=0.5,
        cmap='rocket_r', cbar_kws=dict(pad=0.01), vmin=0, vmax=1)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel(f'predicted ({pred_label_column})')
    plt.ylabel(f'true ({true_label_column})')
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    plt.tight_layout()
    plt.savefig(file, dpi=300, bbox_inches='tight')
    plt.close()

confusion_matrix_plot(
    query_obs, query_obs_mmc, 'class', 'class',
    f'{working_dir}/figures/curio/confusion_matrix_class.png')
confusion_matrix_plot(
    query_obs, query_obs_mmc, 'subclass', 'subclass',
    f'{working_dir}/figures/curio/confusion_matrix_subclass.png')


sample_data = adata_comb.obs[
    adata_comb.obs['source'] == 'curio']['sample'].unique()[0]
plot_df = adata_comb.obs[adata_comb.obs['sample'] == sample_data]
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(plot_df['x_final'], plot_df['y_final'], c='grey', s=1)
random_point = plot_df.sample(n=1)
ax.scatter(random_point['x_final'], random_point['y_final'], c='red', s=10)
from matplotlib.patches import Circle
radius = 0.6133848801255226  # ave_dist_fold=5
radius = 0.24535395205020905  # ave_dist_fold=2
circle = Circle((random_point['x_final'].values[0], 
                 random_point['y_final'].values[0]), 
                radius, fill=False, color='blue')
ax.add_artist(circle)
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
plt.savefig(f'{working_dir}/figures/curio/radius.png', dpi=200)
plt.close()


col = 'parcellation_structure'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
plot_df = adata_comb.obs[adata_comb.obs['sample'] == 'PP_2_2_L']
plot_df[col] = plot_df[col].astype('category').cat.remove_unused_categories()
color_map = plot_df.drop_duplicates(col).set_index(col)[f'{col}_color'].to_dict()

sns.scatterplot(
    data=plot_df, x='x_final', y='y_final', linewidth=0,
    hue=col, palette=color_map, s=35, ax=ax1, legend=False)
ax1.set(xlabel=None, ylabel=None, title='PP_2_2_L')
sns.despine(ax=ax1, bottom=True, left=True)
ax1.axis('equal')
ax1.set_xticks([])
ax1.set_yticks([])

plot_df = adata_comb.obs[adata_comb.obs['sample'] == 'C57BL6J-638850.47_R']
plot_df[col] = plot_df[col].cat.remove_unused_categories()
color_map = plot_df.drop_duplicates(col).set_index(col)[f'{col}_color'].to_dict()

sns.scatterplot(
    data=plot_df, x='x', y='y', linewidth=0,
    hue=col, palette=color_map, s=10, ax=ax2, legend=False)
ax2.set(xlabel=None, ylabel=None, title='C57BL6J-638850.47_R')
sns.despine(ax=ax2, bottom=True, left=True)
ax2.axis('equal')
ax2.set_xticks([])
ax2.set_yticks([])

plt.tight_layout()
plt.savefig(f'{working_dir}/figures/curio/exemplar_{col}.png', 
            dpi=200, bbox_inches='tight')


curio_samples = adata_comb.obs[
    adata_comb.obs['source'] == 'curio']['sample'].unique()
n_samples = len(curio_samples)
n_cols = 6
n_rows = (n_samples + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
axes = axes.flatten()
for i, sample in enumerate(curio_samples):
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
plt.savefig(f'{working_dir}/figures/curio/all_samples_{col}.png', 
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