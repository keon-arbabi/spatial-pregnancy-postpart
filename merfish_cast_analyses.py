import sys, os, warnings
import numpy as np, pandas as pd, anndata as ad
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore")

# Prep raw images ##############################################################

# set paths 
data_dir = 'projects/def-wainberg/spatial/Kalish/pregnancy-postpart/merfish'
working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
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
    'CTRL1': {'L': (72, 6000), 'R': (70, 5200)}, 
    'CTRL2': {'L': (110, 3200), 'R': (110, 2600)},
    'CTRL3': {'L': (-33, 2200), 'R': (-33, 1800)}, 
    'PREG1': {'L': (3, 5800), 'R': (3, 5000)},
    'PREG2': {'L': (-98, -4900), 'R': (-98, -5400)},
    'PREG3': {'L': (-138, -5700), 'R': (-138, -6100)},
    'POSTPART1': {'L': (75, 5800), 'R': (75, 5000)},
    'POSTPART2': {'L': (115, 2600), 'R': (115, 1900)},
    'POSTPART3': {'L': (-65, -1800), 'R': (-65, -2200)}}
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

# CAST_MARK ####################################################################

import sys, os, torch, CAST, warnings, shutil
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import matplotlib.pyplot as plt, seaborn as sns
from scipy import sparse
warnings.filterwarnings("ignore")

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import SingleCell
from ryp import r, to_py

# set paths
data_dir = 'projects/def-wainberg/spatial'
working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/merfish/CAST-MARK', exist_ok=True)

# load rotated and cropped query 
query_dir = f'{data_dir}/Kalish/pregnancy-postpart/merfish/rotate-split-raw'
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

# merge, store raw counts 
adata_query = sc.concat(adatas_query)
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
        .save(f'{working_dir}/output/merfish/data/adata_query.rds', sce=True)
    r('''
    library(scDblFinder)
    library(BiocParallel)
    path = "projects/def-wainberg/karbabi/spatial-pregnancy-postpart/"
    sce = readRDS(paste0(path, "output/merfish/data/adata_query.rds"))
    sce = scDblFinder(sce, samples="sample", BPPARAM=MulticoreParam())
    table(sce$scDblFinder.class)
    #  993425  482712
    coldata = as.data.frame(colData(sce))
    ''')
    coldata = to_py('coldata', format='pandas')
    coldata.to_csv(file)

# qc metrics 
sc.pp.calculate_qc_metrics(
    adata_query, percent_top=[10, 20, 50, 100], inplace=True)

# plot qc metric cdfs
cols = ['scDblFinder.score', 'scDblFinder.weighted', 
        'scDblFinder.cxds_score', 'n_genes_by_counts', 'total_counts']
fig, axs = plt.subplots(3, 2, figsize=(12, 15), squeeze=False)
for ax, col in zip(axs.flat, cols):
    data = adata_query.obs[col].values
    x = np.sort(data)
    ax.plot(x, np.arange(1, len(data) + 1) / len(data))
    p5, p95 = np.percentile(data, [5, 95])
    ax.axvline(p5, color='r', linestyle='--')
    ax.axvline(p95, color='r', linestyle='--')
    ax.text(p5, 0.5, f'{p5:.2f}', rotation=90, va='center', ha='right')
    ax.text(p95, 0.5, f'{p95:.2f}', rotation=90, va='center', ha='left')
    ax.set(title=col, xlabel='Value', ylabel='Cumulative Probability')
for ax in axs.flat[len(cols):]:
    ax.remove()
plt.tight_layout()
plt.savefig(f'{working_dir}/figures/merfish/qc_cdfs_query.png')

# plot relationship between doublet score and number of detected genes 
sns.jointplot(
    data=adata_query.obs,
    x="n_genes_by_counts",
    y="scDblFinder.score",
    kind="hex")
plt.savefig(f'{working_dir}/figures/merfish/qc_joint_query.png')

# # filtered out 
# # doublets: 189088 (12.81%)
# # low counts: 55853 (4.34%)
# # low genes: 98825 (8.03%)
# adata_query = adata_query[adata_query.obs['scDblFinder.score'] <= 0.25]
# sc.pp.filter_cells(adata_query, min_counts=60)
# sc.pp.filter_cells(adata_query, min_genes=10)

# normalize 
sc.pp.normalize_total(adata_query, target_sum=1e4)
sc.pp.log1p(adata_query)
# save
adata_query.write(f'{working_dir}/output/merfish/data/adata_query.h5ad')

######################################

# load reference, output from `merfish_zhuang_prep_atlas.py`
ref_dir = f'{data_dir}/Zhuang/direct-downloads'  
datasets_ref = ['Zhuang-ABCA-1', 'Zhuang-ABCA-2']
samples_ref = [
    'Zhuang-ABCA-1.057', 'Zhuang-ABCA-1.058', 'Zhuang-ABCA-1.059',
    'Zhuang-ABCA-1.060', 'Zhuang-ABCA-1.061', 'Zhuang-ABCA-1.062',
    'Zhuang-ABCA-2.026', 'Zhuang-ABCA-2.027', 'Zhuang-ABCA-2.028',
    'Zhuang-ABCA-2.030']  

# munge each dataset 
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
    adata.obs.index = adata.obs.index.astype(str) + '_' + \
        adata.obs['sample'].astype(str)
    adata.var.reset_index()
    adata.var.index = adata.var['gene_symbol']
    print(f'[{data}] {adata.shape[0]} cells')
    adatas_ref.append(adata)

# merge, convert raw counts to sparse and store
adata_ref = ad.concat(adatas_ref)
adata_ref.X = sparse.csr_matrix(adata_ref.X)
adata_ref.layers['counts'] = adata_ref.X.copy()
adata_ref.var['gene_symbol'] = adata_ref.var.index

# preprocess 
sc.pp.normalize_total(adata_ref, target_sum=1e4)
sc.pp.log1p(adata_ref)
# save
adata_ref.write(f'{working_dir}/output/merfish/data/adata_ref.h5ad')

######################################

# load data, processed above 
adata_query = ad.read_h5ad(
    f'{working_dir}/output/merfish/data/adata_query.h5ad')
adata_ref = ad.read_h5ad(
    f'{working_dir}/output/merfish/data/adata_ref.h5ad')

# combine query and reference dataset
# keep only the same genes and metadata
adata_comb = ad.concat([adata_query, adata_ref], merge='same')

# get the coordinates and expression data for each sample
sample_names = sorted(adata_comb.obs['sample'].unique())
coords_raw = {
    sample: np.array(adata_comb.obs[['x', 'y']])
    [adata_comb.obs['sample'] == sample] for sample in sample_names}
exp_dict = {
    sample: adata_comb[adata_comb.obs['sample'] == sample]
    .X.toarray() for sample in sample_names}
torch.save(coords_raw, f'{working_dir}/output/merfish/data/coords_raw.pt')
torch.save(exp_dict, f'{working_dir}/output/merfish/data/exp_dict.pt')

# get cast-mark embeddings 
file = f'{working_dir}/output/merfish/CAST-MARK/demo_embed_dict.pt'
if os.path.exists(file):
    embed_dict = torch.load(file, map_location='cpu')
else: 
    embed_dict = CAST.CAST_MARK(
        coords_raw, exp_dict, f'{working_dir}/output/merfish/CAST-MARK')
    shutil.move(
        f'{working_dir}/output/merfish/CAST-MARK/demo_embed_dict.pt',
        file)

# run k-means clustering on embeddings 
from sklearn.cluster import KMeans
embed_stack = np.vstack([embed_dict[name].cpu().detach().numpy()
                        for name in sample_names])
# plot
for n_clust in range(6, 24, 2):
    kmeans = KMeans(n_clusters=n_clust, random_state=0).fit(embed_stack)
    cell_label = kmeans.labels_
    cluster_pl = sns.color_palette('Set3', len(np.unique(cell_label)))
    # np.random.shuffle(cluster_pl)

    num_plot = len(sample_names)
    plot_row = int(np.floor(num_plot/5) + 1)
    cell_label_idx = 0
    plt.figure(figsize=((30, 3.5*plot_row)))
    for j in range(num_plot):
        plt.subplot(plot_row, 5, j+1)
        coords_raw0 = coords_raw[sample_names[j]]
        col=coords_raw0[:,0].tolist()
        row=coords_raw0[:,1].tolist()
        cell_type_t = cell_label[cell_label_idx:
            (cell_label_idx + coords_raw0.shape[0])]
        cell_label_idx += coords_raw0.shape[0]
        size = np.log(1e4 / coords_raw0.shape[0]) + 3
        for i in set(cell_type_t):
            plt.scatter(np.array(col)[cell_type_t == i],
            np.array(row)[cell_type_t == i], s=size, edgecolors='none',
            c=np.array(cluster_pl)[cell_type_t[cell_type_t == i]], 
            label = str(i), rasterized=True)
        plt.title(sample_names[j] + 
            ' (KMeans, k = ' + str(n_clust) + ')', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.axis('equal')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.savefig(
        f'{working_dir}/figures/merfish/all_samples_k{str(n_clust)}.png', 
        dpi=200)

    adata_comb.obs[f'k{n_clust}_cluster'] = cell_label
    # color_map = {k: color for k, color in enumerate(cluster_pl.as_hex())}
    # adata_comb.obs[f'k{n_clust}_cluster_colors'] = pd.Series(cell_label)\
    #     .map(color_map).tolist()
# save
adata_comb.write(
    f'{working_dir}/output/merfish/data/adata_comb_cast_mark.h5ad')

# CAST_STACK ###################################################################

import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import sys, os, torch, CAST, warnings
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore")

# set paths 
working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/merfish/CAST-STACK', exist_ok=True)

# function to split the data randomly, while remembering the original cell order
# this is because a single image requires too much GPU mem
def split_dicts(coords_raw, embed_dict, n_split=2, seed=42):
    torch.manual_seed(seed)
    indices_dict = {}
    new_coords = {}; new_embeds = {}; query_reference_list = {}
    for key in coords_raw:
        if not key.startswith('Zhuang'):
            indices = torch.randperm(coords_raw[key].shape[0])
            indices_dict[key] = indices  
            splits = torch.tensor_split(indices, n_split)
            for i, split in enumerate(splits, 1):
                new_key = f"{key}_{i}"
                new_coords[new_key] = coords_raw[key][split]
                new_embeds[new_key] = embed_dict[key][split]
                query_reference_list[new_key] = [new_key, 'Zhuang-ABCA-1.060']
        else:
            new_coords[key] = coords_raw[key]
            new_embeds[key] = embed_dict[key]
    return new_coords, new_embeds, query_reference_list, indices_dict

# after the final coordinates are determined, collapse back at the sample level 
def collapse_dicts(coords_final, indices_dict):
    collapsed = {}
    for base_key, indices in indices_dict.items():
        if base_key.startswith('Zhuang'):
            collapsed[base_key] = coords_final[base_key][base_key]
        else:
            full_array = torch.zeros((len(indices), 2), dtype=torch.float32)
            start_idx = 0
            for i in range(1, len(coords_final) + 1):
                key = f"{base_key}_{i}"
                if key in coords_final:
                    split_data = coords_final[key][key]
                    end_idx = start_idx + len(split_data)
                    split_indices = indices[start_idx:end_idx]
                    full_array[split_indices] = split_data
                    start_idx = end_idx
            collapsed[base_key] = full_array
    return collapsed

# load data 
adata_comb = ad.read_h5ad(f'{working_dir}/output/merfish/data/'
                          'adata_comb_cast_mark.h5ad')
coords_raw = torch.load(
    f'{working_dir}/output/merfish/data/coords_raw.pt', map_location='cpu')
embed_dict = torch.load(
    f'{working_dir}/output/merfish/data/demo_embed_dict.pt', map_location='cpu')

# split data 
coords_raw, embed_dict, query_reference_list, indices_dict = \
      split_dicts(coords_raw, embed_dict, n_split=2)

# run cast-stack, parameters modified for default are commented 
coords_final_split = {}
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

    coords_final_split[sample] = CAST.CAST_STACK(
        coords_raw, embed_dict, f'{working_dir}/output/merfish/CAST-STACK',
        query_reference_list[sample], params_dist, rescale=True)
    
# collapse back, save
coords_final = collapse_dicts(coords_final_split, indices_dict)
torch.save(coords_final, f'{working_dir}/output/merfish/data/coords_final.pt')

# add final coords to anndata object 
sample_names = list(coords_final.keys())
sample_names = sorted(sample_names)
cell_index = adata_comb.obs.index[adata_comb.obs['source'] == 'merfish']
coords_stack = np.vstack([
    coords_final[sample] for sample in sample_names])
coords_df = pd.DataFrame(
    coords_stack, columns=['x_final', 'y_final'], index=cell_index)
adata_comb.obs = adata_comb.obs.join(coords_df)

# df = adata_comb.obs[adata_comb.obs['source'] == 'merfish']
# num_plot = len(sample_names)
# unique_clusters = df['k16_cluster'].unique()
# color_palette = sns.color_palette('Set3', len(unique_clusters))
# color_dict = dict(zip(unique_clusters, color_palette))
# plt.figure(figsize=(30, 40))  
# for j in range(num_plot):
#     plt.subplot(6, 3, j+1)  
#     coords_final0 = coords_final[sample_names[j]]
#     col = coords_final0[:, 0].tolist()
#     row = coords_final0[:, 1].tolist()
#     current_index = df.index[df.index.str.contains(sample_names[j])]
#     cluster_colors = df.loc[current_index, 'k16_cluster'].map(color_dict)
#     plt.scatter(x=col, y=row, s=0.5, c=cluster_colors)
#     plt.title(sample_names[j] + ' (KMeans, k = 16)', fontsize=16)
#     plt.axis('equal')
#     plt.xticks([])
#     plt.yticks([])
# plt.savefig(
#     f'{working_dir}/figures/merfish/all_samples_trained_k16_final.png',
#     dpi=200)

adata_comb.write(
    f'{working_dir}/output/merfish/data/adata_comb_cast_stack.h5ad')

# CAST_PROJECT #################################################################

import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import sys, os, torch, CAST, warnings
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore")

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from utils import debug
debug(third_party=True) 

working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/merfish/CAST-PROJECT', exist_ok=True)

coords_final = torch.load(
    f'{working_dir}/output/merfish/data/coords_final.pt', map_location='cpu')
source_target_list = {
    key: [key, 'Zhuang-ABCA-1.060'] for key in coords_final.keys()}

adata_comb = ad.read_h5ad(
    f'{working_dir}/output/merfish/data/adata_comb_cast_stack.h5ad')
adata_comb.X = adata_comb.layers['counts']
adata_comb = CAST.preprocess_fast(adata_comb, mode='default')
batch_key = 'sample'
color_dict = adata_comb.obs\
    .drop_duplicates().set_index('class')['class_color']\
    .to_dict()
color_dict['Unknown'] = '#A9A9A9'

adata_comb_refs = {}; list_ts = {}

for sample in source_target_list.keys():
    print(sample)
    source_sample, target_sample = source_target_list[sample]
    output_dir_t = f'{working_dir}/output/merfish/CAST-PROJECT/' \
        f'{source_sample}_to_{target_sample}'
    os.makedirs(output_dir_t, exist_ok=True)
    adata_comb_refs[sample], list_ts[sample] = CAST.CAST_PROJECT(
        sdata_inte=adata_comb[
            np.isin(adata_comb.obs[batch_key], 
                    [source_sample, target_sample])],
        source_sample=source_sample,
        target_sample=target_sample, 
        coords_source=np.array(
            adata_comb[np.isin(adata_comb.obs[batch_key], source_sample),:]
            .obs.loc[:,['x_final','y_final']]),
        coords_target=np.array(
            adata_comb[np.isin(adata_comb.obs[batch_key], target_sample),:]
            .obs.loc[:,['x','y']]),
        scaled_layer = 'log1p_norm_scaled', 
        raw_layer = 'raw',
        batch_key=batch_key, 
        use_highly_variable_t=False,  
        ifplot = False,
        n_components = 50,
        k2 = 30,
        source_sample_ctype_col='class', 
        output_path=output_dir_t, 
        integration_strategy='Harmony', 
        ave_dist_fold = 1.5, # 3
        color_dict=color_dict,
        save_result=False)
    
torch.save(adata_comb_refs, 
           f'{working_dir}/output/merfish/data/adata_comb_refs.pt')
torch.save(list_ts, 
           f'{working_dir}/output/merfish/data/list_ts.pt')

import subprocess
subprocess.run(['bash', '-i', '-c', 'c'], capture_output=True, text=True)

list_ts = torch.load(f'{working_dir}/output/merfish/data/list_ts.pt')

new_obs = []; cell_ids = []
for sample in source_target_list.keys():
    source_sample, target_sample = source_target_list[sample]
    project_ind = list_ts[sample][0].flatten()
    source_obs = adata_comb.obs[adata_comb.obs['sample'] == source_sample]
    source_obs = source_obs[[
        'class', 'subclass', 'supertype', 'class_color', 'subclass_color', 
        'supertype_color', 'cluster', 'cluster_color',
        'parcellation_substructure', 'parcellation_substructure_color']]
    source_obs = source_obs.iloc[project_ind].reset_index(drop=True)
    target_obs = adata_comb.obs[adata_comb.obs['sample'] == target_sample]
    target_index = target_obs.index
    target_obs = target_obs[[
        'sample', 'source', 'x', 'y', 'k15_cluster', 'k15_cluster_colors', 
        'x_final', 'y_final']].reset_index(drop=True)
    target_obs = pd.concat([target_obs, source_obs], axis=1)\
        .set_index(target_index)
    cdist = list_ts[sample][2]
    target_obs['cdist'] = cdist
    new_obs.append(target_obs)

new_obs = pd.concat(new_obs)
adata_comb.obs['cdist'] = 0
update_indices = adata_comb.obs.index.isin(new_obs.index)
adata_comb.obs.loc[update_indices] = new_obs
# adata_comb.write('output/CURIO/data/adata_comb_project.h5ad')

tmp = adata_comb.obs[(adata_comb.obs['sample'] == 'CTRL1_L') & 
                     (adata_comb.obs['cdist'] < 0.06)]
tmp['subclass'] = tmp['subclass'].astype('category')\
    .cat.remove_unused_categories()
color_map = tmp.drop_duplicates('subclass')\
    .set_index('subclass')['subclass_color'].to_dict()
plt.clf()
plt.figure(figsize=((15, 15)))
ax  = sns.scatterplot(data=tmp, x='x_final', y='y_final', linewidth=0,
                hue='subclass', palette=color_map, s=10, legend=False)
ax.set(xlabel=None, ylabel=None)
sns.despine(bottom = True, left = True)
plt.legend(fontsize=14, markerscale=3)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(f'{working_dir}/tmp3.png', dpi=200)

tmp = adata_comb.obs[adata_comb.obs['sample'] == 'Zhuang-ABCA-1.060']
tmp['subclass'] = tmp['subclass'].cat.remove_unused_categories()
color_map = tmp.drop_duplicates('subclass')\
    .set_index('subclass')['subclass_color'].to_dict()
plt.clf()
plt.figure(figsize=((15, 15)))
ax  = sns.scatterplot(data=tmp, x='x', y='y', linewidth=0,
                hue='subclass', palette=color_map, s=15, legend=False)
ax.set(xlabel=None, ylabel=None)
sns.despine(bottom = True, left = True)
plt.legend(fontsize=9, markerscale=1)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(f'{working_dir}/tmp4.png', dpi=200)

################################################################################

def duplicate_anndata(adata, n_duplicates):
    adatas = []
    for i in range(n_duplicates):
        adata_copy = adata.copy()
        adata_copy.obs['sample'] = adata_copy.obs['sample'].astype(str) + \
              f'_dup{i:02d}'
        adata_copy.obs.index = adata_copy.obs['sample'] + '_' + \
              adata_copy.obs.index
        adatas.append(adata_copy)
    return ad.concat(adatas, merge='same')


'''
**Affine parameters**
- `iterations` - Iterations of the affine transformation.
- `alpha_basis` - The coefficient for updating the affine transformation
  parameter.
- `dist_penalty1` - Distance penalty parameter in affine transformation. When
  the distance of the query cell to the nearest neighbor in reference sample is
  greater than a distance threshold (by default, average cell distance), CAST
  Stack will add additional distance penalty. The initial cost function value of
  these cells will be multiplied by the `dist_penalty1`. The value `0` indicates
  no additional distance penalty.
- `bleeding` - When the reference sample is larger than the query sample, for
  efficient computation, only the region of the query sample with bleeding
  distance will be considered when calculating the cost function.
- `d_list` - CAST Stack will perform pre-location to find an initial alignment.
  The value in the `d_list` will be multiplied by the query sample to calculate
  the cost function. For example, 2 indicates the two-fold increase of the
  coordinates.
- `attention_params` - The attention mechanism to increase the penalty of the
  cells. It is invalid when the `dist_penalty` = 0.
    - `1st - attention_region` - The True/False index of all the cells of the
      query sample or None.
    - `2nd - double_penalty` - The `average cell distance / double_penalty` will
      be used in distance penalty for the cells with attention.
    - `3rd - penalty_inc_all` - The additional penalty for the attention cells.
      The initial cost function value of these cells will be multiplied by
      `penalty_inc_all`.
    - `4th - penalty_inc_both` - The additional penalty for the cells with
      distance penalty and attention. The initial cost function value of these
      cells will be multiplied by `(penalty_inc_both/dist_penalty + 1)`.

**FFD parameters**

    - `dist_penalty2` - Distance penalty parameter in FFD. Refer to
      `dist_penalty1`.
    - `alpha_basis_bs` - The coefficient for updating the FFD parameter.
    - `meshsize` - mesh size for the FFD.
    - `iterations_bs` - Iterations of the FFD.
    - `attention_params_bs` - The attention mechanism to increase the penalty of
      the cells. Refer to `attention_params`.
    - `mesh_weight` - The weight matrix for the mesh grid. The same size of the
      mesh or None.
'''
