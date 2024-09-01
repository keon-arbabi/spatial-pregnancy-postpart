import sys, os, warnings
import numpy as np, pandas as pd, anndata as ad
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings('ignore')

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

# CAST_MARK ####################################################################

import sys, os, torch, CAST, warnings
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import matplotlib.pyplot as plt, seaborn as sns
from scipy import sparse
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

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

# merge, store raw counts 
adata_query = sc.concat(adatas_query)
adata_query.layers['counts'] = adata_query.X.copy()
adata_query.var['gene_symbol'] = adata_query.var.index

# get qc metrics 
sc.pp.calculate_qc_metrics(
    adata_query, percent_top=[10, 20, 50, 100], inplace=True)
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
plt.savefig(f'{working_dir}/figures/merfish/qc_counts_genes.png')

# filter
print(adata_query.shape[0])
sc.pp.filter_cells(adata_query, min_counts=50)
sc.pp.filter_cells(adata_query, min_genes=10)
print(adata_query.shape[0])

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
    r('''
    library(scDblFinder)
    library(BiocParallel)
    path = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart/'
    sce = readRDS(paste0(path, 'output/merfish/data/adata_query.rds'))
    sce = scDblFinder(sce, samples='sample', BPPARAM=MulticoreParam())
    table(sce$scDblFinder.class)
    # singlet doublet
    #  944707  418018
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
plt.savefig(f'{working_dir}/figures/merfish/qc_doublets.png')

# plot relationship between doublet score and number of detected genes 
sns.jointplot(
    data=adata_query.obs,
    x='n_genes_by_counts',
    y='scDblFinder.score',
    kind='hex')
plt.savefig(f'{working_dir}/figures/merfish/qc_joint.png')

# normalize 
sc.pp.normalize_total(adata_query, target_sum=1e4)
sc.pp.log1p(adata_query, base=2)
# save
adata_query.write(f'{working_dir}/output/merfish/data/adata_query.h5ad')

######################################

# load reference, output from `merfish_zhuang_prep_atlas.py`
ref_dir = 'projects/def-wainberg/single-cell/ABC'
working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'

adata_ref_orig = ad.read_h5ad(
    f'{ref_dir}/expression_matrices/MERFISH-C57BL6J-638850-'
    'imputed/20240831/C57BL6J-638850-imputed-log2.h5ad')
cell_joined = pd.read_csv(f'{ref_dir}/Zeng/cells_joined.csv')

sections = ['C57BL6J-638850.49', 'C57BL6J-638850.48', 
            'C57BL6J-638850.47', 'C57BL6J-638850.46']
z_threshold = 5.5
adatas_ref = []

for section in sections:
    adata = adata_ref_orig[adata_ref_orig.obs['brain_section_label'] == section]
    adata.obs = adata.obs.reset_index()
    adata.obs = pd.merge(adata.obs, cell_joined, on='cell_label', how='left')
    adata.obs = adata.obs.set_index('cell_label', drop=True)
    excluded = ['unassigned', 'brain-unassigned', 'fiber tracts-unassigned']
    adata = adata[~adata.obs['parcellation_division'].isin(excluded)]
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
    adatas_ref.append(subset)

adata_ref = ad.concat(adatas_ref, axis=0, merge='same')
adata_ref.var = adata_ref_orig.var.reset_index().set_index('gene_symbol')
adata_ref.var['gene_symbol'] = adata_ref.var.index
adata_ref.var = adata_ref.var.rename_axis(None)
adata_ref = adata_ref[:, ~adata_ref.var.index.duplicated(keep='first')]

fig, axes = plt.subplots(1, 4, figsize=(25, 7))
fig.suptitle('Zeng ABCA Reference', fontsize=16)
for ax, (sample, data) in zip(axes, adata_ref.obs.groupby('sample')):
    ax.scatter(
        data['x'], data['y'],
        s=1, c=data['parcellation_division_color'])
    ax.set_title(sample)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig(f'{working_dir}/figures/merfish/zeng_reference.png',
            dpi=200, bbox_inches='tight', pad_inches=0)

# save 
adata_ref.X = adata_ref.X.astype(np.float32) # this is buggy, run twice
adata_ref.X = sparse.csr_matrix(adata_ref.X)
adata_ref.write(f'{working_dir}/output/merfish/data/adata_ref.h5ad')

######################################

def duplicate_dict(d, n):
    return {f'{k}_dup{i + 1}' if k in adata_ref.obs['sample'].unique()
        else k: v for k, v in d.items() for i in range(n or 1)}

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

# load anndata objects preprocessed above 
adata_query = ad.read_h5ad(
    f'{working_dir}/output/merfish/data/adata_query.h5ad')
adata_ref = ad.read_h5ad(
    f'{working_dir}/output/merfish/data/adata_ref.h5ad')

# combine, keepingly only the same metadat columns and genes
sample_names = sorted(set(adata_query.obs['sample'].unique()) |
                      set(adata_ref.obs['sample'].unique()))
# CRUICIAL
adata_comb = ad.concat([adata_query, adata_ref], axis=0, merge='same')
adata_comb.obs['sample'] = pd.Categorical(
    adata_comb.obs['sample'], categories=sample_names, ordered=True)
adata_comb = adata_comb[adata_comb.obs.sort_values('sample').index]
adata_comb = adata_comb.copy()

# get the coordinates and expression data for each sample
coords_raw = {
    s: np.array(adata_comb.obs[['x', 'y']])[adata_comb.obs['sample']==s]
    for s in sample_names}
exp_dict = {
    s: adata_comb[adata_comb.obs['sample']==s].X.toarray() 
    for s in sample_names}

# duplicate reference samples so that the embeddings are
# more balanced between query and reference
coords_raw_dup =  duplicate_dict(coords_raw, 3)
exp_dict_dup = duplicate_dict(exp_dict, 3)

# run cast mark
embed_dict_dup = CAST.CAST_MARK(
    coords_raw_dup, exp_dict_dup, 
    f'{working_dir}/output/merfish/CAST-MARK')

# remove duplicated embeddings 
embed_dict = {k.split('_dup')[0]: v.cpu().detach() 
              for k, v in embed_dict_dup.items()}
embed_stack = np.vstack([embed_dict[name].numpy() for name in sample_names])

# k-means cluster on cast embeddings
# and plot 
for n_clust in range(6, 20 + 1, 2):
    print(f'Clustering with k={n_clust}')
    kmeans = KMeans(n_clusters=n_clust, random_state=0).fit(embed_stack)
    cell_label = kmeans.labels_
    cluster_pl = sns.color_palette('Set3', n_clust)
    
    fig = plot_slices(sample_names, coords_raw, cell_label, cluster_pl, n_clust)
    fig.savefig(f'{working_dir}/figures/merfish/all_samples_k{str(n_clust)}.png')
    plt.close(fig)
    
    adata_comb.obs[f'k{n_clust}_cluster'] = cell_label
    color_map = {k: color for k, color in enumerate(cluster_pl.as_hex())}
    adata_comb.obs[f'k{n_clust}_cluster_colors'] = \
        pd.Series(cell_label).map(color_map).tolist()

# save
torch.save(coords_raw, f'{working_dir}/output/merfish/data/coords_raw.pt')
torch.save(exp_dict, f'{working_dir}/output/merfish/data/exp_dict.pt')
torch.save(embed_dict, f'{working_dir}/output/merfish/data/embed_dict.pt')
adata_comb.write(f'{working_dir}/output/merfish/data/adata_comb_cast_mark.h5ad')

# CAST_STACK ###################################################################

import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import sys, os, torch, CAST, warnings
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings('ignore')

# set paths 
working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/merfish/CAST-STACK', exist_ok=True)

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
adata_comb = ad.read_h5ad(f'{working_dir}/output/merfish/data/'
                          'adata_comb_cast_mark.h5ad')
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
        iterations=200, # 500
        dist_penalty1=0,
        bleeding=500,
        d_list=[3,2,1,1/2,1/3],
        attention_params=[None,3,1,0],
        #### FFD parameters                                    
        dist_penalty2=[0],
        alpha_basis_bs=[100], # 500
        meshsize=[8],
        iterations_bs=[50], # 400
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

plot_slices(adata_comb, coords_final, n_clust=16)
plt.savefig(f'{working_dir}/figures/merfish/all_samples_k16_final.png',
            dpi=300)

adata_comb.write(
    f'{working_dir}/output/merfish/data/adata_comb_cast_stack.h5ad')

# CAST_PROJECT #################################################################

import sys, os, torch, pickle, warnings
import matplotlib.pyplot as plt, seaborn as sns
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

sys.path.insert(0, 'projects/def-wainberg/karbabi/CAST')
import CAST
print(CAST.__file__)

warnings.filterwarnings('ignore')

working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/merfish/CAST-PROJECT', exist_ok=True)

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

batch_key = 'sample'
level = 'subclass'
sample = 'CTRL1_L'
source_sample = 'C57BL6J-638850.47_R'
target_sample = sample

color_dict = adata_comb.obs\
    .drop_duplicates()\
    .set_index(level)[f'{level}_color']\
    .to_dict()
color_dict['Unknown'] = '#A9A9A9'

adata_subset = adata_comb[
    adata_comb.obs['sample'].isin([source_sample, target_sample])].copy()

# Run Harmony integration on the subset
adata_subset = CAST.Harmony_integration(
    sdata_inte=adata_subset,
    scaled_layer='log1p_norm_scaled',
    use_highly_variable_t=False,
    batch_key=batch_key,
    umap_n_neighbors=50,
    umap_n_pcs=30,
    min_dist=0.01,
    spread_t=5,
    source_sample_ctype_col=level,
    output_path=f'{working_dir}/output/merfish/CAST-PROJECT',
    n_components=50,
    ifplot=True,
    ifcombat=True
)

# Now use this Harmony-integrated subset for CAST projection
print(f"Processing {sample}")
output_dir_t = f'{working_dir}/output/merfish/CAST-PROJECT/' \
    f'{source_sample}_to_{target_sample}'
os.makedirs(output_dir_t, exist_ok=True)

# Parameters that affect physical distance
ave_dist_fold = 1
alignment_shift_adjustment = 1

_, list_t = CAST.CAST_PROJECT(
    sdata_inte=adata_subset,
    source_sample=source_sample,
    target_sample=target_sample, 
    coords_source=np.array(adata_subset[
        adata_subset.obs['sample'] == \
            source_sample].obs.loc[:,['x','y']]),
    coords_target=np.array(adata_subset[
        adata_subset.obs['sample'] == \
            target_sample].obs.loc[:,['x_final','y_final']]),
    k2=1,
    scaled_layer='log1p_norm_scaled', 
    raw_layer='log1p_norm_scaled',
    batch_key=batch_key, 
    use_highly_variable_t=False,  
    ifplot=False,
    n_components=50,
    source_sample_ctype_col=level, 
    output_path=output_dir_t, 
    integration_strategy=None,  
    pc_feature='X_pca_harmony', 
    umap_feature='X_umap', 
    ave_dist_fold=ave_dist_fold,
    alignment_shift_adjustment=alignment_shift_adjustment,
    color_dict=color_dict,
    save_result=False)

print(list_t)

project_ind = list_t[0][:, 0].flatten()
source_obs = adata_comb.obs[
    adata_comb.obs['sample'] == source_sample].copy()
target_obs = adata_comb.obs[
    adata_comb.obs['sample'] == target_sample].copy()
target_index = target_obs.index
target_obs = target_obs.reset_index(drop=True)

for col in ['parcellation_division', 'parcellation_structure', 
            'parcellation_substructure',
            'class', 'subclass', 'supertype', 'cluster']:
    target_obs[col] = pd.NA
    target_obs[col] = source_obs[col].iloc[project_ind].values
    target_obs[f'{col}_color'] = source_obs[f'{col}_color']\
        .iloc[project_ind].values

target_obs['ref_cell_id'] = source_obs.index[project_ind]
target_obs['cosine_knn_weight'] = list_t[1][:, 0]
target_obs['cosine_knn_cdist'] = list_t[2][:, 0]
target_obs['cosine_knn_physical_dist'] = list_t[3][:, 0]

new_obs = target_obs.set_index(target_index)

adata_comb_i = adata_comb.copy()
for col in new_obs.columns:
    if col not in adata_comb_i.obs.columns:
        adata_comb_i.obs[col] = pd.NA
adata_comb_i.obs.loc[new_obs.index] = new_obs


def create_comparison_plot(adata_comb_i, adata_comb, col, query_sample, 
                           ref_sample, random_state=42):
    
    query_data = adata_comb_i.obs[adata_comb_i.obs['sample'] == query_sample]
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
        ax.set_title(f"{data['sample'].iloc[0]} (Cells: {len(data)})\n"
                     f"{col}: {selected_value}")
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
    plt.savefig(f'{save_dir}{selected_value}.png', dpi=200)
    plt.close(fig)

create_comparison_plot(
    adata_comb_i, adata_comb, col='subclass', 
    query_sample='CTRL1_L', ref_sample='C57BL6J-638850.47_R',
    random_state=None)









































import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import sys, os, torch, pickle, warnings
import matplotlib.pyplot as plt, seaborn as sns

sys.path.insert(0, 'projects/def-wainberg/karbabi/CAST')
import CAST
print(CAST.__file__)

warnings.filterwarnings('ignore')

working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/merfish/CAST-PROJECT', exist_ok=True)

def transfer_region_labels(knn_ind, cosine_weights, physical_dists, 
                           source_region_labels, k=10, alpha=0.7):
    n_target_cells = knn_ind.shape[0]
    knn_ind = knn_ind[:, :k]
    cosine_weights = cosine_weights[:, :k]
    physical_weights = 1 / (physical_dists[:, :k] + 1e-6)
    combined_weights = alpha * physical_weights + (1 - alpha) * cosine_weights
    combined_weights /= np.sum(combined_weights, axis=1, keepdims=True)
    unique_labels, inverse = np.unique(
        source_region_labels, return_inverse=True)
    n_labels = len(unique_labels)
    weighted_counts = np.zeros((n_target_cells, n_labels))
    np.add.at(
        weighted_counts, 
        (np.arange(n_target_cells)[:, None], inverse[knn_ind]), 
        combined_weights)
    
    most_common_indices = np.argmax(weighted_counts, axis=1)
    return unique_labels[most_common_indices]

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

batch_key = 'sample'
level = 'subclass'
source_target_list = {
    key: ['C57BL6J-638850.47_R', key] for key in coords_final.keys()}

color_dict = adata_comb.obs\
    .drop_duplicates()\
    .set_index(level)[f'{level}_color']\
    .to_dict()
color_dict['Unknown'] = '#A9A9A9'

list_ts = {}
for sample in source_target_list.keys():
    print(sample)
    source_sample, target_sample = source_target_list[sample]
    output_dir_t = f'{working_dir}/output/merfish/CAST-PROJECT/' \
        f'{source_sample}_to_{target_sample}'
    os.makedirs(output_dir_t, exist_ok=True)
    _, list_ts[sample] = CAST.CAST_PROJECT(
        sdata_inte=adata_comb[
            np.isin(adata_comb.obs[batch_key], 
                    [source_sample, target_sample])],
        source_sample=source_sample,
        target_sample=target_sample, 
        coords_source=np.array(
            adata_comb[np.isin(adata_comb.obs[batch_key], source_sample),:]
                .obs.loc[:,['x','y']]),
        coords_target=np.array(
            adata_comb[np.isin(adata_comb.obs[batch_key], target_sample),:]
                .obs.loc[:,['x_final','y_final']]),
        k2 = 200,
        scaled_layer = 'log1p_norm_scaled', 
        raw_layer = 'log1p_norm_scaled',
        batch_key=batch_key, 
        use_highly_variable_t=False,  
        ifplot = False,
        n_components = 50,
        source_sample_ctype_col=level, 
        output_path=output_dir_t, 
        integration_strategy='Harmony', 
        ave_dist_fold = 1, 
        alignment_shift_adjustment = 50,
        color_dict=color_dict,
        save_result=False)
    print(list_ts[sample])
    
# with open(f'{working_dir}/output/merfish/data/list_ts.pickle', 'wb') as f:
#     pickle.dump(list_ts, f)
with open(f'{working_dir}/output/merfish/data/list_ts.pickle', 'rb') as f:
    list_ts = pickle.load(f)

new_obs = []
for sample, (source_sample, target_sample) in source_target_list.items():
    project_ind = list_ts[sample][0][:, 0].flatten()
    source_obs = adata_comb.obs[
        adata_comb.obs['sample'] == source_sample].copy()
    target_obs = adata_comb.obs[
        adata_comb.obs['sample'] == target_sample].copy()
    target_index = target_obs.index
    target_obs = target_obs.reset_index(drop=True)

    for col in ['parcellation_division', 'parcellation_structure', 
                'parcellation_substructure',
                'class', 'subclass', 'supertype', 'cluster']:
        target_obs[col] = pd.NA
        target_obs[col] = source_obs[col].iloc[project_ind].values
        target_obs[f'{col}_color'] = source_obs[f'{col}_color']\
            .iloc[project_ind].values

    # for level in ['division', 'structure', 'substructure']:
    #     region_labels = transfer_region_labels(
    #         list_ts[sample][0],
    #         list_ts[sample][1],
    #         list_ts[sample][3],
    #         source_obs[f'parcellation_{level}'].values,
    #         k=50, alpha=0.7)
    #     target_obs[f'parcellation_{level}'] = region_labels
    #     color_mapping = dict(zip(
    #         source_obs[f'parcellation_{level}'],
    #         source_obs[f'parcellation_{level}_color']))
    #     target_obs[f'parcellation_{level}_color'] = \
    #         pd.Series(region_labels).map(color_mapping).values

    target_obs['ref_cell_id'] = source_obs.index[project_ind]
    target_obs['cosine_knn_weight'] = list_ts[sample][1][:, 0]
    target_obs['cosine_knn_cdist'] = list_ts[sample][2][:, 0]
    target_obs['cosine_knn_physical_dist'] = list_ts[sample][3][:, 0]
    new_obs.append(target_obs.set_index(target_index))

new_obs = pd.concat(new_obs)
adata_comb_i = adata_comb.copy()
for col in new_obs.columns:
    if col not in adata_comb_i.obs.columns:
        adata_comb_i.obs[col] = pd.NA
adata_comb_i.obs.loc[new_obs.index] = new_obs

# # save
# adata_comb.write(
#     f'{working_dir}/output/merfish/data/adata_comb_cast_project.h5ad')


def create_comparison_plot(adata_comb_i, adata_comb, col, query_sample, 
                           ref_sample, random_state=42):
    
    query_data = adata_comb_i.obs[adata_comb_i.obs['sample'] == query_sample]
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
        ax.set_title(f"{data['sample'].iloc[0]} (Cells: {len(data)})\n"
                     f"{col}: {selected_value}")
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
    plt.savefig(f'{save_dir}{selected_value}.png', dpi=200)
    plt.close(fig)

create_comparison_plot(
    adata_comb_i, adata_comb, col='subclass', 
    query_sample='CTRL1_L', ref_sample='C57BL6J-638850.47_R',
    random_state=17)











col = 'parcellation_substructure'
plot_df = adata_comb_i.obs[adata_comb.obs['sample'] == 'CTRL1_L']
plot_df[col] = plot_df[col].astype('category')\
    .cat.remove_unused_categories()
color_map = plot_df.drop_duplicates(col)\
    .set_index(col)[f'{col}_color'].to_dict()
plt.clf()
plt.figure(figsize=((15, 15)))
ax  = sns.scatterplot(
    data=plot_df, x='x_final', y='y_final', linewidth=0,
    hue=col, palette=color_map, s=10, legend=False)
ax.set(xlabel=None, ylabel=None)
sns.despine(bottom = True, left = True)
plt.legend(fontsize=14, markerscale=3)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(f'{working_dir}/figures/tmp.png', dpi=200)

plot_df = adata_comb.obs[adata_comb.obs['sample'] == 'C57BL6J-638850.47_R']
plot_df[col] = plot_df[col].cat.remove_unused_categories()
color_map = plot_df.drop_duplicates(col)\
    .set_index(col)[f'{col}_color'].to_dict()
plt.clf()
plt.figure(figsize=((15, 15)))
ax  = sns.scatterplot(
    data=plot_df, x='x', y='y', linewidth=0,
    hue=col, palette=color_map, s=15, legend=False)
ax.set(xlabel=None, ylabel=None)
sns.despine(bottom = True, left = True)
plt.legend(fontsize=9, markerscale=1)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(f'{working_dir}/figures/tmp2.png', dpi=200)

################################################################################

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
