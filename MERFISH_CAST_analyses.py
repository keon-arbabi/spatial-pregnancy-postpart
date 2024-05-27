import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import sys, os, torch, CAST, warnings
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore")

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from utils import Timer, debug
debug(third_party=True)

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
    adata.obs['y'] = -adata.obs['y']
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
    adata = adata[adata.obs['Custom cell groups'] != 'Unclustered']
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

# embed_dict = CAST.CAST_MARK(coords_raw, exp_dict, output_dir)
embed_dict = torch.load(f'{output_dir}/demo_embed_dict.pt', map_location='cpu')

# https://github.com/wanglab-broad/CAST/blob/main/CAST/visualize.py#L7
from sklearn.cluster import KMeans
num_plot = len(sample_names)
plot_row = int(np.floor(num_plot/5) + 1)
embed_stack = embed_dict[sample_names[0]].cpu().detach().numpy()
for i in range(1,num_plot):
    embed_stack = np.row_stack(
        (embed_stack,embed_dict[sample_names[i]].cpu().detach().numpy()))
    
print(f'Perform KMeans clustering on {embed_stack.shape[0]} cells...')
k = 15
kmeans = KMeans(n_clusters=k, random_state=0).fit(embed_stack)
cell_label = kmeans.labels_
cluster_pl = sns.color_palette('Set3', len(np.unique(cell_label)))
np.random.shuffle(cluster_pl)

print(f'Plotting the KMeans clustering results...')
cell_label_idx = 0
plt.figure(figsize=((30, 5*plot_row)))
for j in range(num_plot):
    plt.subplot(plot_row, 5, j+1)
    coords_raw0 = coords_raw[sample_names[j]]
    col=coords_raw0[:,0].tolist()
    row=coords_raw0[:,1].tolist()
    cell_type_t = cell_label[cell_label_idx:
        (cell_label_idx + coords_raw0.shape[0])]
    cell_label_idx += coords_raw0.shape[0]
    size = 1.8 if 'Zhuang-ABCA' in sample_names[j] else 0.3
    for i in set(cell_type_t):
        plt.scatter(np.array(col)[cell_type_t == i],
        np.array(row)[cell_type_t == i], s=size, edgecolors='none',
        c=np.array(cluster_pl)[cell_type_t[cell_type_t == i]], 
        label = str(i), rasterized=True)
    plt.title(sample_names[j] + ' (KMeans, k = ' + str(k) + ')',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
plt.savefig(f'{output_dir}/all_samples_trained_k{str(k)}.png', dpi=200)

adata_comb.obs[f'k{k}_cluster'] = cell_label
color_map = {k: color for k, color in enumerate(cluster_pl.as_hex())}
adata_comb.obs[f'k{k}_cluster_colors'] = pd.Series(cell_label)\
    .map(color_map).tolist()
adata_comb.write(f'{output_dir}/adata_comb.h5ad')

torch.save(coords_raw, f'{output_dir}/coords_raw.pt')
torch.save(exp_dict, f'{output_dir}/exp_dict.pt')

# CAST_STACK ###################################################################

import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import sys, os, torch, CAST, warnings
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore")

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from utils import Timer, debug
debug(third_party=True)

# torch.cuda.set_per_process_memory_fraction(0.5, device=0)
# torch.backends.cuda.matmul.allow_tf32 = False 

work_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
output_dir = 'output/CAST-STACK'
os.chdir(work_dir)
os.makedirs(output_dir, exist_ok=True)

adata_comb = ad.read_h5ad('output/CAST-MARK/adata_comb.h5ad')
sample_names = adata_comb.obs['sample'].unique()

coords_raw = torch.load('output/CAST-MARK/coords_raw.pt', map_location='cpu')
embed_dict = torch.load('output/CAST-MARK/demo_embed_dict.pt', 
                        map_location='cpu')
graph_list = {
    'Ctrl1_L': ['Ctrl1_L', 'Zhuang-ABCA-1.060'],
    'Ctrl1_R': ['Ctrl1_R', 'Zhuang-ABCA-1.060'],
    'Ctrl2_L': ['Ctrl2_L', 'Zhuang-ABCA-1.060'],
    'Ctrl2_R': ['Ctrl2_R', 'Zhuang-ABCA-1.060'],
    'Ctrl3_L': ['Ctrl3_L', 'Zhuang-ABCA-1.060'],
    'Ctrl3_R': ['Ctrl3_R', 'Zhuang-ABCA-1.060'],
    'Preg1_L': ['Preg1_L', 'Zhuang-ABCA-1.060'],
    'Preg1_R': ['Preg1_R', 'Zhuang-ABCA-1.060'],
    'Preg2_L': ['Preg2_L', 'Zhuang-ABCA-1.060'],
    'Preg2_R': ['Preg2_R', 'Zhuang-ABCA-1.060'],
    'Preg3_L': ['Preg3_L', 'Zhuang-ABCA-1.060'],
    'Preg3_R': ['Preg3_R', 'Zhuang-ABCA-1.060'],
    'PostPart1_L': ['PostPart1_L', 'Zhuang-ABCA-1.060'],
    'PostPart1_R': ['PostPart1_R', 'Zhuang-ABCA-1.060'],
    'PostPart2_L': ['PostPart2_L', 'Zhuang-ABCA-1.060'],
    'PostPart2_R': ['PostPart2_R', 'Zhuang-ABCA-1.060'],
    'PostPart3_L': ['PostPart3_L', 'Zhuang-ABCA-1.060'],
    'PostPart3_R': ['PostPart3_R', 'Zhuang-ABCA-1.060']}

rep='Ctrl1_L'

coords_final = {}
for rep in graph_list.keys():
    params_dist = CAST.reg_params(
        dataname = graph_list[rep][0],
        gpu = 0 if torch.cuda.is_available() else -1, 
        diff_step = 5,
        #### Affine parameters
        iterations=500,
        dist_penalty1=0,
        bleeding=500,
        d_list = [3,2,1,1/2,1/3],
        attention_params = [None,3,1,0], 
        #### FFD parameters    
        dist_penalty2 = [0],
        alpha_basis_bs = [500],
        meshsize = [8],
        iterations_bs = [400],
        attention_params_bs = [[None,3,1,0]],
        mesh_weight = [None])
    params_dist.alpha_basis = torch.Tensor(
        [1/1000,1/1000,1/50,5,5]).reshape(5,1).to(params_dist.device)
    
    coords_final[rep] = CAST.CAST_STACK(
        coords_raw, embed_dict, output_dir, graph_list[rep], 
        params_dist, rescale=True)











from torch import nn
from collections import OrderedDict
cos = nn.CosineSimilarity(dim=1)

embed_dict_query = {
    k: v for k, v in embed_dict.items() if 'Zhuang-ABCA' not in k}
embed_dict_ref = {
    k: v for k, v in embed_dict.items() if 'Zhuang-ABCA' in k}

graph_list = {}
for query_key, query_tensor in embed_dict_query.items():
    similarities = {}
    for ref_key, ref_tensor in embed_dict_ref.items():
        min_rows = min(query_tensor.size(0), ref_tensor.size(0))
        similarity = \
            cos( query_tensor[:min_rows], ref_tensor[:min_rows]).mean().item()
        similarities[ref_key] = similarity
    most_similar_ref = max(similarities, key=similarities.get)
    graph_list[query_key] = [query_key, most_similar_ref]

graph_list = OrderedDict(sorted(graph_list.items()))




embed_stack = np.vstack([embed_dict[name].cpu().detach().numpy()
                        for name in sample_names])
adata_embed = sc.AnnData(embed_stack)
adata_embed.obs = adata_comb.obs
sc.pp.pca(adata_embed, n_comps=30)
sc.pp.neighbors(adata_embed)
sc.tl.umap(adata_embed)

sc.set_figure_params(dpi_save=400, vector_friendly=False, figsize=[9,8])
sc.pl.umap(adata_embed,
           color=['k15_cluster'],
           legend_fontoutline=1.2,
           frameon=False,
           ncols=1, size=2,
           save='figures/kmeans_umap.png')


from sklearn.decomposition import PCA
pca_data = PCA(n_components=10).fit_transform(embed_stack)

reduced_tensors = {}; index = 0
for key, tensor in embed_dict.items():
    length = tensor.size(0)
    reduced_tensors[key] = torch.tensor(pca_data[index:index + length])
    index += length
    
from torch import nn
cos = nn.CosineSimilarity(dim=1)
graph_list = {}

query_keys = [k for k in reduced_tensors if 'Zhuang-ABCA' in k]
ref_keys = [k for k in reduced_tensors if 'Zhuang-ABCA' not in k]

for qk in query_keys:
    qt = reduced_tensors[qk]
    similarities = {
        rk: cos(qt, reduced_tensors[rk][:qt.size(0)]).mean().item()
        for rk in ref_keys
    }
    graph_list[qk] = [qk, max(similarities, key=similarities.get)]



for key, k in embed_dict.items():
    print(f"Size of tensor '{key}': {k.size()}")


adata = ad.read_h5ad('../../spatial/Kalish/pregnancy-postpart/raw/Ctrl1.hdf5')
cell_sums = adata.X.sum(axis=1)
no_expression_count = (cell_sums == 0).sum()
total_cells = adata.n_obs

print(f'Number of cells with no expression: {no_expression_count}')
print(f'Total number of cells: {total_cells}')