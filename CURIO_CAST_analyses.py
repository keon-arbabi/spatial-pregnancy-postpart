import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import sys, os, torch, CAST, shutil, warnings
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore")

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from utils import debug
debug(third_party=True)

# Prep raw images ####################################################################

work_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
data_dir = '../../spatial/Kalish/pregnancy-postpart/CURIO'
output_dir = f'{data_dir}/rotate-split-raw'
figure_dir = 'figures/CURIO'
os.chdir(work_dir)
os.makedirs(figure_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

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
    'Preg1_1', 'Preg1_2', 'Virg1_1', 'Virg1_2', 'Virg2_1'] 
params = {
    'Preg1_1': {'L': (180, -7200), 'R': (180, -6100)}, 
    'Preg1_2': {'L': (180, -7200), 'R': (180, -6100)},
    'Virg1_1': {'L': (-55, -500), 'R': (-55, -1000)},
    'Virg1_2': {'L': (-55, -500), 'R': (-55, -1000)},
    'Virg2_1': {'L': (140, -800), 'R': (140, -1500)}}

plot_index = 1
plt.figure(figsize=(3 * 5, 6 * 4))

for sample in sample_names:
    from sklearn.cluster import DBSCAN
    adata = ad.read_h5ad(f'{data_dir}/raw-h5ad/{sample}.h5ad')
    coords = adata.obs[['SPATIAL_1', 'SPATIAL_2']]
    if sample in ['Virg1_1', 'Virg1_2']:
        outliers = DBSCAN(eps=500, min_samples=150).fit(coords) 
    else:
        outliers = DBSCAN(eps=500, min_samples=50).fit(coords) 
    adata = adata[outliers.labels_ != -1]
    coords_filt = adata.obs[['SPATIAL_1', 'SPATIAL_2']]
    print(f"Removed {len(coords) - len(coords_filt)} points.")
    
    for hemi in ['L', 'R']:
        plt.subplot(6, 3, plot_index)
        angle, value = params[sample][hemi]
        coords_hemi = rotate_and_crop(
            coords_filt, angle=angle,
            x_max=value if hemi == 'L' else None,
            x_min=None if hemi == 'L' else value,
            mirror_y=(hemi == 'R'))
        adata_hemi = adata[coords_hemi.index]
        adata_hemi.obs[['x', 'y']] = coords_hemi
        print(f'[{sample}] {adata_hemi.shape[0]} cells')
        adata_hemi.write(f'{data_dir}/rotate-split-raw/{sample}_{hemi}.h5ad')

        spines = plt.gca().spines
        spines['top'].set_visible(False)
        spines['right'].set_visible(False)
        sns.scatterplot(data=coords_hemi, x='x', y='y', color='black', s=5)
        plt.title(f'{sample} - {hemi}')
        plot_index += 1  
        
plt.tight_layout() 
plt.savefig(f'{figure_dir}/crop_and_rotate_all.png', 
            dpi=200, bbox_inches='tight', pad_inches=0)

# CAST_MARK ####################################################################

work_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
os.chdir(work_dir)

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
    adata.obs['source'] = 'Zhuang-ABCA-Reference'
    adata.obs = adata.obs.join(
        metadata, on='cell_label', lsuffix='_l', rsuffix='')
    adata.obs['y'] = -adata.obs['y']
    adata.var.reset_index()
    adata.var.index = adata.var['gene_symbol']
    print(f'[{data}] {adata.shape[0]} cells')
    adatas_ref.append(adata)
adata_ref = ad.concat(adatas_ref)
adata_ref.var['gene_symbol'] = adata_ref.var.index
adata_ref.write('output/CURIO/data/adata_ref.h5ad')

query_dir = '../../spatial/Kalish/pregnancy-postpart/CURIO/rotate-split-raw'
samples_query = [file.replace('.h5ad', '') for file in os.listdir(query_dir)]
samples_query = sorted(samples_query)

adatas_query = []
for sample in samples_query:
    adata = ad.read_h5ad(f'{query_dir}/{sample}.h5ad')
    adata.obs['sample'] = sample
    adata.obs['source'] = 'CURIO'
    adata.obs[[
        'class', 'class_color', 'subclass', 'subclass_color',
        'supertype', 'supertype_color', 'parcellation_substructure',
        'parcellation_substructure_color']] = 'Unknown'
    print(f'[{sample}] {adata.shape[0]} cells')
    adatas_query.append(adata)
adata_query = sc.concat(adatas_query)
adata_query.var['gene_symbol'] = adata_query.var.index
adata_query.write('output/CURIO/data/adata_query.h5ad')

adata_comb = ad.concat([adata_ref, adata_query])
all_var = [a.var for a in [adata_ref, adata_query]]
all_var = pd.concat(all_var, join='inner')
all_var = all_var[~all_var.duplicated()]
adata_comb.var = all_var.loc[adata_comb.var_names]

adata_comb.layers['norm'] = sc.pp.normalize_total(
    adata_comb, target_sum=1e4, inplace=False)['X'].todense()

sample_names = adata_comb.obs['sample'].unique()
coords_raw = {
    sample: np.array(adata_comb.obs[['x', 'y']])
    [adata_comb.obs['sample'] == sample] for sample in sample_names}
exp_dict = {
    sample: adata_comb[adata_comb.obs['sample'] == sample]
    .layers['norm'] for sample in sample_names}

# embed_dict = CAST.CAST_MARK(coords_raw, exp_dict, 'output/CURIO/CAST-MARK')
# shutil.move('output/CURIO/CAST-MARK/demo_embed_dict.pt',
#             'output/CURIO/data/demo_embed_dict.pt')
embed_dict = torch.load('output/CURIO/data/demo_embed_dict.pt', 
                        map_location='cpu')

# https://github.com/wanglab-broad/CAST/blob/main/CAST/visualize.py#L7
from sklearn.cluster import KMeans
num_plot = len(sample_names)
plot_row = int(np.floor(num_plot/5) + 1)
embed_stack = np.vstack([embed_dict[name].cpu().detach().numpy()
                        for name in sample_names])
n_clust = 15
kmeans = KMeans(n_clusters=n_clust, random_state=0).fit(embed_stack)
cell_label = kmeans.labels_
cluster_pl = sns.color_palette('Set3', len(np.unique(cell_label)))
np.random.shuffle(cluster_pl)

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
    size = 1.5 if 'Zhuang-ABCA' in sample_names[j] else 10
    for i in set(cell_type_t):
        plt.scatter(np.array(col)[cell_type_t == i],
        np.array(row)[cell_type_t == i], s=size, edgecolors='none',
        c=np.array(cluster_pl)[cell_type_t[cell_type_t == i]], 
        label = str(i), rasterized=True)
    plt.title(sample_names[j] + 
        ' (KMeans, k = ' + str(n_clust) + ')',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
plt.savefig(f'figures/CURIO/all_samples_trained_k{str(n_clust)}.png', dpi=200)

adata_comb.obs[f'k{n_clust}_cluster'] = cell_label
color_map = {k: color for k, color in enumerate(cluster_pl.as_hex())}
adata_comb.obs[f'k{n_clust}_cluster_colors'] = \
    pd.Series(cell_label).map(color_map).tolist()
adata_comb.obsm['CAST_MARK_embed'] = embed_stack

sc.pp.pca(adata_comb, n_comps=30)
sc.pp.neighbors(adata_comb)
sc.tl.umap(adata_comb)

sc.set_figure_params(dpi_save=400, vector_friendly=True, figsize=[6,6])
sc.pl.umap(adata_comb,
           color=[f'k15_cluster_cat', 'source'],
           legend_fontoutline=1.2, legend_loc='on data',
           ncols=2, size=2, return_fig=True)
plt.savefig('figures/CURIO/CAST_MARK_umap.png')

adata_comb.layers['norm'] = np.array(adata_comb.layers['norm'])
adata_comb.write('output/CURIO/data/adata_comb_mark.h5ad')
torch.save(coords_raw, 'output/CURIO/data/coords_raw.pt')
torch.save(exp_dict, 'output/CURIO/data/exp_dict.pt')

# CAST_STACK ###################################################################

work_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
output_dir = 'output/CURIO/CAST-STACK'
os.chdir(work_dir)
os.makedirs(output_dir, exist_ok=True)

adata_comb = ad.read_h5ad('output/CURIO/data/adata_comb_mark.h5ad')
coords_raw = torch.load('output/CURIO/data/coords_raw.pt')
embed_dict = torch.load('output/CURIO/CAST-MARK/demo_embed_dict.pt',
                        map_location='cpu')

query_reference_list = {
    'Preg1_1_L': ['Preg1_1_L', 'Zhuang-ABCA-1.060'],
    'Preg1_1_R': ['Preg1_1_R', 'Zhuang-ABCA-1.060'],
    'Preg1_2_L': ['Preg1_2_L', 'Zhuang-ABCA-1.060'],
    'Preg1_2_R': ['Preg1_2_R', 'Zhuang-ABCA-1.060'],
    'Virg1_1_L': ['Virg1_1_L', 'Zhuang-ABCA-1.060'],
    'Virg1_1_R': ['Virg1_1_R', 'Zhuang-ABCA-1.060'],
    'Virg1_2_L': ['Virg1_2_L', 'Zhuang-ABCA-1.060'],
    'Virg1_2_R': ['Virg1_2_R', 'Zhuang-ABCA-1.060'],
    'Virg2_1_L': ['Virg2_1_L', 'Zhuang-ABCA-1.060'],
    'Virg2_1_R': ['Virg2_1_R', 'Zhuang-ABCA-1.060']}

coords_final = {}
for sample in sorted(query_reference_list.keys()):
    params_dist = CAST.reg_params(
        dataname = query_reference_list[sample][0],
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
    
    coords_final[sample] = CAST.CAST_STACK(
        coords_raw, embed_dict, output_dir, query_reference_list[sample], 
        params_dist, rescale=True)

sample_names = sorted(list(coords_final.keys()))
adata_comb.obs.index = adata_comb.obs['sample'].astype(str) + '_' + \
    adata_comb.obs.index
cell_index = adata_comb.obs.index[adata_comb.obs['source'] == 'CURIO']
coords_stack = np.vstack([
    coords_final[sample][sample] for sample in sample_names])
coords_df = pd.DataFrame(
    coords_stack, columns=['x_final', 'y_final'], index=cell_index)
adata_comb.obs = adata_comb.obs.join(coords_df)

df = adata_comb.obs[adata_comb.obs['source'] == 'CURIO']
num_plot = len(sample_names)

plt.figure(figsize=((30, 15)))
for j in range(num_plot):
    plt.subplot(3, 5, j+1)
    coords_final0 = coords_final[sample_names[j]][sample_names[j]]
    col=coords_final0[:,0].tolist()
    row=coords_final0[:,1].tolist()
    current_index = df.index[df.index.str.contains(sample_names[j])]
    plt.scatter(x=col, y=row, s=4,
                color=df.loc[current_index, 'k15_cluster_colors'])
    plt.title(sample_names[j] + ' (KMeans, k = 15)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
plt.savefig(
    f'figures/CURIO/all_samples_trained_k15_final.png', dpi=200)

# torch.save(coords_final, 'output/CURIO/data/coords_final.pt')
# coords_final = torch.load('output/CURIO/data/coords_final.pt')
# adata_comb.write('output/CURIO/data/adata_comb_stack.h5ad')

# CAST_PROJECT #################################################################

work_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
output_dir = 'output/CURIO/CAST-PROJECT'
os.chdir(work_dir)
os.makedirs(output_dir, exist_ok=True)

source_target_list = {
    'Preg1_1_L': ['Zhuang-ABCA-1.060', 'Preg1_1_L'],
    'Preg1_1_R': ['Zhuang-ABCA-1.060', 'Preg1_1_R'],
    'Preg1_2_L': ['Zhuang-ABCA-1.060', 'Preg1_2_L'],
    'Preg1_2_R': ['Zhuang-ABCA-1.060', 'Preg1_2_R'],
    'Virg1_1_L': ['Zhuang-ABCA-1.060', 'Virg1_1_L'],
    'Virg1_1_R': ['Zhuang-ABCA-1.060', 'Virg1_1_R'],
    'Virg1_2_L': ['Zhuang-ABCA-1.060', 'Virg1_2_L'],
    'Virg1_2_R': ['Zhuang-ABCA-1.060', 'Virg1_2_R'],
    'Virg2_1_L': ['Zhuang-ABCA-1.060', 'Virg2_1_L'],
    'Virg2_1_R': ['Zhuang-ABCA-1.060', 'Virg2_1_R']}

adata_comb = ad.read_h5ad('output/CURIO/data/adata_comb_stack.h5ad')
adata_comb = CAST.preprocess_fast(adata_comb, mode='default')
batch_key = 'sample'
color_dict = adata_comb.obs\
    .drop_duplicates().set_index('class')['class_color']\
    .to_dict()
color_dict['Unknown'] = '#A9A9A9'

adata_comb_refs = {}
list_ts = {}
for sample in source_target_list.keys():
    print(sample)
    source_sample, target_sample = source_target_list[sample]
    output_dir_t = f'{output_dir}/{source_sample}_to_{target_sample}'
    os.makedirs(output_dir_t, exist_ok=True)
    adata_comb_refs[sample], list_ts[sample] = \
    CAST.CAST_PROJECT(
        sdata_inte=adata_comb[
            np.isin(adata_comb.obs[batch_key], [source_sample, target_sample])],
        use_highly_variable_t=False,  
        source_sample=source_sample,
        target_sample=target_sample, 
        coords_source=np.array(
            adata_comb[np.isin(adata_comb.obs[batch_key], source_sample),:]
            .obs.loc[:,['x','y']]),
        coords_target=np.array(
            adata_comb[np.isin(adata_comb.obs[batch_key], target_sample),:]
            .obs.loc[:,['x_final','y_final']]), # final coords
        scaled_layer='log1p_norm_scaled',
        batch_key=batch_key, 
        source_sample_ctype_col='class', 
        output_path=output_dir_t, 
        integration_strategy='Harmony', 
        color_dict=color_dict,
        save_result=False
)
# torch.save(adata_comb_refs, 'output/CURIO/data/adata_comb_refs.pt')
# torch.save(list_ts, 'output/CURIO/data/list_ts.pt')

list_ts = torch.load('output/CURIO/data/list_ts.pt')

new_obs = []; cell_ids = []
for sample in source_target_list.keys():
    source_sample, target_sample = source_target_list[sample]
    project_ind = list_ts[sample][0].flatten()
    cdist = list_ts[sample][2]
    source_obs = adata_comb.obs[adata_comb.obs['sample'] == source_sample]
    source_obs = source_obs[[
        'class', 'subclass', 'supertype', 'class_color', 'subclass_color', 
        'supertype_color', 'parcellation_substructure', 
        'parcellation_substructure_color']]
    source_obs = source_obs.iloc[project_ind].reset_index(drop=True)
    target_obs = adata_comb.obs[adata_comb.obs['sample'] == target_sample]
    target_index = target_obs.index
    target_obs = target_obs[[
        'sample', 'source', 'x', 'y', 'k15_cluster', 'k15_cluster_colors', 
        'x_final', 'y_final']].reset_index(drop=True)
    target_obs = pd.concat([target_obs, source_obs], axis=1)\
        .set_index(target_index)
    target_obs['cdist'] = cdist
    new_obs.append(target_obs)

new_obs = pd.concat(new_obs)
adata_comb.obs['cdist'] = 0
update_indices = adata_comb.obs.index.isin(new_obs.index)
adata_comb.obs.loc[update_indices] = new_obs
# adata_comb.write('output/CURIO/data/adata_comb_project.h5ad')
# adata_comb = ad.read_h5ad('output/CURIO/data/adata_comb_project.h5ad')

tmp = adata_comb.obs[(adata_comb.obs['sample'] == 'Preg1_1_L') & \
    (adata_comb.obs['cdist'] < 0.1)]
tmp['subclass'] = tmp['subclass'].cat.remove_unused_categories()
color_map = tmp.drop_duplicates('subclass')\
    .set_index('subclass')['subclass_color'].to_dict()
plt.clf()
plt.figure(figsize=((8, 8)))
ax  = sns.scatterplot(data=tmp, x='x_final', y='y_final', linewidth=0,
                hue='subclass', palette=color_map, s=20, legend=False)
ax.set(xlabel=None, ylabel=None)
sns.despine(bottom = True, left = True)
plt.legend(fontsize=14, markerscale=3)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig('tmp.png', dpi=200)


tmp = adata_comb.obs[adata_comb.obs['sample'] == 'Zhuang-ABCA-1.060']
tmp['subclass'] = tmp['subclass'].cat.remove_unused_categories()
color_map = tmp.drop_duplicates('subclass')\
    .set_index('subclass')['subclass_color'].to_dict()
plt.clf()
plt.figure(figsize=((12, 8)))
ax  = sns.scatterplot(data=tmp, x='x', y='y', linewidth=0,
                hue='subclass', palette=color_map, s=10, legend=False)
ax.set(xlabel=None, ylabel=None)
sns.despine(bottom = True, left = True)
plt.legend(fontsize=9, markerscale=1)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig('tmp.png', dpi=200)
