#region pre-processing #########################################################

import sys
import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

# set paths
working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/curio', exist_ok=True)
os.makedirs(f'{working_dir}/figures/curio', exist_ok=True)

# load query anndata objects 
query_dir = 'projects/rrg-wainberg/single-cell/Kalish/pregnancy-postpart/' \
    'curio/raw-anndata'
samples_query = [file.replace('.h5ad', '') for file in os.listdir(query_dir)]
samples_query = sorted(samples_query)

# munge each sample, adding placeholders for later concat
file = f'{working_dir}/output/curio/adata_query_curio_tmp.h5ad'
if os.path.exists(file):
    adata_query = sc.read_h5ad(file)
else:
    adatas_query = []
    for sample in samples_query:
        # load sample 
        adata = sc.read_h5ad(f'{query_dir}/{sample}.h5ad')
        # filtering for spatial outliers
        coords = adata.obs[['x', 'y']]
        if sample.startswith('PREG_3'): 
            outliers = DBSCAN(eps=800, min_samples=90).fit(coords)
        else:
            outliers = DBSCAN(eps=500, min_samples=110).fit(coords)    
        adata = adata[outliers.labels_ == 0].copy()
        print(f'[{sample}] {adata.shape[0]} cells after DBSCAN filtering')
        # add metadata
        adata.obs['sample_rep'] = sample
        adata.obs['sample'] = sample.rsplit('_', 1)[0]
        adata.obs['condition'] = sample.split('_')[0]
        adata.obs['source'] = 'curio'
        adata.obs = adata.obs[[
            'sample', 'sample_rep', 'condition', 'source', 'cell_id', 'x', 'y']]
        adata.obs = adata.obs.rename(columns={'x': 'x_raw', 'y': 'y_raw'})
        adata.obs[['class', 'class_color', 'subclass', 'subclass_color']] = \
            'Unknown'
        adata.obs.index = adata.obs['sample_rep'].astype(str) + '_' + \
            adata.obs.index.str.split('_', n=3).str[3] + '-' + \
            adata.obs.index.factorize()[0].astype(str)
        # detect doublets 
        adata = sc.pp.scrublet(adata, copy=True)
        adatas_query.append(adata)

    # concat and store raw counts 
    adata_query = sc.concat(adatas_query, axis=0, merge='same')
    adata_query.layers['counts'] = adata_query.X.copy()
    adata_query.var = adata_query.var.rename(columns={'gene': 'gene_symbol'})
    adata_query.write(file)

# get qc metrics
adata_query.var['mt'] = adata_query.var_names.str.startswith('mt-')
sc.pp.calculate_qc_metrics(
    adata_query, qc_vars=['mt'], percent_top=None, log1p=True, inplace=True)

# plot qc metrics
metrics = [
    'n_genes_by_counts', 'total_counts', 'doublet_score',
    'pct_counts_mt']
titles = [
    'Genes per Cell', 'Total UMI Counts', 
    'Doublet Score', 'Mitochondrial %']
y_labels = [
    'Number of Genes', 'UMI Counts', 'Doublet Score', 'MT %']

sample_order = [
    'CTRL_1', 'CTRL_2', 'CTRL_3', 'PREG_1', 'PREG_2', 'PREG_3',
    'POSTPART_1', 'POSTPART_2'
]
sample_labels = [
    'Control 1', 'Control 2', 'Control 3',
    'Pregnant 1', 'Pregnant 2', 'Pregnant 3',
    'Postpartum 1', 'Postpartum 2'
]

fig, axes = plt.subplots(len(metrics), 1, figsize=(5, 3*len(metrics)))
pink = sns.color_palette('PiYG')[0]
configs = {
    'n_genes_by_counts': dict(
        log=True, lines=(500, None),
        ticks=[300, 500, 1000, 3000, 5000],
        ylim=(200, 6000)),
    'total_counts': dict(
        log=True, lines=(500, None),
        ticks=[300, 500, 1000, 3000, 5000, 10000],
        ylim=(200, 10000)),
    'doublet_score': dict(
        log=False, lines=(0.20, None),
        ticks=[0, 0.1, 0.2, 0.3],
        ylim=(-0.05, 0.30)),
    'pct_counts_mt': dict(
        log=False, lines=(10, None),
        ticks=[0, 2, 4, 6, 8, 10, 12],
        ylim=(-0.5, 12))
}
for i, (m, title, ylabel) in enumerate(zip(metrics, titles, y_labels)):
    cfg = configs[m]
    plot_df = pd.DataFrame({
        'sample': adata_query.obs['sample'].values,
        m: adata_query.obs[m].values
    })
    sns.boxplot(data=plot_df, x='sample', y=m, ax=axes[i],
              color=pink, linewidth=1, width=0.4, showfliers=False, 
              order=sample_order)
    
    if cfg['log']:
        axes[i].set_yscale('log')
        axes[i].set_yticks(cfg['ticks'])
        axes[i].set_yticklabels([str(int(x)) for x in cfg['ticks']])
    if cfg.get('invert', False):
        axes[i].invert_yaxis()
    for val in cfg['lines']:
        if val:
            axes[i].axhline(y=val, ls='--', color=pink, alpha=0.5)
            axes[i].text(1.02, val, f'{val:.1f}', va='center',
                        transform=axes[i].get_yaxis_transform())
    if i < len(metrics) - 1:
        axes[i].set_xticklabels([])
        axes[i].set_xlabel('')
        axes[i].set_xticks([])
        axes[i].xaxis.label.set_visible(False)
    else:
        axes[i].set_xticklabels(sample_labels, rotation=45, ha='right', va='top')
        axes[i].set_xlabel('Sample', fontsize=11)
    
    axes[i].set_title(title, fontsize=12)
    axes[i].set_ylabel(ylabel, fontsize=11)
    axes[i].set_ylim(*cfg['ylim'])

plt.tight_layout()
plt.savefig(f'{working_dir}/figures/curio/qc_scores_boxplots.svg',
            bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/curio/qc_scores_boxplots.png',
            dpi=150, bbox_inches='tight')
plt.close()

# get filters
total = len(adata_query)
for name, mask in {
   'high_doublet': adata_query.obs['doublet_score'] >= 0.20,
   'low_n_genes': adata_query.obs['n_genes_by_counts'] < 500, 
   'low_counts': adata_query.obs['total_counts'] <= 500,
   'high_mt_pct': adata_query.obs['pct_counts_mt'] >= 10
}.items():
   print(f'{name}: {mask.sum()} ({mask.sum()/total*100:.1f}%) cells dropped')

keep_idx = []
for sample in adata_query.obs['sample'].unique():
   mask = adata_query.obs['sample'] == sample
   cells = adata_query.obs.loc[mask]
   fail_mask = ~((cells['doublet_score'] < 0.20) & 
                 (cells['n_genes_by_counts'] >= 500) & 
                 (cells['total_counts'] > 500) & 
                 (cells['pct_counts_mt'] < 10))
   keep_idx.extend(cells.index[~fail_mask])

cells_dropped = total - len(keep_idx)
print(f'\ntotal cells dropped: {cells_dropped} ({cells_dropped/total*100:.1f}%)')

'''
high_doublet: 5725 (4.5%) cells dropped
low_n_genes: 6199 (4.9%) cells dropped
low_counts: 807 (0.6%) cells dropped
high_mt_pct: 241 (0.2%) cells dropped

total cells dropped: 12107 (9.5%)
'''

# filter cells and save
adata_query = adata_query[keep_idx].copy()
adata_query.write(f'{working_dir}/output/data/adata_query_curio.h5ad')

#endregion

#region CAST-MARK ##############################################################

import os
import warnings
warnings.filterwarnings('ignore')

import scanorama
import torch
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import CAST
from CAST.models.model_GCNII import Args

working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/curio/CAST-MARK', exist_ok=True)

# load query data
adata_query = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_curio.h5ad')
adata_query.obs['x'] = adata_query.obs['x_raw']
adata_query.obs['y'] = adata_query.obs['y_raw']
adata_query.X = adata_query.layers['counts']
sc.pp.normalize_total(adata_query)
sc.pp.log1p(adata_query)

# load and preprocess reference data (raw counts)
adata_ref = sc.read_h5ad(
    f'{working_dir}/output/data/adata_ref_zeng_raw.h5ad')
adata_ref.obs['sample_rep'] = adata_ref.obs['sample']
sc.pp.normalize_total(adata_ref)
sc.pp.log1p(adata_ref)

# batch correction
adata_query_s, adata_ref_s = scanorama.correct_scanpy([
    adata_query, adata_ref])

# combine data for CAST-MARK input
adata_comb = ad.concat([adata_query, adata_ref], axis=0, merge='same')
adata_comb = adata_comb[:, adata_comb.var_names.sort_values()]
adata_comb.layers['X_scanorama'] = ad.concat(
    [adata_query_s, adata_ref_s], axis=0, merge='same').X.copy()
del adata_query_s, adata_ref_s

# order by sample names
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

# check if embeddings already exist
embed_dict_path = f'{working_dir}/output/curio/embed_dict.pt'
if os.path.exists(embed_dict_path):
    embed_dict = torch.load(embed_dict_path, weights_only=False)
else:
    # run cast mark
    embed_dict = CAST.CAST_MARK(
        coords_raw, exp_dict, 
        f'{working_dir}/output/curio/CAST-MARK',
        graph_strategy='delaunay', 
        args = Args(
            dataname='curio', # name of the dataset, used to save the log file
            gpu = 0, # gpu id, set to zero for single-GPU nodes
            epochs=100, # number of epochs for training
            lr1=1e-3, # learning rate
            wd1=0, # weight decay
            lambd=1e-3, # lambda in the loss function
            n_layers=9, # number of GCNII layers
            der=0.5, # edge dropout rate in CCA-SSG
            dfr=0.3, # feature dropout rate in CCA-SSG
            use_encoder=False, # perform single-layer dimension reduction before 
                            # GNNs, helps save VRAM and time if gene panel large
            encoder_dim=512 # encoder dimension, ignore if use_encoder is False
        )
    )
    # save 
    torch.save(embed_dict, embed_dict_path)

# detach, remove duplicated embeddings, and stack 
embed_dict = {k.split('_dup')[0]: v.cpu().detach() 
              for k, v in embed_dict.items()}
embed_stack = np.vstack([embed_dict[name].numpy() for name in sample_names])

# kmeans clustering and plotting
plot_dir = f'{working_dir}/figures/curio/k_clusters'
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

for n_clust in list(range(4, 20 + 1, 4)) + [50]:
    print(f'Clustering with k={n_clust}')
    kmeans = KMeans(n_clusters=n_clust, random_state=0).fit(embed_stack)
    cell_label = kmeans.labels_
    cluster_pl = sns.color_palette('Set3', n_clust)
    fig = plot_slices(sample_names, coords_raw, cell_label, cluster_pl, n_clust)
    fig.savefig(f'{plot_dir}/all_samples_k{str(n_clust)}.png', dpi=150)
    plt.close(fig)
    adata_comb.obs[f'k{n_clust}_cluster'] = cell_label

# Save results
torch.save(coords_raw, f'{working_dir}/output/curio/coords_raw.pt')
torch.save(exp_dict, f'{working_dir}/output/curio/exp_dict.pt')
adata_comb.write_h5ad(f'{working_dir}/output/curio/adata_comb_cast_mark.h5ad')

#endregion

#region CAST-STACK #############################################################

import os
import sys
import torch
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# import custom CAST
cast_path = os.path.abspath('projects/rrg-wainberg/karbabi/CAST-keon')
sys.path.insert(0, cast_path)
if 'CAST' in sys.modules:
    del sys.modules['CAST']
import CAST
print(CAST.__file__)

# set paths 
working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/curio/CAST-STACK', exist_ok=True)

# rotate query coords to help with registration
def rotate_coords(coords, angle):
    theta = np.radians(angle)
    rot_mat = torch.tensor([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]],
        dtype=torch.float32)
    return torch.mm(torch.from_numpy(coords).float(), rot_mat).numpy()

# generate query_reference_list
def generate_reference_list(coords_raw):
    query_reference_list = {}
    for key in coords_raw:
        if not key.startswith('C57BL6J-638850'):
            query_reference_list[key] = [key, 'C57BL6J-638850.46_R']
    return query_reference_list

# load data 
adata_comb = sc.read_h5ad(
    f'{working_dir}/output/curio/adata_comb_cast_mark.h5ad')
coords_raw = torch.load(
    f'{working_dir}/output/curio/coords_raw.pt', weights_only=False)
embed_dict = torch.load(
    f'{working_dir}/output/curio/embed_dict.pt', weights_only=False)

# rotate 
rotation_angles = {
    'CTRL_1': -55,
    'CTRL_2': 140,
    'CTRL_3': 10, 
    'PREG_1': 180, 
    'PREG_2': 5,
    'PREG_3': 5, 
    'POSTPART_1': -110,
    'POSTPART_2': 90, 
    'POSTPART_3': 10
}
coords_raw = {
    k: rotate_coords(v, rotation_angles[k])
    if not k.startswith('C57BL6J') else v 
    for k, v in coords_raw.items()
}

# generate reference list
query_reference_list = generate_reference_list(coords_raw)

coords_affine = {}
coords_ffd = {}
for sample in sorted(query_reference_list.keys()):
    folder_path = f'{working_dir}/output/curio/CAST-STACK/{sample}'
    os.makedirs(folder_path, exist_ok=True)

    cache_path = f'{folder_path}/{sample}.pt'
    if os.path.exists(cache_path):
        print(f'Loading cached coordinates for {sample}')
        coords_affine[sample], coords_ffd[sample] = \
            torch.load(cache_path, weights_only=False)
        continue

    params_dist = CAST.reg_params(
        dataname = query_reference_list[sample],
        gpu = 0 if torch.cuda.is_available() else -1,
        diff_step = 5,
        #### Affine parameters
        iterations=100,
        dist_penalty1=0.1,
        bleeding=500,
        d_list = [3,2,1,1/2,1/3],
        attention_params = [None,3,1,0],
        #### FFD parameters
        dist_penalty2 = [0.1],
        alpha_basis_bs = [500],
        meshsize = [8],
        iterations_bs = [50], 
        attention_params_bs = [[None,3,1,0]],
        mesh_weight = [None])
    
    params_dist.alpha_basis = torch.Tensor(
        [1/1000,1/1000,1/50,5,5]).reshape(5,1).to(params_dist.device)

    coords_affine[sample], coords_ffd[sample] = \
        CAST.CAST_STACK(
            coords_raw, 
            embed_dict, 
            f'{working_dir}/output/curio/CAST-STACK/{sample}',
            query_reference_list[sample],
            params_dist, 
            mid_visual=False,
            rescale=True)
    
    print(coords_affine[sample])
    print(coords_ffd[sample])
    torch.save((coords_affine[sample], coords_ffd[sample]), cache_path)

# add coords to adata
sample_names = sorted(list(coords_ffd.keys()))
cell_index = adata_comb.obs.index[adata_comb.obs['source'] == 'curio']

coords_stack = np.vstack([coords_affine[s][s] for s in sample_names])
coords_df = pd.DataFrame(
    coords_stack, 
    columns=['x_affine', 'y_affine'], 
    index=cell_index)
adata_comb.obs = adata_comb.obs.join(coords_df)

coords_stack_ffd = np.vstack([coords_ffd[s][s] for s in sample_names])
coords_df_ffd = pd.DataFrame(
    coords_stack_ffd, 
    columns=['x_ffd', 'y_ffd'], 
    index=cell_index)
adata_comb.obs = adata_comb.obs.join(coords_df_ffd)

mask = adata_comb.obs['source'] == 'Zeng-ABCA-Reference'
for coord in ['affine', 'ffd']:
    adata_comb.obs.loc[mask, f'x_{coord}'] = adata_comb.obs.loc[mask, 'x']
    adata_comb.obs.loc[mask, f'y_{coord}'] = adata_comb.obs.loc[mask, 'y']

# plot each coordinate type (affine and ffd)
for coord_type in ['affine', 'ffd']:
    fig, axs = plt.subplots(3, 3, figsize=(20, 15))
    axs = axs.flatten()
    samples = sorted(query_reference_list.keys())
    ref_mask = adata_comb.obs['sample'] == 'C57BL6J-638850.46_R'

    for i, sample in enumerate(samples):
        axs[i].scatter(
            adata_comb.obs.loc[ref_mask, f'x_{coord_type}'],
            adata_comb.obs.loc[ref_mask, f'y_{coord_type}'],
            s=0.5, c='lightgray', alpha=0.2)
        mask = adata_comb.obs['sample'] == sample
        axs[i].scatter(
            adata_comb.obs.loc[mask, f'x_{coord_type}'],
            adata_comb.obs.loc[mask, f'y_{coord_type}'],
            s=1, c='red')
        axs[i].set_title(sample)
        axs[i].axis('off')
    
    for j in range(len(samples), len(axs)):
        axs[j].set_visible(False)
    
    plt.savefig(f'{working_dir}/figures/curio/stack_{coord_type}.png', dpi=300)
    plt.close()

torch.save(coords_affine, f'{working_dir}/output/curio/coords_affine.pt')
torch.save(coords_ffd, f'{working_dir}/output/curio/coords_ffd.pt')
adata_comb.write(f'{working_dir}/output/curio/adata_comb_cast_stack.h5ad')

#endregion

#region CAST-PROJECT ###########################################################

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import sys
import os
import torch
import warnings
import gc
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# import custom CAST
cast_path = os.path.abspath('projects/rrg-wainberg/karbabi/CAST-keon')
sys.path.insert(0, cast_path)
if 'CAST' in sys.modules:
    del sys.modules['CAST']
import CAST
print(CAST.__file__)

# set paths 
working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/curio/CAST-PROJECT', exist_ok=True)

# load data
adata_comb = sc.read_h5ad(
    f'{working_dir}/output/curio/adata_comb_cast_stack.h5ad')

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
        adata_comb.obs['source'] == 'curio', 'sample'].unique()
}
color_dict = (
    adata_comb.obs
    .drop_duplicates()
    .set_index(level)[f'{level}_color']
    .to_dict()
)
color_dict['Unknown'] = 'black'

list_ts = {}
for _, (source_sample, target_sample) in source_target_list.items():
    print(f'Processing {target_sample}')
    output_dir_t = f'{working_dir}/output/curio/CAST-PROJECT/' \
        f'{source_sample}_to_{target_sample}'
    os.makedirs(output_dir_t, exist_ok=True)
    
    list_ts_file = f'{output_dir_t}/list_ts_{target_sample}.pt'
    if os.path.exists(list_ts_file):
        print(f'Loading cached list_ts for {target_sample}')
        list_ts[target_sample] = torch.load(list_ts_file, weights_only=False)
        continue
        
    harmony_file = f'{output_dir_t}/X_harmony_{source_sample}_to_{target_sample}.h5ad'
    if os.path.exists(harmony_file):
        print(f'Loading precomputed harmony from {harmony_file}')
        adata_subset = sc.read_h5ad(harmony_file)
    else:
        print('Computing harmony')
        adata_subset = adata_comb[
            (adata_comb.obs[batch_key] == target_sample) |
            (adata_comb.obs[batch_key] == source_sample)]
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
            color_dict=color_dict,
            output_path=output_dir_t,
            ifplot=False,
            ifcombat=False)
        adata_subset.write_h5ad(harmony_file)
    
    print(f'Running CAST_PROJECT for {target_sample}')
    _, list_ts[target_sample] = CAST.CAST_PROJECT(
        sdata_inte=adata_subset,
        source_sample=source_sample,
        target_sample=target_sample,
        coords_source=np.array(
            adata_subset[adata_subset.obs[batch_key] == source_sample,:]
                .obs.loc[:,['x_ffd','y_ffd']]),
        coords_target=np.array(
            adata_subset[adata_subset.obs[batch_key] == target_sample,:]
                .obs.loc[:,['x_ffd','y_ffd']]),
        k2=50,
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
        ave_dist_fold=5,
        alignment_shift_adjustment=0,
        color_dict=color_dict,
        adjust_shift=False,
        metric_t='cosine',
        working_memory_t=1000)
    torch.save(list_ts[target_sample], list_ts_file)
    print(list_ts[target_sample])
    del adata_subset; gc.collect()

# transfer cell type
new_obs_list = []
for sample, (source_sample, target_sample) in source_target_list.items():
    project_ind, project_weight, cdists, physical_dist = list_ts[sample]
    source_obs = adata_comb.obs[adata_comb.obs[batch_key] == source_sample].copy()
    target_obs = adata_comb.obs[adata_comb.obs[batch_key] == target_sample].copy()
    target_index = target_obs.index
    target_obs = target_obs.reset_index(drop=True)
    
    print(f'Processing {target_sample}')
    k = 10
    
    for i in range(len(target_obs)):
        weights = project_weight[i][:k]
        target_obs.loc[i, 'avg_weight'] = np.mean(weights)
        target_obs.loc[i, 'avg_cdist'] = np.mean(cdists[i][:k])
        target_obs.loc[i, 'avg_pdist'] = np.mean(physical_dist[i][:k])
    
    for col in ['class', 'subclass']:
        source_labels = source_obs[col].to_numpy()
        neighbor_labels = source_labels[project_ind]
        cell_types = []
        confidences = []
        
        for i in range(len(target_obs)):
            top_k = neighbor_labels[i][:k]
            labels, counts = np.unique(top_k, return_counts=True)
            winners = labels[counts == counts.max()]
            cell_type = (winners[0] if len(winners) == 1 else
                        winners[np.argmax([np.sum(source_labels == l)
                                         for l in winners])])
            confidences.append(np.sum(top_k == cell_type) / k)
            cell_types.append(cell_type)
        
        target_obs[col] = cell_types
        target_obs[f'{col}_confidence'] = confidences
        target_obs[f'{col}_color'] = target_obs[col].map(
            dict(zip(source_obs[col], source_obs[f'{col}_color']))
        )
    
    new_obs_list.append(target_obs.set_index(target_index))

new_obs = pd.concat(new_obs_list)
new_obs.to_csv(f'{working_dir}/output/curio/new_obs.csv', index_label='cell_id')

#endregion

#region post-processing ########################################################

import numpy as np
import pandas as pd 
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'

# add new obs columns
adata_query = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_curio.h5ad')
adata_query.obs = adata_query.obs.drop(columns=[
    'class', 'subclass', 'class_color', 'subclass_color'])

new_obs = pd.read_csv(
    f'{working_dir}/output/curio/new_obs.csv',index_col='cell_id')
for col in new_obs.columns:
    if col not in adata_query.obs.columns:
        adata_query.obs[col] = new_obs[col]

# plot metrics
metrics = [
   'avg_cdist', 'avg_pdist',
   'class_confidence', 'subclass_confidence'
]
titles = [
   'Expression Distance', 'Spatial Distance',
   'Class Assignment Confidence', 'Subclass Assignment Confidence'
]
y_labels = [
   'Cosine Distance', 'Physical Distance (AU)',
   'Confidence', 'Confidence'
]
sample_order = [
   'CTRL_1', 'CTRL_2', 'CTRL_3',
   'PREG_1', 'PREG_2', 'PREG_3',
   'POSTPART_1', 'POSTPART_2'
]
sample_labels = [
   'Control 1', 'Control 2', 'Control 3',
   'Pregnant 1', 'Pregnant 2', 'Pregnant 3',
   'Postpartum 1', 'Postpartum 2'
]

fig, axes = plt.subplots(len(metrics), 1, figsize=(5, 3*len(metrics)))
pink = sns.color_palette('PiYG')[0]

configs = {
   'avg_cdist': dict(
       lines=(0.8, None),
       ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
       ylim=(0, 1.2)),
   'avg_pdist': dict(
       lines=(None, None),
       ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
       ylim=(0, 0.8)),
   'class_confidence': dict(
       lines=(0.6, None),
       ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
       ylim=(0, 1.2)),
   'subclass_confidence': dict(
       lines=(0.5, None),
       ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
       ylim=(0, 1.2))
}
plot_data = new_obs
for i, (m, title, ylabel) in enumerate(zip(metrics, titles, y_labels)):
   cfg = configs[m]
   if m == 'avg_pdist':
       data = plot_data[~np.isinf(plot_data[m])]
   else:
       data = plot_data
   sns.boxplot(
       data=data, x='sample', y=m, ax=axes[i],
       color=pink, linewidth=1, width=0.4, showfliers=False,
       order=sample_order)
   
   if cfg.get('invert', False):
       axes[i].invert_yaxis()
   if cfg['lines'][0]:
       axes[i].axhline(y=cfg['lines'][0], ls='--', color=pink, alpha=0.5)
       axes[i].text(1.02, cfg['lines'][0], f'{cfg["lines"][0]:.1f}', va='center',
                   transform=axes[i].get_yaxis_transform())
   if i < len(metrics) - 1:
       axes[i].set_xticklabels([])
       axes[i].set_xlabel('')
       axes[i].set_xticks([])
       axes[i].xaxis.label.set_visible(False)
   else:
       axes[i].set_xticklabels(sample_labels, rotation=45, ha='right', va='top')
       axes[i].set_xlabel('Sample', fontsize=11)
   
   axes[i].set_title(title, fontsize=12)
   axes[i].set_ylabel(ylabel, fontsize=11)
   axes[i].set_yticks(cfg['ticks'])
   axes[i].set_ylim(*cfg['ylim'])

plt.tight_layout()
plt.savefig(f'{working_dir}/figures/curio/cast_metrics_boxplot.svg',
           bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/curio/cast_metrics_boxplot.png',
           dpi=150, bbox_inches='tight')
plt.close()

# filter cells 
total = len(adata_query)
for name, mask in {
    'infinite pdist': np.isinf(adata_query.obs['avg_pdist']),
    'low class confidence': adata_query.obs['class_confidence'] <= 0.6,
    'low subclass confidence': adata_query.obs['subclass_confidence'] <= 0.5,
    'high expression dist': adata_query.obs['avg_cdist'] >= 0.8
}.items():
    print(f'{name}: {mask.sum()} ({mask.sum()/total*100:.1f}%) cells dropped')

mask = ((~np.isinf(adata_query.obs['avg_pdist'])) &
        (adata_query.obs['class_confidence'] >= 0.6) &
        (adata_query.obs['subclass_confidence'] >= 0.5) &
        (adata_query.obs['avg_cdist'] <= 0.8))
        
cells_dropped = total - mask.sum()
print(f'\ntotal cells dropped: {cells_dropped} '
      f'({cells_dropped/total*100:.1f}%)')

adata_query = adata_query[mask].copy()
print(f'cells after filtering: {len(adata_query)}')

'''
infinite pdist: 891 (0.8%) cells dropped
low class confidence: 16068 (14.0%) cells dropped
low subclass confidence: 23094 (20.1%) cells dropped
high expression dist: 9086 (7.9%) cells dropped

total cells dropped: 21629 (18.8%)
cells after filtering: 93217
'''

# remove noise cells 
broad_group = {
   '01 IT-ET Glut': 'neuronal',
   '02 NP-CT-L6b Glut': 'neuronal', 
   '05 OB-IMN GABA': 'neuronal',
   '06 CTX-CGE GABA': 'neuronal',
   '07 CTX-MGE GABA': 'neuronal',
   '08 CNU-MGE GABA': 'neuronal',
   '09 CNU-LGE GABA': 'neuronal',
   '10 LSX GABA': 'neuronal',
   '11 CNU-HYa GABA': 'neuronal',
   '12 HY GABA': 'neuronal',
   '13 CNU-HYa Glut': 'neuronal',
   '14 HY Glut': 'neuronal',
   '30 Astro-Epen': 'non-neuronal',
   '31 OPC-Oligo': 'non-neuronal',
   '33 Vascular': 'non-neuronal',
   '34 Immune': 'non-neuronal'
}

# preprocessing
adata_i = adata_query.copy()
adata_i.obs['broad'] = adata_i.obs['class'].map(broad_group)
adata_i.X = adata_query.layers['counts'].copy()
sc.pp.normalize_total(adata_i)
sc.pp.log1p(adata_i)
sc.pp.highly_variable_genes(adata_i, batch_key='sample')
sc.tl.pca(adata_i)

# calculate initial global confidence
sc.pp.neighbors(adata_i, n_neighbors=30)
nn_mat = adata_i.obsp['distances'].astype(bool)
broad_labels = adata_i.obs['broad']
same_broad = nn_mat.multiply(broad_labels.values[:, None] == broad_labels.values)
global_conf = same_broad.sum(1).A1 / nn_mat.sum(1).A1
adata_i.obs['global_conf'] = global_conf

# filter and recompute embeddings
n_cells_before = len(adata_i)
adata_i = adata_i[adata_i.obs['global_conf'] > 0.7].copy()
n_cells_after = len(adata_i)
cells_dropped = n_cells_before - n_cells_after
print(f'dropped {cells_dropped} cells ({cells_dropped/n_cells_before*100:.1f}%) '
      f'with low global confidence')
print(f'cells after global confidence filtering: {n_cells_after}')

sc.pp.neighbors(adata_i, n_neighbors=30)
sc.tl.umap(adata_i)

# calculate local confidence for class and subclass
neighbors = adata_i.obsp['connectivities']
obs_index = adata_i.obs.index
for level in ['class', 'subclass']:
    cell_labels = adata_i.obs[level].values
    same_cell = neighbors.multiply(cell_labels[:, None] == cell_labels)
    local_conf = same_cell.sum(1).A1 / neighbors.sum(1).A1
    adata_i.obs[f'{level}_local_conf'] = local_conf

# filter cells with low confidence scores (bottom 2% in any confidence metric)
conf_metrics = ['class_local_conf', 'subclass_local_conf']
cells_before = len(adata_i)
thresholds = {m: np.percentile(adata_i.obs[m], 2) for m in conf_metrics}
keep = np.all([adata_i.obs[m] > thresholds[m] for m in conf_metrics], axis=0)
adata_i = adata_i[keep].copy()
cells_after = len(adata_i)
cells_dropped = cells_before - cells_after
print(f'dropped {cells_dropped} cells ({cells_dropped/cells_before*100:.1f}%) '
      f'with low local confidence')
print(f'cells after local confidence filtering: {cells_after}')

# recompute neighbors once after filtering
sc.pp.neighbors(adata_i, n_neighbors=30)
neighbors = adata_i.obsp['connectivities']
obs_index = adata_i.obs.index

# perform corrections for both levels
for level in ['class', 'subclass']:
    # store original labels
    adata_i.obs[f'original_{level}'] = adata_i.obs[level].copy()
    
    # add boolean column to track corrections
    adata_i.obs[f'{level}_corrected'] = False
    
    # recalculate local confidence with new neighbors
    cell_labels = adata_i.obs[level].values
    same_cell = neighbors.multiply(cell_labels[:, None] == cell_labels)
    local_conf = same_cell.sum(1).A1 / neighbors.sum(1).A1
    adata_i.obs[f'{level}_local_conf'] = local_conf
    
    # correction
    corrections = []
    for i in range(len(adata_i)):
        if local_conf[i] < 0.6:  
            neighbor_idx = neighbors[i].indices
            neighbor_labels = cell_labels[neighbor_idx]
            label_counts = pd.Series(neighbor_labels).value_counts()
            majority = label_counts.index[0]
            maj_frac = label_counts.iloc[0] / len(neighbor_idx)
            
            cell_idx = obs_index[i]
            nbr_idx = obs_index[neighbor_idx[0]]
            
            if (maj_frac > 0.5 and majority != cell_labels[i]):
                if level == 'subclass':
                    parent_matches = (
                        adata_i.obs.loc[cell_idx, 'class'] == 
                        adata_i.obs.loc[nbr_idx, 'class'])
                    if not parent_matches:
                        continue
                corrections.append({
                    'cell': cell_idx,
                    'old': cell_labels[i],
                    'new': majority,
                    'conf': maj_frac
                })
                adata_i.obs.loc[cell_idx, level] = majority
                adata_i.obs.loc[cell_idx, f'{level}_corrected'] = True
    
    # calculate and print correction statistics
    n_corrected = adata_i.obs[f'{level}_corrected'].sum()
    pct_corrected = n_corrected/len(adata_i)*100
    print(f'total {level} corrections: {n_corrected} '
          f'({pct_corrected:.1f}% of cells)')

# drop the bottom 2% most disconnected cells
connectivity_scores = neighbors.sum(1).A1
connectivity_before = len(adata_i)
connectivity_threshold = np.percentile(connectivity_scores, 2)
keep_connected = connectivity_scores > connectivity_threshold
adata_i = adata_i[keep_connected].copy()
connectivity_after = len(adata_i)
cells_dropped = connectivity_before - connectivity_after
print(f'dropped {cells_dropped} cells ({cells_dropped/connectivity_before*100:.1f}%) '
      f'poorly connected')
print(f'final cells after all filtering: {len(adata_i)}')

'''
total cells dropped: 21629 (18.8%)
cells after filtering: 93217

dropped 10953 cells (11.8%) with low global confidence
cells after global confidence filtering: 82264

dropped 4538 cells (5.5%) with low local confidence
cells after local confidence filtering: 77726

total class corrections: 4070 (5.2% of cells)
total subclass corrections: 8744 (11.2% of cells)

dropped 1618 cells (2.1%) poorly connected
final cells after all filtering: 76108
'''

# plot results for both levels
fig, axes = plt.subplots(5, 2, figsize=(20, 40))
size = 8
sc.pl.umap(adata_i, color='original_class', size=size, ax=axes[0,0], 
           show=False, legend_loc='none', title='original class')
sc.pl.umap(adata_i, color='class', size=size, ax=axes[0,1], 
           show=False, title='corrected class')
sc.pl.umap(adata_i, color='original_subclass', size=size, ax=axes[1,0], 
           show=False, legend_loc='none', title='original subclass')
sc.pl.umap(adata_i, color='subclass', size=size, ax=axes[1,1], 
           show=False, title='corrected subclass')

# add plots to show which cells were corrected
sc.pl.umap(adata_i, color='class_corrected', size=size, ax=axes[2,0], 
           show=False, title='class corrected cells', 
           color_map='RdBu_r')
sc.pl.umap(adata_i, color='subclass_corrected', size=size, ax=axes[2,1], 
           show=False, title='subclass corrected cells', 
           color_map='RdBu_r')

# confidence plots
sc.pl.umap(adata_i, color='global_conf', size=size, ax=axes[3,0], 
           show=False, title='global confidence')
sc.pl.umap(adata_i, color='class_local_conf', size=size, ax=axes[3,1], 
           show=False, title='class local confidence')
sc.pl.umap(adata_i, color='subclass_local_conf', size=size, ax=axes[4,0], 
           show=False, title='subclass local confidence')
plt.savefig(f'{working_dir}/figures/curio/umap_correction.png', dpi=200)

# update original adata
adata_query = adata_query[adata_i.obs.index].copy()
adata_query.obs['class'] = adata_i.obs['class']
adata_query.obs['subclass'] = adata_i.obs['subclass']
adata_query.obs['class_corrected'] = adata_i.obs['class_corrected']
adata_query.obs['subclass_corrected'] = adata_i.obs['subclass_corrected']

# keep cell types with at least 5 cells in at least 2 samples per condition
min_cells, min_samples = 5, 2
conditions = ['CTRL', 'PREG', 'POSTPART']

for level in ['class', 'subclass']:
    kept = []
    for type_ in adata_query.obs[level].unique():
        passes = True
        for cond in conditions:
            samples = adata_query.obs.loc[
                adata_query.obs['sample'].str.contains(cond), 'sample'].unique()
            n_valid = sum(sum((adata_query.obs['sample'] == s) & 
                (adata_query.obs[level] == type_)) >= min_cells 
                for s in samples)
            if n_valid < min_samples:
                passes = False
                break
        if passes:
            kept.append(type_)
    col_name = f'{level}_keep'
    adata_query.obs[col_name] = adata_query.obs[level].isin(kept)
    print(f'Kept {level}:')
    for k in sorted(kept, key=lambda x: int(x.split()[0])):
        print(f'  {k}')

# umap
seed = 0
adata_query.X = adata_query.layers['counts'].copy()
sc.pp.normalize_total(adata_query)
sc.pp.log1p(adata_query)
adata_query.layers['log1p'] = adata_query.X.copy()

sc.pp.highly_variable_genes(adata_query, batch_key='sample')
sc.tl.pca(adata_query, random_state=seed)
sc.pp.neighbors(adata_query, n_neighbors=30, random_state=seed)
sc.tl.umap(adata_query, random_state=seed)

# add protein coding genes
protein_coding_genes = pd.read_csv(
    'projects/rrg-wainberg/single-cell/Kalish/pregnancy-postpart/'
    'MRK_ENSEMBL.csv', header=None)
protein_coding_genes = protein_coding_genes[
    protein_coding_genes[8] == 'protein coding gene'][1].to_list()
adata_query.var['protein_coding'] = adata_query.var['gene_symbol']\
    .isin(protein_coding_genes)

# save
adata_query.X = adata_query.layers['counts'].copy()
adata_query.write(
    f'{working_dir}/output/data/adata_query_curio_final.h5ad')

#endregion

#region plotting ###############################################################

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'

cells_joined = pd.read_csv(
  'projects/rrg-wainberg/single-cell/ABC/metadata/MERFISH-C57BL6J-638850/'
  '20231215/views/cells_joined.csv')
color_mappings = {
   'class': dict(zip(cells_joined['class'].str.replace('/', '_'), 
                     cells_joined['class_color'])),
   'subclass': {k.replace('_', '/'): v for k,v in dict(zip(
       cells_joined['subclass'].str.replace('/', '_'), 
       cells_joined['subclass_color'])).items()}
}

adata_query = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_curio_final.h5ad')

for level in ['class', 'subclass']:
    # umap
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(
        adata_query.obsm['X_umap'][:, 0], adata_query.obsm['X_umap'][:, 1],
        c=[color_mappings[level][c] for c in adata_query.obs[level]],
        s=4, linewidths=0, rasterized=True)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
    plt.savefig(f'{working_dir}/figures/curio/umap_{level}.png', dpi=400)
    plt.savefig(f'{working_dir}/figures/curio/umap_{level}.svg', format='svg')
    plt.close()
    
    # spatial exemplar 
    sample = 'CTRL_2'
    plot_color = adata_query[(adata_query.obs['sample'] == sample)].obs
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        plot_color['x_ffd'], plot_color['y_ffd'],
        c=[color_mappings[level][c] for c in plot_color[level]], 
        s=16, linewidths=0, rasterized=True)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{working_dir}/figures/curio/spatial_example_{level}.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{working_dir}/figures/curio/spatial_example_{level}.svg',
                format='svg', bbox_inches='tight')
    plt.savefig(f'{working_dir}/figures/curio/spatial_example_{level}.pdf',
                bbox_inches='tight')
    plt.close()

# create multi-sample plots
adata_comb = sc.read_h5ad(
    f'{working_dir}/output/curio/adata_comb_cast_stack.h5ad')
obs_ref = adata_comb[adata_comb.obs['source'] == 'Zeng-ABCA-Reference'].obs
obs_query = adata_query.obs

def create_multi_sample_plot(ref_obs, query_obs, col, cell_type, output_dir):
    ref_samples = ref_obs['sample'].unique()
    query_samples = query_obs['sample'].unique() 
    n_cols = 4
    n_rows = 1 + -(-len(query_samples) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.flatten()
    
    cell_color = color_mappings[col][cell_type]
    coord_cols = ['x_ffd', 'y_ffd']
    
    for i, (sample, obs) in enumerate(
            [(s, ref_obs) for s in ref_samples] +
            [(s, query_obs) for s in query_samples]):
        if i >= len(axes):
            break
        ax = axes[i]
        plot_df = obs[obs['sample'] == sample]
        mask = plot_df[col] == cell_type
        if mask.sum() > 0:
            ax.scatter(plot_df[~mask][coord_cols[0]], 
                      plot_df[~mask][coord_cols[1]], 
                      c='grey', s=1, alpha=0.1)
            ax.scatter(plot_df[mask][coord_cols[0]], 
                      plot_df[mask][coord_cols[1]], 
                      c=cell_color, s=4)
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
    plt.savefig(f'{output_dir}/{safe_filename}.png', dpi=200, 
                bbox_inches='tight')
    plt.close(fig)

col = 'subclass'
output_dir = f'{working_dir}/figures/curio/spatial_cell_types_{col}'
os.makedirs(output_dir, exist_ok=True)
cell_types = obs_query[col].unique()
for cell_type in cell_types:
    create_multi_sample_plot(obs_ref, obs_query, col, cell_type, output_dir)

# # radius plot 
# plot_df = adata_comb.obs[adata_comb.obs['sample'] == 'CTRL_1']
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.scatter(plot_df['x_ffd'], plot_df['y_ffd'], c='grey', s=1)
# random_point = plot_df.sample(n=1)
# ax.scatter(random_point['x_ffd'], random_point['y_ffd'], c='red', s=10)
# from matplotlib.patches import Circle
# radius = 0.624520173821159  # ave_dist_fold=10
# circle = Circle((random_point['x_ffd'].values[0], 
#                 random_point['y_ffd'].values[0]), 
#                 radius, fill=False, color='red')
# ax.add_artist(circle)
# ax.set_aspect('equal')
# ax.axis('off')
# plt.tight_layout()
# plt.savefig(f'{working_dir}/figures/curio/radius.png', dpi=200)


#endregion



