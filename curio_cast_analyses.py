#region pre-processing #########################################################

import sys
import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import mygene
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from ryp import r, to_py, to_r
from single_cell import SingleCell

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

# set paths
working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/curio', exist_ok=True)
os.makedirs(f'{working_dir}/figures/curio', exist_ok=True)

# load query anndata objects
query_dir = 'projects/rrg-wainberg/single-cell/Kalish/' \
    'pregnancy-postpart/curio/cellbender_positioned'
files = [f for f in os.listdir(query_dir) if f.endswith('.h5ad')]
samples_query = sorted([
    f.split('_Positioned_anndata_matched')[0] for f in files])

# munge each sample, adding placeholders for later concat
# load temp data if exists
file = f'{working_dir}/output/curio/adata_query_curio_tmp.h5ad'
if os.path.exists(file):
    adata_query = sc.read_h5ad(file)
else:
    adatas_query = []
    for sample in samples_query:
        adata = sc.read_h5ad(
            f'{query_dir}/{sample}_Positioned_anndata_matched.h5ad')
        coords_data = pd.read_csv(
            f'projects/rrg-wainberg/single-cell/Kalish/pregnancy-postpart/'
            f'curio/coords/coords_{sample}.txt', delim_whitespace=True)\
            .rename(columns={'number_clusters': 'SB_number_cluster'})
        coords_data = coords_data[
            ~((coords_data['x_um'] == 0) & (coords_data['y_um'] == 0))]
        
        # dbscan filtering for spatial outliers
        coords = adata.obsm['X_spatial']
        eps = 800 if sample.startswith('PREG_3') else 500
        min_samples = 90 if sample.startswith('PREG_3') else 110
        outliers = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        n = adata.shape[0]
        adata = adata[outliers.labels_ == 0].copy()
        print(f'[{sample}] {n - adata.shape[0]} cells filtered by dbscan')
        
        # add metadata
        adata.obs['sample_rep'] = sample
        adata.obs['sample'] = sample.rsplit('_', 1)[0]
        adata.obs['condition'] = sample.split('_')[0]
        adata.obs['source'] = 'curio'
        adata.obs['cell_id_orig'] = adata.obs.index.str.rstrip('-1')
        adata.obs.reset_index(drop=True, inplace=True)
        adata.obs['cell_id'] = adata.obs['sample'] + '_' + \
            adata.obs['cell_id_orig'] + '-' + adata.obs.index.astype(str)
        adata.obs['x_raw'] = adata.obsm['X_spatial'][:, 0]
        adata.obs['y_raw'] = adata.obsm['X_spatial'][:, 1]
        adata.obs = adata.obs[[
            'sample', 'sample_rep', 'condition', 'source', 'cell_id',
            'cell_id_orig', 'x_raw', 'y_raw']]
        
        # find duplicate coordinates
        dupes = coords_data.duplicated(subset=['x_um', 'y_um'], keep=False)
        coords_unique = coords_data[~dupes].copy()
        print(f'[{sample}] removed {dupes.sum()} coords with duplicate positions')
        
        coords_unique['x_abs'] = coords_unique['x_um'].abs()
        coords_unique['y_abs'] = coords_unique['y_um'].abs()
        adata.obs['x_abs'] = adata.obs['x_raw'].abs()
        adata.obs['y_abs'] = adata.obs['y_raw'].abs()
        
        # merge coords data into adata
        n_before = len(adata)
        adata.obs = pd.merge(
            adata.obs, coords_unique,
            left_on=['x_abs', 'y_abs'],
            right_on=['x_abs', 'y_abs'],
            how='left')
        matched_count = (~adata.obs['cell_bc'].isna()).sum()
        print(f'[{sample}] matched {matched_count}/{n_before} cells')
        
        adata.obs = adata.obs.drop(['x_abs', 'y_abs', 'x_um', 'y_um'], axis=1)
        adata.obs.index = adata.obs['cell_id']
        adata.obs.index.name = None
        del adata.obsm
        
        # detect doublets 
        adata = sc.pp.scrublet(adata, copy=True)
        print(f'[{sample}] detected {adata.obs['predicted_doublet'].sum()} doublets')
        adatas_query.append(adata)

    # concat and store raw counts 
    adata_query = sc.concat(adatas_query, axis=0, merge='same')
    adata_query.layers['counts'] = adata_query.X.copy()
    adata_query.var = adata_query.var.rename(columns={'name': 'gene_symbol'})
    adata_query.obs['predicted_doublet'] = \
        adata_query.obs['predicted_doublet'].astype(str)
    # adata_query.write(file)

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

fig, axes = plt.subplots(len(metrics), 1, figsize=(6, 3*len(metrics)))
pink = sns.color_palette('PiYG')[0]
configs = {
    'n_genes_by_counts': dict(
        log=True, lines=(100, None),
        ticks=[10, 100, 500, 2000, 5000],
        ylim=(-5, 6000)),
    'total_counts': dict(
        log=True, lines=(100, None),
        ticks=[10, 100, 300, 1000, 3000, 10000],
        ylim=(-5, 12000)),
    'doublet_score': dict(
        log=False, lines=(0.10, None),
        ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
        ylim=(0.2, -0.05)),
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
   'high_doublet': adata_query.obs['doublet_score'] >= 0.10,
   'low_n_genes': adata_query.obs['n_genes_by_counts'] < 100, 
   'low_counts': adata_query.obs['total_counts'] <= 100,
   'high_mt_pct': adata_query.obs['pct_counts_mt'] >= 10
}.items():
   print(f'{name}: {mask.sum()} ({mask.sum()/total*100:.1f}%) cells dropped')

keep_idx = []
for sample in adata_query.obs['sample'].unique():
   mask = adata_query.obs['sample'] == sample
   cells = adata_query.obs.loc[mask]
   fail_mask = ~((cells['doublet_score'] < 0.10) & 
                 (cells['n_genes_by_counts'] >= 100) & 
                 (cells['total_counts'] > 100) & 
                 (cells['pct_counts_mt'] < 10))
   keep_idx.extend(cells.index[~fail_mask])

cells_dropped = total - len(keep_idx)
print(f'\ntotal cells dropped: {cells_dropped} ({cells_dropped/total*100:.1f}%)')

'''
high_doublet: 14505 (10.8%) cells dropped
low_n_genes: 10602 (7.9%) cells dropped
low_counts: 6212 (4.6%) cells dropped
high_mt_pct: 11126 (8.3%) cells dropped

total cells dropped: 29937 (22.3%)
'''

# filter cells
adata_query = adata_query[keep_idx].copy()

# normalize and log transform
sc.pp.normalize_total(adata_query)
sc.pp.log1p(adata_query)
adata_query.layers['log1p'] = adata_query.X.copy()

# save
adata_query.write(f'{working_dir}/output/data/adata_query_curio.h5ad')

#endregion

#region CAST_MARK ####################################################################

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

# load and preprocess reference data (raw counts)
adata_ref = sc.read_h5ad(
    f'{working_dir}/output/data/adata_ref_zeng_raw.h5ad')
# normalize 
sc.pp.normalize_total(adata_ref)
sc.pp.log1p(adata_ref)

# batch correction
adata_query_s, adata_ref_s = scanorama.correct_scanpy([adata_query, adata_ref])

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
    embed_dict = torch.load(embed_dict_path)
else:
    # run cast mark
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

for n_clust in list(range(4, 20 + 1, 2)) + [30, 40, 50, 100, 200]:
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

#region CAST_STACK ###################################################################

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

# modified CAST_Projection.py 
sys.path.insert(0, 'project/CAST')
import CAST
print(CAST.__file__)

# set paths 
working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'
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
            query_reference_list[key] = [key, 'C57BL6J-638850.46']
    return query_reference_list

# load data 
adata_comb = sc.read_h5ad(
    f'{working_dir}/output/curio/adata_comb_cast_mark.h5ad')
coords_raw = torch.load(
    f'{working_dir}/output/curio/coords_raw.pt')
embed_dict = torch.load(
    f'{working_dir}/output/curio/embed_dict.pt')

# rotate 
rotation_angles = {
    'CTRL_1_1': -55, 'CTRL_1_2': -55, 
    'CTRL_2_1': 140, 'CTRL_2_2': 140, 
    'CTRL_3_1': 10, 'CTRL_3_2': 10, 
    'PREG_1_1': 180, 'PREG_1_2': 180, 
    'PREG_2_1': 5, 'PREG_2_2': 5, 
    'PREG_3_1': 5, 'PREG_3_2': 5, 
    'POSTPART_1_1': -110, 'POSTPART_1_2': -110, 
    'POSTPART_2_1': 90, 'POSTPART_2_2': 90, 
    'POSTPART_3_1': 10, 'POSTPART_3_2': 10, 
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
    cache_path = f'{working_dir}/output/curio/CAST-STACK/{sample}.pt'
    if os.path.exists(cache_path):
        print(f'Loading cached coordinates for {sample}')
        coords_affine[sample], coords_ffd[sample] = torch.load(cache_path)
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
            f'{working_dir}/output/curio/CAST-STACK',
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
coords_df = pd.DataFrame(coords_stack, 
                        columns=['x_affine', 'y_affine'], 
                        index=cell_index)
adata_comb.obs = adata_comb.obs.join(coords_df)

coords_stack_ffd = np.vstack([coords_ffd[s][s] for s in sample_names])
coords_df_ffd = pd.DataFrame(coords_stack_ffd, 
                           columns=['x_ffd', 'y_ffd'], 
                           index=cell_index)
adata_comb.obs = adata_comb.obs.join(coords_df_ffd)

mask = adata_comb.obs['source'] == 'Zeng-ABCA-Reference'
for coord in ['affine', 'ffd']:
    adata_comb.obs.loc[mask, f'x_{coord}'] = adata_comb.obs.loc[mask, 'x']
    adata_comb.obs.loc[mask, f'y_{coord}'] = adata_comb.obs.loc[mask, 'y']

torch.save(coords_affine, f'{working_dir}/output/curio/coords_affine.pt')
torch.save(coords_ffd, f'{working_dir}/output/curio/coords_ffd.pt')
adata_comb.write(f'{working_dir}/output/curio/adata_comb_cast_stack.h5ad')


# CAST_PROJECT #################################################################

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

# modified CAST_Projection.py 
sys.path.insert(0, 'project/CAST')
import CAST
print(CAST.__file__)

# set paths 
working_dir = 'project/spatial-pregnancy-postpart'
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
        list_ts[target_sample] = torch.load(list_ts_file)
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
            ifplot=True,
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
        ifplot=True,
        source_sample_ctype_col=level,
        output_path=output_dir_t,
        umap_feature='X_umap',
        pc_feature='X_pca_harmony',
        integration_strategy=None, 
        ave_dist_fold=3,
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
            dict(zip(source_obs[col], source_obs[f'{col}_color'])))
    
    new_obs_list.append(target_obs.set_index(target_index))

new_obs = pd.concat(new_obs_list)
new_obs.to_csv(f'{working_dir}/output/curio/new_obs.csv', index_label='cell_id')

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
   'Cosine Distance', 'Physical Distance (μm)',
   'Confidence', 'Confidence'
]
sample_order = [
   'CTRL_1_1', 'CTRL_1_2', 'CTRL_2_1', 'CTRL_3_1', 'CTRL_3_2',
   'PREG_1_1', 'PREG_1_2', 'PREG_2_1', 'PREG_2_2', 'PREG_3_1', 'PREG_3_2',
   'POSTPART_1_1', 'POSTPART_1_2', 'POSTPART_2_1', 'POSTPART_2_2'
]
sample_labels = [
   'Control 1.1', 'Control 1.2', 'Control 2.1', 'Control 3.1', 'Control 3.2',
   'Pregnant 1.1', 'Pregnant 1.2', 'Pregnant 2.1', 'Pregnant 2.2', 'Pregnant 3.1',
   'Pregnant 3.2', 'Postpartum 1.1', 'Postpartum 1.2', 'Postpartum 2.1',
   'Postpartum 2.2'
]
pink = sns.color_palette("PiYG")[0]
fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3*len(metrics)))
configs = {
   'avg_cdist': dict(
       lines=(0.8, None),
       ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
       invert=True),
   'avg_pdist': dict(
       lines=(None, None),
       ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
       invert=True),
   'class_confidence': dict(
       lines=(0.7, None),
       ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0]),
   'subclass_confidence': dict(
       lines=(None, None),
       ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
}
plot_data = pd.concat(new_obs_list)
for i, (m, title, ylabel) in enumerate(zip(metrics, titles, y_labels)):
   cfg = configs[m]
   if m == 'avg_pdist':
       data = plot_data[~np.isinf(plot_data[m])]
   else:
       data = plot_data
   sns.violinplot(
       data=data, x='sample', y=m, ax=axes[i],
       color=pink, alpha=0.5, linewidth=1, linecolor=pink,
       order=sample_order)
   if cfg.get('invert', False):
       axes[i].invert_yaxis()
   if cfg['lines'][0]:
       axes[i].axhline(y=cfg['lines'][0], ls='--', color=pink, alpha=0.5)
       axes[i].text(1.02, cfg['lines'][0], f'{cfg["lines"][0]:.1f}',
                   va='center', transform=axes[i].get_yaxis_transform())
   if i < len(metrics) - 1:
       axes[i].set_xticklabels([])
       axes[i].set_xlabel('')
       axes[i].set_xticks([])
   else:
       axes[i].set_xticklabels(sample_labels, rotation=45, ha='right', va='top')
       axes[i].set_xlabel('Sample', fontsize=11, fontweight='bold')
   
   axes[i].set_title(title, fontsize=12, fontweight='bold')
   axes[i].set_ylabel(ylabel, fontsize=11, fontweight='bold')
   axes[i].set_yticks(cfg['ticks'])
plt.tight_layout()
plt.savefig(f'{working_dir}/figures/curio/cast_metrics_violin.svg',
           bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/curio/cast_metrics_violin.png',
           dpi=150, bbox_inches='tight')

# post-processing ##############################################################

import numpy as np
import pandas as pd 
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

working_dir = 'project/spatial-pregnancy-postpart'

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

# filter cells 
total = len(adata_query)
for name, mask in {
    'infinite pdist': np.isinf(adata_query.obs['avg_pdist']),
    'low class confidence': adata_query.obs['class_confidence'] <= 0.7,
    'high expression dist': adata_query.obs['avg_cdist'] >= 0.8
}.items():
    print(f'{name}: {mask.sum()} ({mask.sum()/total*100:.1f}%) cells dropped')
'''
infinite pdist: 1499 (1.6%) cells dropped
low class confidence: 18466 (19.7%) cells dropped
high expression dist: 8033 (8.6%) cells dropped
'''
mask = ((~np.isinf(adata_query.obs['avg_pdist'])) &
        (adata_query.obs['class_confidence'] >= 0.7) &
        (adata_query.obs['avg_cdist'] <= 0.8))
        
cells_dropped = total - mask.sum()
print(f'\nTotal cells dropped: {cells_dropped} '
      f'({cells_dropped/total*100:.1f}%)')
'''Total cells dropped: 18051 (19.2%)'''

adata_query = adata_query[mask].copy()

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
adata_i = adata_i[adata_i.obs['global_conf'] > 0.8].copy()
sc.pp.neighbors(adata_i, n_neighbors=30)
sc.tl.umap(adata_i)

# correct both class and subclass labels
neighbors = adata_i.obsp['connectivities']
obs_index = adata_i.obs.index

for level in ['class', 'subclass']:
    # calculate local confidence
    cell_labels = adata_i.obs[level].values
    same_cell = neighbors.multiply(cell_labels[:, None] == cell_labels)
    local_conf = same_cell.sum(1).A1 / neighbors.sum(1).A1
    adata_i.obs[f'{level}_local_conf'] = local_conf
    
    # store original labels
    adata_i.obs[f'original_{level}'] = adata_i.obs[level].copy()
    
    # correction
    corrections = []
    for i in range(len(adata_i)):
        if local_conf[i] < 0.5:
            neighbor_idx = neighbors[i].indices
            neighbor_labels = cell_labels[neighbor_idx]
            label_counts = pd.Series(neighbor_labels).value_counts()
            majority = label_counts.index[0]
            maj_frac = label_counts.iloc[0] / len(neighbor_idx)
            
            cell_idx = obs_index[i]
            nbr_idx = obs_index[neighbor_idx[0]]
            
            if (maj_frac > 0.6 and majority != cell_labels[i]):
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
    print(f"total {level} corrections made: {len(corrections)}")

'''
total class corrections made: 4318
total subclass corrections made: 7047
'''

# conf_metrics = ['global_conf', 'class_local_conf', 'subclass_local_conf']
# thresholds = {m: np.percentile(adata_i.obs[m], 5) for m in conf_metrics}
# keep = np.all([adata_i.obs[m] > thresholds[m] for m in conf_metrics], axis=0)

# adata_i = adata_i[keep].copy()
# print(f"cells after filtering low confidence: {len(adata_i)}")

# plot results for both levels
fig, axes = plt.subplots(4, 2, figsize=(20, 32))
size = 8
sc.pl.umap(adata_i, color='original_class', size=size, ax=axes[0,0], 
           show=False, legend_loc='none', title='original class')
sc.pl.umap(adata_i, color='class', size=size, ax=axes[0,1], 
           show=False, title='corrected class')
sc.pl.umap(adata_i, color='original_subclass', size=size, ax=axes[1,0], 
           show=False, legend_loc='none', title='original subclass')
sc.pl.umap(adata_i, color='subclass', size=size, ax=axes[1,1], 
           show=False, title='corrected subclass')

sc.pl.umap(adata_i, color='global_conf', size=size, ax=axes[2,0], 
           show=False, title='global confidence')
sc.pl.umap(adata_i, color='class_local_conf', size=size, ax=axes[2,1], 
           show=False, title='class local confidence')
sc.pl.umap(adata_i, color='subclass_local_conf', size=size, ax=axes[3,0], 
           show=False, title='subclass local confidence')
plt.savefig(f'{working_dir}/figures/curio/umap_correction.png', dpi=200)

# update original adata
adata_query = adata_query[adata_i.obs.index].copy()
adata_query.obs['class'] = adata_i.obs['class']
adata_query.obs['subclass'] = adata_i.obs['subclass']

# keep cell types with at least 5 cells in at least 3 samples per condition

min_cells, min_samples = 5, 3
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
sc.pp.neighbors(adata_query, metric='cosine', random_state=seed)
sc.tl.umap(adata_query, min_dist=0.5, spread=1.0, random_state=seed)

# add protein coding genes
protein_coding_genes = pd.read_csv(
    'project/single-cell/Kalish/pregnancy-postpart/MRK_ENSEMBL.csv', 
    header=None)
protein_coding_genes = protein_coding_genes[
    protein_coding_genes[8] == 'protein coding gene'][1].to_list()
adata_query.var['protein_coding'] = adata_query.var['gene_symbol']\
    .isin(protein_coding_genes)

# save
adata_query.X = adata_query.layers['counts'].copy()
adata_query.write(
    f'{working_dir}/output/data/adata_query_curio_final.h5ad')

# plotting #####################################################################

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

working_dir = 'project/spatial-pregnancy-postpart'

cells_joined = pd.read_csv(
  'project/single-cell/ABC/metadata/MERFISH-C57BL6J-638850/20231215/'
  'views/cells_joined.csv')
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
        s=6, linewidths=0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
    plt.savefig(f'{working_dir}/figures/curio/umap_{level}.png', dpi=400)
    plt.savefig(f'{working_dir}/figures/curio/umap_{level}.svg', format='svg')

    # spatial exemplar 

    sample = 'CTRL_2_1'
    plot_color = adata_query[(adata_query.obs['sample'] == sample)].obs
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        plot_color['x_ffd'], plot_color['y_ffd'],
        c=[color_mappings[level][c] for c in plot_color[level]], 
        s=12, linewidths=0)
    if level == 'subclass':
        unique_classes = sorted(plot_color[
            plot_color['subclass_keep'] == True][level].unique(),
            key=lambda x: int(x.split()[0]))
    else:
        unique_classes = sorted(plot_color[level].unique(),
                                key=lambda x: int(x.split()[0]))
    legend_elements = [plt.Line2D(
        [0], [0], marker='o', color='w',
        markerfacecolor=color_mappings[level][class_],
        label=class_, markersize=8)
        for class_ in unique_classes]
    ax.legend(handles=legend_elements, loc='center left',
            bbox_to_anchor=(1, 0.5), frameon=False)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{working_dir}/figures/curio/spatial_example_{level}.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{working_dir}/figures/curio/spatial_example_{level}.svg',
                format='svg', bbox_inches='tight')
    plt.savefig(f'{working_dir}/figures/curio/spatial_example_{level}.pdf',
                bbox_inches='tight')




