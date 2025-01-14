# Pre-processing ##############################################################

import sys
import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from ryp import r, to_py, to_r
import warnings

warnings.filterwarnings('ignore')

sys.path.append('project/utils')
from single_cell import SingleCell

# set paths
working_dir = 'project/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/curio', exist_ok=True)
os.makedirs(f'{working_dir}/figures/curio', exist_ok=True)

# load rotated and cropped query anndata objects 
query_dir = 'project/single-cell/Kalish/pregnancy-postpart/curio/raw-anndata'
samples_query = [file.replace('.h5ad', '') for file in os.listdir(query_dir)]
samples_query = sorted(samples_query)


# munge each sample, adding placeholders for metadata columns to be added 
adatas_query = []
for sample in samples_query:
    adata = ad.read_h5ad(f'{query_dir}/{sample}.h5ad')
    # DBSCAN filtering for spatial outliers
    coords = adata.obs[['x', 'y']]
    if sample.startswith('PREG_3'):  # matches both PREG_3_1 and PREG_3_2
        outliers = DBSCAN(eps=800, min_samples=90).fit(coords)
    else:
        outliers = DBSCAN(eps=500, min_samples=110).fit(coords)    
    adata = adata[outliers.labels_ == 0].copy()
    
    adata.obs['sample'] = sample
    adata.obs['condition'] = sample.split('_')[0]
    adata.obs['source'] = 'curio'
    adata.obs = adata.obs[[
        'sample', 'condition', 'source', 'cell_id', 'x', 'y']]
    adata.obs = adata.obs.rename(columns={'x': 'x_raw', 'y': 'y_raw'})
    adata.obs[['class', 'class_color', 'subclass', 'subclass_color']] = 'Unknown'
    adata.obs.index = adata.obs.index.str.split('_', n=3).str[3] + '_' + \
        adata.obs['sample'].astype(str)
    print(f'[{sample}] {adata.shape[0]} cells after DBSCAN filtering')
    adatas_query.append(adata)

# concat and store raw counts 
adata_query = sc.concat(adatas_query, axis=0, merge='same')
adata_query.layers['counts'] = adata_query.X.copy()
adata_query.var = adata_query.var.rename(columns={'gene': 'gene_symbol'})

# detect doublets 
# https://github.com/plger/scDblFinder
file = f'{working_dir}/output/curio/coldata.csv'
if os.path.exists(file):
    coldata = pd.read_csv(f'{working_dir}/output/curio/coldata.csv')
    adata_query.obs = coldata.set_index('index')
else:
    SingleCell(adata_query).to_sce('sce')
    to_r(working_dir, 'working_dir')
    r('''
    library(scDblFinder)
    library(BiocParallel)
    set.seed(123)
    sce = scDblFinder(sce, samples='sample', BPPARAM=MulticoreParam())
    table(sce$scDblFinder.class)
    # singlet doublet 
    #  111180   15773 
    coldata = as.data.frame(colData(sce))
    ''')
    coldata = to_py('coldata', format='pandas')
    adata_query.obs = coldata
    coldata.to_csv(file)

# add expression-based qc score 
# see methods: 
# https://www.nature.com/articles/s41586-023-06812-z#Sec49
qc_genes = ['Atp5g1', 'Guk1', 'Coa3', 'Hras', 'Heph', 'Naca', 'Atp5k', '1810037I17Rik', 'Atp5g3', 'Cycs', 'Fau', 'Nlrc4', 'Rtl8a', 'Uqcrq', 'Pdxp', 'Atp5j2', 'Ndufa4', 'Tpt1', 'Fkbp3', 'Edf1', 'Necap1', 'Cox7a2', 'Cox6b1', 'Polr2m', 'Slc16a2', 'Mif', 'Cox7a2l', 'Ndufb7', 'Ndufa5', 'Acadl', 'Snu13', 'Taf1c', 'Ndufc1', 'Atp5d', 'Cox5b', 'Eef1b2', 'Eif5a', 'Atp5l', 'Mrfap1', 'Chchd10', 'Atp6v1f', 'Cox7b', 'Atp5e', 'Snrpd2', 'Ftl1', 'Ndufv3', 'Usp50', 'Pfn2', 'Rab3a', 'Uqcr11', 'Ndufs7', 'Uqcrb', 'Ubb', 'Atp5j', 'Ndufa1', 'Cox6a1', 'Cox6c', 'Timm8b', 'Ap2s1', 'Ndufa2']
sc.pp.normalize_total(adata_query, target_sum=1e4)
adata_query.obs['qc_score'] = np.log1p(
    adata_query[:, qc_genes].X).mean(axis=1)
adata_query.X = adata_query.layers['counts'].copy()

# plot qc metrics
metrics = ['qc_score', 'n_genes_by_counts', 'total_counts', 'scDblFinder.score',
          'pct_counts_mt']
titles = ['Expression QC Score', 'Genes per Cell', 'Total UMI Counts', 
          'Doublet Score', 'Mitochondrial %']
y_labels = ['QC Score', 'Number of Genes', 'UMI Counts', 'scDblFinder Score',
           'MT %']

# get qc metrics
adata_query.var['mt'] = adata_query.var_names.str.startswith('mt-')
sc.pp.calculate_qc_metrics(
    adata_query, qc_vars=['mt'], percent_top=None, log1p=True, inplace=True)

# plot qc metrics
metrics = ['qc_score', 'n_genes_by_counts', 'total_counts', 'scDblFinder.score',
          'pct_counts_mt']
titles = ['Expression QC Score', 'Genes per Cell', 'Total UMI Counts', 
          'Doublet Score', 'Mitochondrial %']
y_labels = ['QC Score', 'Number of Genes', 'UMI Counts', 'scDblFinder Score',
           'MT %']

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

pink = sns.color_palette('PiYG')[0]
fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3*len(metrics)))

configs = {
    'qc_score': dict(
        log=False, lines=(0.1, None),
        ticks=[0, 2, 4, 6, 8, 10]),
    'n_genes_by_counts': dict(
        log=True, lines=(500, None),
        ticks=[100, 200, 500, 1000, 2000]),
    'total_counts': dict(
        log=True, lines=(300, None),
        ticks=[100, 1000, 10000]),
    'scDblFinder.score': dict(
        log=False, lines=(0.4, None),
        ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], invert=True),
    'pct_counts_mt': dict(
        log=False, lines=(10, None),
        ticks=[0, 5, 10, 15, 20])
}

for i, (m, title, ylabel) in enumerate(zip(metrics, titles, y_labels)):
    cfg = configs[m]
    sns.violinplot(data=adata_query.obs, x='sample', y=m, ax=axes[i],
                  color=pink, alpha=0.5, linewidth=1, linecolor=pink,
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
        axes[i].set_xlabel('Sample', fontsize=11, fontweight='bold')
    
    axes[i].set_title(title, fontsize=12, fontweight='bold')
    axes[i].set_ylabel(ylabel, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{working_dir}/figures/curio/qc_scores_violin.svg',
            dpi=300, bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/curio/qc_scores_violin.png',
            dpi=300, bbox_inches='tight')

# filter cells per sample 
keep_idx = []  
for sample in adata_query.obs['sample'].unique():
    mask = adata_query.obs['sample'] == sample
    cells = adata_query.obs.loc[mask]
    qc_mask = (
        (cells['scDblFinder.score'] < 0.4) & 
        (cells['qc_score'] > 0.1) & 
        (cells['n_genes_by_counts'] >= 500) & 
        (cells['total_counts'] > 300) & 
        (cells['pct_counts_mt'] < 10)) 
    keep_idx.extend(cells.index[qc_mask])
print(f'Keeping {len(keep_idx)} cells out of {len(adata_query)} '
      f'({len(keep_idx)/len(adata_query)*100:.1f}%)')
adata_query = adata_query[keep_idx].copy()

# normalize and log transform
sc.pp.normalize_total(adata_query)
sc.pp.log1p(adata_query)
adata_query.layers['log1p'] = adata_query.X.copy()

# add ensembl ids 
import mygene
mg = mygene.MyGeneInfo()
mapping = {
    r['query']: (r['ensembl']['gene'] if isinstance(r['ensembl'], dict) 
    else r['ensembl'][0]['gene']) 
    for r in mg.querymany(
        adata_query.var['gene_symbol'].to_list(), 
        scopes='symbol',
        fields='ensembl.gene',
        species='mouse')
    if 'ensembl' in r
}
adata_query.var['gene_id'] = adata_query.var['gene_symbol'].map(mapping)

# save
adata_query.write(f'{working_dir}/output/data/adata_query_curio.h5ad')

# CAST_MARK ####################################################################

import os
import warnings
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

warnings.filterwarnings('ignore')

working_dir = 'project/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/curio/CAST-MARK', exist_ok=True)

# load query data
adata_query = ad.read_h5ad(
    f'{working_dir}/output/data/adata_query_curio.h5ad')
adata_query.obs['x'] = adata_query.obs['x_raw']
adata_query.obs['y'] = adata_query.obs['y_raw']

# load and preprocess reference data (raw counts)
adata_ref = ad.read_h5ad(
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


# CAST_STACK ###################################################################

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
working_dir = 'project/spatial-pregnancy-postpart'
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
adata_comb = ad.read_h5ad(
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
adata_comb = ad.read_h5ad(
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
color_dict['Unknown'] = '#A9A9A9'

list_ts = {}
for _, (source_sample, target_sample) in source_target_list.items():
    print(f'Processing {target_sample}')
    output_dir_t = f'{working_dir}/output/curio/CAST-PROJECT/{source_sample}_to_{target_sample}'
    os.makedirs(output_dir_t, exist_ok=True)
    
    list_ts_file = f'{output_dir_t}/list_ts_{target_sample}.pt'
    if os.path.exists(list_ts_file):
        print(f'Loading cached list_ts for {target_sample}')
        list_ts[target_sample] = torch.load(list_ts_file)
        continue
        
    harmony_file = f'{output_dir_t}/X_harmony_{source_sample}_to_{target_sample}.h5ad'
    if os.path.exists(harmony_file):
        print(f'Loading precomputed harmony from {harmony_file}')
        adata_subset = ad.read_h5ad(harmony_file)
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
        k2=20,
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
        ave_dist_fold=3,
        alignment_shift_adjustment=0,
        color_dict=color_dict,
        adjust_shift=False,
        metric_t='cosine',
        working_memory_t=1000
    )
    torch.save(list_ts[target_sample], list_ts_file)
    print(list_ts[target_sample])
    del adata_subset; gc.collect()

# transfer cell type 
new_obs_list = []
for sample, (source_sample, target_sample) in source_target_list.items():
    print(f'Processing {target_sample}')
    # get nearest neighbor results from cast
    project_ind = list_ts[sample][0]  
    project_weight = list_ts[sample][1]      
    cdists = list_ts[sample][2]
    physical_dist = list_ts[sample][3]
    
    source_obs = adata_comb.obs[
        adata_comb.obs[batch_key] == source_sample].copy()
    target_obs = adata_comb.obs[
        adata_comb.obs[batch_key] == target_sample].copy()
    target_index = target_obs.index
    target_obs = target_obs.reset_index(drop=True)

    for col in ['class', 'subclass']:
        source_labels = source_obs[col].to_numpy()        
        neighbor_labels = source_labels[project_ind]
        
        num_cells = len(target_obs)
        cell_types = []
        confidences = []
        avg_weights = []
        avg_cdists = []
        avg_pdists = []
        
        for i in range(num_cells):
            # get most common label among neighbors
            unique_labels, label_counts = np.unique(
                neighbor_labels[i], return_counts=True)
            max_count = np.max(label_counts)
            most_common_mask = label_counts == max_count
            most_common_labels = unique_labels[most_common_mask]
            
            # break ties by overall frequency in source
            if len(most_common_labels) > 1:
                label_freqs = [
                    np.sum(source_labels == label)
                    for label in most_common_labels]
                cell_type = most_common_labels[np.argmax(label_freqs)]
            else:
                cell_type = most_common_labels[0]
            
            # calculate metrics using contributing neighbors
            contributing_mask = neighbor_labels[i] == cell_type
            neighbor_weights = project_weight[i][contributing_mask]
            confidence = np.sum(neighbor_weights) / np.sum(project_weight[i])
            
            avg_weights.append(np.mean(project_weight[i][contributing_mask]))
            avg_cdists.append(np.mean(cdists[i][contributing_mask]))
            avg_pdists.append(np.mean(physical_dist[i][contributing_mask]))
            
            cell_types.append(cell_type)
            confidences.append(confidence)
        
        # store results
        target_obs[col] = cell_types
        target_obs[f'{col}_confidence'] = confidences
        target_obs[f'{col}_avg_weight'] = avg_weights
        target_obs[f'{col}_avg_cdist'] = avg_cdists
        target_obs[f'{col}_avg_pdist'] = avg_pdists
        
        # map colors
        color_mapping = dict(zip(source_obs[col], source_obs[f'{col}_color']))
        target_obs[f'{col}_color'] = target_obs[col].map(color_mapping)
    
    new_obs_list.append(target_obs.set_index(target_index))

# plot cast metrics
metrics = ['subclass_confidence', 'subclass_avg_cdist', 'subclass_avg_pdist']
titles = ['Subclass Assignment Confidence', 'Subclass Average Cosine Distance', 
          'Subclass Average Physical Distance']
y_labels = ['Confidence Score', 'Cosine Distance', 'Physical Distance (μm)']

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
fig, axes = plt.subplots(len(metrics), 1, figsize=(6, 3*len(metrics)))

configs = {
    'subclass_confidence': dict(
        log=False, 
        lines=(0.6, None),
        ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    'subclass_avg_cdist': dict(
        log=False, 
        lines=(0.7, None),
        ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
        invert=True),
    'subclass_avg_pdist': dict(
        log=False,
        lines=(None, None), 
        ticks=[0, 1], 
        invert=True)
}

for i, (m, title, ylabel) in enumerate(zip(metrics, titles, y_labels)):
    cfg = configs[m]
    
    # Filter out inf values for physical distance plot
    if m == 'subclass_avg_pdist':
        plot_data = pd.concat(new_obs_list)
        plot_data = plot_data[~np.isinf(plot_data[m])]
    else:
        plot_data = pd.concat(new_obs_list)
    
    sns.violinplot(
        data=plot_data, x='sample', y=m, ax=axes[i],
        color=pink, alpha=0.5, linewidth=1, linecolor=pink,
        order=sample_order)
    
    if cfg['log']:
        axes[i].set_yscale('log')
        axes[i].set_yticks(cfg['ticks'])
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
        axes[i].set_xlabel('Sample', fontsize=11, fontweight='bold')
    
    axes[i].set_title(title, fontsize=12, fontweight='bold')
    axes[i].set_ylabel(ylabel, fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{working_dir}/figures/curio/cast_metrics_violin.svg',
            dpi=300, bbox_inches='tight')

# add new obs columns
adata_query = ad.read_h5ad(
    f'{working_dir}/output/data/adata_query_curio.h5ad')
adata_query.obs = adata_query.obs.drop(columns=[
    'class', 'subclass', 'class_color', 'subclass_color'])

new_obs = pd.concat(new_obs_list)
new_obs = new_obs.reindex(index=adata_query.obs_names)
for col in new_obs.columns:
    if col not in adata_query.obs.columns:
        adata_query.obs[col] = new_obs[col]

print('Initial cells:', len(adata_query))
drop_mask = (
    np.isinf(adata_query.obs['subclass_avg_cdist']) | 
    np.isinf(adata_query.obs['subclass_avg_pdist']) |
    (adata_query.obs['subclass_avg_weight'] == 0)
)
mask = (
    (~drop_mask) &
    (adata_query.obs['subclass_confidence'] >= 0.6) &
    (adata_query.obs['subclass_avg_cdist'] <= 0.7)
)

adata_query = adata_query[mask].copy()
print('Remaining cells:', len(adata_query))

# save
adata_query.X = adata_query.layers['counts']
adata_query.write(
    f'{working_dir}/output/data/adata_query_curio_final.h5ad')


# plotting #####################################################################

import os
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns

working_dir = 'project/spatial-pregnancy-postpart'

# create multi-sample plots
adata_comb = ad.read_h5ad(
    f'{working_dir}/output/curio/adata_comb_cast_stack.h5ad')
obs_ref = adata_comb[adata_comb.obs['source'] == 'Zeng-ABCA-Reference'].obs

adata_query = ad.read_h5ad(
    f'{working_dir}/output/data/adata_query_curio_final.h5ad')
obs_query = adata_query.obs

def create_multi_sample_plot(ref_obs, query_obs, col, cell_type, output_dir):
    ref_samples = ref_obs['sample'].unique()
    query_samples = query_obs['sample'].unique() 
    n_cols = 4
    n_rows = 1 + -(-len(query_samples) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.flatten()
    
    # get color from query obs
    cell_color = query_obs[f'{col}_color'].loc[query_obs[col] == cell_type].iloc[0]
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
                      c=cell_color, s=6)
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
output_dir = f'{working_dir}/figures/curio/spatial_cell_types_{col}_final'
os.makedirs(output_dir, exist_ok=True)
cell_types = pd.concat([obs_ref[col], obs_query[col]]).unique()

for cell_type in cell_types:
    if obs_query[col].value_counts().get(cell_type, 0) > 0:
        create_multi_sample_plot(
            obs_ref, obs_query, col, cell_type, output_dir)


# radius plot 
adata_comb = ad.read_h5ad(
    f'{working_dir}/output/curio/adata_comb_cast_stack.h5ad')

plot_df = adata_comb.obs[adata_comb.obs['sample'] == 'PREG_3_2']

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(plot_df['x_ffd'], plot_df['y_ffd'], c='grey', s=1)

random_point = plot_df.sample(n=1)
ax.scatter(random_point['x_ffd'], random_point['y_ffd'], c='red', s=10)

from matplotlib.patches import Circle
radius = 0.5536779033973138  # ave_dist_fold=5
circle = Circle((random_point['x_ffd'].values[0], 
                random_point['y_ffd'].values[0]), 
                radius, fill=False, color='red')
ax.add_artist(circle)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig(f'{working_dir}/figures/curio/radius.png', dpi=200)