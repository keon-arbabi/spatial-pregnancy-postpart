# Pre-processing ###############################################################
import sys
import os
import json
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from ryp import r, to_r, to_py
import warnings

warnings.filterwarnings('ignore')

sys.path.append('project/utils')
from single_cell import SingleCell
from utils import run

# set paths
working_dir = 'project/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/output/merfish/CAST-MARK', exist_ok=True)

# load query data 
query_dir = 'project/single-cell/Kalish/pregnancy-postpart/merfish/raw-anndata'
samples_query = [file.replace('.h5ad', '') for file in os.listdir(query_dir)]
samples_query = sorted(samples_query)

adatas_query = []
for sample in samples_query:
    adata = ad.read_h5ad(f'{query_dir}/{sample}.h5ad')
    adata.obs['sample'] = sample
    adata.obs['condition'] = sample[:-1]
    adata.obs['source'] = 'merfish'
    adata.obs = adata.obs[[
        'sample', 'condition', 'source', 'cell_id',
        'Custom_regions', 'Datasets', 'volume', 'center_x', 'center_y']]
    adata.obs = adata.obs.rename(columns={
        'Custom_regions': 'custom_regions', 'Datasets': 'datasets',
        'center_x': 'x_raw', 'center_y': 'y_raw'})
    adata.obs.index = adata.obs['sample'] + '_' + \
        adata.obs.index.str.split('_').str[1]
    del adata.layers['orig_norm']
    print(f'[{sample}] {adata.shape[0]} cells')
    adatas_query.append(adata)

# concat and store raw counts 
adata_query = sc.concat(adatas_query, axis=0, merge='same')
adata_query.layers['counts'] = adata_query.X.copy()
adata_query.var = adata_query.var.rename(columns={'gene': 'gene_symbol'})

# detect doublets 
# https://github.com/plger/scDblFinder
file = f'{working_dir}/output/merfish/coldata.csv'
if os.path.exists(file):
    coldata = pd.read_csv(f'{working_dir}/output/merfish/coldata.csv')
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
    # 867938  571048 
    coldata = as.data.frame(colData(sce))
    ''')
    coldata = to_py('coldata', format='pandas')
    adata_query.obs = coldata
    coldata.to_csv(file)

# get qc metrics
sc.pp.calculate_qc_metrics(
    adata_query, percent_top=None, log1p=True, inplace=True)

# plot qc metrics
metrics = ['volume', 'n_genes_by_counts', 'total_counts', 'scDblFinder.score']
titles = ['Cell Volume', 'Genes per Cell', 'Total UMI Counts', 'Doublet Score']
y_labels = ['Volume (μm³)', 'Number of Genes', 'UMI Counts', 'scDblFinder Score']

sample_order = [
    'CTRL1', 'CTRL2', 'CTRL3', 
    'PREG1', 'PREG2', 'PREG3',
    'POSTPART1', 'POSTPART2', 'POSTPART3']
sample_labels = [
    'Control 1', 'Control 2', 'Control 3',
    'Pregnant 1', 'Pregnant 2', 'Pregnant 3',
    'Postpartum 1', 'Postpartum 2', 'Postpartum 3']

pink = sns.color_palette("PiYG")[0]
fig, axes = plt.subplots(len(metrics), 1, figsize=(6, 3*len(metrics)))

configs = {
    'volume': dict(
        log=True, lines=(100, 2000),
        ticks=[10, 100, 1000, 5000]),
    'n_genes_by_counts': dict(
        log=True, lines=(5, None),
        ticks=[1, 10, 100]),
    'total_counts': dict(
        log=True, lines=(20, None),
        ticks=[10, 20, 40, 80, 160, 320, 640]),
    'scDblFinder.score': dict(
        log=False, lines=(0.2, None),
        ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5], invert=True)
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
plt.savefig(f'{working_dir}/figures/merfish/qc_scores_violin.png',
            dpi=300, bbox_inches='tight')

# filter cells per sample 
keep_idx = []  
for sample in adata_query.obs['sample'].unique():
    mask = adata_query.obs['sample'] == sample
    cells = adata_query.obs.loc[mask]
    median_vol = np.median(cells['volume'])
    qc_mask = (
        (cells['scDblFinder.score'] < 0.2) &
        (cells['volume'] > 100) &
        (cells['volume'] <= 3 * median_vol) &
        (cells['total_counts'] > 20) &
        (cells['n_genes_by_counts'] >= 5))
    keep_idx.extend(cells.index[qc_mask])
adata_query = adata_query[keep_idx].copy()

# normalize counts by volume
volumes = adata_query.obs['volume'].to_numpy()
adata_query.X = adata_query.X.multiply(1/volumes[:, None]).tocsr()

# filter global expression outliers (2% tails)
total_exp = adata_query.X.sum(1).A1
low, high = np.percentile(total_exp, [2, 98])
adata_query = adata_query[(total_exp >= low) & (total_exp <= high)].copy()

# normalize to 250 and log transform
sc.pp.normalize_total(adata_query, target_sum=250)
sc.pp.log1p(adata_query)
adata_query.layers['volume_log1p'] = adata_query.X.copy()

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
mapping.update({
   'Il1f6': 'ENSMUSG00000026984',
   'Tdgf1': 'ENSMUSG00000032494', 
   'Il1f5': 'ENSMUSG00000026983',
   'Fcrls': 'ENSMUSG00000015852'
})
adata_query.var['gene_id'] = adata_query.var['gene_symbol'].map(mapping)
adata_query.var.index = adata_query.var['gene_id']

# temp save
# adata_query.write(f'{working_dir}/output/data/adata_query_merfish.h5ad')

# get cell type labels 
# https://github.com/AllenInstitute/cell_type_mapper/tree/main
# run('''
#     python -m cell_type_mapper.cli.from_specified_markers \
#         --query_path project/spatial-pregnancy-postpart/output/data/adata_query_merfish.h5ad \
#         --extended_result_path project/spatial-pregnancy-postpart/output/merfish/mapper_output.json \
#         --log_path project/spatial-pregnancy-postpart/output/merfish/mapper_log.txt \
#         --csv_result_path project/spatial-pregnancy-postpart/output/merfish/mapper_output.csv \
#         --drop_level CCN20230722_SUPT \
#         --cloud_safe False \
#         --query_markers.serialized_lookup cell_type_mapper/mouse_markers_230821.json \
#         --precomputed_stats.path cell_type_mapper/precomputed_stats_ABC_revision_230821.h5 \
#         --type_assignment.normalization log2CPM \
#         --type_assignment.n_processors 64
# ''')

# load mapper results and munge 
with open(f'{working_dir}/output/merfish/mapper_output.json') as f:
    mapper_json = json.load(f)

mapper_df = pd.DataFrame([dict(m, cell_id=c['cell_id'], level=l) 
    for c in mapper_json['results'] 
    for l,m in c.items() if isinstance(m, dict)])

cols_to_drop = ['runner_up_assignment', 'runner_up_correlation', 
                'runner_up_probability']
levels = ['CCN20230722_CLAS', 'CCN20230722_SUBC']
values = ['assignment', 'bootstrapping_probability', 'avg_correlation', 
          'aggregate_probability', 'directly_assigned']

mapper_df = (mapper_df[mapper_df['level'].isin(levels)]
    .drop(columns=cols_to_drop)
    .assign(level=lambda x: x.level.str.extract('CCN20230722_(.*)')[0].str.lower())
    .pivot(index='cell_id', columns='level', values=values))
mapper_df.columns = [f'{col[0]}_{col[1]}' for col in mapper_df.columns]

for metric in ['bootstrapping_probability', 'avg_correlation', 
               'aggregate_probability']:
    for level in ['clas', 'subc']:
        mapper_df[f'{metric}_{level}'] = mapper_df[f'{metric}_{level}'
            ].astype(float)

for level in ['clas', 'subc']:
    mapper_df[f'directly_assigned_{level}'] = mapper_df[
        f'directly_assigned_{level}'].astype('bool')
    mapper_df[f'assignment_{level}'] = mapper_df[
        f'assignment_{level}'].astype('category')

mapper_names = {level: mapper_json['taxonomy_tree']['name_mapper'][
    f'CCN20230722_{level.upper()}'] for level in ['clas', 'subc']}

for level, new_col in [('clas', 'class'), ('subc', 'subclass')]:
    mapper_df[new_col] = mapper_df[f'assignment_{level}'].map(
        lambda x: mapper_names[level].get(x, {}).get('name', 'Unknown')
        ).astype('category')
    
# plot mapping metrics
metrics = ['bootstrapping_probability', 'avg_correlation']
titles = ['Bootstrapping Probability', 'Average Correlation']
pink = sns.color_palette("PiYG")[0]

fig, axes = plt.subplots(len(metrics), 1, figsize=(6, 3*len(metrics)))
for i, (metric, title) in enumerate(zip(metrics, titles)):
    plot_df = pd.DataFrame({
        'value': mapper_df[f'{metric}_subc'],
        'sample': adata_query.obs['sample'],
        'cell_type': mapper_df['subclass']
    })
    sns.violinplot(data=plot_df, x='sample', y='value', ax=axes[i],
                  color=pink, alpha=0.5, linewidth=1, linecolor=pink,
                  order=sample_order)
    
    line_value = 0.8 if 'probability' in metric else 0.3
    axes[i].axhline(y=line_value, ls='--', color=pink, alpha=0.5)
    axes[i].text(1.02, line_value, f'{line_value:.1f}', va='center', 
                transform=axes[i].get_yaxis_transform())
    
    axes[i].set_title(title, fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Score', fontsize=11, fontweight='bold')
    if i < len(metrics) - 1:
        axes[i].set_xticklabels([])
        axes[i].set_xlabel('')
        axes[i].set_xticks([])
    else:
        axes[i].set_xticklabels(sample_labels, rotation=45, ha='right', va='top')
        axes[i].tick_params(axis='x', rotation=45)
        plt.setp(axes[i].get_xticklabels(), ha='right', va='top')

plt.tight_layout()
plt.savefig(f'{working_dir}/figures/merfish/mapping_scores_violin.png',
            dpi=300, bbox_inches='tight')

# join mapper results to anndata
adata_query.obs = adata_query.obs.join(mapper_df)

# plot mapping metrics vs qc metrics
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
pink = sns.color_palette('PiYG')[0]
plots = [
    ('n_genes_by_counts', 'avg_correlation_subc', 'Genes vs Correlation'),
    ('n_genes_by_counts', 'bootstrapping_probability_subc', 'Genes vs Probability'),
    ('avg_correlation_subc', 'bootstrapping_probability_subc', 
     'Correlation vs Probability')]
for i, (x, y, title) in enumerate(plots):
    sns.scatterplot(
        data=adata_query.obs, x=x, y=y, ax=axes[i], color=pink, 
        alpha=0.005, s=10, linewidth=0)
    if x == 'n_genes_by_counts':
        axes[i].set_xscale('log')
    axes[i].set_title(title, fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{working_dir}/figures/merfish/qc_vs_mapping_scatter.png',
           dpi=300, bbox_inches='tight')

# filter cells based on mapping metrics
mask = (
    (adata_query.obs['directly_assigned_subc'] == True) &
    (adata_query.obs['bootstrapping_probability_subc'] > 0.8) &
    (adata_query.obs['avg_correlation_subc'] > 0.3) &
    (adata_query.obs['class'] != 'Unknown') &
    (adata_query.obs['subclass'] != 'Unknown'))
print(sum(mask))
adata_query = adata_query[mask].copy()

# save
adata_query.write(f'{working_dir}/output/data/adata_query_merfish.h5ad')

# CAST-MARK ####################################################################

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
os.makedirs(f'{working_dir}/output/merfish/CAST-MARK', exist_ok=True)

# load query data
adata_query = ad.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish.h5ad')
adata_query.obs['x'] = adata_query.obs['x_raw']
adata_query.obs['y'] = adata_query.obs['y_raw']

# load reference data (imputed)
adata_ref = ad.read_h5ad(
    f'{working_dir}/output/data/adata_ref_zeng_imputed.h5ad')
adata_ref.var.index = adata_ref.var['gene_identifier']

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

# run cast mark
embed_dict = CAST.CAST_MARK(
    coords_raw, exp_dict, 
    f'{working_dir}/output/merfish/CAST-MARK',
    graph_strategy='delaunay', 
    args = Args(
        dataname='merfish',
        gpu = 0, 
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
              for k, v in embed_dict.items()}
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

for n_clust in list(range(4, 20 + 1, 2)) + [30, 40, 50, 100, 200]:
    print(f'Clustering with k={n_clust}')
    kmeans = KMeans(n_clusters=n_clust, random_state=0).fit(embed_stack)
    cell_label = kmeans.labels_
    cluster_pl = sns.color_palette('Set3', n_clust)
    fig = plot_slices(sample_names, coords_raw, cell_label, cluster_pl, n_clust)
    fig.savefig(f'{plot_dir}/all_samples_k{str(n_clust)}.png', dpi=300)
    plt.close(fig)
    adata_comb.obs[f'k{n_clust}_cluster'] = cell_label

# Save results
torch.save(coords_raw, f'{working_dir}/output/merfish/coords_raw.pt')
torch.save(exp_dict, f'{working_dir}/output/merfish/exp_dict.pt')
torch.save(embed_dict, f'{working_dir}/output/merfish/embed_dict.pt')
adata_comb.write_h5ad(f'{working_dir}/output/merfish/adata_comb_cast_mark.h5ad')

# CAST_STACK ###################################################################

import os
import sys
import torch
import CAST
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# set paths 
working_dir = 'project/spatial-pregnancy-postpart'
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
                query_reference_list[new_key] = [new_key, 'C57BL6J-638850.46']
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

# rotate query coords to help with registration
def rotate_coords(coords, angle):
    theta = np.radians(angle)
    rot_mat = torch.tensor([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]],
        dtype=torch.float32)
    return torch.mm(torch.from_numpy(coords).float(), rot_mat).numpy()

# load data 
adata_comb = ad.read_h5ad(
    f'{working_dir}/output/merfish/adata_comb_cast_mark.h5ad')
coords_raw = torch.load(
    f'{working_dir}/output/merfish/coords_raw.pt')
embed_dict = torch.load(
    f'{working_dir}/output/merfish/embed_dict.pt')

# rotate 
rotation_angles = {
    'CTRL1': 72, 'CTRL2': 110, 'CTRL3': -33,
    'PREG1': 3, 'PREG2': -98, 'PREG3': -138,
    'POSTPART1': 75, 'POSTPART2': 115, 'POSTPART3': -65
}
coords_raw = {
    k: rotate_coords(v, rotation_angles[k.split('_')[0]]) 
    if not k.startswith('C57BL6J') else v 
    for k, v in coords_raw.items()
}

# split  
coords_raw, embed_dict, indices_dict, query_reference_list  = \
      split_dicts(coords_raw, embed_dict, n_split=10)

# run cast-stack, parameters modified for default are commented 
coords_final_split = {}
for sample in sorted(query_reference_list.keys()):
    params_dist = CAST.reg_params(
        dataname = query_reference_list[sample],
        gpu = 0 if torch.cuda.is_available() else -1,
        diff_step = 5,
        #### Affine parameters
        iterations=50,
        dist_penalty1=0,
        bleeding=500,
        d_list = [3,2,1,1/2,1/3],
        attention_params = [None,3,1,0],
        #### FFD parameters
        dist_penalty2 = [0],
        alpha_basis_bs = [500],
        meshsize = [8],
        iterations_bs = [0], # No FFD
        attention_params_bs = [[None,3,1,0]],
        mesh_weight = [None])
    
    params_dist.alpha_basis = torch.Tensor(
        [1/1000,1/1000,1/50,5,5]).reshape(5,1).to(params_dist.device)

    coords_final_split[sample] = CAST.CAST_STACK(
        coords_raw, 
        embed_dict, 
        f'{working_dir}/output/merfish/CAST-STACK',
        query_reference_list[sample],
        params_dist, 
        rescale=True)

# collapse back
coords_final = collapse_dicts(coords_final_split, indices_dict)

# add final coords to anndata object 
sample_names = sorted(list(coords_final.keys()))
cell_index = adata_comb.obs.index[adata_comb.obs['source'] == 'merfish']
coords_stack = np.vstack([coords_final[s] for s in sample_names])
coords_df = pd.DataFrame(coords_stack, 
                         columns=['x_final', 'y_final'], 
                         index=cell_index)
adata_comb.obs = adata_comb.obs.join(coords_df)
mask = adata_comb.obs['source'] == 'Zeng-ABCA-Reference'
adata_comb.obs.loc[mask, 'x_final'] = adata_comb.obs.loc[mask, 'x']
adata_comb.obs.loc[mask, 'y_final'] = adata_comb.obs.loc[mask, 'y']

# save
torch.save(coords_final, f'{working_dir}/output/merfish/coords_final.pt')
adata_comb.write(f'{working_dir}/output/merfish/adata_comb_cast_stack.h5ad')

# Post-processing ##############################################################

import os
import sys
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

working_dir = 'project/spatial-pregnancy-postpart'

# add new obs columns to query data
adata_comb = ad.read_h5ad(
    f'{working_dir}/output/merfish/adata_comb_cast_stack.h5ad')
adata_query = ad.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish.h5ad')

adata_query_i = adata_comb[adata_comb.obs['source'] == 'merfish'].copy()
adata_query = adata_query[adata_query_i.obs_names]

for col in adata_query_i.obs.columns.drop(['x', 'y']):
    if col not in adata_query.obs.columns:
        adata_query.obs[col] = adata_query_i.obs[col]

# add cell type colors
cells_joined = pd.read_csv(
    'project/single-cell/ABC/metadata/MERFISH-C57BL6J-638850/20231215/'
    'views/cells_joined.csv')







adata_query.X = adata_query.layers['counts']

adata_query.write(f'{working_dir}/output/data/adata_query_merfish_final.h5ad')

# save the reference obs
adata_ref = adata_comb[adata_comb.obs['source'] == 'Zeng-ABCA-Reference']
adata_ref_obs = adata_ref.obs
adata_ref_obs.to_csv(
    f'{working_dir}/output/data/adata_ref_final_merfish_obs.csv')







































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

ref_obs = adata_comb[adata_comb.obs['source'] == 'Zeng-ABCA-Reference'].obs
query_obs = adata_comb[adata_comb.obs['source'] == 'merfish'].obs

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
    plt.savefig(f'{output_dir}/{safe_filename}.png', dpi=200, 
                bbox_inches='tight')
    plt.close(fig)

col = 'subclass'
output_dir = f'{working_dir}/figures/merfish/spatial_cell_types_{col}_final'
os.makedirs(output_dir, exist_ok=True)
cell_types = pd.concat([ref_obs[col], query_obs[col]]).unique()

for cell_type in cell_types:
    if (ref_obs[col].value_counts().get(cell_type, 0) > 0 or 
        query_obs[col].value_counts().get(cell_type, 0) > 0):
        create_multi_sample_plot(ref_obs, query_obs, col, cell_type, output_dir)