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
    adata.obs[[
        'class', 'class_color', 'subclass', 'subclass_color',
        'parcellation_division', 'parcellation_division_color',
        'parcellation_structure', 'parcellation_structure_color']] = 'Unknown'
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
            dpi=150, bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/merfish/qc_scores_violin.svg',
            bbox_inches='tight')

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

# load mapper results  
with open(f'{working_dir}/output/merfish/mapper_output.json') as f:
    mapper_json = json.load(f)

# get reference cell types and map to taxonomy ids
obs_ref_zeng = ad.read_h5ad(
    f'{working_dir}/output/data/adata_ref_zeng_raw.h5ad').obs
obs_ref_zhuang = ad.read_h5ad(
    f'{working_dir}/output/data/adata_ref_zhuang.h5ad').obs

keep_cell_types = (obs_ref_zeng['subclass'].value_counts()[
    obs_ref_zeng['subclass'].value_counts() > 20].index.union(
    obs_ref_zhuang['subclass'].value_counts()[
        obs_ref_zhuang['subclass'].value_counts() > 20].index))
print(len(keep_cell_types))

# get reverse mapping from names to ids for filtering
mapper_names = {
    level: mapper_json['taxonomy_tree']['name_mapper'][
        f'CCN20230722_{level.upper()}'] 
    for level in ['clas', 'subc']
}
name_to_id = {
    name: id 
    for id, data in mapper_names['subc'].items() 
    for name in [data.get('name')]
}
keep_cell_type_ids = [
    name_to_id[name] for name in keep_cell_types if name in name_to_id
]

# create expanded dataframe including runner-up info
rows = []
for result in mapper_json['results']:
    cell_id = result['cell_id']
    for level, data in result.items():
        if isinstance(data, dict):
            # add main assignment
            row = {
                'cell_id': cell_id,
                'level': level,
                'assignment': data['assignment'],
                'bootstrapping_probability': data['bootstrapping_probability'],
                'avg_correlation': data['avg_correlation']
            }
            rows.append(row)
            # add runner-up assignments if they exist
            if 'runner_up_assignment' in data:
                for i, runner_up in enumerate(data['runner_up_assignment']):
                    row = {
                        'cell_id': cell_id,
                        'level': level,
                        'assignment': runner_up,
                        'bootstrapping_probability': 
                            data['runner_up_probability'][i],
                        'avg_correlation': data['runner_up_correlation'][i]
                    }
                    rows.append(row)

mapper_df = pd.DataFrame(rows)

# pre-filter and sort data frames 
subc_df = (mapper_df[mapper_df['level'] == 'subc']
           .sort_values('cell_id')
           .set_index('cell_id'))
class_df = (mapper_df[mapper_df['level'] == 'clas']
           .groupby('cell_id').first())

# get main assignments (first row for each cell)
main_assignments = subc_df.groupby('cell_id').first()
valid_main = main_assignments[
    main_assignments['assignment'].isin(keep_cell_type_ids)]

# for cells without valid main, get first valid runner-up
invalid_cells = set(subc_df.index) - set(valid_main.index)
runner_ups = subc_df.loc[list(invalid_cells)].groupby('cell_id').apply(
    lambda x: x[x['assignment'].isin(keep_cell_type_ids)].iloc[0] 
    if any(x['assignment'].isin(keep_cell_type_ids)) else None
).dropna()

# combine valid assignments
valid_subc = pd.concat([valid_main, runner_ups])
valid_class = class_df.loc[valid_subc.index]
valid_assignments = pd.concat([valid_subc, valid_class])

# reset index to make cell_id a column before pivoting
mapper_df = valid_assignments.reset_index()

# pivot and flatten column names
mapper_df = mapper_df.pivot(
    index='cell_id',
    columns='level',
    values=['assignment', 'bootstrapping_probability', 'avg_correlation'])
mapper_df.columns = [f'{col[0]}_{col[1]}' for col in mapper_df.columns]

# mark invalid assignments as NA
invalid_mask = ~mapper_df['assignment_subc'].isin(keep_cell_type_ids)
mapper_df.loc[invalid_mask] = np.nan

# join mapper results to anndata
adata_query.obs = adata_query.obs.join(mapper_df)

# map ids back to names
for level, new_col in [('clas', 'class_mapper'), ('subc', 'subclass_mapper')]:
    mapper_df[new_col] = mapper_df[f'assignment_{level}'].map(
        lambda x: mapper_names[level].get(x, {}).get('name', 'Unknown')
        ).astype('category')

# convert data types
for metric in ['bootstrapping_probability', 'avg_correlation']:
    for level in ['clas', 'subc']:
        mapper_df[f'{metric}_{level}'] = mapper_df[
            f'{metric}_{level}'].astype(float)
for level in ['clas', 'subc']:
    mapper_df[f'assignment_{level}'] = mapper_df[
        f'assignment_{level}'].astype('category')

# plot mapping metrics
metrics = ['bootstrapping_probability', 'avg_correlation']
titles = ['Bootstrapping Probability', 'Average Correlation']
pink = sns.color_palette('PiYG')[0]

fig, axes = plt.subplots(len(metrics), 1, figsize=(6, 3*len(metrics)))
for i, (metric, title) in enumerate(zip(metrics, titles)):
    plot_df = pd.DataFrame({
        'value': mapper_df[f'{metric}_subc'],
        'sample': adata_query.obs['sample'],
        'cell_type': mapper_df['subclass_mapper']
    })
    sns.violinplot(
        data=plot_df, x='sample', y='value', ax=axes[i],
        color=pink, alpha=0.5, linewidth=1, linecolor=pink,
        order=sample_order)

    axes[i].set_title(title, fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Score', fontsize=11, fontweight='bold')
    if i < len(metrics) - 1:
        axes[i].set_xticklabels([])
        axes[i].set_xlabel('')
        axes[i].set_xticks([])
    else:
        axes[i].set_xticklabels(
            sample_labels, rotation=45, ha='right', va='top')
        axes[i].tick_params(axis='x', rotation=45)
        plt.setp(axes[i].get_xticklabels(), ha='right', va='top')

plt.tight_layout()
plt.savefig(
    f'{working_dir}/figures/merfish/mapping_scores_violin.svg',
    bbox_inches='tight')
plt.savefig(
    f'{working_dir}/figures/merfish/mapping_scores_violin.png',
    dpi=150, bbox_inches='tight')

# join mapper results to anndata, keeping only mapped cells
adata_query = adata_query[mapper_df.index]
adata_query.obs = adata_query.obs.join(mapper_df)

# plot mapping metrics vs qc metrics
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
plots = [
    ('n_genes_by_counts', 'avg_correlation_subc', 'Genes vs Correlation'),
    ('n_genes_by_counts', 'bootstrapping_probability_subc', 
     'Genes vs Probability'),
    ('avg_correlation_subc', 'bootstrapping_probability_subc', 
     'Correlation vs Probability')
]
for i, (x, y, title) in enumerate(plots):
    sns.scatterplot(
        data=adata_query.obs, x=x, y=y, ax=axes[i], 
        color=pink, alpha=0.005, s=10, linewidth=0)
    if x == 'n_genes_by_counts':
        axes[i].set_xscale('log')
    axes[i].set_title(title, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(
    f'{working_dir}/figures/merfish/qc_vs_mapping_scatter.png',
    dpi=150, bbox_inches='tight')

# # filter cells based on mapping metrics
# print(adata_query.shape[0])
# mask = (
#     (adata_query.obs['bootstrapping_probability_subc'] > 0.4) &
#     (adata_query.obs['avg_correlation_subc'] > 0.3))
# print(sum(mask))
# adata_query = adata_query[mask].copy()

# save
adata_query.write(f'{working_dir}/output/adata_query_merfish.h5ad')

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

# check if embeddings already exist
embed_dict_path = f'{working_dir}/output/merfish/embed_dict.pt'
if os.path.exists(embed_dict_path):
    embed_dict = torch.load(embed_dict_path)
else:
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
            n_layers=12, # number of GCNII layers, more layers mean a deeper model,
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
    fig.savefig(f'{plot_dir}/all_samples_k{str(n_clust)}.png', dpi=150)
    plt.close(fig)
    adata_comb.obs[f'k{n_clust}_cluster'] = cell_label

# Save results
torch.save(coords_raw, f'{working_dir}/output/merfish/coords_raw.pt')
torch.save(exp_dict, f'{working_dir}/output/merfish/exp_dict.pt')
adata_comb.write_h5ad(f'{working_dir}/output/merfish/adata_comb_cast_mark.h5ad')

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
os.makedirs(f'{working_dir}/output/merfish/CAST-STACK', exist_ok=True)

# rotate query coords to help with registration
def rotate_coords(coords, angle):
    theta = np.radians(angle)
    rot_mat = torch.tensor([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]],
        dtype=torch.float32)
    return torch.mm(torch.from_numpy(coords).float(), rot_mat).numpy()

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
            data = next(iter(coords_final[base_key].values()))
            collapsed[base_key] = np.asarray(data, dtype=np.float32)
        else:
            full_array = np.zeros((len(indices), 2), dtype=np.float32)
            start_idx = 0
            for i in range(1, len(coords_final) + 1):
                key = f'{base_key}_{i}'
                if key in coords_final:
                    split_data = next(v for k, v in coords_final[key].items() 
                                   if not k.startswith('C57BL6J'))
                    split_data = np.asarray(split_data, dtype=np.float32)
                    end_idx = start_idx + len(split_data)
                    split_indices = indices[start_idx:end_idx]
                    full_array[split_indices] = split_data
                    start_idx = end_idx
            collapsed[base_key] = full_array
    return collapsed

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
coords_raw, embed_dict, indices_dict, query_reference_list = \
    split_dicts(coords_raw, embed_dict, n_split=15)

coords_affine_split = {}
coords_ffd_split = {}
for sample in sorted(query_reference_list.keys()):
    cache_path = f'{working_dir}/output/merfish/CAST-STACK/{sample}.pt'
    if os.path.exists(cache_path):
        print(f'Loading cached coordinates for {sample}')
        coords_affine_split[sample], coords_ffd_split[sample] = \
            torch.load(cache_path)
        continue

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
        dist_penalty2 = [0.2],
        alpha_basis_bs = [500],
        meshsize = [8],
        iterations_bs = [80], 
        attention_params_bs = [[None,3,1,0]],
        mesh_weight = [None])
    
    params_dist.alpha_basis = torch.Tensor(
        [1/1000,1/1000,1/50,5,5]).reshape(5,1).to(params_dist.device)

    coords_affine_split[sample], coords_ffd_split[sample] = \
        CAST.CAST_STACK(
            coords_raw, 
            embed_dict, 
            f'{working_dir}/output/merfish/CAST-STACK',
            query_reference_list[sample],
            params_dist, 
            mid_visual=False,
            rescale=True)
    
    print(coords_affine_split[sample])
    print(coords_ffd_split[sample])
    torch.save((coords_affine_split[sample], coords_ffd_split[sample]), 
               cache_path)

# collapse replicates
coords_affine = collapse_dicts(coords_affine_split, indices_dict)
coords_ffd = collapse_dicts(coords_ffd_split, indices_dict)

# add coords to adata
sample_names = sorted(list(coords_ffd.keys()))
cell_index = adata_comb.obs.index[adata_comb.obs['source'] == 'merfish']

coords_stack = np.vstack([coords_affine[s] for s in sample_names])
coords_df = pd.DataFrame(coords_stack, 
                        columns=['x_affine', 'y_affine'], 
                        index=cell_index)
adata_comb.obs = adata_comb.obs.join(coords_df)

coords_stack_ffd = np.vstack([coords_ffd[s] for s in sample_names])
coords_df_ffd = pd.DataFrame(coords_stack_ffd, 
                           columns=['x_ffd', 'y_ffd'], 
                           index=cell_index)
adata_comb.obs = adata_comb.obs.join(coords_df_ffd)

mask = adata_comb.obs['source'] == 'Zeng-ABCA-Reference'
for coord in ['affine', 'ffd']:
    adata_comb.obs.loc[mask, f'x_{coord}'] = adata_comb.obs.loc[mask, 'x']
    adata_comb.obs.loc[mask, f'y_{coord}'] = adata_comb.obs.loc[mask, 'y']

torch.save(coords_affine, f'{working_dir}/output/merfish/coords_affine.pt')
torch.save(coords_ffd, f'{working_dir}/output/merfish/coords_ffd.pt')
adata_comb.write(f'{working_dir}/output/merfish/adata_comb_cast_stack.h5ad')

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
os.makedirs(f'{working_dir}/output/merfish/CAST-PROJECT', exist_ok=True)

# load data
adata_comb = ad.read_h5ad(
    f'{working_dir}/output/merfish/adata_comb_cast_stack.h5ad')

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
    output_dir_t = f'{working_dir}/output/merfish/CAST-PROJECT/{source_sample}_to_{target_sample}'
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
        ave_dist_fold=30,
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
new_obs.to_csv(f'{working_dir}/output/merfish/new_obs.csv', index_label='cell_id')

# plot cast metrics
metrics = ['subclass_confidence', 'avg_cdist', 'avg_pdist']
titles = ['Subclass Assignment Confidence', 'Average Cosine Distance',
          'Average Physical Distance']
y_labels = ['Confidence Score', 'Cosine Distance', 'Physical Distance (μm)']

sample_order = [
    'CTRL1', 'CTRL2', 'CTRL3', 
    'PREG1', 'PREG2', 'PREG3',
    'POSTPART1', 'POSTPART2', 'POSTPART3']
sample_labels = [
    'Control 1', 'Control 2', 'Control 3',
    'Pregnant 1', 'Pregnant 2', 'Pregnant 3',
    'Postpartum 1', 'Postpartum 2', 'Postpartum 3']

configs = {
    'subclass_confidence': dict(
        log=False, lines=(0.7, None),
        ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    'avg_cdist': dict(
        log=False, lines=(0.7, None),
        ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], invert=True),
    'avg_pdist': dict(
        log=False, lines=(None, None),
        ticks=[0, 1], invert=True)
}

pink = sns.color_palette("PiYG")[0]
fig, axes = plt.subplots(len(metrics), 1, figsize=(6, 3*len(metrics)))

for i, (m, title, ylabel) in enumerate(zip(metrics, titles, y_labels)):
    cfg = configs[m]  
    plot_data = pd.concat(new_obs_list)
    plot_data = plot_data[~np.isinf(plot_data[m])]
    sns.violinplot(
        data=plot_data, x='sample', y=m, ax=axes[i],
        color=pink, alpha=0.5, linewidth=1, linecolor=pink,
        order=sample_order)
    if cfg['log']:
        axes[i].set_yscale('log')
        axes[i].set_yticks(cfg['ticks'])
        axes[i].set_yticklabels([str(x) for x in cfg['ticks']])
    if cfg.get('invert', False):
        axes[i].invert_yaxis()
    for val in cfg['lines']:
        if val is not None:
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
plt.savefig(f'{working_dir}/figures/merfish/cast_metrics_violin.svg',
            bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/merfish/cast_metrics_violin.png',
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
    f'{working_dir}/output/data/adata_query_merfish.h5ad')
adata_query.obs = adata_query.obs.drop(columns=[
    'class', 'subclass', 'class_color', 'subclass_color'])

new_obs = pd.read_csv(
    f'{working_dir}/output/merfish/new_obs.csv',index_col='cell_id')
for col in new_obs.columns:
    if col not in adata_query.obs.columns:
        adata_query.obs[col] = new_obs[col]

# filter cells 
total = len(adata_query)
for name, mask in {
    'low subclass confidence': adata_query.obs['subclass_confidence'] <= 0.7,
    'high expression dist': adata_query.obs['avg_cdist'] >= 0.7
}.items():
    print(f'{name}: {mask.sum()} ({mask.sum()/total*100:.1f}%) cells dropped')
'''
low subclass confidence: 276418 (23.6%) cells dropped
high expression dist: 42198 (3.6%) cells dropped

'''
mask = ((adata_query.obs['class_confidence'] >= 0.7) &
        (adata_query.obs['avg_cdist'] <= 0.7))
        
cells_dropped = total - mask.sum()
print(f'\nTotal cells dropped: {cells_dropped} '
      f'({cells_dropped/total*100:.1f}%)')
'''Total cells dropped: 115347 (9.8%)'''

adata_query = adata_query[mask].copy()

# remove noise cells 
mapping_group_1 = {
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
mapping_group_2 = {
   '01 IT-ET Glut': 'Glut',
   '02 NP-CT-L6b Glut': 'Glut', 
   '05 OB-IMN GABA': 'GABA',
   '06 CTX-CGE GABA': 'GABA',
   '07 CTX-MGE GABA': 'GABA',
   '08 CNU-MGE GABA': 'GABA',
   '09 CNU-LGE GABA': 'GABA',
   '10 LSX GABA': 'GABA',
   '11 CNU-HYa GABA': 'GABA',
   '12 HY GABA': 'GABA',
   '13 CNU-HYa Glut': 'Glut',
   '14 HY Glut': 'Glut',
   '30 Astro-Epen': 'Glia',
   '31 OPC-Oligo': 'Glia',
   '33 Vascular': 'Glia',
   '34 Immune': 'Glia'
}
adata_query = adata_query.copy()
adata_query.obs['group_1'] = adata_query.obs['class'].map(mapping_group_1)
adata_query.obs['group_2'] = adata_query.obs['class'].map(mapping_group_2)

adata_query.X = adata_query.layers['volume_log1p']
sc.pp.highly_variable_genes(adata_query, n_top_genes=2000, batch_key='sample')
sc.tl.pca(adata_query)
sc.pp.neighbors(adata_query)

nn_mat = adata_query.obsp['distances'].astype(bool)
labels = adata_query.obs['group_2'].values
confidences = []
for i in range(len(labels)):
    neighbor_idx = nn_mat[i].indices
    confidence = np.mean(labels[neighbor_idx] == labels[i])
    confidences.append(confidence)
adata_query.obs['group_2_confidence'] = confidences

sns.ecdfplot(data=adata_query.obs, x='group_2_confidence')
plt.savefig(f'{working_dir}/figures/merfish/broad_class_confidence_ecdf.png',
            dpi=200, bbox_inches='tight')

mask = adata_query.obs['group_2_confidence'] < 0.8
print(sum(mask))
# 151541

sc.tl.umap(adata_query)
sc.pl.umap(adata_query, color=['group_2', 'group_2_confidence'])
plt.savefig(f'{working_dir}/figures/merfish/broad_class_confidence_umap.png',
           dpi=200, bbox_inches='tight')
plt.close()

adata_query = adata_query[~mask]

# keep cell types with at least 5 cells in at least 3 samples per condition
min_cells, min_samples = 10, 3
conditions = ['CTRL', 'PREG', 'POSTPART']
kept_types = [
    subclass for subclass in adata_query.obs['subclass'].unique()
    if all(sum(sum((adata_query.obs['sample'] == s) & 
              (adata_query.obs['subclass'] == subclass)) >= min_cells
          for s in adata_query.obs['sample'][
              adata_query.obs['sample'].str.contains(c)].unique()) >= min_samples
        for c in conditions)
]

print("Kept cell types:")
for t in sorted(kept_types):
    print(f"- {t}")

print("\nDropped cell types:")
for t in sorted(set(adata_query.obs['subclass']) - set(kept_types)):
    print(f"- {t}")

adata_query.obs['keep_subclass'] = adata_query.obs['subclass'].isin(kept_types)
print(adata_query.obs['keep_subclass'].value_counts())
'''
True     902685
False      1952
'''

# add colors 
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
for level in ['class', 'subclass']:
  unique_categories = adata_query.obs[level].unique()
  category_colors = [color_mappings[level][cat] for cat in unique_categories]
  adata_query.uns[f'{level}_colors'] = category_colors
  adata_query.uns[f'{level}_color_dict'] = color_mappings[level]

# umap
seed = 0
sc.pp.neighbors(adata_query, metric='cosine', random_state=seed)
sc.tl.umap(adata_query, min_dist=0.4, spread=1.0, random_state=seed)

sc.pl.umap(adata_query, color='class', title=None)
plt.savefig(f'{working_dir}/figures/merfish/umap_class.png',
            dpi=300, bbox_inches='tight')

# save
adata_query.X = adata_query.layers['counts']
adata_query.write(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')

# plotting #####################################################################

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

working_dir = 'project/spatial-pregnancy-postpart'

adata_comb = sc.read_h5ad(
    f'{working_dir}/output/curio/adata_comb_cast_stack.h5ad')
obs_ref = adata_comb[adata_comb.obs['source'] == 'Zeng-ABCA-Reference'].obs

adata_query = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')
obs_query = adata_query.obs

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

# create multi-sample plots
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
                      c='grey', s=0.2, alpha=0.1)
            ax.scatter(plot_df[mask][coord_cols[0]], 
                      plot_df[mask][coord_cols[1]], 
                      c=cell_color, s=1)
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
output_dir = f'{working_dir}/figures/merfish/spatial_cell_types_{col}'
os.makedirs(output_dir, exist_ok=True)
cell_types = obs_query[col].unique()
for cell_type in cell_types:
    create_multi_sample_plot(obs_ref, obs_query, col, cell_type, output_dir)

# radius plot 
plot_df = adata_comb.obs[adata_comb.obs['sample'] == 'PREG3']

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(plot_df['x_ffd'], plot_df['y_ffd'], c='grey', s=1)
random_point = plot_df.sample(n=1)
ax.scatter(random_point['x_ffd'], random_point['y_ffd'], c='red', s=10)
from matplotlib.patches import Circle
radius = 0.546297587762388  # ave_dist_fold=30
circle = Circle((random_point['x_ffd'].values[0], 
                random_point['y_ffd'].values[0]), 
                radius, fill=False, color='red')
ax.add_artist(circle)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig(f'{working_dir}/figures/merfish/radius.png', dpi=200)



