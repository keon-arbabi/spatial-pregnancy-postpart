import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import sys, os, torch, CAST, shutil, warnings
import matplotlib.pyplot as plt, seaborn as sns
from scipy import sparse
warnings.filterwarnings("ignore")

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from utils import debug
debug(third_party=True)

# CAST_MARK ####################################################################

data_dir = 'projects/def-wainberg/spatial'
working_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart'

ref_dir = f'{data_dir}/Zhuang/direct-downloads'  
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
    adata.obs['y'] = -adata.obs['y'] # invert y
    adata.var.reset_index()
    adata.var.index = adata.var['gene_symbol']
    print(f'[{data}] {adata.shape[0]} cells')
    adatas_ref.append(adata)

adata_ref = ad.concat(adatas_ref)
adata_ref.X = sparse.csr_matrix(adata_ref.X)
adata_ref.layers['counts'] = adata_ref.X.copy()
sc.pp.normalize_total(adata_ref)
sc.pp.log1p(adata_ref)
adata_ref.var['gene_symbol'] = adata_ref.var.index
adata_ref.write(f'{working_dir}/output/data/adata_ref.h5ad')

adatas_query = []
for source in ['MERFISH', 'CURIO']:
    query_dir = f'{data_dir}/Kalish/pregnancy-postpart/' \
        f'{source}/rotate-split-raw'
    samples_query = [
        file.replace('.h5ad', '') 
        for file in os.listdir(query_dir)]
    samples_query = sorted(samples_query)

    for sample in samples_query:
        adata = ad.read_h5ad(f'{query_dir}/{sample}.h5ad')
        adata.obs['sample'] = source + '_' + sample
        adata.obs['source'] = source
        adata.obs[[
            'class', 'class_color', 'subclass', 'subclass_color',
            'supertype', 'supertype_color', 'cluster', 'cluster_color',
            'parcellation_substructure', 
            'parcellation_substructure_color']] = 'Unknown'
        print(f'[{source} {sample}] {adata.shape[0]} cells')
        adatas_query.append(adata)

adata_query = sc.concat(adatas_query)
adata_query.layers['counts'] = adata_query.X.copy()
sc.pp.normalize_total(adata_query)
sc.pp.log1p(adata_query)
adata_query.var['gene_symbol'] = adata_query.var.index
adata_query.write(f'{working_dir}/output/data/adata_query.h5ad')

adata_comb = ad.concat([adata_query, adata_ref], merge='same')
adata_comb.obs.index = adata_comb.obs['sample'].astype(str) + '_' + \
      adata_comb.obs.index

sample_names = sorted(adata_comb.obs['sample'].unique())
adata_comb.obs['sample'] = pd.Categorical(
    adata_comb.obs['sample'], categories=sample_names, ordered=True)
adata_comb = adata_comb[
    adata_comb.obs.sort_values('sample', kind='stable').index]

coords_raw = {
    sample: np.array(adata_comb.obs[['x', 'y']])
    [adata_comb.obs['sample'] == sample] for sample in sample_names}
exp_dict = {
    sample: adata_comb[adata_comb.obs['sample'] == sample]
    .X.toarray() for sample in sample_names}

embed_dict = CAST.CAST_MARK(
    coords_raw, exp_dict, f'{working_dir}/output/CAST-MARK')

embed_dict = torch.load(
    f'{working_dir}/output/CAST-MARK/demo_embed_dict.pt',
    map_location='cpu')

from sklearn.cluster import KMeans
embed_stack = np.vstack([embed_dict[name].cpu().detach().numpy()
                        for name in sample_names])
n_clust = 15
kmeans = KMeans(n_clusters=n_clust, random_state=0).fit(embed_stack)
cell_label = kmeans.labels_
cluster_pl = sns.color_palette('Set3', len(np.unique(cell_label)))
np.random.shuffle(cluster_pl)

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
        ' (KMeans, k = ' + str(n_clust) + ')',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
plt.savefig(
    f'{working_dir}/figures/all_samples_trained_k{str(n_clust)}.png', 
    dpi=200)

adata_comb.obs[f'k{n_clust}_cluster'] = cell_label
color_map = {k: color for k, color in enumerate(cluster_pl.as_hex())}
adata_comb.obs[f'k{n_clust}_cluster_colors'] = pd.Series(cell_label)\
    .map(color_map).tolist()
adata_comb.write(f'{working_dir}/output/MERFISH/data/adata_comb.h5ad')

torch.save(coords_raw, f'{working_dir}/output/MERFISH/data/coords_raw.pt')
torch.save(exp_dict, f'{working_dir}/output/MERFISH/data/exp_dict.pt')