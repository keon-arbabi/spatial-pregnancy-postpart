import os, sys, gc
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import coo_array
from scipy.spatial import KDTree
from typing import Tuple, List
from ryp import r, to_r, to_py
from tqdm.auto import tqdm
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

################################################################################
#region functions

def calculate_distance_scale(coords: np.ndarray) -> float:
    tree = KDTree(coords)
    d_scale = np.median(tree.query(coords, k=2)[0][:, 1])
    return d_scale, tree

def plot_sample_radii(
    spatial_data: pd.DataFrame, 
    coords_cols: Tuple[str, str],
    sample_col: str,
    d_max_scale: float,
    *,
    figsize: Tuple[int, int] = (15, 10)) -> Tuple[plt.Figure, np.ndarray]:

    samples = spatial_data[sample_col].unique()
    n_cols = 3
    n_rows = int(np.ceil(len(samples) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for ax, sample in zip(axes, samples):
        sample_data = spatial_data[spatial_data[sample_col] == sample]
        coords = sample_data[list(coords_cols)].values
        d_scale, _ = calculate_distance_scale(coords)
        d_max = d_max_scale * d_scale
        ax.scatter(
            coords[:, 0], coords[:, 1], s=0.05, alpha=1, c='gray',
            linewidth=0)
        random_idx = np.random.randint(len(coords))
        random_point = coords[random_idx]
        circle = plt.Circle(
            random_point, d_max, fill=False, color='red', 
            linewidth=0.1)
        ax.add_patch(circle)
        ax.scatter(*random_point, c='red', s=0.05)
        ax.set_title(f'Sample: {sample}')
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
    
    for ax in axes[len(samples):]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig, axes

def get_spatial_stats(
    spatial_data: pd.DataFrame,
    coords_cols: Tuple[str, str],
    cell_type_col: str,
    condition_col: str,
    sample_col: str,
    d_min_scale: float = 0,
    d_max_scale: float = 5) -> pd.DataFrame:
    
    coords = spatial_data[list(coords_cols)].values
    d_scale, tree = calculate_distance_scale(coords)
    
    d_min = d_min_scale * d_scale
    d_max = d_max_scale * d_scale
    
    pairs = tree.query_pairs(d_max)
    if d_min > 0:
        pairs -= tree.query_pairs(d_min)
    mat = np.array(list(pairs))
    mat = np.concatenate((mat, mat[:, ::-1]))
    sparse_mat = coo_array((np.ones(len(mat), dtype=bool), mat.T),
        shape=(len(spatial_data), len(spatial_data))).tocsr()
    
    condition = spatial_data[condition_col].iloc[0]
    sample_id = spatial_data[sample_col].iloc[0]
    results = []
    for cell_type_b in spatial_data[cell_type_col].unique():
        cell_type_mask = spatial_data[cell_type_col].values == cell_type_b
        cell_b_count = sparse_mat[:, cell_type_mask].sum(axis=1)
        all_count = sparse_mat.sum(axis=1)
        with np.errstate(invalid='ignore'):
            cell_b_ratio = cell_b_count / all_count
            
        results.append(pd.DataFrame({
            'cell_id': spatial_data.index,
            'cell_type_a': spatial_data[cell_type_col].values,
            'cell_type_b': cell_type_b,
            'b_count': cell_b_count,
            'all_count': all_count,
            'b_ratio': cell_b_ratio,
            'd_min': d_min,
            'd_max': d_max,
            'condition': condition,
            'sample_id': sample_id
        }))
    return pd.concat(results).reset_index(drop=True)

def get_spatial_diff(
    spatial_stats: pd.DataFrame,
    cell_type_a: str,
    cell_type_b: str) -> List[pd.DataFrame]:

    sub = spatial_stats[
        (spatial_stats['cell_type_a'] == cell_type_a) &
        (spatial_stats['cell_type_b'] == cell_type_b)
    ].copy().set_index('cell_id')
    
    counts = pd.DataFrame({
        'b_count': sub['b_count'].astype(int),
        'other_count': (sub['all_count'] - sub['b_count']).astype(int)
    }, index=sub.index)
    meta = sub[['condition', 'sample_id']]
    
    to_r(counts, 'counts', format='data.frame')
    to_r(meta, 'meta', format='data.frame')
    r('''
    suppressPackageStartupMessages({
        library(crumblr)
        library(variancePartition)
        library(parallel)
    })
    param = SnowParam(detectCores(), 'SOCK', progressbar = TRUE)
    cobj = crumblr(counts, method='clr_2class')
    form = ~ 0 + condition + (1 | sample_id)
    L = makeContrastsDream(form, meta,
        contrasts = c(
        PREG_vs_CTRL="conditionPREG-conditionCTRL",
        POST_vs_PREG="conditionPOSTPART-conditionPREG"
        )
    )
    fit = dream(cobj, form, meta, L, param=param)
    fit = eBayes(fit)
    tt = list(
    PREG_vs_CTRL=topTable(fit,coef="PREG_vs_CTRL",number=Inf),
    POST_vs_PREG=topTable(fit,coef="POST_vs_PREG",number=Inf)
    )
    ''')
    
    tt = to_py('tt', format='pandas')
    pair_results = []
    for contrast, df in tt.items():
        df = df.loc[df.index == "b_count"].copy()
        df["contrast"] = contrast
        df["cell_type_a"] = cell_type_a
        df["cell_type_b"] = cell_type_b
        pair_results.append(df)
    return pair_results

def plot_spatial_diff_heatmap(
    df, tested_pairs, contrast, sig=0.10, figsize=(15, 15)):
    
    s1 = set(df['cell_type_a'].unique())
    s2 = set(df['cell_type_b'].unique())
    all_ct = sorted(list(s1.union(s2)))
    n = len(all_ct)
    mat = np.full((n, n), np.nan)
    sigs = np.full((n, n), '', dtype=object)
    idx = {c: i for i, c in enumerate(all_ct)}
    for _, row in df.iterrows():
        i, j = idx[row['cell_type_a']], idx[row['cell_type_b']]
        mat[i, j] = row['logFC']
        if row['adj.P.Val'] < sig: sigs[i, j] = '*'
    fig, ax = plt.subplots(figsize=figsize)
    cmap = LinearSegmentedColormap.from_list(
        "custom_diverging",
        ["#156a2f", "#66b66b", "white", "#813e8f", "#4b0857"], N=100)
    mabs = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap=cmap, vmin=-mabs, vmax=mabs, aspect='equal')
    for i in range(n):
        for j in range(n):
            a, b = all_ct[i], all_ct[j]
            if (a, b) not in tested_pairs:
                ax.text(j, i, 'X', ha='center', va='center', 
                        color='gray', size=10)
            elif sigs[i, j]=='*':
                ax.text(j, i, '*', ha='center', va='center',
                        color='white', size=14)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(all_ct, rotation=45, ha='right')
    ax.set_yticklabels(all_ct)
    ax.set_xlabel('Query (B)')
    ax.set_ylabel('Reference (A)')
    ax.set_title(f'{contrast} Spatial Diff', size=14)
    cbar = fig.colorbar(im, ax=ax, shrink=0.2) 
    cbar.set_label('logFC')
    plt.tight_layout()
    return fig, ax

#endregion

################################################################################
#region global proportions

working_dir = 'project/spatial-pregnancy-postpart'
adata = sc.read_h5ad(f'{working_dir}/output/data/adata_query_merfish_final.h5ad')

cell_type_col = 'subclass'
adata = adata[adata.obs[f'{cell_type_col}_keep']].copy()
adata.obs[cell_type_col] = adata.obs[cell_type_col].astype(str)\
    .str.extract(r'^(\d+)\s+(.*)', expand=False)[1]

df = pd.DataFrame({
    'sample': adata.obs['sample'], 
    'condition': adata.obs['condition'],
    'cell_type': adata.obs[cell_type_col]
})
counts = pd.crosstab(index=df['sample'], columns=df['cell_type'])
meta = df[['sample','condition']]\
    .drop_duplicates().set_index('sample', drop=False)

to_r(counts, 'counts', format='data.frame')
to_r(meta, 'meta', format='data.frame')
to_r(working_dir, 'working_dir')

r('''
suppressPackageStartupMessages({
    library(crumblr)
    library(variancePartition)
})

cobj = crumblr(counts)
    
form = ~ (1 | condition)
vp = fitExtractVarPartModel(cobj, form, meta)
png(file.path(working_dir, 'figures/merfish/varpart_fractions.png'))
plotPercentBars(vp)
dev.off()

form = ~ 0 + condition
L = makeContrastsDream(form, meta,
    contrasts = c(
        PREG_vs_CTRL = "conditionPREG - conditionCTRL",
        POST_vs_PREG = "conditionPOSTPART - conditionPREG"
    )
)
fit = dream(cobj, form, meta, L)
fit = eBayes(fit)

tt_preg_ctrl = topTable(fit, coef="PREG_vs_CTRL", number=Inf)
tt_post_preg = topTable(fit, coef="POST_vs_PREG", number=Inf)
''')

tt_preg_ctrl = to_py('tt_preg_ctrl', format='pandas') 
tt_post_preg = to_py('tt_post_preg', format='pandas')

from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy as hc
clust_ids = sorted(list(adata.obs[cell_type_col].unique()))
clust_avg = np.vstack([
    adata[adata.obs[cell_type_col] == i].layers['volume_log1p'].mean(0)
    for i in clust_ids
])

D = pdist(clust_avg, 'correlation')
Z = hc.linkage(D, 'complete', optimal_ordering=False)
n = len(clust_ids)

merge_matrix = np.zeros((n-1, 2), dtype=int)
for i in range(n-1):
    for j in range(2):
        val = Z[i,j]
        merge_matrix[i,j] = -int(val + 1) if val < n else int(val) - n + 1

hc_dict = {
    'merge': merge_matrix,
    'height': Z[:,2],
    'order': np.array([x+1 for x in hc.leaves_list(Z)]),
    'labels': np.array(clust_ids),
    'method': 'complete',
    'call': {},
    'dist.method': 'correlation'
}

to_r(hc_dict, 'hc')

r('''
library(crumblr)
library(patchwork)
  
hc = structure(hc, class = "hclust")
hc$call = NULL

res = treeTest(fit, cobj, hc, coef = "PREG_vs_CTRL", method = "FE") 
png(file.path(working_dir, 'figures/merfish/tree_test_ctrl_vs_preg.png'),
    width=10, height=12, units='in', res=300)
plotTreeTestBeta(res) + plotForest(res, hide = TRUE) +
    plot_layout(nrow = 1, widths = c(2, 1))
dev.off()
  
res = treeTest(fit, cobj, hc, coef = "POST_vs_PREG", method = "FE") 
png(file.path(working_dir, 'figures/merfish/tree_test_preg_vs_post.png'),
    width=10, height=12, units='in', res=300)
plotTreeTestBeta(res) + plotForest(res, hide = TRUE) +
    plot_layout(nrow = 1, widths = c(2, 1))
dev.off()
''')

#endregion

################################################################################
#region local proportions

working_dir = 'project/spatial-pregnancy-postpart'
adata = sc.read_h5ad(f'{working_dir}/output/data/adata_query_merfish_final.h5ad')

cell_type_col = 'subclass'
adata = adata[adata.obs[f'{cell_type_col}_keep']].copy()
adata.obs[cell_type_col] = adata.obs[cell_type_col].astype(str)\
    .str.extract(r'^(\d+)\s+(.*)', expand=False)[1]

d_max_scale = 15

# plot radius visualization for all samples
fig, axes = plot_sample_radii(
    spatial_data=adata.obs,
    coords_cols=('x_affine', 'y_affine'),
    sample_col='sample',
    d_max_scale=d_max_scale)
fig.savefig(f'{working_dir}/figures/merfish/proximity_radii.png', 
            dpi=300, bbox_inches='tight')
plt.close(fig)

# get spatial stats per sample
file = f'{working_dir}/output/merfish/spatial_stats_{cell_type_col}.pkl'
if os.path.exists(file):
    spatial_stats = pd.read_pickle(file)
else:
    results = []
    for sample in adata.obs['sample'].unique():
        sample_data = adata.obs[adata.obs['sample'] == sample]
        stats = get_spatial_stats(
            spatial_data=sample_data,
            coords_cols=('x_affine', 'y_affine'),
            cell_type_col=cell_type_col,
            condition_col='condition',
            sample_col='sample',
            d_min_scale=0,
            d_max_scale=d_max_scale
        )
        results.append(stats)
    spatial_stats = pd.concat(results)    
    spatial_stats.to_pickle(file)

# minimum number of nonzero interactions required 
# in each sample for a cell type pair
min_nonzero = 5  
pairs = spatial_stats[['cell_type_a', 'cell_type_b']].drop_duplicates()
sample_stats = spatial_stats\
    .groupby(['sample_id', 'cell_type_a', 'cell_type_b'])\
    .agg(n_nonzero=('b_count', lambda x: (x>0).sum()))
filtered_pairs = sample_stats\
    .groupby(['cell_type_a', 'cell_type_b'])\
    .agg(min_nonzero_count=('n_nonzero', 'min'))\
    .query('min_nonzero_count >= @min_nonzero')\
    .reset_index()[['cell_type_a', 'cell_type_b']]

pairs_tested = set(tuple(x) for x in filtered_pairs.values)
print(f'testing {len(filtered_pairs)} pairs out of {len(pairs)} pairs')
del pairs, sample_stats; gc.collect()

# testing 2055 pairs out of 6241 pairs

# get differential testing results 
file = f'{working_dir}/output/merfish/spatial_diff_{cell_type_col}.csv'
if os.path.exists(file):
    spatial_diff = pd.read_csv(file)
else:
    res = []
    with tqdm(total=len(filtered_pairs), desc='Processing pairs') as pbar:
        for _, row in filtered_pairs.iterrows():
            pair_result = get_spatial_diff(
                spatial_stats=spatial_stats,
                cell_type_a=row['cell_type_a'],
                cell_type_b=row['cell_type_b'])
            if pair_result: 
                res.extend(pair_result)
            pbar.update(1)
    spatial_diff = pd.concat(res, ignore_index=True)
    spatial_diff['adj.P.Val'] = fdrcorrection(spatial_diff['P.Value'])[1]
    spatial_diff.to_csv(file, index=False)

# heatmaps for each contrast
for contrast in spatial_diff['contrast'].unique():
    contrast_data = spatial_diff[spatial_diff['contrast'] == contrast].copy()
    if not contrast_data.empty:
        fig, ax = plot_spatial_diff_heatmap(
            contrast_data, 
            pairs_tested, 
            contrast, 
            sig=0.10,
            figsize=(16, 22) if cell_type_col == 'subclass' else (10, 12))
        plt.savefig(
            f'{working_dir}/figures/merfish/'
            f'heatmap_{cell_type_col}_{contrast}.png',
            dpi=300, bbox_inches='tight')
        plt.close()

#endregion

















# region scratch

# determining get_spatial_diff() thresholds based on rare cell types 
ct_counts = spatial_stats.groupby('cell_type_a')['cell_id'].nunique()
print("\nRarest cell types:")
print(ct_counts.sort_values().head(10))

pair_stats = spatial_stats.groupby(['cell_type_a','cell_type_b']).agg(
   total_b_count=('b_count','sum'),
   num_nonzero=('b_count',lambda x:(x>0).sum())
).reset_index()

pair_stats['a_abundance'] = pair_stats['cell_type_a'].map(ct_counts)
pair_stats['b_abundance'] = pair_stats['cell_type_b'].map(ct_counts)

print("\nInteraction count percentiles:")
print(pair_stats['total_b_count'].describe(percentiles=[0.1,0.25,0.5,0.75,0.9]))

rare_pairs = pair_stats[(pair_stats['a_abundance'] < ct_counts.median()/5) | 
                      (pair_stats['b_abundance'] < ct_counts.median()/5)]

for t in [100,250,500,1000]:
   for nz in [10,25,50,100]:
       n_pairs = len(pair_stats[
           (pair_stats['total_b_count'] >= t) &
           (pair_stats['num_nonzero'] >= nz)
       ])
       n_rare = len(rare_pairs[
           (rare_pairs['total_b_count'] >= t) &
           (rare_pairs['num_nonzero'] >= nz)
       ])
       print(f'total_b_count >= {t:4d}, nonzero >= {nz:3d}: '
             f'{n_pairs:4d} total pairs, {n_rare:4d} rare pairs')

'''
Rare cell types have ~100-350 cells
With total_b_count ≥250 and ≥25 nonzero cells captures 1,407 pairs (43 rare)

cell_type_a
MEA Slc17a7 Glut                104
L6b/CT ENT Glut                 109
MPN-MPO-LPO Lhx6 Zfhx3 Gaba     144
BST-SI-AAA Six3 Slc22a3 Gaba    172
CA2-FC-IG Glut                  197
SI-MA-ACB Ebf1 Bnc2 Gaba        205
IT AON-TT-DP Glut               265
GPe-SI Sox6 Cyp26b1 Gaba        272
BST Tac2 Gaba                   349
COAa-PAA-MEA Barhl2 Glut        352

count    6.241000e+03
mean     2.760008e+03
std      4.287628e+04
min      0.000000e+00
10%      0.000000e+00
25%      0.000000e+00
50%      1.200000e+01
75%      1.860000e+02
90%      1.625000e+03
max      2.435302e+06

total_b_count >=  100, nonzero >=  10: 1924 total pairs,   80 rare pairs
total_b_count >=  100, nonzero >=  25: 1922 total pairs,   80 rare pairs
total_b_count >=  100, nonzero >=  50: 1895 total pairs,   77 rare pairs
total_b_count >=  100, nonzero >= 100: 1709 total pairs,   59 rare pairs
total_b_count >=  250, nonzero >=  10: 1407 total pairs,   43 rare pairs
total_b_count >=  250, nonzero >=  25: 1407 total pairs,   43 rare pairs
total_b_count >=  250, nonzero >=  50: 1405 total pairs,   43 rare pairs
total_b_count >=  250, nonzero >= 100: 1383 total pairs,   37 rare pairs
total_b_count >=  500, nonzero >=  10: 1074 total pairs,   27 rare pairs
total_b_count >=  500, nonzero >=  25: 1074 total pairs,   27 rare pairs
total_b_count >=  500, nonzero >=  50: 1074 total pairs,   27 rare pairs
total_b_count >=  500, nonzero >= 100: 1070 total pairs,   26 rare pairs
total_b_count >= 1000, nonzero >=  10:  822 total pairs,   12 rare pairs
total_b_count >= 1000, nonzero >=  25:  822 total pairs,   12 rare pairs
total_b_count >= 1000, nonzero >=  50:  822 total pairs,   12 rare pairs
total_b_count >= 1000, nonzero >= 100:  822 total pairs,   12 rare pairs
'''

#endregion
