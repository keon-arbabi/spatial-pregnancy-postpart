import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import coo_array
from scipy.spatial import KDTree
from typing import Tuple
from ryp import r, to_r, to_py
import multiprocessing as mp
from functools import partial

################################################################################
#region functions

def get_spatial_stats(
    spatial_data: pd.DataFrame,
    coords_cols: Tuple[str, str],
    cell_type_col: str,
    condition_col: str,
    sample_col: str,
    d_min_scale: float = 0,
    d_max_scale: float = 5) -> pd.DataFrame:

    # Setup KDTree
    tree = KDTree(spatial_data[list(coords_cols)].values)
    d_scale = np.median(tree.query(
        spatial_data[list(coords_cols)].values, k=2)[0][:, 1])
    
    d_min = d_min_scale * d_scale
    d_max = d_max_scale * d_scale
    
    # Get cell pairs
    pairs = tree.query_pairs(d_max)
    if d_min > 0:
        pairs -= tree.query_pairs(d_min)
    mat = np.array(list(pairs))
    mat = np.concatenate((mat, mat[:, ::-1]))
    sparse_mat = coo_array((np.ones(len(mat), dtype=bool), mat.T),
        shape=(len(spatial_data), len(spatial_data))).tocsr()
    
    # Calculate stats for each cell type
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

def process_cell_pair(row, spatial_stats):
   a = row['cell_type_a']
   b = row['cell_type_b']
   print(f'Processing pair: {a} vs {b}')
   
   sub = spatial_stats[
       (spatial_stats['cell_type_a'] == a) &
       (spatial_stats['cell_type_b'] == b)
   ].copy().set_index('cell_id')
   
   if sub.shape[0] < 1:
       print(f'Skipping {a} vs {b}: No valid data')
       return None
       
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
   })
   cobj = crumblr(counts, method="clr_2class")
   form = ~ 0 + condition + (1 | sample_id)
   L = makeContrastsDream(form, meta,
       contrasts = c(
           PREG_vs_CTRL = "conditionPREG - conditionCTRL",
           POST_vs_PREG = "conditionPOSTPART - conditionPREG"
       )
   )
   fit = dream(cobj, form, meta, L)
   fit = eBayes(fit)
   tt = list(
       PREG_vs_CTRL = topTable(fit, coef="PREG_vs_CTRL", number=Inf),
       POST_vs_PREG = topTable(fit, coef="POST_vs_PREG", number=Inf)
   )
   ''')
   
   tt = to_py('tt', format='pandas')
   pair_results = []
   for contrast, df in tt.items():
       df = df.loc[df.index == "b_count"].copy()
       df["contrast"] = contrast
       df["cell_type_a"] = a
       df["cell_type_b"] = b
       pair_results.append(df)
   
   return pair_results

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

cell_type_col = 'class'
adata = adata[adata.obs[f'{cell_type_col}_keep']].copy()
adata.obs[cell_type_col] = adata.obs[cell_type_col].astype(str)\
    .str.extract(r'^(\d+)\s+(.*)', expand=False)[1]

# get spatial stats per sample
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
        d_max_scale=5
    )
    results.append(stats)

spatial_stats = pd.concat(results)

unique_pairs = spatial_stats[['cell_type_a', 'cell_type_b']].drop_duplicates()
ctx = mp.get_context('spawn')
num_cores = mp.cpu_count() - 1

pool = ctx.Pool(processes=num_cores)
process_func = partial(process_cell_pair, spatial_stats=spatial_stats)
all_results = pool.map(process_func, [row for _, row in unique_pairs.iterrows()])
pool.close()
pool.join()

results = [result for sublist in all_results if sublist is not None for result in sublist]
spatial_diff = pd.concat(results, ignore_index=True)