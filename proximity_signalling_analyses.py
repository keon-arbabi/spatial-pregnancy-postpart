import os, gc, pickle, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.sparse import coo_array
from scipy.spatial import KDTree
from scipy.stats import ttest_ind
from typing import Tuple, List
from ryp import r, to_r, to_py
from tqdm.auto import tqdm
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection
from matplotlib.colors import LinearSegmentedColormap
from single_cell import SingleCell
from scipy.spatial.distance import pdist
from gprofiler import GProfiler
from typing import List, Optional
from scipy.cluster.hierarchy import linkage, leaves_list

warnings.filterwarnings('ignore')

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

condition_colors = {
    'CTRL': '#7209b7',
    'PREG': '#b5179e',
    'POSTPART': '#f72585'
}
modality_colors = {
    'merfish': '#4361ee',
    'curio': '#4cc9f0'
}

#region functions #############################################################

def get_global_diff(
    adata: sc.AnnData,
    dataset_name: str,
    cell_type_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df = pd.DataFrame({
        'sample': adata.obs['sample'], 
        'condition': adata.obs['condition'],
        'cell_type': adata.obs[cell_type_col]
    })
    counts = pd.crosstab(index=df['sample'], columns=df['cell_type'])
    meta = df[['sample','condition']].drop_duplicates()
    
    to_r(counts, 'counts', format='data.frame')
    to_r(meta, 'meta', format='data.frame')
    to_r(dataset_name, 'dataset_name')
    
    r('''
    suppressPackageStartupMessages({
        library(crumblr)
        library(variancePartition)
    })
    meta = meta[meta$sample %in% rownames(counts),]
    rownames(meta) = meta$sample
    
    cobj = crumblr(counts)
    form = ~ 0 + condition
    L = makeContrastsDream(form, meta,
        contrasts = c(
            PREG_vs_CTRL = 'conditionPREG - conditionCTRL',
            POSTPART_vs_PREG = 'conditionPOSTPART - conditionPREG',
            POST_vs_CTRL = 'conditionPOSTPART - conditionCTRL'
        )
    )
    fit = dream(cobj, form, meta, L, useWeights=TRUE)
    fit = eBayes(fit)
    
    norm_props = cobj$E
    norm_props_df = as.data.frame(norm_props)
    norm_props_df$cell_type = rownames(norm_props_df)
    
    norm_props_by_condition = list()
    for (cond in unique(meta$condition)) {
        samples = rownames(meta)[meta$condition == cond]
        if (length(samples) > 1) {
            means = rowMeans(norm_props[, samples, drop=FALSE])
            ses = apply(norm_props[, samples, drop=FALSE], 1, sd) / 
                    sqrt(length(samples))
            norm_props_by_condition[[cond]] = data.frame(
                cell_type = rownames(norm_props),
                condition = cond,
                mean = means,
                se = ses
            )
        } else if (length(samples) == 1) {
            means = norm_props[, samples, drop=FALSE]
            ses = rep(0, length(means))
            norm_props_by_condition[[cond]] = data.frame(
                cell_type = rownames(norm_props),
                condition = cond,
                mean = means[,1],
                se = ses
            )
        }
    }
    norm_props_summary = do.call(rbind, norm_props_by_condition)
    
    results = list()
    for(coef in c('PREG_vs_CTRL', 'POSTPART_vs_PREG')) {
        tt = topTable(fit, coef=coef, number=Inf)
        tt$SE = fit$stdev.unscaled[,coef] * fit$sigma
        tt$contrast = coef
        tt$dataset = dataset_name
        results[[coef]] = tt
    }
    tt_all = do.call(rbind, results)
    ''')
    
    result = to_py('tt_all', format='pandas').reset_index()
    result['cell_type'] = result['index'].str.split('.', expand=True)[1]
    result.drop('index', axis=1, inplace=True)
    
    norm_props = to_py('norm_props_df', format='pandas')
    norm_props_long = pd.melt(
        norm_props, 
        id_vars=['cell_type'],
        var_name='sample',
        value_name='normalized_proportion')
    norm_props_long = norm_props_long.merge(
        meta[['sample', 'condition']], 
        on='sample', 
        how='left'
    )
    norm_props_summary = to_py('norm_props_summary', format='pandas')
    norm_props_long = norm_props_long.merge(
        norm_props_summary[['cell_type', 'condition', 'mean', 'se']],
        on=['cell_type', 'condition'],
        how='left'
    )
    norm_props_long['dataset'] = dataset_name

    t_test_results = []
    contrasts_map = {
        'PREG_vs_CTRL': ('PREG', 'CTRL'),
        'POSTPART_vs_PREG': ('POSTPART', 'PREG')
    }

    for cell_type_val in norm_props_long['cell_type'].unique():
        ct_data = norm_props_long[
            norm_props_long['cell_type'] == cell_type_val
        ]
        for contrast_name, (cond2_str, cond1_str) in contrasts_map.items():
            group1_values = ct_data[
                ct_data['condition'] == cond1_str
            ]['normalized_proportion'].dropna()
            group2_values = ct_data[
                ct_data['condition'] == cond2_str
            ]['normalized_proportion'].dropna()

            t_stat, p_val = ttest_ind(
                group1_values, 
                group2_values, 
                equal_var=True,  # Student's t-test
                nan_policy='omit'
            )
            t_test_results.append({
                'cell_type': cell_type_val,
                'contrast': contrast_name,
                't_test_P.Value': p_val
            })
    
    t_test_df = pd.DataFrame(t_test_results)
    result = result.merge(
        t_test_df, 
        on=['cell_type', 'contrast'], 
        how='left'
    )
    return result, norm_props_long

def _insert_gap(arr, indices, axis=0):
    if isinstance(arr, pd.DataFrame):
        for i in sorted(indices, reverse=True):
            arr = pd.concat([
                arr.iloc[:i],
                pd.DataFrame([[np.nan]*arr.shape[1]], columns=arr.columns),
                arr.iloc[i:]
            ]).reset_index(drop=True)
    else:
        val = np.nan
        if arr.dtype == object or arr.dtype.kind in ['U', 'S']:
            val = ''
        for i in sorted(indices, reverse=True):
            arr = np.insert(arr, i, val, axis=axis)
    return arr

def _segments(mask):
    seg, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            seg.append((s, i))
            s = None
    if s is not None:
        seg.append((s, len(mask)))
    return seg

def plot_cell_type_proportions(
    norm_props_df: pd.DataFrame,
    tt_combined: pd.DataFrame = None,
    cell_types: List[str] = None,
    nrows: int = 2,
    base_figsize: Tuple[float, float] = (2.0, 2.0),
    palette: dict = {'curio': '#4cc9f0', 'merfish': '#4361ee'},
    condition_order: List[str] = ['CTRL', 'PREG', 'POSTPART'],
    legend_position: Tuple[str, Tuple[float, float]] = (
        'center right', (1.10, 0.5)),
    datasets_to_plot: List[str] = ['merfish']) -> plt.Figure:

    if cell_types:
        cell_types = sorted(cell_types)
    else:
        cell_types = sorted(norm_props_df['cell_type'].unique())

    ncols = int(np.ceil(len(cell_types) / nrows))
    figsize = (base_figsize[0] * ncols, base_figsize[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    condition_labels = {
        'CTRL': 'Control', 'PREG': 'Pregnancy', 'POSTPART': 'Postpartum'
    }
    legend_handles, legend_labels = [], []

    for i, cell_type_val in enumerate(cell_types):
        ax = axes[i]
        ct_data = norm_props_df[norm_props_df['cell_type'] == cell_type_val]
        plotted_conditions_on_ax = []

        for dataset_name, line_color in palette.items():
            if datasets_to_plot and dataset_name not in datasets_to_plot:
                continue

            ds_data = ct_data[ct_data['dataset'] == dataset_name]
            cond_data_init = ds_data.drop_duplicates(
                subset=['condition', 'mean', 'se']
            )
            ordered_conds = [
                c for c in condition_order
                if c in cond_data_init['condition'].values
            ]
            cond_data = (cond_data_init.set_index('condition')
                         .loc[ordered_conds].reset_index())

            if not plotted_conditions_on_ax:
                plotted_conditions_on_ax = cond_data['condition'].tolist()

            if len(cond_data) > 1:
                mean = cond_data['mean'].mean()
                std = cond_data['mean'].std()
                cond_data['mean'] = (cond_data['mean'] - mean) / std
                cond_data['se'] = cond_data['se'] / std

            line = ax.errorbar(
                x=cond_data['condition'], y=cond_data['mean'],
                yerr=cond_data['se'], fmt='o-', color=line_color, alpha=0.7,
                label=dataset_name, capsize=3, markersize=5,
                linewidth=1.5, elinewidth=1, ecolor=line_color,
                capthick=1
            )
            if dataset_name not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(dataset_name)

            if tt_combined is not None:
                asts_to_plot = []
                for j in range(len(condition_order) - 1):
                    c1, c2 = condition_order[j], condition_order[j+1]
                    sig_row = tt_combined[
                        (tt_combined['cell_type'] == cell_type_val) &
                        (tt_combined['dataset'] == dataset_name) &
                        (tt_combined['contrast'] == f'{c2}_vs_{c1}')
                    ]
                    if sig_row.empty: continue
                    p_val = sig_row['t_test_P.Value'].iloc[0]
                    sig_str = '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                    if sig_str:
                        conds = cond_data['condition'].tolist()
                        x_pos = (conds.index(c1) + conds.index(c2)) / 2.0
                        asts_to_plot.append({'x': x_pos, 's': sig_str})

                if asts_to_plot:
                    se = cond_data['se'].fillna(0)
                    y_coord = (cond_data['mean'] + se).max() - 0.5
                    for info in asts_to_plot:
                        ax.text(
                            info['x'], y_coord, info['s'], ha='center',
                            va='bottom', fontsize=12, color='black',
                            fontweight='bold', clip_on=True, zorder=20
                        )

        ax.set_title(cell_type_val, fontsize=10)
        is_bottom = (i // ncols == nrows - 1) or \
                    (i >= len(cell_types) - ncols)
        xtick_lbls = plotted_conditions_on_ax or condition_order

        if is_bottom:
            ax.set_xticks(range(len(xtick_lbls)))
            ax.set_xticklabels(
                [condition_labels.get(c, c) for c in xtick_lbls],
                rotation=45, ha='right', fontsize=9,
                rotation_mode='anchor'
            )
        else:
            ax.set_xticks([])

    for i in range(len(cell_types), len(axes)):
        axes[i].set_visible(False)

    fig.text(
        0.035, 0.5, 'Normalized Proportion\n(Z-score)', va='center',
        ha='center', rotation='vertical', fontsize=9
    )
    if legend_handles:
        fig.legend(
            handles=legend_handles, labels=legend_labels,
            loc=legend_position[0], bbox_to_anchor=legend_position[1],
            fontsize=9
        )
    plt.tight_layout(rect=[0.05, 0, 0.95, 1])
    fig.subplots_adjust(wspace=0.3, hspace=0.35)

    return fig

def calculate_distance_scale(coords: np.ndarray) -> float:
    tree = KDTree(coords)
    d_scale = np.median(tree.query(coords, k=2)[0][:, 1])
    return d_scale, tree

def plot_spatial_diff_radii(
    spatial_data: pd.DataFrame, 
    coords_cols: Tuple[str, str],
    sample_col: str,
    d_max_scale: float,
    s: float = 0.05,
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
            coords[:, 0], coords[:, 1], s=s, alpha=1, c='gray',
            linewidth=0)
        random_idx = np.random.randint(len(coords))
        random_point = coords[random_idx]
        circle = plt.Circle(
            random_point, d_max, fill=False, color='red', 
            linewidth=0.3)
        ax.add_patch(circle)
        ax.scatter(*random_point, c='red', s=s)
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
        PREG_vs_CTRL='conditionPREG-conditionCTRL',
        POSTPART_vs_PREG='conditionPOSTPART-conditionPREG'
        )
    )
    fit = dream(cobj, form, meta, L, param=param)
    fit = eBayes(fit)
    tt = list(
    PREG_vs_CTRL=topTable(fit,coef='PREG_vs_CTRL',number=Inf),
    POSTPART_vs_PREG=topTable(fit,coef='POSTPART_vs_PREG',number=Inf)
    )
    ''')
    
    tt = to_py('tt', format='pandas')
    pair_results = []
    for contrast, df in tt.items():
        df = df.loc[df.index == 'b_count'].copy()
        df['contrast'] = contrast
        df['cell_type_a'] = cell_type_a
        df['cell_type_b'] = cell_type_b
        pair_results.append(df)
    return pair_results

def plot_spatial_diff_heatmap(
    df, 
    tested_pairs,
    sig=0.10, 
    figsize=(15, 15),
    cell_types_a=None, 
    cell_types_b=None,
    recompute_fdr=True, 
    ax=None, 
    vmin=None, vmax=None,
    row_order=None, 
    adata=None) -> Tuple[
        plt.Figure, plt.Axes, plt.cm.ScalarMappable, List[str]]:
    
    filtered_df = df.copy()
    if cell_types_a is not None:
        filtered_df = filtered_df[
            filtered_df['cell_type_a'].isin(cell_types_a)
        ]
    if cell_types_b is not None:
        filtered_df = filtered_df[
            filtered_df['cell_type_b'].isin(cell_types_b)
        ]
        
    if recompute_fdr and len(filtered_df) > 0:
        for cell_type_b in filtered_df['cell_type_b'].unique():
            b_mask = filtered_df['cell_type_b'] == cell_type_b
            if b_mask.sum() > 0:
                filtered_df.loc[b_mask, 'adj.P.Val'] = fdrcorrection(
                    filtered_df.loc[b_mask, 'P.Value'])[1]
    
    a_types = sorted(filtered_df['cell_type_a'].unique())
    b_types = sorted(filtered_df['cell_type_b'].unique())
    
    mat = pd.DataFrame(
        index=a_types, columns=b_types, 
        data=np.nan
    )
    sigs = pd.DataFrame(
        index=a_types, columns=b_types,
        data=''
    )
    
    for _, row in filtered_df.iterrows():
        i, j = row['cell_type_a'], row['cell_type_b']
        mat.loc[i, j] = row['logFC']
        if row['adj.P.Val'] < sig: 
            sigs.loc[i, j] = '*'

    if row_order is None and adata is not None:
        type_info = adata.obs[['subclass', 'type']]\
            .drop_duplicates().set_index('subclass')
        mat = mat.join(type_info)
        
        type_order = ['Glut', 'Gaba', 'NN']
        mat['type'] = pd.Categorical(
            mat['type'], categories=type_order, ordered=True
        )
        mat = mat.sort_values('type')
        
        ordered_a_types = []
        for type_name, group in mat.groupby('type'):
            cluster_data = group.drop('type', axis=1).fillna(0)
            if len(group) > 1:
                link = linkage(cluster_data.values, method='average')
                order = leaves_list(link)
                ordered_a_types.extend(cluster_data.index[order])
            else:
                ordered_a_types.extend(cluster_data.index)
        mat = mat.reindex(ordered_a_types)
    elif row_order is not None:
        mat = mat.reindex(row_order)
        mat = mat.join(
            adata.obs[['subclass', 'type']].drop_duplicates()
            .set_index('subclass'))
    else:
        ordered_a_types = a_types

    row_order = mat.index.tolist()
    a_types = row_order
    
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    cmap = 'PRGn'
    
    if vmin is None or vmax is None:
        mabs = np.nanmax(np.abs(mat.drop('type', axis=1).values))
        mabs = 1.0 if (np.isnan(mabs) or mabs == 0) else mabs
        if vmin is None: vmin = -mabs
        if vmax is None: vmax = mabs
    
    plot_mat = mat.drop('type', axis=1)
    type_boundaries = mat['type'].ne(mat['type'].shift()).cumsum()
    gaps = np.where(type_boundaries.diff() > 0)[0]
    plot_mat = _insert_gap(plot_mat, gaps, axis=0)

    plot_sigs = sigs.reindex(mat.index)
    plot_sigs = _insert_gap(plot_sigs, gaps, axis=0)
    
    a_types_gapped = _insert_gap(
        pd.Series(a_types, dtype=object), gaps, axis=0
    )

    im = ax.pcolormesh(plot_mat.values, cmap=cmap, vmin=vmin, vmax=vmax,
                      rasterized=False)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    row_seg = _segments(~plot_mat.isna().all(1))
    col_seg = [(0, len(b_types))]

    for r0, r1 in row_seg:
        for c0, c1 in col_seg:
            ax.add_patch(
                patches.Rectangle(
                    (c0, r0), c1 - c0, r1 - r0,
                    fill=False, ec='black', lw=0.5
                )
            )

    mask = pd.isna(mat)
    for i, a in enumerate(a_types_gapped):
        if a == '': continue
        for j, b in enumerate(b_types):
            is_tested = (a, b) in tested_pairs
            
            if not is_tested:
                ax.text(j + 0.5, i + 0.5, 'X', ha='center', va='center',
                        color='gray', size=10)
            elif plot_sigs.iloc[i, j] == '*':
                ax.text(j + 0.5, i + 0.5, '*', ha='center', va='center',
                        color='black', size=14, weight='bold')

    ax.set_xlim(0, len(b_types))
    ax.set_ylim(len(a_types_gapped), 0)
    ax.set_xticks(np.arange(len(b_types)) + 0.5)
    ax.set_yticks(np.arange(len(a_types_gapped)) + 0.5)
    ax.set_xticklabels(b_types, rotation=45, ha='right')
    ax.set_yticklabels(a_types_gapped)
    ax.tick_params(length=0)

    ax.set_xlabel('Surround Cell Type')
    ax.set_ylabel('Center Cell Type')
    
    if fig is not None:
        cbar = fig.colorbar(im, ax=ax, shrink=0.2) 
        cbar.set_label('logFC')
        plt.tight_layout()    
    return fig, ax, im, row_order

def get_spatial_map_intermediate_data(
    adata,
    spatial_stats,
    cell_type_pair,
    contrast,
    cache_dir,
    cell_type_col,
    coords_cols):

    cell_type_a, cell_type_b = cell_type_pair
    safe_cta = cell_type_a.replace(' ', '_').replace('/', '_')
    safe_ctb = cell_type_b.replace(' ', '_').replace('/', '_')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir, f'intermediate_{safe_cta}_{safe_ctb}_{contrast}.pkl'
    )
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    base_sample = 'PREG1'
    base_cells = adata[adata.obs['sample'] == base_sample]
    
    cond_B_name, cond_A_name = contrast.split('_vs_')
    
    cond_A_samples = adata.obs[
        adata.obs['condition'] == cond_A_name]['sample'].unique()
    cond_B_samples = adata.obs[
        adata.obs['condition'] == cond_B_name]['sample'].unique()
    
    pair_stats = spatial_stats[
        (spatial_stats['cell_type_a'] == cell_type_a) &
        (spatial_stats['cell_type_b'] == cell_type_b)
    ]

    a_cells_all = adata[adata.obs[cell_type_col] == cell_type_a]

    all_coords, all_b_counts, all_all_counts, all_conditions = \
        [], [], [], []

    for cond, samples in [
        (cond_A_name, cond_A_samples), (cond_B_name, cond_B_samples)
    ]:
        for sample in samples:
            a_cells = adata.obs[
                (adata.obs['sample'] == sample) &
                (adata.obs[cell_type_col] == cell_type_a)
            ]
            if len(a_cells) == 0:
                continue

            stats = pair_stats[
                (pair_stats['sample_id'] == sample) &
                (pair_stats['cell_id'].isin(a_cells.index))
            ]
            if len(stats) == 0:
                continue

            coords = a_cells[list(coords_cols)].values
            b_counts = np.zeros(len(coords))
            tot_counts = np.zeros(len(coords))

            for i, idx in enumerate(a_cells.index):
                stat = stats[stats['cell_id'] == idx]
                if len(stat) > 0:
                    b_counts[i] = stat['b_count'].values[0]
                    tot_counts[i] = stat['all_count'].values[0]

            valid = tot_counts > 0
            if not any(valid):
                continue

            all_coords.append(coords[valid])
            all_b_counts.append(b_counts[valid])
            all_all_counts.append(tot_counts[valid])
            all_conditions.extend([cond] * np.sum(valid))

    if not all_coords:
        return None

    other_counts = np.concatenate(all_all_counts) - \
        np.concatenate(all_b_counts)
    clr_values = np.log(np.concatenate(all_b_counts) + 0.5) - 0.5 * (
        np.log(np.concatenate(all_b_counts) + 0.5) + \
        np.log(other_counts + 0.5)
    )

    map_data = {
        'base_coords': base_cells.obs[list(coords_cols)].values,
        'a_coords_all': a_cells_all.obs[list(coords_cols)].values,
        'all_coords': np.vstack(all_coords),
        'clr_values': clr_values,
        'all_conditions': np.array(all_conditions),
        'cond_A_name': cond_A_name,
        'cond_B_name': cond_B_name
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(map_data, f)
        
    return map_data

def plot_spatial_diff_map(
    adata,
    spatial_stats,
    cell_type_pair,
    contrast,
    cache_dir,
    coords_cols=('x_ffd', 'y_ffd'),
    cell_type_col='subclass',
    influence_radius=8,
    resolution=400,
    vmax=None,
    ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    map_data = get_spatial_map_intermediate_data(
        adata, spatial_stats, cell_type_pair,
        contrast, cache_dir, cell_type_col, coords_cols
    )

    if map_data is None:
        return fig, ax, None

    base_coords = map_data['base_coords']
    a_coords_all = map_data['a_coords_all']
    all_coords = map_data['all_coords']
    clr_values = map_data['clr_values']
    all_conditions = map_data['all_conditions']
    cond_A_name = map_data['cond_A_name']
    cond_B_name = map_data['cond_B_name']

    base_coords_tree = KDTree(base_coords)
    all_coords_tree = KDTree(all_coords)

    x_min, x_max = base_coords[:, 0].min(), base_coords[:, 0].max()
    y_min, y_max = base_coords[:, 1].min(), base_coords[:, 1].max()

    padding = 0.05
    x_pad = (x_max - x_min) * padding
    y_pad = (y_max - y_min) * padding
    plot_x_min, plot_x_max = x_min - x_pad, x_max + x_pad
    plot_y_min, plot_y_max = y_min - y_pad, y_max + y_pad

    pixel_res = resolution
    x_grid = np.linspace(x_min, x_max, pixel_res)
    y_grid = np.linspace(y_min, y_max, pixel_res)
    X, Y = np.meshgrid(x_grid, y_grid)

    tissue_mask = np.zeros((pixel_res, pixel_res))
    for x, y in base_coords:
        i = int((y - y_min) / (y_max - y_min) * (pixel_res - 1))
        j = int((x - x_min) / (x_max - x_min) * (pixel_res - 1))
        if 0 <= i < pixel_res and 0 <= j < pixel_res:
            tissue_mask[i, j] = 1

    a_mask = np.zeros_like(tissue_mask)
    for x, y in a_coords_all:
        i = int((y - y_min) / (y_max - y_min) * (pixel_res - 1))
        j = int((x - x_min) / (x_max - x_min) * (pixel_res - 1))
        if 0 <= i < pixel_res and 0 <= j < pixel_res:
            a_mask[i, j] = 1

    d_scale, _ = calculate_distance_scale(all_coords)
    d_max = influence_radius * d_scale
    grid_step = (x_max - x_min) / resolution

    Z_diff = np.zeros_like(X)
    Z_weight = np.zeros_like(X)
    base_mask = np.zeros_like(X, dtype=bool)

    grid_points = np.vstack([X.ravel(), Y.ravel()]).T

    for idx, point in enumerate(grid_points):
        row, col = np.unravel_index(idx, X.shape)
        
        dist, _ = base_coords_tree.query(point, k=1)
        if dist >= grid_step * 3:
            continue
        base_mask[row, col] = True
        
        search_radius = d_max * 1.2
        neighbors = all_coords_tree.query_ball_point(point, r=search_radius)
        
        if not neighbors:
            continue
            
        nb_coords = all_coords[neighbors]
        nb_conditions = all_conditions[neighbors]
        nb_clr_values = clr_values[neighbors]
        
        dists = np.sqrt(np.sum((nb_coords - point)**2, axis=1))
        weights = np.exp(-0.7 * dists / d_max)

        mask_A = nb_conditions == cond_A_name
        mask_B = nb_conditions == cond_B_name

        sum_wA = np.sum(weights[mask_A])
        sum_wB = np.sum(weights[mask_B])
        
        if sum_wA > 0 and sum_wB > 0:
            avg_clr_A = np.sum(
                weights[mask_A] * nb_clr_values[mask_A]) / sum_wA
            avg_clr_B = np.sum(
                weights[mask_B] * nb_clr_values[mask_B]) / sum_wB
            Z_diff[row, col] = avg_clr_B - avg_clr_A
            Z_weight[row, col] = min(sum_wA, sum_wB)

    Z_diff = np.where((Z_weight > 0) & base_mask, Z_diff, np.nan)

    from scipy.ndimage import gaussian_filter
    valid_mask = ~np.isnan(Z_diff)
    Z_smooth = np.copy(Z_diff)
    if np.any(valid_mask):
        Z_smooth[valid_mask] = gaussian_filter(
            Z_diff[valid_mask], sigma=1.5, mode='constant')
    Z_diff = Z_smooth

    if vmax is None and np.any(valid_mask):
        vmax = np.nanmax(np.abs(Z_diff))
    elif vmax is None:
        vmax = 1.0
    vmin = -vmax

    cmap = 'PRGn'
    
    ax.pcolormesh(X, Y, tissue_mask,
                 cmap='Greys', alpha=0.4, rasterized=True)
    
    im = ax.pcolormesh(X, Y, Z_diff, cmap=cmap, vmin=vmin, vmax=vmax,
                      alpha=0.9, shading='gouraud', rasterized=True)
    
    ax.pcolormesh(X, Y, a_mask, cmap='binary',
                 alpha=0.5, vmin=0, vmax=1, rasterized=True)
    
    ax.set_aspect('equal')
    ax.set_xlim(plot_x_min, plot_x_max)
    ax.set_ylim(plot_y_min, plot_y_max)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    if fig.get_axes().__len__() == 1:
        cbar = fig.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('logFC')

    return fig, ax, im

def plot_spatial_maps_grid(
    adata,
    spatial_stats,
    cell_type_pairs,
    cache_dir,
    resolution=350,
    influence_radius=3.5,
    vmax=0.4):

    n_rows = len(cell_type_pairs)
    n_cols = 2
    contrasts = ['PREG_vs_CTRL', 'POSTPART_vs_PREG']

    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.8, n_rows * 2.2)
    )
    if n_rows == 1:
        axs = np.array([axs])
    if n_cols == 1:
        axs = axs.reshape(-1, 1)

    images = []
    for i, pair in enumerate(tqdm(cell_type_pairs, desc='Plotting maps')):
        for j, contrast in enumerate(contrasts):
            ax = axs[i, j]
            _, _, im = plot_spatial_diff_map(
                adata, spatial_stats,
                cell_type_pair=pair,
                contrast=contrast,
                cache_dir=cache_dir,
                resolution=resolution,
                influence_radius=influence_radius,
                vmax=vmax,
                ax=ax
            )
            if im:
                images.append(im)
            if i == 0:
                title = 'Pregnancy vs Control' if contrast == 'PREG_vs_CTRL' \
                    else 'Postpartum vs Pregnancy'
                ax.set_title(title, fontsize=10)
            if j == 0:
                ylabel_str = f'{pair[0]} (Center)\n{pair[1]} (Surround)'
                ax.set_ylabel(
                    ylabel_str, fontsize=7, 
                    rotation='vertical', va='center', labelpad=15)

    if not images:
        plt.close(fig)
        return
    for im in images:
        im.set_clim(-vmax, vmax)

    fig.subplots_adjust(wspace=0, hspace=0)
    return fig, axs

def plot_cellchat_diff_radii(
    spatial_data: pd.DataFrame,
    coords_cols: tuple = ('x', 'y'),
    sample_col: str = 'sample',
    interaction_range: float = 600,
    s: float = 0.2):

    samples = sorted(spatial_data[sample_col].unique())
    n_cols = 3
    n_rows = int(np.ceil(len(samples) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(15, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for ax, sample in zip(axes, samples):
        sample_data = spatial_data[spatial_data[sample_col] == sample].copy()
        coords = sample_data[list(coords_cols)].values

        ax.scatter(
            coords[:, 0], coords[:, 1], s=s, alpha=0.8, c='gray', linewidth=0)

        random_idx = np.random.randint(len(coords))
        random_point = coords[random_idx]
        circle = plt.Circle(
            random_point, interaction_range, fill=False, color='red', 
            linewidth=1.5, zorder=10)
        ax.add_patch(circle)
        ax.scatter(*random_point, c='red', s=s*10, zorder=10)

        ax.set_title(f'Sample: {sample}')
        ax.set_xlabel('X coordinate (microns)')
        ax.set_ylabel('Y coordinate (microns)')
        ax.set_aspect('equal', adjustable='box')

    for i in range(len(samples), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        f'CellChat Interaction Radius ({interaction_range} microns)', 
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def update_cellchat_db(
    cellphonedb_path: str,
    cache_path: str,
    updated_db_name: str = 'CellChatDB_mouse_extended') -> str:

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            data_for_r = pickle.load(f)
        interaction_input = data_for_r['interaction_input']
        complex_input = data_for_r['complex_input']
        cpdb_gene_info_mouse = data_for_r['cpdb_gene_info_mouse']
    else:
        interaction_input = pd.read_csv(
            os.path.join(cellphonedb_path, 'interaction_input.csv'))
        complex_input = pd.read_csv(
            os.path.join(cellphonedb_path, 'complex_input.csv'))
        gene_input = pd.read_csv(
            os.path.join(cellphonedb_path, 'gene_input.csv'))

        gp = GProfiler(user_agent='cellchat_update_tool',
                       return_dataframe=True)
        human_genes = gene_input['hgnc_symbol'].dropna().unique().tolist()
        
        ortholog_results = gp.orth(
            organism='hsapiens',
            target='mmusculus',
            query=human_genes
        )

        mapping_df = ortholog_results.drop_duplicates(subset=['incoming'])        
        mapping = mapping_df.set_index('incoming')['name']
        gene_input['Symbol'] = gene_input['hgnc_symbol'].map(mapping)

        cpdb_gene_info_mouse = gene_input[
            ['uniprot', 'Symbol']].dropna().drop_duplicates()

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'interaction_input': interaction_input,
                'complex_input': complex_input,
                'cpdb_gene_info_mouse': cpdb_gene_info_mouse
            }, f)

    to_r(interaction_input, 'cpdb_interaction_input')
    to_r(complex_input, 'cpdb_complex_input')
    to_r(cpdb_gene_info_mouse.reset_index(drop=True), 'cpdb_gene_info_mouse')
    to_r(updated_db_name, 'updated_db_name')

    r_script = '''
    suppressPackageStartupMessages({
        library(CellChat)
        library(NeuronChat)
        library(dplyr)
        library(purrr)
    })

    base_interaction <- CellChatDB.mouse$interaction
    base_complex <- CellChatDB.mouse$complex
    base_geneInfo <- CellChatDB.mouse$geneInfo
    geneInfo_cpdb <- cpdb_gene_info_mouse

    cpdb_interactions_filtered <- cpdb_interaction_input %>%
        filter(partner_a %in% geneInfo_cpdb$uniprot & 
               partner_b %in% geneInfo_cpdb$uniprot)
    
    idx_partnerA <- match(
        cpdb_interactions_filtered$partner_a, geneInfo_cpdb$uniprot)
    cpdb_interactions_filtered$ligand <- geneInfo_cpdb$Symbol[idx_partnerA]
    idx_partnerB <- match(
        cpdb_interactions_filtered$partner_b, geneInfo_cpdb$uniprot)
    cpdb_interactions_filtered$receptor <- geneInfo_cpdb$Symbol[idx_partnerB]

    cpdb_formatted <- cpdb_interactions_filtered %>%
        rename(interaction_name = interactors, 
               pathway_name = classification) %>%
        mutate(
            annotation = case_when(
                directionality == 'secreted' ~ 'Secreted Signaling',
                directionality == 'transmembrane' ~ 'Cell-Cell Contact',
                TRUE ~ 'Cell-Cell Contact'
            ),
            interaction_name_2 = interaction_name
        )
    
    data(list='interactionDB_mouse')
    neuron_chat_db_list <- eval(parse(text = 'interactionDB_mouse'))
    neuron_chat_formatted <- purrr::map_dfr(
        neuron_chat_db_list,
        ~ expand.grid(
            interaction_name = .x$interaction_name,
            ligand = .x$lig_contributor,
            receptor = .x$receptor_subunit,
            stringsAsFactors = FALSE
        )
    ) %>%
    mutate(pathway_name = ligand, annotation = 'Secreted Signaling')

    required_cols <- colnames(base_interaction)
    for (col in required_cols) {
        if (!col %in% names(cpdb_formatted)) {
            cpdb_formatted[[col]] <- ''
        }
        if (!col %in% names(neuron_chat_formatted)) {
            neuron_chat_formatted[[col]] <- ''
        }
    }
    cpdb_final <- cpdb_formatted[, required_cols]
    neuron_chat_final <- neuron_chat_formatted[, required_cols]

    final_interactions <- bind_rows(
        base_interaction, cpdb_final, neuron_chat_final) %>%
        mutate(ligand_upper = toupper(ligand), 
               receptor_upper = toupper(receptor)) %>%
        distinct(ligand_upper, receptor_upper, .keep_all = TRUE) %>%
        select(-ligand_upper, -receptor_upper)

    final_geneInfo <- bind_rows(base_geneInfo, geneInfo_cpdb) %>%
        distinct(Symbol, .keep_all = TRUE)

    db_extended <- list(
        interaction = final_interactions,
        complex = base_complex,
        cofactor = CellChatDB.mouse$cofactor,
        geneInfo = final_geneInfo
    )
    
    assign(updated_db_name, db_extended, envir = .GlobalEnv)
    '''
    r(r_script)
    return updated_db_name

def prepare_cellchat_object(
    adata: sc.AnnData,
    cell_type_col: str,
    conditions: Tuple[str, str],
    output_path: str,
    db_name: str):
    
    if os.path.exists(output_path):
        return

    adata_cleaned = adata.copy()
    
    np.random.seed(0)
    sample_indices = np.random.choice(adata_cleaned.n_obs, 1000, 
                                      replace=False)
    sample_obs = adata_cleaned.obs.iloc[sample_indices]
    
    coords_orig = sample_obs[['x', 'y']].values
    coords_affine = sample_obs[['x_affine', 'y_affine']].values
    
    conversion_factor = np.median(pdist(coords_orig)) / \
                        np.median(pdist(coords_affine))

    spatial_cols = ['x_affine', 'y_affine']
    obs_cols = ['sample', 'condition', cell_type_col, 'x', 'y'] + spatial_cols
    adata_cleaned.obs = adata_cleaned.obs[obs_cols]
    adata_cleaned = adata_cleaned[:, adata_cleaned.var['protein_coding']]
    adata_cleaned = adata_cleaned[
        :, ~adata_cleaned.var['gene_symbol']
        .str.lower().str.startswith('mt-', na=False)].copy()
    adata_cleaned.var = adata_cleaned.var[['gene_symbol']]
    adata_cleaned.uns, adata_cleaned.varm, \
        adata_cleaned.obsp, adata_cleaned.obsm = {}, {}, {}, {}

    SingleCell(adata_cleaned).to_seurat('sobj', v3=True)
    spatial_locs = adata_cleaned.obs[spatial_cols]
    to_r(spatial_locs, 'spatial_locs', format='data.frame')
    to_r(conditions[0], 'cond_1')
    to_r(conditions[1], 'cond_2')
    to_r(cell_type_col, 'cell_type_col')
    to_r(conversion_factor, 'conversion_factor')
    to_r(output_path, 'output_path')
    to_r(db_name, 'db_name')

    r_script = f'''
        suppressPackageStartupMessages({{
            library(tidyverse)
            library(Seurat)
            library(CellChat)
        }})

        cellchat_db <- get(db_name)

        sobj$samples <- sobj$sample
        sobj <- NormalizeData(sobj)
        
        sobj_1 <- subset(sobj, subset = condition == cond_1)
        sobj_2 <- subset(sobj, subset = condition == cond_2)
        
        spot.size = 15

        spatial_locs_1 <- spatial_locs[colnames(sobj_1), ]
        spatial_factors_1 <- data.frame(
            ratio = conversion_factor,
            tol = spot.size / 2)
        cobj_1 <- createCellChat(
            object = sobj_1, group.by = cell_type_col, assay = 'RNA',
            datatype = 'spatial', coordinates = spatial_locs_1,
            spatial.factors = spatial_factors_1)
        
        spatial_locs_2 <- spatial_locs[colnames(sobj_2), ]
        spatial_factors_2 <- data.frame(
            ratio = conversion_factor,
            tol = spot.size / 2)
        cobj_2 <- createCellChat(
            object = sobj_2, group.by = cell_type_col, assay = 'RNA',
            datatype = 'spatial', coordinates = spatial_locs_2,
            spatial.factors = spatial_factors_2)

        cobj_1@DB <- cellchat_db
        cobj_2@DB <- cellchat_db
        
        cobj_1 <- subsetData(cobj_1)
        cobj_2 <- subsetData(cobj_2)
        
        cobj_1 <- identifyOverExpressedGenes(cobj_1)
        cobj_1 <- identifyOverExpressedInteractions(cobj_1)
        cobj_2 <- identifyOverExpressedGenes(cobj_2)
        cobj_2 <- identifyOverExpressedInteractions(cobj_2)
        
        cobj_1 <- computeCommunProb(
            cobj_1, type = 'truncatedMean', trim = 0.1, 
            distance.use = TRUE, interaction.range = 600, 
            scale.distance = 1.8, contact.range = 10)
        cobj_2 <- computeCommunProb(
            cobj_2, type = 'truncatedMean', trim = 0.1, 
            distance.use = TRUE, interaction.range = 600, 
            scale.distance = 1.8, contact.range = 10)

        cobjs <- list(cobj_1, cobj_2)
        saveRDS(cobjs, file = output_path)
    '''
    r(r_script)

def get_cellchat_cell_type_pair_diff(
    cobj_rds_path: str,
    conditions: Tuple[str, str]) -> pd.DataFrame:

    to_r(cobj_rds_path, 'cobj_rds_path')
    to_r(conditions[0], 'cond_1')
    to_r(conditions[1], 'cond_2')

    r_script = f'''
        suppressPackageStartupMessages({{
            library(tidyverse)
            library(Seurat)
            library(CellChat)
        }})
        
        cobjs <- readRDS(cobj_rds_path)
        cobj_1 <- cobjs[[1]]
        cobj_2 <- cobjs[[2]]

        cobj_1 <- filterCommunication(cobj_1, min.cells = 10)
        cobj_2 <- filterCommunication(cobj_2, min.cells = 10)
        
        cobj_1 <- aggregateNet(cobj_1)
        cobj_2 <- aggregateNet(cobj_2)

        cellchat <- mergeCellChat(
            list(cobj_1, cobj_2), add.names = c(cond_1, cond_2))

        g1_w <- cellchat@net[[cond_1]]$weight
        g2_w <- cellchat@net[[cond_2]]$weight
        g1_c <- cellchat@net[[cond_1]]$count
        g2_c <- cellchat@net[[cond_2]]$count

        all_sources <- unique(c(rownames(g1_w), rownames(g2_w),
            rownames(g1_c), rownames(g2_c)))
        all_targets <- unique(c(colnames(g1_w), colnames(g2_w),
            colnames(g1_c), colnames(g2_c)))

        g1_w_full <- matrix(0,
            nrow=length(all_sources), ncol=length(all_targets),
            dimnames=list(all_sources, all_targets))
        if (!is.null(g1_w)) g1_w_full[rownames(g1_w), colnames(g1_w)] <- g1_w
        
        g2_w_full <- matrix(0,
            nrow=length(all_sources),
            ncol=length(all_targets),
            dimnames=list(all_sources, all_targets))
        if (!is.null(g2_w)) g2_w_full[rownames(g2_w), colnames(g2_w)] <- g2_w

        g1_c_full <- matrix(0,
            nrow=length(all_sources),
            ncol=length(all_targets), dimnames=list(all_sources, all_targets))
        if (!is.null(g1_c)) g1_c_full[rownames(g1_c), colnames(g1_c)] <- g1_c

        g2_c_full <- matrix(0,
            nrow=length(all_sources),
            ncol=length(all_targets), dimnames=list(all_sources, all_targets))
        if (!is.null(g2_c)) g2_c_full[rownames(g2_c), colnames(g2_c)] <- g2_c
        
        net_df_w <- reshape2::melt(g2_w_full - g1_w_full, value.name = 'logFC')
        net_df_w$measure <- 'weight'
        
        net_df_c <- reshape2::melt(g2_c_full - g1_c_full, value.name = 'logFC')
        net_df_c$measure <- 'count'

        net_df <- rbind(net_df_w, net_df_c)
        colnames(net_df)[1:2] <- c('cell_type_a', 'cell_type_b')
        net_df$contrast <- paste0(cond_2, '_vs_', cond_1)
    '''
    r(r_script)
    diff_df = to_py('net_df', format='pandas')
    return diff_df

def get_cellchat_pathway_pair_diff(
    cobj_rds_path: str,
    conditions: tuple,
    contrast_name: str) -> pd.DataFrame:
    
    to_r(cobj_rds_path, 'cobj_rds_path')
    to_r(conditions[0], 'cond_1')
    to_r(conditions[1], 'cond_2')
    to_r(contrast_name, 'contrast_name')

    r_script = f'''
        suppressPackageStartupMessages({{
            library(tidyverse)
            library(Seurat)
            library(CellChat)
            library(reshape2)
        }})
        
        cobjs <- readRDS(cobj_rds_path)
        cobj_1 <- cobjs[[1]]
        cobj_2 <- cobjs[[2]]

        cobj_1 <- filterCommunication(cobj_1, min.cells = 10)
        cobj_2 <- filterCommunication(cobj_2, min.cells = 10)

        cobj_1 <- computeCommunProbPathway(cobj_1)
        cobj_2 <- computeCommunProbPathway(cobj_2)

        prob1 <- cobj_1@netP$prob
        prob2 <- cobj_2@netP$prob
        
        pathways1 <- if(!is.null(prob1)) dimnames(prob1)[[3]] else character(0)
        pathways2 <- if(!is.null(prob2)) dimnames(prob2)[[3]] else character(0)
        all_pathways <- unique(c(pathways1, pathways2))

        pathway_pair_diff_df <- data.frame(
            source=character(), target=character(), 
            strength_diff=numeric(), pathway=character(), 
            contrast=character())

        if (length(all_pathways) > 0) {{
            all_cell_types <- sort(unique(c(
                levels(cobj_1@idents), levels(cobj_2@idents)
            )))
            
            if (length(all_cell_types) > 0) {{
                all_diffs <- list()
                for (pathway in all_pathways) {{
                    mat1 <- matrix(
                        0, nrow = length(all_cell_types), 
                        ncol = length(all_cell_types),
                        dimnames = list(all_cell_types, all_cell_types))
                    mat2 <- matrix(
                        0, nrow = length(all_cell_types),
                        ncol = length(all_cell_types),
                        dimnames = list(all_cell_types, all_cell_types))

                    if (pathway %in% pathways1) {{
                        p1 <- prob1[,,pathway]
                        if (!is.null(p1)) mat1[rownames(p1), colnames(p1)] <- p1
                    }}
                    if (pathway %in% pathways2) {{
                        p2 <- prob2[,,pathway]
                        if (!is.null(p2)) mat2[rownames(p2), colnames(p2)] <- p2
                    }}

                    diff_mat <- mat2 - mat1
                    diff_df <- melt(diff_mat, value.name = 'strength_diff')
                    colnames(diff_df)[1:2] <- c('source', 'target')
                    diff_df$pathway <- pathway
                    all_diffs[[pathway]] <- diff_df
                }}

                if (length(all_diffs) > 0) {{
                    pathway_pair_diff_df <- do.call(rbind, all_diffs)
                    pathway_pair_diff_df$contrast <- contrast_name
                }}
            }}
        }}
    '''
    r(r_script)

    pathway_pair_diff_df = to_py('pathway_pair_diff_df', format='pandas')
    if pathway_pair_diff_df is None:
        return pd.DataFrame()
    return pathway_pair_diff_df

def plot_cellchat_cell_type_heatmap(
    df: pd.DataFrame,
    ax: plt.Axes,
    tested_pairs: Optional[set] = None,
    value_col: str = 'logFC',
    x_axis_cell_types: Optional[List[str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = '',
    x_axis_is_sender: bool = True,
    row_order: Optional[List[str]] = None,
    adata: Optional[sc.AnnData] = None
) -> Optional[plt.cm.ScalarMappable]:

    sender_col, receiver_col = ('cell_type_a', 'cell_type_b') if \
        x_axis_is_sender else ('cell_type_b', 'cell_type_a')
    
    df_plot = df.copy()
    if x_axis_cell_types:
        df_plot = df_plot[df_plot[sender_col].isin(x_axis_cell_types)]

    heatmap_data = df_plot.pivot_table(
        index=receiver_col, columns=sender_col, values=value_col,
        aggfunc='sum', fill_value=0
    )
    if heatmap_data.empty:
        return None

    y_types = [r for r in row_order if r in heatmap_data.index] if \
        row_order else sorted(heatmap_data.index)
    x_types = [ct for ct in x_axis_cell_types if ct in heatmap_data.columns] \
        if x_axis_cell_types else sorted(heatmap_data.columns)

    mat = heatmap_data.reindex(
        index=y_types, columns=x_types, fill_value=0
    ).values.astype(float)
    if not mat.size: return None

    if tested_pairs:
        for i, y_label in enumerate(y_types):
            for j, x_label in enumerate(x_types):
                sender = x_label if x_axis_is_sender else y_label
                receiver = y_label if x_axis_is_sender else x_label
                if (receiver, sender) not in tested_pairs:
                    mat[i, j] = np.nan

    gaps = []
    if row_order and adata:
        type_info = adata.obs[['subclass', 'type']].drop_duplicates()\
            .set_index('subclass').reindex(y_types)
        gaps = np.where(type_info['type'].ne(type_info['type'].shift()))[0]
    
    mat = _insert_gap(mat, gaps, axis=0)
    y_types_gapped = _insert_gap(pd.Series(y_types, dtype=object), gaps)

    cmap = plt.get_cmap('seismic')
    cmap.set_bad(color='white')
    
    mabs = np.nanmax(np.abs(mat)) if np.any(~np.isnan(mat)) else 1.0
    vmin = -mabs if vmin is None else vmin
    vmax = mabs if vmax is None else vmax

    im = ax.pcolormesh(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    for spine in ax.spines.values(): spine.set_visible(False)

    for r0, r1 in _segments(~np.isnan(mat).all(1)):
        ax.add_patch(patches.Rectangle(
            (0, r0), len(x_types), r1 - r0, fill=False, ec='black', lw=0.5
        ))

    if tested_pairs:
        for i, y_label in enumerate(y_types_gapped):
            if not y_label: continue
            for j, x_label in enumerate(x_types):
                sender = x_label if x_axis_is_sender else y_label
                receiver = y_label if x_axis_is_sender else x_label
                if (receiver, sender) not in tested_pairs:
                    ax.text(j + 0.5, i + 0.5, 'X', ha='center',
                            va='center', color='gray', size=8)

    ax.set_xticks(np.arange(len(x_types)) + 0.5)
    ax.set_yticks(np.arange(len(y_types_gapped)) + 0.5)
    ax.set_xticklabels(x_types, rotation=45, ha='right')
    ax.set_yticklabels(y_types_gapped)
    ax.tick_params(length=0)
    ax.set_xlabel('Sender' if x_axis_is_sender else 'Target')
    ax.set_ylabel('Target' if x_axis_is_sender else 'Sender')
    ax.set_title(title)
    ax.set_xlim(0, len(x_types))
    ax.set_ylim(len(y_types_gapped), 0)

    return im

def plot_cellchat_vs_proximity_scatter(
    cellchat_df: pd.DataFrame,
    spatial_df: pd.DataFrame,
    contrast: str,
    ax: plt.Axes,
    tested_pairs: set,
    color: str,
    value_col: str = 'logFC',
    cell_types_to_include: List[str] = None,
    align_sender_to_center: bool = False):
    contrast_map = {
        'PREG_vs_CTRL': 'Pregnancy vs Control',
        'POSTPART_vs_PREG': 'Postpartum vs Pregnancy'
    }

    cc_data = cellchat_df[cellchat_df['contrast'] == contrast].copy()
    if cell_types_to_include:
        cc_data = cc_data[
            cc_data['cell_type_a'].isin(cell_types_to_include) |
            cc_data['cell_type_b'].isin(cell_types_to_include)
        ]

    cc_agg = (
        cc_data.groupby(['cell_type_a', 'cell_type_b'])[value_col]
        .sum().reset_index()
        .rename(columns={value_col: 'cellchat_value'})
    )

    spatial_contrast_df = spatial_df[
        spatial_df['contrast'] == contrast
    ].copy()
    spatial_contrast_df = spatial_contrast_df[
        ['cell_type_a', 'cell_type_b', 'logFC']
    ].rename(columns={'logFC': 'proximity_logFC'})

    if align_sender_to_center:
        left_on = ['cell_type_a', 'cell_type_b']
        right_on = ['cell_type_a', 'cell_type_b']
    else:
        left_on = ['cell_type_a', 'cell_type_b']
        right_on = ['cell_type_b', 'cell_type_a']

    merged_df = pd.merge(
        cc_agg,
        spatial_contrast_df,
        left_on=left_on,
        right_on=right_on,
        how='inner'
    )

    if tested_pairs and not merged_df.empty:
        if 'cell_type_a_y' in merged_df.columns:
            pair_cols = ['cell_type_a_y', 'cell_type_b_y']
        else:
            pair_cols = ['cell_type_a', 'cell_type_b']

        merged_df = merged_df[
            merged_df.apply(
                lambda row: (row[pair_cols[0]], row[pair_cols[1]])
                in tested_pairs,
                axis=1
            )
        ]
        
    merged_df = merged_df[
        (merged_df['cellchat_value'] != 0) & (merged_df['proximity_logFC'] != 0)
    ]

    if merged_df.empty:
        ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        return

    x_vals = merged_df['cellchat_value']
    y_vals = merged_df['proximity_logFC']

    ax.scatter(x_vals, y_vals, alpha=0.6, s=15, c=color)

    m, b = np.polyfit(x_vals, y_vals, 1)
    x_fit = np.unique(x_vals)
    ax.plot(x_fit, m * x_fit + b, color=color, linestyle='--')

    r_val, p_val = pearsonr(x_vals, y_vals)

    display_contrast = contrast_map.get(contrast, contrast)
    legend_text = f'R={r_val:.2f}, p={p_val:.4f}'
    ax.text(
        0.95, 0.95, legend_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5)
    )

    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)

    ax.set_title(display_contrast, fontsize=11)
    ax.set_xlabel(f'Signaling Change ({value_col})', fontsize=9)
    ax.set_ylabel('Proximity Change (logFC)', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)

def plot_pathway_diff_dotplot(
    pathway_pair_diff_df: pd.DataFrame,
    cell_type_pairs: list,
    contrasts: list,
    top_n: int = 10,
    z_score: bool = True,
    pathways_to_exclude: Optional[List[str]] = None,
    pathways_to_include: Optional[List[str]] = None,
    cluster_rows: bool = True):

    pathway_pair_diff_df['canonical_pair'] = pathway_pair_diff_df.apply(
        lambda row: frozenset([row['source'], row['target']]), axis=1
    )
    pairs_df = pd.DataFrame(cell_type_pairs, columns=['c1', 'c2'])
    pairs_df['canonical_pair'] = pairs_df.apply(
        lambda row: frozenset([row['c1'], row['c2']]), axis=1
    )
    plot_data = pd.merge(
        pathway_pair_diff_df, pairs_df[['canonical_pair']], on='canonical_pair'
    )
    
    pair_str_map = {
        frozenset(p): f'{p[0]} ↔ {p[1]}' for p in cell_type_pairs
    }
    plot_data['pair_str'] = plot_data['canonical_pair'].map(pair_str_map)

    if plot_data.empty:
        print('No matching data found for the provided cell type pairs.')
        return None

    if z_score:
        grouped = plot_data.groupby(['contrast', 'pair_str'])['strength_diff']
        transform_func = lambda x: (x - x.mean()) / x.std()
        plot_data['plot_value'] = grouped.transform(transform_func).fillna(0)
        cbar_label_base = 'Signaling Change\n(Pair-wise Z-score'
    else:
        plot_data['plot_value'] = plot_data['strength_diff']
        cbar_label_base = 'Change in Signaling Strength'

    plot_data['abs_plot_value'] = plot_data['plot_value'].abs()

    top_pathways = set()
    unique_pairs = plot_data['canonical_pair'].unique()
    for pair_key in unique_pairs:
        for contrast in contrasts:
            subset = plot_data[
                (plot_data['canonical_pair'] == pair_key)
                & (plot_data['contrast'] == contrast)
            ]
            subset = subset.sort_values('abs_plot_value', ascending=False)
            top_pathways.update(subset.head(top_n)['pathway'])

    if pathways_to_exclude is not None:
        top_pathways.difference_update(set(pathways_to_exclude))

    if pathways_to_include is not None:
        top_pathways.update(set(pathways_to_include))

    plot_data = plot_data[plot_data['pathway'].isin(top_pathways)].copy()
    
    if plot_data.empty:
        print('No data to plot for the selected pathways.')
        return None

    idx_max = plot_data.groupby(
        ['contrast', 'canonical_pair', 'pathway']
    )['abs_plot_value'].idxmax()
    plot_data = plot_data.loc[idx_max]

    lower_thresh = plot_data['plot_value'].quantile(0.01)
    upper_thresh = plot_data['plot_value'].quantile(0.99)
    plot_data['plot_value_winsorized'] = plot_data['plot_value'].clip(
        lower_thresh, upper_thresh
    )
    cbar_label = f'{cbar_label_base}, Winsorized)'

    n_contrasts = len(contrasts)
    fig_height = len(top_pathways) * 0.4
    fig_width = 3.2 * n_contrasts 
    fig, axes = plt.subplots(
        1, n_contrasts, figsize=(fig_width, fig_height), sharey=True
    )
    if n_contrasts == 1:
        axes = [axes]

    cmap = plt.get_cmap('seismic')
    vmax = plot_data['plot_value_winsorized'].abs().max()
    if pd.isna(vmax) or vmax == 0:
        vmax = 1.5
    
    vmin = -vmax

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    ordered_pair_strs = [f'{p[0]} ↔ {p[1]}' for p in cell_type_pairs]
    plotted_pairs = plot_data['pair_str'].unique()
    x_cats = [p for p in ordered_pair_strs if p in plotted_pairs]
    
    if cluster_rows:
        pivot_df = plot_data.pivot_table(
            index='pathway',
            columns=['pair_str', 'contrast'],
            values='plot_value_winsorized'
        ).fillna(0)
        
        pivot_df = pivot_df.reindex(columns=pd.MultiIndex.from_product(
            [x_cats, contrasts], names=['pair_str', 'contrast']), fill_value=0
        )

        if len(pivot_df) > 1:
            row_linkage = linkage(
                pdist(pivot_df.values, metric='euclidean'), method='average'
            )
            ordered_indices = leaves_list(row_linkage)
            y_cats = pivot_df.index[ordered_indices].tolist()
        else:
            y_cats = pivot_df.index.tolist()
    else:
        y_cats = sorted(list(top_pathways), reverse=True)
    
    y_map = {cat: j for j, cat in enumerate(y_cats)}
    x_map = {cat: j for j, cat in enumerate(x_cats)}

    title_map = {
        'PREG_vs_CTRL': 'Pregnancy vs Control',
        'POSTPART_vs_PREG': 'Postpartum vs Pregnancy',
    }

    for i, (ax, contrast) in enumerate(zip(axes, contrasts)):
        contrast_data = plot_data[plot_data['contrast'] == contrast].copy()

        grid_df = pd.DataFrame(
            [(p, pair) for p in y_cats for pair in x_cats],
            columns=['pathway', 'pair_str'],
        )
        grid_df['y_coord'] = grid_df['pathway'].map(y_map)
        grid_df['x_coord'] = grid_df['pair_str'].map(x_map)
        ax.scatter(
            x=grid_df['x_coord'],
            y=grid_df['y_coord'],
            s=100,
            facecolors='none',
            edgecolors='#eeeeee',
            zorder=1,
        )

        if not contrast_data.empty:
            contrast_data['y_coord'] = contrast_data['pathway'].map(y_map)
            contrast_data['x_coord'] = contrast_data['pair_str'].map(x_map)
            scatter_colors = contrast_data['plot_value_winsorized']
            ax.scatter(
                x=contrast_data['x_coord'],
                y=contrast_data['y_coord'],
                c=scatter_colors,
                s=120,
                cmap=cmap,
                norm=norm,
                linewidth=1,
                zorder=2,
            )

        ax.set_xticks(range(len(x_cats)))
        ax.set_xticklabels(
            x_cats,
            rotation=45,
            ha='right',
            rotation_mode='anchor',
        )
        ax.set_title(
            title_map.get(contrast, contrast.replace('_', ' vs ')), fontsize=11
        )
        ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)

        if i == 0:
            ax.set_yticks(range(len(y_cats)))
            ax.set_yticklabels(y_cats, fontsize=9)

        ax.tick_params(axis='x', length=0)
        ax.tick_params(axis='y', length=0)
        ax.set_xlim(-0.5, len(x_cats) - 0.5)
        ax.set_ylim(-0.5, len(y_cats) - 0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(pad=1.0)
    fig.subplots_adjust(right=0.85, bottom=0.25)
    cbar_ax = fig.add_axes([0.88, 0.25, 0.02, 0.5])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=10)

    return fig

def get_cellchat_interaction_df(
    cellphonedb_path: str,
    cache_path: str,
    updated_db_name: str = 'CellChatDB_mouse_extended') -> pd.DataFrame:

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            data_for_r = pickle.load(f)
        interaction_input = data_for_r['interaction_input']
        complex_input = data_for_r['complex_input']
        cpdb_gene_info_mouse = data_for_r['cpdb_gene_info_mouse']
    else:
        interaction_input = pd.read_csv(
            os.path.join(cellphonedb_path, 'interaction_input.csv'))
        complex_input = pd.read_csv(
            os.path.join(cellphonedb_path, 'complex_input.csv'))
        gene_input = pd.read_csv(
            os.path.join(cellphonedb_path, 'gene_input.csv'))

        gp = GProfiler(user_agent='cellchat_update_tool',
                       return_dataframe=True)
        human_genes = gene_input['hgnc_symbol'].dropna().unique().tolist()
        
        ortholog_results = gp.orth(
            organism='hsapiens', target='mmusculus', query=human_genes)

        mapping_df = ortholog_results.drop_duplicates(subset=['incoming'])        
        mapping = mapping_df.set_index('incoming')['name']
        gene_input['Symbol'] = gene_input['hgnc_symbol'].map(mapping)

        cpdb_gene_info_mouse = gene_input[
            ['uniprot', 'Symbol']].dropna().drop_duplicates()

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'interaction_input': interaction_input,
                'complex_input': complex_input,
                'cpdb_gene_info_mouse': cpdb_gene_info_mouse
            }, f)

    to_r(interaction_input, 'cpdb_interaction_input')
    to_r(complex_input, 'cpdb_complex_input')
    to_r(cpdb_gene_info_mouse.reset_index(drop=True), 'cpdb_gene_info_mouse')
    to_r(updated_db_name, 'updated_db_name')

    r('''
    suppressPackageStartupMessages({
        library(CellChat)
        library(NeuronChat)
        library(dplyr)
        library(purrr)
    })
    base_interaction <- CellChatDB.mouse$interaction
    base_complex <- CellChatDB.mouse$complex
    base_geneInfo <- CellChatDB.mouse$geneInfo
    geneInfo_cpdb <- cpdb_gene_info_mouse
    cpdb_interactions_filtered <- cpdb_interaction_input %>%
        filter(partner_a %in% geneInfo_cpdb$uniprot & 
               partner_b %in% geneInfo_cpdb$uniprot)
    idx_partnerA <- match(
        cpdb_interactions_filtered$partner_a, geneInfo_cpdb$uniprot)
    cpdb_interactions_filtered$ligand <- geneInfo_cpdb$Symbol[idx_partnerA]
    idx_partnerB <- match(
        cpdb_interactions_filtered$partner_b, geneInfo_cpdb$uniprot)
    cpdb_interactions_filtered$receptor <- geneInfo_cpdb$Symbol[idx_partnerB]
    cpdb_formatted <- cpdb_interactions_filtered %>%
        rename(interaction_name = interactors, 
               pathway_name = classification) %>%
        mutate(
            annotation = case_when(
                directionality == 'secreted' ~ 'Secreted Signaling',
                directionality == 'transmembrane' ~ 'Cell-Cell Contact',
                TRUE ~ 'Cell-Cell Contact'
            ),
            interaction_name_2 = interaction_name
        )
    data(list='interactionDB_mouse')
    neuron_chat_db_list <- eval(parse(text = 'interactionDB_mouse'))
    neuron_chat_formatted <- purrr::map_dfr(
        neuron_chat_db_list,
        ~ expand.grid(
            interaction_name = .x$interaction_name,
            ligand = .x$lig_contributor,
            receptor = .x$receptor_subunit,
            stringsAsFactors = FALSE
        )
    ) %>%
    mutate(pathway_name = ligand, annotation = 'Secreted Signaling')
    required_cols <- colnames(base_interaction)
    for (col in required_cols) {
        if (!col %in% names(cpdb_formatted)) {
            cpdb_formatted[[col]] <- ''
        }
        if (!col %in% names(neuron_chat_formatted)) {
            neuron_chat_formatted[[col]] <- ''
        }
    }
    cpdb_final <- cpdb_formatted[, required_cols]
    neuron_chat_final <- neuron_chat_formatted[, required_cols]
    final_interactions <- bind_rows(
        base_interaction, cpdb_final, neuron_chat_final) %>%
        mutate(ligand_upper = toupper(ligand), 
               receptor_upper = toupper(receptor)) %>%
        distinct(ligand_upper, receptor_upper, .keep_all = TRUE) %>%
        select(-ligand_upper, -receptor_upper)
    final_geneInfo <- bind_rows(base_geneInfo, geneInfo_cpdb) %>%
        distinct(Symbol, .keep_all = TRUE)
    db_extended <- list(
        interaction = final_interactions,
        complex = base_complex,
        cofactor = CellChatDB.mouse$cofactor,
        geneInfo = final_geneInfo
    )
    assign(updated_db_name, db_extended, envir = .GlobalEnv)
    interaction_df_r <- db_extended$interaction
    ''')
    
    interaction_df = to_py('interaction_df_r', format='pandas')
    return interaction_df

def get_pathway_genes(
    pathway_name: str,
    interaction_df: pd.DataFrame) -> List[str]:

    pathway_interactions = interaction_df[
        interaction_df['pathway_name'] == pathway_name
    ]
    
    if pathway_interactions.empty:
        return []

    ligands = pathway_interactions['ligand.symbol'].dropna().unique()
    receptors = pathway_interactions['receptor.symbol'].dropna().unique()
    
    all_genes = set()
    for gene_group in np.concatenate([ligands, receptors]):
        all_genes.update(
            g for g in str(gene_group).replace('_', ',')
                .replace(' ', '').split(',') if g
        )
    
    return sorted(list(all_genes))

def get_pathway_interface_intermediate_data(
    adata: sc.AnnData,
    pathway_name: str,
    cell_type_a: str,
    cell_type_b: str,
    contrast: str,
    interaction_df: pd.DataFrame,
    cache_dir: str,
    coords_cols: Tuple[str, str] = ('x_ffd', 'y_ffd'),
    cell_type_col: str = 'subclass') -> Optional[dict]:

    safe_p = pathway_name.replace(' ', '_').replace('/', '_')
    safe_a = cell_type_a.replace(' ', '_').replace('/', '_')
    safe_b = cell_type_b.replace(' ', '_').replace('/', '_')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir, f'interface_{safe_p}_{safe_a}_{safe_b}_{contrast}.pkl'
    )

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f: return pickle.load(f)

    pathway_genes = get_pathway_genes(pathway_name, interaction_df)
    available_genes = [
        g for g in pathway_genes 
        if g.upper() in adata.var['gene_symbol'].str.upper().values
    ]
    if not available_genes: return None

    upper_to_symbol = pd.Series(
        adata.var['gene_symbol'].values,
        index=adata.var['gene_symbol'].str.upper())
    genes_to_use = upper_to_symbol[
        upper_to_symbol.index.isin([g.upper() for g in available_genes])
    ].unique().tolist()
    
    pathway_expr = np.array(adata[:, genes_to_use].X.mean(axis=1)).flatten()

    is_a = adata.obs[cell_type_col] == cell_type_a
    is_b = adata.obs[cell_type_col] == cell_type_b
    
    cells_a = adata.obs[is_a]
    cells_b = adata.obs[is_b]

    if cells_a.empty or cells_b.empty: return None
        
    cond_B_name, cond_A_name = contrast.split('_vs_')

    map_data = {
        'base_coords': adata.obs[list(coords_cols)].values,
        'coords_a': cells_a[list(coords_cols)].values,
        'expr_a': pathway_expr[is_a],
        'conditions_a': cells_a['condition'].values,
        'coords_b': cells_b[list(coords_cols)].values,
        'expr_b': pathway_expr[is_b],
        'conditions_b': cells_b['condition'].values,
        'cond_A_name': cond_A_name, 'cond_B_name': cond_B_name,
    }

    with open(cache_file, 'wb') as f: pickle.dump(map_data, f)
    return map_data

def plot_pathway_interface_diff_map(
    adata: sc.AnnData,
    pathway_name: str,
    cell_type_a: str,
    cell_type_b: str,
    contrast: str,
    interaction_df: pd.DataFrame,
    cache_dir: str,
    coords_cols: Tuple[str, str] = ('x_ffd', 'y_ffd'),
    cell_type_col: str = 'subclass',
    influence_radius: float = 8.0,
    resolution: int = 400,
    vmax: Optional[float] = None,
    ax: Optional[plt.Axes] = None) -> Tuple[
        plt.Figure, plt.Axes, Optional[plt.cm.ScalarMappable]]:
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    map_data = get_pathway_interface_intermediate_data(
        adata, pathway_name, cell_type_a, cell_type_b, contrast,
        interaction_df, cache_dir, coords_cols, cell_type_col)

    if map_data is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig, ax, None

    base_tree = KDTree(map_data['base_coords'])
    tree_a = KDTree(map_data['coords_a'])
    tree_b = KDTree(map_data['coords_b'])

    x_min, x_max = map_data['base_coords'][:, 0].min(), \
                   map_data['base_coords'][:, 0].max()
    y_min, y_max = map_data['base_coords'][:, 1].min(), \
                   map_data['base_coords'][:, 1].max()
    
    padding = 0.05
    x_pad = (x_max - x_min) * padding
    y_pad = (y_max - y_min) * padding
    plot_x_min, plot_x_max = x_min - x_pad, x_max + x_pad
    plot_y_min, plot_y_max = y_min - y_pad, y_max + y_pad

    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T

    d_scale, _ = calculate_distance_scale(map_data['base_coords'])
    d_max = influence_radius * d_scale
    grid_step = (x_max - x_min) / resolution

    Z_diff = np.zeros_like(X)
    Z_weight = np.zeros_like(X)
    base_mask = np.zeros_like(X, dtype=bool)

    for idx, point in enumerate(grid_points):
        row, col = np.unravel_index(idx, X.shape)
        if base_tree.query(point, k=1)[0] >= grid_step * 3: continue
        base_mask[row, col] = True
        
        neighbors_a = tree_a.query_ball_point(point, d_max)
        neighbors_b = tree_b.query_ball_point(point, d_max)
        if not neighbors_a or not neighbors_b: continue

        total_diff = 0
        min_weight = float('inf')

        for n_indices, coords, exprs, conds in [
            (neighbors_a, map_data['coords_a'], map_data['expr_a'], 
             map_data['conditions_a']),
            (neighbors_b, map_data['coords_b'], map_data['expr_b'],
             map_data['conditions_b'])
        ]:
            dists = np.sqrt(np.sum((coords[n_indices] - point)**2, axis=1))
            weights = np.exp(-0.7 * dists / d_max)
            
            is_A = conds[n_indices] == map_data['cond_A_name']
            is_B = conds[n_indices] == map_data['cond_B_name']
            
            w_A, w_B = weights[is_A], weights[is_B]
            sum_w_A, sum_w_B = np.sum(w_A), np.sum(w_B)

            if sum_w_A > 0 and sum_w_B > 0:
                avg_A = np.sum(w_A * exprs[n_indices][is_A]) / sum_w_A
                avg_B = np.sum(w_B * exprs[n_indices][is_B]) / sum_w_B
                total_diff += (avg_B - avg_A)
                min_weight = min(min_weight, sum_w_A, sum_w_B)
            else:
                min_weight = 0; break
        
        if min_weight > 0:
            Z_diff[row, col] = total_diff
            Z_weight[row, col] = min_weight

    Z_diff = np.where((Z_weight > 0) & base_mask, Z_diff, np.nan)
    
    from scipy.ndimage import gaussian_filter
    valid_mask = ~np.isnan(Z_diff)
    if np.any(valid_mask):
        Z_smooth = np.copy(Z_diff)
        Z_smooth[valid_mask] = gaussian_filter(
            Z_diff[valid_mask], sigma=1.5, mode='constant')
        Z_diff = Z_smooth

    if vmax is None:
        vmax = np.nanpercentile(
            np.abs(Z_diff), 98) if ~np.all(np.isnan(Z_diff)) else 1.0
    vmin = -vmax
    
    cmap = plt.get_cmap('seismic')
    
    ax.pcolormesh(X, Y, base_mask, cmap='Greys', 
                  vmin=0, vmax=2.0, alpha=0.2, rasterized=True)

    im = ax.pcolormesh(X, Y, Z_diff, cmap=cmap, vmin=vmin, vmax=vmax,
                      shading='gouraud', rasterized=True)
    
    ax.set_aspect('equal')
    ax.set_xlim(plot_x_min, plot_x_max)
    ax.set_ylim(plot_y_min, plot_y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    
    return fig, ax, im

def plot_pathway_expression_maps_grid(
    adata: sc.AnnData,
    interaction_triplets: List[Tuple[str, str, str]],
    contrasts: List[str],
    interaction_df: pd.DataFrame,
    cache_dir: str,
    coords_cols: Tuple[str, str] = ('x_ffd', 'y_ffd'),
    cell_type_col: str = 'subclass',
    influence_radius: float = 8.0,
    resolution: int = 300,
    vmax: Optional[float] = None):

    n_triplets = len(interaction_triplets)
    n_rows = n_triplets
    n_cols = len(contrasts)
    
    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.8, n_rows * 2.2)
    )
    if n_rows == 1: axs = np.array([axs])
    if n_cols == 1: axs = axs.reshape(-1, 1)

    images = []
    
    pbar_desc='Plotting maps'
    with tqdm(total=n_rows * n_cols, desc=pbar_desc) as pbar:
        for i, (p_name, ct_a, ct_b) in enumerate(interaction_triplets):
            for j, contrast in enumerate(contrasts):
                ax = axs[i, j]
                _, _, im = plot_pathway_interface_diff_map(
                    adata, p_name, ct_a, ct_b, contrast, interaction_df,
                    cache_dir, coords_cols, cell_type_col,
                    influence_radius, resolution, vmax, ax)
                
                if im: images.append(im)
                
                if i == 0:
                    if contrast == 'PREG_vs_CTRL':
                        title = 'Pregnancy vs Control'
                    else:
                        title = 'Postpartum vs Pregnancy'
                    ax.set_title(title, fontsize=10)

                if j == 0:
                    ylabel_str = f'{p_name.upper()}\n{ct_a} ↔ {ct_b}'
                    ax.set_ylabel(
                        ylabel_str, fontsize=7, 
                        rotation='vertical', va='center', labelpad=15)
                pbar.update(1)

    if images:
        vmax_used = images[0].get_clim()[1] if vmax is None else vmax
        for im in images: im.set_clim(-vmax_used, vmax_used)
        
        fig.subplots_adjust(wspace=0, hspace=0)

    return fig, axs

def plot_proximity_violins(
    adata: sc.AnnData,
    spatial_stats: pd.DataFrame,
    cell_type_pairs: List[tuple],
    cache_dir: str,
    cell_type_col: str,
    coords_cols: Tuple[str, str],
    condition_palette: dict):

    n_rows = len(cell_type_pairs)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(1.8, n_rows * 1.8), sharey='row'
    )
    if n_rows == 1: axes = axes.reshape(1, 2)
    
    comparisons = [['CTRL', 'PREG'], ['PREG', 'POSTPART']]
    pbar_desc = 'Plotting proximity violins'
    for i, pair in enumerate(tqdm(cell_type_pairs, desc=pbar_desc)):
        all_data = [
            get_spatial_map_intermediate_data(
                adata, spatial_stats, pair, c, cache_dir,
                cell_type_col, coords_cols
            ) for c in ['PREG_vs_CTRL', 'POSTPART_vs_PREG']
        ]
        valid_data = [pd.DataFrame(
            {'clr': d['clr_values'], 'condition': d['all_conditions']}
        ) for d in all_data if d]

        if not valid_data:
            for ax in axes[i]: ax.set_visible(False)
            continue
            
        df = pd.concat(valid_data).drop_duplicates()
        df['x'] = ''

        for j, comp in enumerate(comparisons):
            ax = axes[i, j]
            sns.violinplot(
                data=df[df['condition'].isin(comp)], x='x', y='clr',
                hue='condition', hue_order=comp, palette=condition_palette,
                split=True, ax=ax, cut=0, width=0.6,
                linewidth=1.0, inner=None
            )
            ax.get_legend().remove()
            ax.set(ylabel='', xlabel='', yticklabels=[], yticks=[])
            ax.spines[:].set_visible(False)
            ax.set_xticks([])

    legend_patches = [patches.Patch(color=c, label=l) for l, c in \
               condition_palette.items()]
    fig.legend(
        handles=legend_patches, loc='lower center', ncol=3,
        bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=8
    )
    fig.tight_layout()
    return fig, axes

def get_pathway_violin_data(
    adata: sc.AnnData,
    interaction_df: pd.DataFrame,
    triplet: Tuple[str, str, str]
) -> pd.DataFrame:
    
    p_name, ct_a, ct_b = triplet
    pathway_genes = get_pathway_genes(p_name, interaction_df)
    if not pathway_genes: return pd.DataFrame()

    adata_var_upper = adata.var['gene_symbol'].str.upper()
    available_genes = [
        g for g in pathway_genes if g.upper() in adata_var_upper.values
    ]
    if not available_genes: return pd.DataFrame()

    upper_to_symbol = pd.Series(
        adata.var['gene_symbol'].values, index=adata_var_upper
    )
    genes_to_use = upper_to_symbol[
        upper_to_symbol.index.isin([g.upper() for g in available_genes])
    ].unique()

    pathway_expr = np.array(adata[:, genes_to_use].X.mean(axis=1)).flatten()
    log_pathway_expr = np.log1p(pathway_expr) # Apply log1p transformation

    mask = adata.obs['subclass'].isin([ct_a, ct_b])
    df = pd.DataFrame({
        'pathway_expr': log_pathway_expr[mask],
        'condition': adata.obs['condition'][mask],
        'cell_type': adata.obs['subclass'][mask]
    })
    return df

def plot_pathway_violins(
    adata: sc.AnnData,
    interaction_df: pd.DataFrame,
    interaction_triplets: List[Tuple[str, str, str]],
    condition_palette: dict):
    
    n_rows = len(interaction_triplets)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(1.8, n_rows * 1.8), sharey='row'
    )
    if n_rows == 1: axes = axes.reshape(1, 2)
    
    comparisons = [['CTRL', 'PREG'], ['PREG', 'POSTPART']]
    pbar_desc = 'Plotting pathway violins'
    for i, triplet in enumerate(tqdm(interaction_triplets, desc=pbar_desc)):
        df = get_pathway_violin_data(adata, interaction_df, triplet)

        if df.empty:
            for ax in axes[i]: ax.set_visible(False)
            continue
        
        df['x'] = ''
        axes[i, 0].set_ylabel(
            f'{triplet[0]}', fontsize=8, rotation=0,
            ha='right', va='center', labelpad=25
        )
        for j, comp in enumerate(comparisons):
            ax = axes[i, j]
            sub_df = df[df['condition'].isin(comp)]
            sns.violinplot(
                data=sub_df, x='x', y='pathway_expr', hue='condition',
                hue_order=comp, palette=condition_palette,
                split=True, ax=ax, cut=0, width=0.6,
                linewidth=1.0, inner=None
            )
            ax.get_legend().remove()
            ax.set(ylabel='', xlabel='', yticklabels=[], yticks=[])
            ax.spines[:].set_visible(False)
            ax.set_xticks([])

    legend_patches = [patches.Patch(color=c, label=l) for l, c in \
               condition_palette.items()]
    fig.legend(
        handles=legend_patches, loc='lower center', ncol=3,
        bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=8
    )
    fig.tight_layout()
    return fig, axes

#endregion 

#region analysis configuration #################################################

working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'

cell_type_col = 'subclass'

selected_cell_types = [
    'Astro-NT NN', 'Astro-TE NN', 'Endo NN', 'Ependymal NN', 'Microglia NN',
    'Oligo NN', 'OPC NN', 'Peri NN', 'VLMC NN'
]

contrasts = ['PREG_vs_CTRL', 'POSTPART_vs_PREG']
comparisons = [
    ('PREG_vs_CTRL', ('CTRL', 'PREG')),
    ('POSTPART_vs_PREG', ('PREG', 'POSTPART'))
]

all_cell_type_pairs = [
    ('MPO-ADP Lhx8 Gaba', 'Endo NN'),
    ('SI-MPO-LPO Lhx8 Gaba', 'Endo NN'),
    ('Pvalb Gaba', 'Oligo NN'),
    ('Astro-NT NN', 'OPC NN'),
    ('STR D1 Gaba', 'OPC NN'),
]

interaction_triplets_to_plot = [
    # 1. Hypothalamic Neuro-Vascular Remodeling
    ('VEGF', 'MPO-ADP Lhx8 Gaba', 'Endo NN'),
    # 2. Neurotransmitter-Based Neuro-Vascular Modulation
    ('GABA-A', 'SI-MPO-LPO Lhx8 Gaba', 'Endo NN'),
    # 3. Interneuron Trophic Support to Oligodendrocytes
    ('FGF', 'Pvalb Gaba', 'Oligo NN'),
    # 4. Astrocyte-Driven Regulation of Glial Precursors
    ('PTN', 'Astro-NT NN', 'OPC NN'),
    # 5. Neuron-Glia Synaptic Adhesion in the Striatum
    ('NRXN', 'STR D1 Gaba', 'OPC NN'),
]

#endregion

#region load data ##############################################################

cells_joined = pd.read_csv(
    'projects/rrg-wainberg/single-cell/ABC/metadata/MERFISH-C57BL6J-638850/'
    '20231215/views/cells_joined.csv')
color_mappings = {
    'class': dict(zip(
        cells_joined['class'].str.replace('/', '_'), 
        cells_joined['class_color'])),
    'subclass': {k.replace('_', '/'): v for k,v in dict(zip(
        cells_joined['subclass'].str.replace('/', '_'), 
        cells_joined['subclass_color'])).items()}
}
for level in color_mappings:
    color_mappings[level] = {
        k.split(' ', 1)[1]: v for k, v in color_mappings[level].items()
}

adata_curio = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_curio_final.h5ad')

adata_merfish = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')
adata_merfish.var.index = adata_merfish.var['gene_symbol']

for adata in [adata_curio, adata_merfish]:
    for col in ['class', 'subclass']:
        adata.obs[col] = adata.obs[col].astype(str)\
            .str.extract(r'^(\d+)\s+(.*)', expand=False)[1]
    adata.obs['type'] = adata.obs[cell_type_col]\
        .astype(str).str.extract(r'(\w+)$', expand=False)
    adata.obs['type'] = adata.obs['type'].replace({'IMN': 'Gaba'})
    adata.obs['type'] = adata.obs['type'].replace({'Chol': 'Gaba'})

common_cell_types = (
    set(adata_curio.obs[
        adata_curio.obs[f'{cell_type_col}_keep']][cell_type_col])
    & set(adata_merfish.obs[
        adata_merfish.obs[f'{cell_type_col}_keep']][cell_type_col]))

adata_curio = adata_curio[
    adata_curio.obs[cell_type_col].isin(common_cell_types)].copy()
adata_merfish = adata_merfish[
    adata_merfish.obs[cell_type_col].isin(common_cell_types)].copy()

#endregion 

#region global proportions merfish #############################################

tt_curio, norm_props_curio = \
    get_global_diff(adata_curio, 'curio', cell_type_col)
tt_merfish, norm_props_merfish = \
    get_global_diff(adata_merfish, 'merfish', cell_type_col)

tt_combined = pd.concat([tt_curio, tt_merfish])
norm_props_combined = pd.concat([norm_props_curio, norm_props_merfish])

selected_cell_types = [
    'Astro-NT NN', 'Astro-TE NN', 'Endo NN', 'Ependymal NN', 'Microglia NN',
    'Oligo NN', 'OPC NN', 'Peri NN', 'VLMC NN']

tt_combined.sort_values('t_test_P.Value')
tt_combined[tt_combined['cell_type'].eq('Endo NN')]\
    .sort_values('t_test_P.Value')

fig = plot_cell_type_proportions(
    norm_props_combined,
    tt_combined,
    selected_cell_types,
    datasets_to_plot=['merfish'],
    base_figsize=(2.1, 2.0),
    nrows=int(np.ceil(len(selected_cell_types)/2)),
    legend_position=('center left', (0, 0))
)
fig.savefig(
    f'{working_dir}/figures/cell_type_proportions.png',
    dpi=300,
    bbox_inches='tight'
)
fig.savefig(
    f'{working_dir}/figures/cell_type_proportions.svg',
    dpi=300,
    bbox_inches='tight'
)

#endregion

#region local proportions merfish ##############################################

dataset_name = 'merfish'
d_max_scale = 20

file_path = (
    f'{working_dir}/output/{dataset_name}/spatial_stats_{cell_type_col}.pkl'
)
if os.path.exists(file_path):
    spatial_stats = pd.read_pickle(file_path)
else:
    results = []
    for sample in adata_merfish.obs['sample'].unique():
        sample_data = adata_merfish.obs[adata_merfish.obs['sample'] == sample]
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
    # spatial_stats.to_pickle(file_path)

min_nonzero = 5
pairs = spatial_stats[['cell_type_a', 'cell_type_b']].drop_duplicates()
sample_stats = (
    spatial_stats.groupby(['sample_id', 'cell_type_a', 'cell_type_b'])
    .agg(n_nonzero=('b_count', lambda x: (x > 0).sum()))
)
filtered_pairs = (
    sample_stats.groupby(['cell_type_a', 'cell_type_b'])
    .agg(min_nonzero_count=('n_nonzero', 'min'))
    .query('min_nonzero_count >= @min_nonzero')
    .reset_index()[['cell_type_a', 'cell_type_b']]
)
pairs_tested = set(tuple(x) for x in filtered_pairs.values)
print(f'testing {len(filtered_pairs)} pairs out of {len(pairs)} pairs')
del pairs, sample_stats; gc.collect()

selected_cell_types = [
    'Astro-NT NN', 'Astro-TE NN', 'Endo NN', 'Ependymal NN', 'Microglia NN',
    'Oligo NN', 'OPC NN', 'Peri NN', 'VLMC NN']

pairs_to_process = filtered_pairs[
    filtered_pairs['cell_type_b'].isin(selected_cell_types)].copy()
print(
    f'testing {len(pairs_to_process)} pairs out of {len(filtered_pairs)} pairs'
)

file_path = (
    f'{working_dir}/output/{dataset_name}/spatial_diff_{cell_type_col}.pkl'
)
if os.path.exists(file_path):
    spatial_diff = pd.read_pickle(file_path)
else:
    res = []
    pbar_desc = 'Processing pairs'
    with tqdm(total=len(pairs_to_process), desc=pbar_desc) as pbar:
        for _, row in pairs_to_process.iterrows():
            pair_result = get_spatial_diff(
                spatial_stats=spatial_stats,
                cell_type_a=row['cell_type_a'],
                cell_type_b=row['cell_type_b']
            )
            if pair_result:
                res.extend(pair_result)
            pbar.update(1)
    spatial_diff = pd.concat(res, ignore_index=True)
    spatial_diff['adj.P.Val'] = fdrcorrection(spatial_diff['P.Value'])[1]
    spatial_diff['contrast'] = (
        spatial_diff['contrast']
        .str.replace('POST_vs_PREG', 'POSTPART_vs_PREG')
    )
    # spatial_diff.to_pickle(file_path)

spatial_diff.to_csv(
    f'{working_dir}/output/{dataset_name}/spatial_diff_{cell_type_col}.csv',
    index=False
)

fig, axes = plt.subplots(1, 2, figsize=(7.5, 11))
all_logFC = spatial_diff[spatial_diff['contrast'].isin(contrasts)]['logFC']
vmax = all_logFC.abs().max()
vmin = -vmax

for ax, contrast in zip(axes, contrasts):
    contrast_data = spatial_diff[spatial_diff['contrast'] == contrast].copy()
    _, _, im, row_order = plot_spatial_diff_heatmap(
        contrast_data,
        pairs_tested,
        sig=0.01,
        cell_types_b=selected_cell_types,
        recompute_fdr=True,
        ax=ax,
        vmin=-0.45,
        vmax=0.45,
        adata=adata_merfish
    )
    ax.set_title('')

axes[1].set_yticks([])
axes[1].set_yticklabels([])
axes[1].set_ylabel('')

cbar_ax = fig.add_axes([0.05, 0.04, 0.2, 0.008])
fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='logFC')

plt.tight_layout(rect=[0, 0.05, 0.91, 1])
plt.savefig(
    f'{working_dir}/figures/heatmap_{dataset_name}_{cell_type_col}.png',
    dpi=300, bbox_inches='tight'
)
plt.savefig(
    f'{working_dir}/figures/heatmap_{dataset_name}_{cell_type_col}.svg',
    dpi=300, bbox_inches='tight'
)
plt.close(fig)

cache_dir = f'{working_dir}/output/merfish/spatial_maps'
for pair in tqdm(all_cell_type_pairs, desc='Preparing map data'):
    for contrast in contrasts:
        get_spatial_map_intermediate_data(
            adata=adata_merfish,
            spatial_stats=spatial_stats,
            cell_type_pair=pair,
            contrast=contrast,
            cache_dir=cache_dir,
            cell_type_col=cell_type_col,
            coords_cols=('x_ffd', 'y_ffd')
        )

fig, _ = plot_spatial_maps_grid(
    adata_merfish,
    spatial_stats,
    all_cell_type_pairs,
    cache_dir=cache_dir,
    resolution=250,
    influence_radius=8,
    vmax=0.5
)
fig.savefig(f'{working_dir}/figures/spatial_maps.png',
            bbox_inches='tight', dpi=300)
fig.savefig(f'{working_dir}/figures/spatial_maps.svg',
            bbox_inches='tight')
plt.close(fig)

fig_violins, _ = plot_proximity_violins(
    adata=adata_merfish,
    spatial_stats=spatial_stats,
    cell_type_pairs=all_cell_type_pairs,
    cache_dir=cache_dir,
    cell_type_col=cell_type_col,
    coords_cols=('x_affine', 'y_affine'),
    condition_palette=condition_colors
)
fig_violins.savefig(
    f'{working_dir}/figures/proximity_violins_final.svg',
    bbox_inches='tight'
)
plt.close(fig_violins)

fig, axes = plot_spatial_diff_radii(
    spatial_data=adata_merfish.obs,
    coords_cols=('x_affine', 'y_affine'),
    sample_col='sample',
    d_max_scale=d_max_scale,
    s=0.05
)
fig.savefig(
    f'{working_dir}/figures/{dataset_name}/proximity_radii.png', 
    dpi=300, bbox_inches='tight'
)
plt.close(fig)

#endregion

#region cell type differential signalling curio ################################

cellphonedb_v4_mouse_path = f'{working_dir}/cellphonedb'
ortholog_cache_file = f'{working_dir}/cellphonedb/gprofiler_orthologs.pkl'

extended_db_name = update_cellchat_db(
    cellphonedb_path=cellphonedb_v4_mouse_path,
    cache_path=ortholog_cache_file,
    updated_db_name='CellChatDB_mouse_extended'
)

cell_type_diffs = {}
for name, conditions in comparisons:
    cobj_rds_path = (
        f'{working_dir}/output/curio/cellchat_{name}_{cell_type_col}.rds'
    )
    prepare_cellchat_object(
        adata=adata_curio, cell_type_col=cell_type_col,
        conditions=conditions, output_path=cobj_rds_path,
        db_name=extended_db_name
    )

    diff_path = (
        f'{working_dir}/output/curio/'
        f'cellchat_cell_type_diff_{name}_{cell_type_col}_spatial.pkl'
    )
    if os.path.exists(diff_path):
        cell_type_diffs[name] = pd.read_pickle(diff_path)
    else:
        cell_type_diffs[name] = get_cellchat_cell_type_pair_diff(
            cobj_rds_path=cobj_rds_path, conditions=conditions
        )
        pd.to_pickle(cell_type_diffs[name], diff_path)

cellchat_count_df = pd.concat(
    [df[df['measure'] == 'count'] for df in cell_type_diffs.values()]
)
cellchat_weight_df = pd.concat(
    [df[df['measure'] == 'weight'] for df in cell_type_diffs.values()]
)

cellchat_count_df[
    cellchat_count_df['cell_type_b'].isin(selected_cell_types)
].to_csv(
    f'{working_dir}/output/curio/cellchat_cell_type_count_diff.csv',
    index=False
)

cellchat_weight_df[
    cellchat_weight_df['cell_type_b'].isin(selected_cell_types)
].to_csv(
    f'{working_dir}/output/curio/cellchat_cell_type_weight_diff.csv',
    index=False
)

subplot_color = '#4361ee'

for align_sender_to_center_flag in [True, False]:
    for df, measure in [
        (cellchat_count_df, 'count'), (cellchat_weight_df, 'weight')
    ]:
        fig, axes = plt.subplots(
            len(contrasts), 1, figsize=(4.5, 4 * len(contrasts))
        )
        axes = [axes] if len(contrasts) == 1 else axes
        for i, contrast in enumerate(contrasts):
            plot_cellchat_vs_proximity_scatter(
                cellchat_df=df,
                spatial_df=spatial_diff,
                contrast=contrast,
                ax=axes[i],
                tested_pairs=pairs_tested,
                color=subplot_color,
                value_col='logFC',
                cell_types_to_include=selected_cell_types,
                align_sender_to_center=align_sender_to_center_flag
            )
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        
        base_name = (
            f'{working_dir}/figures/scatter_prox_vs_cellchat_{measure}'
            f'_sender_is_center_{align_sender_to_center_flag}'
        )
        fig.savefig(f'{base_name}.png', dpi=200, bbox_inches='tight')
        fig.savefig(f'{base_name}.svg', bbox_inches='tight')
        plt.close(fig)

for x_axis_are_senders in [True, False]:
    for df, measure in [
        (cellchat_count_df, 'count'), (cellchat_weight_df, 'weight')]:
        
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 11))
        row_order_contrast = {}
        for i, contrast in enumerate(contrasts):
            
            spatial_fig, spatial_ax = plt.subplots()
            _, _, _, row_order = plot_spatial_diff_heatmap(
                spatial_diff[spatial_diff['contrast'] == contrast],
                pairs_tested,
                sig=0.01,
                cell_types_b=selected_cell_types,
                recompute_fdr=True,
                ax=spatial_ax,
                vmin=-0.45,
                vmax=0.45,
                adata=adata_merfish
            )
            plt.close(spatial_fig)
            row_order_contrast[contrast] = row_order

            im = plot_cellchat_cell_type_heatmap(
                df=df[df['contrast'] == contrast],
                ax=axes[i],
                vmin=-0.01,
                vmax=0.01,
                x_axis_is_sender=x_axis_are_senders,
                x_axis_cell_types=selected_cell_types,
                tested_pairs=pairs_tested,
                title='',
                row_order=row_order_contrast[contrast],
                adata=adata_curio
            )
        
        axes[1].set_ylabel('')
        axes[1].set_yticklabels([])
        axes[1].set_yticks([])

        if im:
            cbar_ax = fig.add_axes([0.05, 0.04, 0.2, 0.008])
            label = ('Difference in Count' if measure == 'count' else 
                     'Difference in Strength (logFC)')
            fig.colorbar(
                im, cax=cbar_ax, orientation='horizontal', label=label
            )
            plt.tight_layout(rect=[0, 0.05, 0.91, 1])

        base_name = (
            f'{working_dir}/figures/cellchat_diff_{measure}_combined_'
            f'xaxis_is_sender_{x_axis_are_senders}'
        )
        fig.savefig(f'{base_name}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{base_name}.svg', dpi=300, bbox_inches='tight')
        plt.close(fig)

#endregion

#region pathway differential signalling curio #################################

pathway_diffs = {}
for name, conditions in comparisons:
    pathway_diff_path = (
        f'{working_dir}/output/curio/'
        f'cellchat_pathway_diff_{name}_{cell_type_col}_spatial.pkl'
    )
    if os.path.exists(pathway_diff_path):
        pathway_diffs[name] = pd.read_pickle(pathway_diff_path)
    else:
        pathway_diffs[name] = get_cellchat_pathway_pair_diff(
            cobj_rds_path=cobj_rds_path,
            conditions=conditions,
            contrast_name=name
        )
        pd.to_pickle(pathway_diffs[name], pathway_diff_path)

exclude_pathways = [
    'CD39', 'CEACAM', 'DHEA', 'DHT', 'Ddc', 'Gjb6', 'PROS', 'TGFb', 'WNT',
    'PSAP', 'SLITRK', 'Slc17a7', 'COLLAGEN', 'Signaling by Calsyntenin'
]
include_pathways = []

pathway_pair_diff_df = pd.concat(pathway_diffs.values())

fig = plot_pathway_diff_dotplot(
    pathway_pair_diff_df=pathway_pair_diff_df,
    cell_type_pairs=all_cell_type_pairs,
    contrasts=contrasts,
    top_n=8,
    pathways_to_exclude=exclude_pathways,
    pathways_to_include=include_pathways,
    cluster_rows=False
)
fig.savefig(
    f'{working_dir}/figures/cellchat_pathway_diff_dotplot.png',
    dpi=300, bbox_inches='tight'
)
fig.savefig(
    f'{working_dir}/figures/cellchat_pathway_diff_dotplot.svg',
    dpi=300, bbox_inches='tight'
)
plt.close(fig)

adata = adata_curio.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

cellchat_interaction_df = get_cellchat_interaction_df(
    cellphonedb_path=cellphonedb_v4_mouse_path,
    cache_path=ortholog_cache_file
)
cellchat_interaction_df.to_csv(
    f'{working_dir}/output/curio/cellchat_interaction_df.csv', index=False
)
pathway_map_cache_dir = f'{working_dir}/output/curio/pathway_maps'

genes = sorted(set(
    g for t in interaction_triplets_to_plot
    for g in get_pathway_genes(t[0], cellchat_interaction_df)
    if g in adata.var_names
))
pd.DataFrame({'gene': genes}).to_csv(
    f'{working_dir}/output/curio/cellchat_interaction_df_select.csv', index=False)

for p_name, ct_a, ct_b in tqdm(
    interaction_triplets_to_plot, desc='Caching interface data'):
    for contrast in contrasts:
        get_pathway_interface_intermediate_data(
            adata=adata,
            pathway_name=p_name,
            cell_type_a=ct_a,
            cell_type_b=ct_b,
            contrast=contrast,
            interaction_df=cellchat_interaction_df,
            cache_dir=pathway_map_cache_dir,
            cell_type_col=cell_type_col
        )

fig, axs = plot_pathway_expression_maps_grid(
    adata=adata,
    interaction_triplets=interaction_triplets_to_plot,
    contrasts=contrasts,
    interaction_df=cellchat_interaction_df,
    cache_dir=pathway_map_cache_dir,
    resolution=300, 
    influence_radius=18,
    vmax=0.8
)
fig.savefig(
    f'{working_dir}/figures/pathway_interface_maps_grid.png',
    dpi=300, bbox_inches='tight'
)
fig.savefig(
    f'{working_dir}/figures/pathway_interface_maps_grid.svg',
    bbox_inches='tight'
)
plt.close(fig)

fig_pathway_violins, _ = plot_pathway_violins(
    adata=adata_curio,
    interaction_df=cellchat_interaction_df,
    interaction_triplets=interaction_triplets_to_plot,
    condition_palette=condition_colors
)
if fig_pathway_violins:
    fig_pathway_violins.savefig(
        f'{working_dir}/figures/pathway_violins_final.svg',
        bbox_inches='tight'
    )
    plt.close(fig_pathway_violins)

#endregion

#region scratchpad #############################################################

df = pathway_pair_diff_df.copy()

top_n_per_pair = 8
pathways_to_include_from_plot = ['APP']
cell_type_pairs_from_plot = all_cell_type_pairs
contrasts_from_plot = contrasts

df['canonical_pair'] = df.apply(
    lambda row: frozenset([row['source'], row['target']]), axis=1
)
pairs_df = pd.DataFrame(cell_type_pairs_from_plot, columns=['c1', 'c2'])
pairs_df['canonical_pair'] = pairs_df.apply(
    lambda row: frozenset([row['c1'], row['c2']]), axis=1
)
plot_data = pd.merge(df, pairs_df[['canonical_pair']], on='canonical_pair')
plot_data = plot_data[plot_data['contrast'].isin(contrasts_from_plot)]

pair_str_map = {
    frozenset(p): f'{p[0]} ↔ {p[1]}' for p in cell_type_pairs_from_plot
}
plot_data['pair_str'] = plot_data['canonical_pair'].map(pair_str_map)

z_transformer = lambda x: (x - x.mean()) / x.std()
plot_data['plot_value'] = (
    plot_data.groupby(['contrast', 'pair_str'])['strength_diff']
    .transform(z_transformer)
    .fillna(0)
)
plot_data['abs_plot_value'] = plot_data['plot_value'].abs()

top_pathways = set()
unique_pairs = plot_data['canonical_pair'].unique()
for pair_key in unique_pairs:
    for contrast in contrasts_from_plot:
        subset = plot_data[
            (plot_data['canonical_pair'] == pair_key)
            & (plot_data['contrast'] == contrast)
        ]
        subset = subset.sort_values('abs_plot_value', ascending=False)
        top_pathways.update(subset.head(top_n_per_pair)['pathway'])

if 'exclude_pathways' in locals() and exclude_pathways is not None:
    top_pathways.difference_update(set(exclude_pathways))

top_pathways.update(set(pathways_to_include_from_plot))

plot_data = plot_data[plot_data['pathway'].isin(top_pathways)].copy()

idx_max = plot_data.groupby(
    ['contrast', 'canonical_pair', 'pathway']
)['abs_plot_value'].idxmax()
csv_df = plot_data.loc[idx_max].copy()

csv_df.to_csv(
    f'{working_dir}/output/curio/cellchat_pathway_pair_diff.csv',
    index=False
)

#endregion

