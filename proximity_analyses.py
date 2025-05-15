import os, re, gc
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import coo_array
from scipy.spatial import KDTree
from scipy.stats import ttest_ind
from typing import Tuple, List
from ryp import r, to_r, to_py
from tqdm.auto import tqdm
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

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
            PREG_vs_CTRL = "conditionPREG - conditionCTRL",
            POST_vs_PREG = "conditionPOSTPART - conditionPREG",
            POST_vs_CTRL = "conditionPOSTPART - conditionCTRL"
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
    for(coef in c("PREG_vs_CTRL", "POST_vs_PREG")) {
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
        "PREG_vs_CTRL": ("PREG", "CTRL"),
        "POST_vs_PREG": ("POSTPART", "PREG")
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

            if len(group1_values) >= 2 and len(group2_values) >= 2:
                t_stat, p_val = ttest_ind(
                    group1_values, 
                    group2_values, 
                    equal_var=False, 
                    nan_policy='omit'
                )
            else:
                p_val = np.nan
            
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

def plot_cell_type_proportions(
    norm_props_df: pd.DataFrame,
    tt_combined: pd.DataFrame = None,
    cell_types: List[str] = None,
    nrows: int = 2,
    base_figsize: Tuple[float, float] = (2.0, 2.0),
    palette: dict = {'curio': '#4cc9f0', 'merfish': '#4361ee'},
    condition_order: List[str] = ['CTRL', 'PREG', 'POSTPART'],
    legend_position: Tuple[str, Tuple[float, float]] =
        ('center right', (1.10, 0.5)),
    datasets_to_plot: List[str] = ['merfish']
    ) -> plt.Figure:

    if cell_types is None:
        cell_types = sorted(norm_props_df['cell_type'].unique())
    else:
        cell_types = sorted(cell_types)

    ncols = int(np.ceil(len(cell_types) / nrows))
    figsize = (base_figsize[0] * ncols, base_figsize[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    condition_labels = {
        'CTRL': 'Control',
        'PREG': 'Pregnant',
        'POSTPART': 'Postpartem'
    }

    legend_handles = []
    legend_labels = []

    for i, cell_type_val in enumerate(cell_types):
        ax = axes[i]
        ct_data = norm_props_df[
            norm_props_df['cell_type'] == cell_type_val
        ].copy()

        plotted_conditions_on_ax = []

        for dataset_name, line_color in palette.items():
            if datasets_to_plot and dataset_name not in datasets_to_plot:
                continue

            ds_data = ct_data[ct_data['dataset'] == dataset_name].copy()
            if ds_data.empty:
                continue

            cond_data_initial = ds_data.drop_duplicates(
                subset=['condition', 'mean', 'se']
            )
            ordered_cond_values_for_ds = [
                c for c in condition_order
                if c in cond_data_initial['condition'].values
            ]
            if not ordered_cond_values_for_ds:
                continue
            
            cond_data = cond_data_initial.set_index('condition').loc[
                ordered_cond_values_for_ds
            ].reset_index()

            if not plotted_conditions_on_ax:
                plotted_conditions_on_ax = cond_data['condition'].tolist()

            if len(cond_data) > 1:
                mean_val = cond_data['mean'].mean()
                std_val = cond_data['mean'].std()
                if pd.notna(std_val) and std_val > 0:
                    cond_data['mean'] = (cond_data['mean']-mean_val)/std_val
                    cond_data['se'] = cond_data['se'] / std_val

            current_line = ax.errorbar(
                x=cond_data['condition'],
                y=cond_data['mean'],
                yerr=cond_data['se'],
                fmt='o-', color=line_color, alpha=0.7, label=dataset_name,
                capsize=3, markersize=5, linewidth=1.5,
                elinewidth=1, ecolor=line_color, capthick=1
            )
            if dataset_name not in legend_labels:
                legend_handles.append(current_line)
                legend_labels.append(dataset_name)

            asterisks_to_plot_on_line = []
            if tt_combined is not None:
                for j in range(len(condition_order)-1):
                    cond1_plot_name = condition_order[j]
                    cond2_plot_name = condition_order[j+1]

                    c1_lookup = cond1_plot_name
                    c2_lookup = cond2_plot_name
                    if cond2_plot_name == 'POSTPART':
                        c2_lookup = 'POST'
                    contrast_lookup = f'{c2_lookup}_vs_{c1_lookup}'

                    filter_ct = (tt_combined['cell_type'] == cell_type_val)
                    filter_ds = (tt_combined['dataset'] == dataset_name)
                    filter_cn = (tt_combined['contrast'] == contrast_lookup)
                    sig_data_row = tt_combined[
                        filter_ct & filter_ds & filter_cn
                    ]
                    
                    if sig_data_row.empty:
                        continue

                    p_val_col = 't_test_P.Value'
                    p_val = sig_data_row[p_val_col].iloc[0]
                    
                    sig_str = ''
                    if p_val < 0.01: sig_str = '**'
                    elif p_val < 0.05: sig_str = '*'

                    if sig_str:
                        current_plot_conds = cond_data['condition'].tolist()
                        idx1 = current_plot_conds.index(cond1_plot_name)
                        idx2 = current_plot_conds.index(cond2_plot_name)
                        x_txt_pos = (idx1 + idx2) / 2.0
                        asterisks_to_plot_on_line.append(
                            {'x': x_txt_pos, 's': sig_str}
                        )
            
            if asterisks_to_plot_on_line:
                se_safe = cond_data['se'].fillna(0)
                line_data_tops = cond_data['mean'] + se_safe
                max_error_bar_top = line_data_tops.max()
                
                if pd.notna(max_error_bar_top):
                    asterisk_y_coord = max_error_bar_top - 0.5
                    for ast_info in asterisks_to_plot_on_line:
                        ax.text(
                            ast_info['x'], asterisk_y_coord, ast_info['s'],
                            ha='center', va='bottom',
                            fontsize=12, color='black',
                            fontweight='bold', clip_on=True, zorder=20
                        )

        ax.set_title(cell_type_val, fontsize=10)
        row_idx = i // ncols
        is_bottom_row = (row_idx == nrows - 1 or 
                         i >= len(cell_types) - ncols)
        
        ax_xtick_lbls_use = (plotted_conditions_on_ax if 
                             plotted_conditions_on_ax else condition_order)
        ax.set_xticks(range(len(ax_xtick_lbls_use)))

        if is_bottom_row:
            ax.set_xticklabels(
                [condition_labels.get(c, c) for c in ax_xtick_lbls_use],
                rotation=45, ha='right', fontsize=9,
                rotation_mode='anchor'
            )
        else:
            ax.set_xticklabels([])
            
        for spine in ax.spines.values():
            spine.set_visible(True)
    
    for i in range(len(cell_types), len(axes)):
        axes[i].set_visible(False)
    
    fig.text(0.035, 0.5, 'Normalized Proportion\n(Z-score)', 
             va='center', ha='center', rotation='vertical', fontsize=9)
    
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

def plot_sample_radii(
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
    df, tested_pairs, sig=0.10, figsize=(15, 15),
    cell_types_a=None, cell_types_b=None,
    recompute_fdr=True, ax=None, vmin=None, vmax=None) -> Tuple[
        plt.Figure, plt.Axes, plt.cm.ScalarMappable]:
    
    filtered_df = df.copy()
    if cell_types_a is not None:
        filtered_df = filtered_df[filtered_df['cell_type_a'].isin(cell_types_a)]
    if cell_types_b is not None:
        filtered_df = filtered_df[filtered_df['cell_type_b'].isin(cell_types_b)]
        
    if recompute_fdr and len(filtered_df) > 0:
        for cell_type_b in filtered_df['cell_type_b'].unique():
            b_mask = filtered_df['cell_type_b'] == cell_type_b
            if b_mask.sum() > 0:
                filtered_df.loc[b_mask, 'adj.P.Val'] = fdrcorrection(
                    filtered_df.loc[b_mask, 'P.Value'])[1]
    
    a_types = sorted(filtered_df['cell_type_a'].unique())
    b_types = sorted(filtered_df['cell_type_b'].unique())
    
    mat = np.full((len(a_types), len(b_types)), np.nan)
    sigs = np.full((len(a_types), len(b_types)), '', dtype=object)
    a_idx = {c: i for i, c in enumerate(a_types)}
    b_idx = {c: i for i, c in enumerate(b_types)}
    
    for _, row in filtered_df.iterrows():
        i = a_idx[row['cell_type_a']]
        j = b_idx[row['cell_type_b']]
        mat[i, j] = row['logFC']
        if row['adj.P.Val'] < sig: 
            sigs[i, j] = '*'
    
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    cmap = LinearSegmentedColormap.from_list(
        "custom_diverging",
        ["#4b0857", "#813e8f", "white", "#66b66b", "#156a2f"], N=100)
    
    if vmin is None or vmax is None:
        if np.all(np.isnan(mat)):
            mabs = 1.0
        else:
            mabs = np.nanmax(np.abs(mat))
            if np.isnan(mabs) or mabs == 0:
                mabs = 1.0
        if vmin is None:
            vmin = -mabs
        if vmax is None:
            vmax = mabs
    
    x = np.arange(len(b_types) + 1)
    y = np.arange(len(a_types) + 1)
    X, Y = np.meshgrid(x, y)
    im = ax.pcolormesh(X, Y, mat, cmap=cmap, vmin=vmin, vmax=vmax, 
                      rasterized=False)
    
    ax.set_xlim(0, len(b_types))
    ax.set_ylim(len(a_types), 0)
    
    for i in range(len(a_types)):
        for j in range(len(b_types)):
            a, b = a_types[i], b_types[j]
            if (a, b) not in tested_pairs:
                ax.text(j + 0.5, i + 0.5, 'X', ha='center', va='center', 
                        color='gray', size=10)
            elif sigs[i, j]=='*':
                ax.text(j + 0.5, i + 0.5, '*', ha='center', va='center',
                        color='black', size=14, weight='bold')
    
    ax.set_xticks(np.arange(len(b_types)) + 0.5)
    ax.set_yticks(np.arange(len(a_types)) + 0.5)
    ax.set_xticklabels(b_types, rotation=45, ha='right')
    ax.set_yticklabels(a_types)
    ax.set_xlabel('Surround Cell Type')
    ax.set_ylabel('Center Cell Type')
    
    if fig is not None:
        cbar = fig.colorbar(im, ax=ax, shrink=0.2) 
        cbar.set_label('logFC')
        plt.tight_layout()    
    return fig, ax, im

def plot_spatial_diff_maps(
    adata, 
    spatial_stats,
    spatial_diff,
    cell_type_pairs,
    contrast,
    coords_cols=('x_ffd', 'y_ffd'),
    cell_type_col='subclass',
    influence_radius=8,
    resolution=350,
    vmax=None):
    
    n_pairs = len(cell_type_pairs)
    fig = plt.figure(figsize=(6, 5 * n_pairs))
    
    base_sample = 'PREG1'
    base_cells = adata[adata.obs['sample'] == base_sample]
    base_coords = base_cells.obs[list(coords_cols)].values
    
    x_min, x_max = base_coords[:, 0].min(), base_coords[:, 0].max()
    y_min, y_max = base_coords[:, 1].min(), base_coords[:, 1].max()
    
    padding = 0.15
    x_pad, y_pad = (x_max - x_min) * padding, (y_max - y_min) * padding
    plot_x_min, plot_x_max = x_min - x_pad, x_max + x_pad
    plot_y_min, plot_y_max = y_min - y_pad, y_max + y_pad
    
    condition_map = {'POST': 'POSTPART', 'PREG': 'PREG', 'CTRL': 'CTRL'}
    cond1, cond2 = contrast.split('_vs_')
    mapped_cond1 = condition_map.get(cond1, cond1)
    mapped_cond2 = condition_map.get(cond2, cond2)
    
    cond1_samples = adata.obs[adata.obs['condition'] == mapped_cond1]['sample']
    cond1_samples = cond1_samples.unique()
    cond2_samples = adata.obs[adata.obs['condition'] == mapped_cond2]['sample']
    cond2_samples = cond2_samples.unique()
    
    max_abs_all = 0
    plot_data = []
    
    for cell_type_a, cell_type_b in cell_type_pairs:
        diff_data = spatial_diff[
            (spatial_diff['cell_type_a'] == cell_type_a) &
            (spatial_diff['cell_type_b'] == cell_type_b) &
            (spatial_diff['contrast'] == contrast)
        ]
        
        if len(diff_data) == 0:
            plot_data.append(None)
            continue
            
        pair_stats = spatial_stats[
            (spatial_stats['cell_type_a'] == cell_type_a) &
            (spatial_stats['cell_type_b'] == cell_type_b)
        ]
        
        a_cells_all = adata[adata.obs[cell_type_col] == cell_type_a]
        a_coords_all = a_cells_all.obs[list(coords_cols)].values
        
        all_coords, all_b_counts, all_all_counts, all_conditions = [], [], [], []
        
        for cond, mapped_cond, samples in [
            (cond1, mapped_cond1, cond1_samples), 
            (cond2, mapped_cond2, cond2_samples)
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
            plot_data.append(None)
            continue
            
        all_coords = np.vstack(all_coords)
        all_b_counts = np.concatenate(all_b_counts)
        all_all_counts = np.concatenate(all_all_counts)
        all_conditions = np.array(all_conditions)
        
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
        
        other_counts = all_all_counts - all_b_counts
        clr_values = np.log(all_b_counts + 0.5) - 0.5 * (
            np.log(all_b_counts + 0.5) + np.log(other_counts + 0.5))
        
        base_mask = np.zeros_like(X, dtype=bool)
        for i in range(len(X)):
            for j in range(len(X[0])):
                point = np.array([X[i, j], Y[i, j]])
                dists = np.sqrt(np.sum((base_coords - point)**2, axis=1))
                base_mask[i, j] = np.min(dists) < grid_step * 3
                
                dists = np.sqrt(np.sum((all_coords - point)**2, axis=1))
                weights = np.exp(-0.7 * dists/d_max) * (dists <= d_max * 1.2)
                
                if np.sum(weights) > 0:
                    mask1 = all_conditions == cond1
                    mask2 = all_conditions == cond2
                    
                    if np.sum(weights[mask1]) > 0 and np.sum(weights[mask2]) > 0:
                        avg_clr1 = np.sum(weights[mask1] * clr_values[mask1])
                        avg_clr1 /= np.sum(weights[mask1])
                        avg_clr2 = np.sum(weights[mask2] * clr_values[mask2])
                        avg_clr2 /= np.sum(weights[mask2])
                        
                        Z_diff[i, j] = avg_clr2 - avg_clr1
                        Z_weight[i, j] = min(np.sum(weights[mask1]), 
                                           np.sum(weights[mask2]))
        
        Z_diff = np.where((Z_weight > 0) & base_mask, Z_diff, np.nan)
        
        from scipy.ndimage import gaussian_filter
        valid_mask = ~np.isnan(Z_diff)
        Z_smooth = np.copy(Z_diff)
        Z_smooth[valid_mask] = gaussian_filter(
            Z_diff[valid_mask], sigma=1.5, mode='constant')
        Z_diff = Z_smooth
        
        max_abs = np.nanmax(np.abs(Z_diff))
        max_abs_all = max(max_abs_all, max_abs)
        
        plot_data.append({
            'X': X, 'Y': Y, 
            'Z_diff': Z_diff, 
            'tissue_mask': tissue_mask, 
            'a_mask': a_mask,
            'cell_type_a': cell_type_a,
            'cell_type_b': cell_type_b
        })
    
    if vmax is None:
        vmax = max_abs_all
    vmin = -vmax
    
    cmap = LinearSegmentedColormap.from_list(
        'custom_diverging',
        ['#4b0857', '#813e8f', '#ffffff', '#66b66b', '#156a2f'], N=100)
    
    im = None
    for idx, data in enumerate(plot_data):
        if data is None:
            continue
            
        ax = fig.add_subplot(n_pairs, 1, idx + 1)
        
        title = f"{data['cell_type_a']} (Center)\n{data['cell_type_b']} (Surround)"
        ax.set_title(title, loc='left', pad=5)
        
        ax.pcolormesh(data['X'], data['Y'], data['tissue_mask'], 
                     cmap='Greys', alpha=0.4, rasterized=True)
        
        im = ax.pcolormesh(data['X'], data['Y'], data['Z_diff'], 
                          cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.9,
                          shading='gouraud', rasterized=True)
        
        ax.pcolormesh(data['X'], data['Y'], data['a_mask'], 
                     cmap='binary', alpha=0.5, vmin=0, vmax=1, rasterized=True)
        
        ax.set_aspect('equal')
        ax.set_xlim(plot_x_min, plot_x_max)
        ax.set_ylim(plot_y_min, plot_y_max)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_color('black')
    
    if im is not None:
        cbar_ax = fig.add_axes([0.4, 0.06, 0.2, 0.01])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('logFC')
        cbar.set_ticks([-0.25, 0.0, 0.25])
    
    plt.subplots_adjust(hspace=0.05, bottom=0.08)
    return fig

#endregion 

#region load data ##############################################################

working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'

cell_type_col = 'subclass'

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

adata_curio.X = adata_curio.layers['log1p'].copy()
adata_merfish.X = adata_merfish.layers['volume_log1p'].copy()

common_subclasses_numbered = (
    set(adata_curio.obs[adata_curio.obs['subclass_keep']]['subclass'])
    & set(adata_merfish.obs[adata_merfish.obs['subclass_keep']]['subclass']))

subclass_map = {}
for subclass in common_subclasses_numbered:
    if isinstance(subclass, str) and re.match(r'^\d+\s+', subclass):
        clean_name = re.sub(r'^\d+\s+', '', subclass)
        subclass_map[clean_name] = subclass

for adata in [adata_curio, adata_merfish]:
    for col in ['class', 'subclass']:
        adata.obs[col] = adata.obs[col].astype(str)\
            .str.extract(r'^(\d+)\s+(.*)', expand=False)[1]

common_subclasses = (
    set(adata_curio.obs[adata_curio.obs['subclass_keep']]['subclass'])
    & set(adata_merfish.obs[adata_merfish.obs['subclass_keep']]['subclass']))

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

#region global proportions #####################################################

tt_curio, norm_props_curio = \
    get_global_diff(adata_curio, 'curio', cell_type_col)
tt_merfish, norm_props_merfish = \
    get_global_diff(adata_merfish, 'merfish', cell_type_col)

tt_combined = pd.concat([tt_curio, tt_merfish])
norm_props_combined = pd.concat([norm_props_curio, norm_props_merfish])

selected_cell_types = [
    'VLMC NN', 'Astro-TE NN', 'Endo NN', 'Microglia NN', 'OPC NN',
    'Peri NN', 'Oligo NN', 'Astro-NT NN']

tt_combined.sort_values('t_test_P.Value')
tt_combined[tt_combined['cell_type'].eq('Endo NN')]\
    .sort_values('t_test_P.Value')

fig = plot_cell_type_proportions(
    norm_props_combined,
    tt_combined,
    selected_cell_types,
    datasets_to_plot=['merfish'],
    base_figsize=(2, 2),
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

#region local proportions ######################################################

dataset_name = 'merfish'
d_max_scale = 20

# get spatial stats per sample
file = f'{working_dir}/output/{dataset_name}/spatial_stats_{cell_type_col}.pkl'
if os.path.exists(file):
    spatial_stats = pd.read_pickle(file)
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
    spatial_stats.to_pickle(file)

# minimum number of nonzero interactions required
# in each sample for a cell type pair
min_nonzero = 5
pairs = spatial_stats[['cell_type_a', 'cell_type_b']].drop_duplicates()
sample_stats = spatial_stats\
    .groupby(['sample_id', 'cell_type_a', 'cell_type_b'])\
    .agg(n_nonzero=('b_count', lambda x: (x > 0).sum()))
filtered_pairs = sample_stats\
    .groupby(['cell_type_a', 'cell_type_b'])\
    .agg(min_nonzero_count=('n_nonzero', 'min'))\
    .query('min_nonzero_count >= @min_nonzero')\
    .reset_index()[['cell_type_a', 'cell_type_b']]

pairs_tested = set(tuple(x) for x in filtered_pairs.values)
print(f'testing {len(filtered_pairs)} pairs out of {len(pairs)} pairs')
del pairs, sample_stats; gc.collect()

selected_cell_types = [
    'VLMC NN', 'Astro-TE NN', 'Endo NN', 'Microglia NN', 'OPC NN',
    'Peri NN', 'Oligo NN', 'Astro-NT NN'
]
pairs_to_process = filtered_pairs[
    filtered_pairs['cell_type_b'].isin(selected_cell_types)].copy()
print(f'testing {len(pairs_to_process)} pairs out of {len(filtered_pairs)} pairs')

# get differential testing results
file = f'{working_dir}/output/{dataset_name}/spatial_diff_{cell_type_col}.csv'
if os.path.exists(file):
    spatial_diff = pd.read_csv(file)
else:
    res = []
    with tqdm(total=len(pairs_to_process), desc='Processing pairs') as pbar:
        for _, row in pairs_to_process.iterrows():
            pair_result = get_spatial_diff(
                spatial_stats=spatial_stats,
                cell_type_a=row['cell_type_a'],
                cell_type_b=row['cell_type_b'])
            if pair_result:
                res.extend(pair_result)
            pbar.update(1)
    spatial_diff = pd.concat(res, ignore_index=True)
    spatial_diff['adj.P.Val'] = fdrcorrection(spatial_diff['P.Value'])[1]
    # spatial_diff.to_csv(file, index=False)

# plot heatmaps for both contrasts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 9))

contrasts = list(spatial_diff['contrast'].unique())
contrast1, contrast2 = contrasts[0], contrasts[1]
contrast1_data = spatial_diff[spatial_diff['contrast'] == contrast1].copy()
contrast2_data = spatial_diff[spatial_diff['contrast'] == contrast2].copy()

all_data = pd.concat([contrast1_data, contrast2_data])
max_abs_val = np.max(np.abs(all_data['logFC']))
vmin, vmax = -max_abs_val, max_abs_val

_, _, im1 = plot_spatial_diff_heatmap(
    contrast1_data, pairs_tested, sig=0.10,
    cell_types_a=None, cell_types_b=selected_cell_types,
    recompute_fdr=True, ax=ax1, vmin=vmin, vmax=vmax)

_, _, im2 = plot_spatial_diff_heatmap(
    contrast2_data, pairs_tested, sig=0.10,
    cell_types_a=None, cell_types_b=selected_cell_types,
    recompute_fdr=True, ax=ax2, vmin=vmin, vmax=vmax)

ax1.set_title('')
ax2.set_title('')
ax2.set_yticks([])
ax2.set_yticklabels([])
ax2.set_ylabel('')

cbar_ax = fig.add_axes([0.05, 0.04, 0.2, 0.008])
cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
cbar.set_label('logFC')

plt.tight_layout(rect=[0, 0.05, 0.91, 1])
plt.savefig(
    f'{working_dir}/figures/heatmap_{dataset_name}_{cell_type_col}.png',
    dpi=300, bbox_inches='tight')
plt.savefig(
    f'{working_dir}/figures/heatmap_{dataset_name}_{cell_type_col}.svg',
    dpi=300, bbox_inches='tight')
plt.close(fig)

# plot spatial maps for both contrasts
cell_type_pairs = [
    ('CEA-AAA-BST Six3 Sp9 Gaba', 'Endo NN'),
    ('Sst Chodl Gaba', 'Endo NN'),
    ('BAM NN', 'Astro-NT NN')
]
plot_spatial_diff_maps(
    adata_merfish,
    spatial_stats,
    spatial_diff,
    cell_type_pairs=cell_type_pairs,
    contrast='PREG_vs_CTRL',
    resolution=250,
    influence_radius=3.5,
    vmax=0.4
)
plt.savefig(f'{working_dir}/figures/spatial_maps_preg_vs_ctrl.svg', 
            bbox_inches='tight')

cell_type_pairs = [
    ('CEA-AAA-BST Six3 Sp9 Gaba', 'Endo NN'),
    ('MPO-ADP Lhx8 Gaba', 'Endo NN'),
    ('Tanycyte NN', 'Oligo NN')
]
plot_spatial_diff_maps(
    adata_merfish,
    spatial_stats,
    spatial_diff,
    cell_type_pairs=cell_type_pairs,
    contrast='POST_vs_PREG',
    resolution=250,
    influence_radius=3.5,
    vmax=0.4
)
plt.savefig(f'{working_dir}/figures/spatial_maps_postpart_vs_preg.svg', 
            bbox_inches='tight')

# plot sample radii
fig, axes = plot_sample_radii(
    spatial_data=adata_merfish.obs,
    coords_cols=('x_affine', 'y_affine'),
    sample_col='sample',
    d_max_scale=d_max_scale,
    s=0.05)
fig.savefig(
    f'{working_dir}/figures/{dataset_name}/proximity_radii.png', 
    dpi=300, bbox_inches='tight')
plt.close(fig)

#endregion

#region scratch ################################################################

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

# generate filtered heatmaps for each contrast
for contrast in spatial_diff['contrast'].unique():
    contrast_data = spatial_diff[spatial_diff['contrast'] == contrast].copy()
    if not contrast_data.empty:
        fig, ax = plot_spatial_diff_heatmap(
            contrast_data, pairs_tested, contrast, sig=0.10,
            cell_types_a=None, 
            cell_types_b=selected_cell_types, 
            recompute_fdr=True, figsize=(25, 15))
        plt.savefig(
            f'{working_dir}/figures/'
            f'heatmap_{dataset_name}_{cell_type_col}_{contrast}.png',
            dpi=300, bbox_inches='tight')
        plt.close()

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
