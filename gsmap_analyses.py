import os
import scanpy as sc
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as patches
from utils import run
from pathlib import Path
from ryp import r, to_r
from scipy.cluster.hierarchy import linkage, leaves_list
from typing import Optional, List

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 500

workdir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'
input_dir = f'{workdir}/gsmap/input'
output_dir = f'{workdir}/gsmap/output'
figures_dir = f'{workdir}/figures/gsmap'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

#region functions ##############################################################

def _insert_gap(arr, indices, axis=0):
    if isinstance(arr, pd.DataFrame):
        res = arr.copy()
        if axis == 0:
            for i in sorted(indices, reverse=True):
                res = pd.concat([
                    res.iloc[:i],
                    pd.DataFrame([[np.nan]*res.shape[1]], columns=res.columns),
                    res.iloc[i:]
                ]).reset_index(drop=True)
        elif axis == 1:
            for i in sorted(indices, reverse=True):
                res.insert(loc=int(i), column=f'gap_{i}', value=np.nan)
        return res

    is_series = isinstance(arr, pd.Series)
    if is_series:
        arr = arr.values

    val = np.nan
    if arr.dtype == object or arr.dtype.kind in ['U', 'S']:
        val = ''
    for i in sorted(indices, reverse=True):
        arr = np.insert(arr, i, val, axis=axis)
    
    if is_series:
        return pd.Series(arr)
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

def plot_trait_ranking(output_dir, figures_dir, conditions):
    sample_map = {s: c for c, L in conditions.items() for s in L}
    files = Path(output_dir).glob('*/cauchy_combination/*.Cauchy.csv.gz')
    df = pl.concat([
        pl.scan_csv(f).with_columns(
            condition=pl.lit(sample_map.get(f.parts[-3])),
            trait=pl.lit(
                f.name.replace(f.parts[-3] + '_', '')
                .replace('.Cauchy.csv.gz', '')
            )
        ) for f in files if sample_map.get(f.parts[-3])
    ])

    stats_df = df.select(['trait', 'annotation']).collect()
    n_traits = stats_df['trait'].n_unique()
    n_annotations = stats_df['annotation'].n_unique()
    bonferroni_p = 0.05 / (n_traits * n_annotations)
    log_p_threshold = -np.log10(bonferroni_p)

    ranking_data = df\
        .with_columns(p_log=(-pl.col('p_cauchy').log10()))\
        .group_by('trait', 'annotation', 'condition')\
        .agg(pl.median('p_log').alias('median_log_p'))\
        .group_by('trait')\
        .agg(pl.max('median_log_p').alias('max_median_log_p'))\
        .sort('max_median_log_p', descending=True)\
        .collect()\
        .to_pandas()

    fig, ax = plt.subplots(figsize=(2.5, 3.5), facecolor='white')

    scores = ranking_data['max_median_log_p']
    norm = colors.Normalize(vmin=scores.min(), vmax=scores.max())
    cmap = plt.get_cmap('GnBu')

    sns.barplot(
        x='max_median_log_p', y='trait', data=ranking_data,
        hue='trait', palette=list(cmap(norm(scores.values))), 
        ax=ax, orient='h', legend=False
    )

    ax.axvline(
        x=log_p_threshold, color='black', linestyle='--',
        linewidth=2
    )

    ax.set_xlabel('Peak Score')
    ax.set_ylabel('')
    sns.despine(ax=ax)
    ax.tick_params(axis='y', length=0)

    fig.tight_layout()
    fig.savefig(f'{figures_dir}/trait_ranking.svg', bbox_inches='tight')
    fig.savefig(
        f'{figures_dir}/trait_ranking.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_gwas_heatmap(
    adata, output_dir, figures_dir, conditions, 
    traits_to_include: Optional[List[str]] = None
):
    sample_map = {s: c for c, L in conditions.items() for s in L}
    files = [f for f in Path(output_dir).glob(
        '*/cauchy_combination/*.Cauchy.csv.gz'
    ) if sample_map.get(f.parts[-3])]

    def get_trait(f):
        return f.name.replace(f.parts[-3] + '_', '')\
            .replace('.Cauchy.csv.gz', '')

    df = pl.concat([
        pl.scan_csv(f).with_columns(
            condition=pl.lit(sample_map.get(f.parts[-3])),
            trait=pl.lit(get_trait(f))
        ) for f in files
    ])

    agg_data = df\
        .with_columns(
            p_log=(-pl.col('p_cauchy').log10()).fill_null(0.0))\
        .group_by(['condition', 'trait', 'annotation'])\
        .agg(pl.col('p_log').median())\
        .collect()\
        .to_pandas()
    
    avg_data = agg_data.groupby(
        ['trait', 'annotation'])['p_log'].mean().reset_index()
    avg_data.columns = ['trait', 'annotation', 'score']

    mat_order = avg_data.pivot(
        index='annotation', columns='trait', values='score'
    ).dropna().sort_index()

    if traits_to_include:
        mat_order = mat_order[[t for t in traits_to_include 
                              if t in mat_order.columns]]

    mat_order = mat_order.T
    if not mat_order.empty:
        mat_order = mat_order.reindex(
            mat_order.mean(axis=1).sort_values(
                ascending=False).index)

    type_info = adata.obs[['subclass', 'type']]\
        .drop_duplicates().set_index('subclass')
    col_df = pd.DataFrame(index=mat_order.columns).join(type_info)
    col_df['type'] = pd.Categorical(
        col_df['type'], categories=['Glut', 'Gaba', 'NN'], ordered=True
    )
    col_df = col_df.sort_values('type')

    ordered_cols = []
    for _, group in col_df.groupby('type', sort=False):
        subtypes = group.index
        if len(subtypes) > 1:
            avg_scores = mat_order[subtypes].mean(axis=0)
            subtypes = avg_scores.sort_values(ascending=False).index
        ordered_cols.extend(subtypes)
    
    a_types = mat_order.index.tolist()
    b_types = ordered_cols

    condition_data = {}
    for cond in ['control', 'pregnant', 'postpartum']:
        cond_df = agg_data[agg_data['condition'] == cond]
        cond_mat = cond_df.pivot(
            index='annotation', columns='trait', values='p_log'
        )
        if traits_to_include:
            cond_mat = cond_mat[[t for t in traits_to_include 
                               if t in cond_mat.columns]]
        cond_mat = cond_mat.T
        cond_mat = cond_mat.reindex(index=a_types, columns=b_types)
        condition_data[cond] = cond_mat
    
    complete_data_mask = pd.DataFrame(True, index=a_types, columns=b_types)
    for cond_mat in condition_data.values():
        complete_data_mask &= ~cond_mat.isna()
    
    valid_cols = complete_data_mask.columns[complete_data_mask.any()]
    b_types = [col for col in b_types if col in valid_cols]
    
    for cond in condition_data:
        condition_data[cond] = condition_data[cond][b_types]

    fig, ax = plt.subplots(figsize=(15.5, 3.0), facecolor='white')

    col_df_sorted = col_df.reindex(b_types)
    type_boundaries = col_df_sorted['type'].ne(
        col_df_sorted['type'].shift()
    ).cumsum()
    gaps = np.where(type_boundaries.diff() > 0)[0]

    all_values = []
    for cond_mat in condition_data.values():
        all_values.extend(cond_mat.values.flatten())
    all_values = [v for v in all_values if pd.notna(v)]
    vmin, vmax = min(all_values), max(all_values)
    
    cmap = plt.get_cmap('GnBu')
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    gap_offset = 0
    for col_idx, col in enumerate(b_types):
        if col_idx in gaps:
            gap_offset += 1
        
        for row_idx, row in enumerate(a_types):
            x = col_idx + gap_offset
            y = row_idx
            
            ctrl_val = condition_data['control'].loc[row, col]
            preg_val = condition_data['pregnant'].loc[row, col]
            post_val = condition_data['postpartum'].loc[row, col]
            
            values = {
                'control': ctrl_val,
                'pregnant': preg_val,
                'postpartum': post_val
            }
            
            cx, cy = x + 0.5, y + 0.5
            corners = [(x, y), (x+1, y), (x+1, y+1), (x, y+1)]
            
            angles = [np.pi/6, np.pi/6 + 2*np.pi/3, np.pi/6 + 4*np.pi/3]
            
            def get_edge_points(angle):
                dx, dy = np.cos(angle), np.sin(angle)
                edge_pts = []
                
                if abs(dx) > 1e-10:
                    t_right = (x + 1 - cx) / dx
                    y_right = cy + t_right * dy
                    if y <= y_right <= y + 1 and t_right > 0:
                        edge_pts.append((x + 1, y_right))
                    
                    t_left = (x - cx) / dx
                    y_left = cy + t_left * dy
                    if y <= y_left <= y + 1 and t_left > 0:
                        edge_pts.append((x, y_left))
                
                if abs(dy) > 1e-10:
                    t_bottom = (y + 1 - cy) / dy
                    x_bottom = cx + t_bottom * dx
                    if x <= x_bottom <= x + 1 and t_bottom > 0:
                        edge_pts.append((x_bottom, y + 1))
                    
                    t_top = (y - cy) / dy
                    x_top = cx + t_top * dx
                    if x <= x_top <= x + 1 and t_top > 0:
                        edge_pts.append((x_top, y))
                
                return edge_pts[0] if edge_pts else (cx, cy)
            
            edge_points = [get_edge_points(a) for a in angles]
            
            def angle_from_center(pt):
                return np.arctan2(pt[1] - cy, pt[0] - cx)
            
            wedges = []
            for i in range(3):
                p1 = edge_points[i]
                p2 = edge_points[(i + 1) % 3]
                a1 = angle_from_center(p1)
                a2 = angle_from_center(p2)
                
                if a2 < a1:
                    a2 += 2 * np.pi
                
                poly_points = [(cx, cy), p1]
                
                for corner in corners:
                    ca = angle_from_center(corner)
                    if ca < a1:
                        ca += 2 * np.pi
                    if a1 <= ca <= a2:
                        poly_points.append(corner)
                
                poly_points.append(p2)
                
                def polar_sort_key(pt):
                    angle = angle_from_center(pt)
                    if angle < a1:
                        angle += 2 * np.pi
                    return angle
                
                poly_points = [poly_points[0]] + sorted(
                    poly_points[1:], key=polar_sort_key)
                wedges.append(poly_points)
            
            conditions = ['control', 'pregnant', 'postpartum']
            for i, (cond, poly_pts) in enumerate(zip(conditions, wedges)):
                wedge = patches.Polygon(
                    poly_pts,
                    facecolor=cmap(norm(values[cond])),
                    edgecolor='white', linewidth=0.1
                )
                ax.add_patch(wedge)

    plot_mat = _insert_gap(mat_order, gaps, axis=1)
    b_types_gapped = _insert_gap(pd.Series(b_types, dtype=object), gaps)
    
    for r0, r1 in [(0, len(a_types))]:
        for c0, c1 in _segments(~plot_mat.isna().all(0)):
            ax.add_patch(patches.Rectangle(
                (c0, r0), c1 - c0, r1 - r0,
                fill=False, ec='black', lw=0.5
            ))

    ax.set_xlim(0, len(b_types_gapped))
    ax.set_ylim(len(a_types), 0)
    ax.set_xticks(np.arange(len(b_types_gapped)) + 0.5)
    ax.set_yticks(np.arange(len(a_types)) + 0.5)
    ax.set_xticklabels(b_types_gapped, rotation=45, ha='right')
    ax.set_yticklabels(a_types)
    ax.tick_params(length=0)
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('GWAS Trait')
    
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(right=0.88, bottom=0.3, left=0.2)

    cbar_ax = fig.add_axes([0.9, 0.3, 0.015, 0.4])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r'Median $-\log_{10}(P\text{-value})$')
    
    legend_ax = fig.add_axes([0.92, 0.75, 0.05, 0.1])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis('off')
    
    cell_border = patches.Rectangle(
        (0.2, 0.2), 0.6, 0.6,
        facecolor='none', edgecolor='black', linewidth=1
    )
    legend_ax.add_patch(cell_border)
    
    cx, cy = 0.5, 0.5
    x, y = 0.2, 0.8
    size = 0.6
    
    angles = [np.pi/6, np.pi/6 + 2*np.pi/3, np.pi/6 + 4*np.pi/3]
    
    def get_legend_edge(angle):
        dx, dy = np.cos(angle), -np.sin(angle)
        edge_pts = []
        
        if abs(dx) > 1e-10:
            t_right = (x + size - cx) / dx
            y_right = cy + t_right * dy
            if y - size <= y_right <= y and t_right > 0:
                edge_pts.append((x + size, y_right))
            
            t_left = (x - cx) / dx
            y_left = cy + t_left * dy
            if y - size <= y_left <= y and t_left > 0:
                edge_pts.append((x, y_left))
        
        if abs(dy) > 1e-10:
            t_top = (y - cy) / dy
            x_top = cx + t_top * dx
            if x <= x_top <= x + size and t_top > 0:
                edge_pts.append((x_top, y))
            
            t_bottom = (y - size - cy) / dy
            x_bottom = cx + t_bottom * dx
            if x <= x_bottom <= x + size and t_bottom > 0:
                edge_pts.append((x_bottom, y - size))
        
        return edge_pts[0] if edge_pts else (cx, cy)
    
    edge_points = [get_legend_edge(a) for a in angles]
    corners = [(x, y), (x+size, y), (x+size, y-size), (x, y-size)]
    
    def angle_from_center(pt):
        return np.arctan2(-(pt[1] - cy), pt[0] - cx)
    
    wedges = []
    labels = ['C', 'P', 'PP']
    
    for i in range(3):
        p1 = edge_points[i]
        p2 = edge_points[(i + 1) % 3]
        a1 = angle_from_center(p1)
        a2 = angle_from_center(p2)
        
        if a2 < a1:
            a2 += 2 * np.pi
        
        poly_points = [(cx, cy), p1]
        
        for corner in corners:
            ca = angle_from_center(corner)
            if ca < a1:
                ca += 2 * np.pi
            if a1 <= ca <= a2:
                poly_points.append(corner)
        
        poly_points.append(p2)
        
        def polar_sort_key(pt):
            angle = angle_from_center(pt)
            if angle < a1:
                angle += 2 * np.pi
            return angle
        
        poly_points = [poly_points[0]] + sorted(
            poly_points[1:], key=polar_sort_key)
        
        wedge = patches.Polygon(
            poly_points,
            facecolor='lightgray', edgecolor='black', linewidth=0.5
        )
        legend_ax.add_patch(wedge)
        
        label_x = sum(pt[0] for pt in poly_points) / len(poly_points)
        label_y = sum(pt[1] for pt in poly_points) / len(poly_points)
        legend_ax.text(label_x, label_y, labels[i], 
                      ha='center', va='center', fontsize=6)
    
    legend_ax.text(0.5, -0.1, 'Conditions', 
                   ha='center', va='top', fontsize=8)

    fig.savefig(
        f'{figures_dir}/gwas_association_heatmap.svg', 
        bbox_inches='tight')
    fig.savefig(
        f'{figures_dir}/gwas_association_heatmap.png', 
        dpi=300, bbox_inches='tight'
    )
    plt.close(fig)

def analyze_trait_associations(
    output_dir, conditions, traits_of_interest: List[str]
):
    from scipy.stats import mannwhitneyu, kruskal
    sample_map = {s: c for c, L in conditions.items() for s in L}
    all_samples = sorted([s for L in conditions.values() for s in L])

    files = [
        p for sample in all_samples for trait in traits_of_interest
        if (p := Path(output_dir) / sample / 'report' / trait /
                'gsMap_plot' / f'{sample}_{trait}_gsMap_plot.csv').exists()
    ]
    df = pl.concat([
        pl.scan_csv(f).with_columns(
            sample=pl.lit(f.parts[-5]),
            trait=pl.lit(f.parts[-3]),
            condition=pl.lit(sample_map.get(f.parts[-5]))
        ) for f in files
    ])\
    .with_columns(gsmap_score=-pl.col('p').log10())\
    .select(['gsmap_score', 'annotation', 'sample', 'trait', 'condition'])\
    .collect()\
    .to_pandas()

    plot_df = df.reset_index(drop=True)
    
    results = []
    for (trait, annotation), group in plot_df.groupby(['trait', 'annotation']):
        ctrl = group[group['condition'] == 'control']['gsmap_score'].values
        preg = group[group['condition'] == 'pregnant']['gsmap_score'].values
        post = group[group['condition'] == 'postpartum']['gsmap_score'].values
        
        if len(ctrl) >= 10 and len(preg) >= 10 and len(post) >= 10:
            _, kruskal_p = kruskal(ctrl, preg, post)
        else:
            kruskal_p = np.nan
        
        if len(ctrl) >= 10 and len(preg) >= 10:
            _, preg_ctrl_p = mannwhitneyu(
                preg, ctrl, alternative='two-sided')
            results.append({
                'trait': trait, 'annotation': annotation,
                'comparison': 'pregnant_vs_control',
                'logfc': np.median(preg) - np.median(ctrl),
                'p_value': preg_ctrl_p, 'kruskal_p': kruskal_p,
                'ctrl_n': len(ctrl), 'test_n': len(preg)
            })
        
        if len(ctrl) >= 10 and len(post) >= 10:
            _, post_ctrl_p = mannwhitneyu(
                post, ctrl, alternative='two-sided')
            results.append({
                'trait': trait, 'annotation': annotation,
                'comparison': 'postpartum_vs_control',
                'logfc': np.median(post) - np.median(ctrl),
                'p_value': post_ctrl_p, 'kruskal_p': kruskal_p,
                'ctrl_n': len(ctrl), 'test_n': len(post)
            })
        
        if len(preg) >= 10 and len(post) >= 10:
            _, post_preg_p = mannwhitneyu(
                post, preg, alternative='two-sided')
            results.append({
                'trait': trait, 'annotation': annotation,
                'comparison': 'postpartum_vs_pregnant',
                'logfc': np.median(post) - np.median(preg),
                'p_value': post_preg_p, 'kruskal_p': kruskal_p,
                'ctrl_n': len(preg), 'test_n': len(post)
            })
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df['p_adj'] = results_df.groupby('comparison')['p_value']\
            .transform(lambda x: pd.Series(
                np.minimum(x * len(x) / (~x.isna()).sum(), 1)
            ))
    return results_df

def plot_trait_boxplots(
    results_df, output_dir, conditions, figures_dir, 
    traits_of_interest: List[str], cell_types_to_plot: List[str], 
    p_threshold: float = 0.05
):
    condition_colors = {
        'control': '#7209b7', 'pregnant': '#b5179e', 'postpartum': '#f72585'
    }
    sample_map = {s: c for c, L in conditions.items() for s in L}
    all_samples = sorted([s for L in conditions.values() for s in L])
    
    files = [
        p for sample in all_samples for trait in traits_of_interest
        if (p := Path(output_dir) / sample / 'report' / trait /
                'gsMap_plot' / f'{sample}_{trait}_gsMap_plot.csv').exists()
    ]
    df = pl.concat([
        pl.scan_csv(f).with_columns(
            sample=pl.lit(f.parts[-5]),
            trait=pl.lit(f.parts[-3]),
            condition=pl.lit(sample_map.get(f.parts[-5]))
        ) for f in files
    ])\
    .with_columns(gsmap_score=-pl.col('p').log10())\
    .select(['gsmap_score', 'annotation', 'sample', 'trait', 'condition'])\
    .collect()\
    .to_pandas()
    
    if results_df.empty:
        return
    
    plot_annotations = pd.DataFrame({
        'trait': [traits_of_interest[0]] * len(cell_types_to_plot),
        'annotation': cell_types_to_plot
    })
    
    plot_df = pd.merge(df, plot_annotations, on=['trait', 'annotation'])
    
    n_annotations = plot_annotations['annotation'].nunique()
    fig, axes = plt.subplots(
        1, n_annotations, figsize=(2.2 * n_annotations, 3.5), 
        squeeze=False, constrained_layout=True
    )
    axes = axes.flatten()
    
    annotation_order = [
        ct for ct in cell_types_to_plot if ct in plot_df['annotation'].values]
    
    for idx, annotation in enumerate(annotation_order):
        ax = axes[idx]
        ann_df = plot_df[plot_df['annotation'] == annotation]
        
        positions = []
        for i, cond in enumerate(['control', 'pregnant', 'postpartum']):
            cond_data = ann_df[ann_df['condition'] == cond]['gsmap_score']
            if len(cond_data) > 0:
                y = cond_data.values
                x = np.random.normal(i, 0.04, size=len(y))
                ax.scatter(x, y, alpha=1.0, color=condition_colors[cond], s=4, 
                          edgecolors='none')
                positions.append((i, cond_data))
            else:
                positions.append((i, pd.Series(dtype=float)))
        
        bp_data = [pos[1] for pos in positions if len(pos[1]) > 0]
        bp_positions = [pos[0] for pos in positions if len(pos[1]) > 0]
        
        if bp_data:
            bp = ax.boxplot(
                bp_data, positions=bp_positions, widths=0.4,
                patch_artist=True, showfliers=False
            )
            for patch, pos in zip(bp['boxes'], bp_positions):
                cond = ['control', 'pregnant', 'postpartum'][pos]
                patch.set_facecolor(condition_colors[cond])
                patch.set_alpha(0.4)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.0)
            
            for element in ['whiskers', 'medians', 'caps']:
                for item in bp[element]:
                    item.set_color('black')
                    item.set_alpha(1.0)
        
        ann_results = results_df[
            (results_df['annotation'] == annotation) & 
            (results_df['p_adj'] < p_threshold)
        ]
        
        y_max = ann_df['gsmap_score'].max()
        y_offset = y_max * 0.05
        sig_y = y_max + y_offset
        
        for _, row in ann_results.iterrows():
            if row['comparison'] == 'pregnant_vs_control':
                x1, x2 = 0, 1
            elif row['comparison'] == 'postpartum_vs_control':
                x1, x2 = 0, 2
            else:
                x1, x2 = 1, 2
            
            ax.plot([x1, x1, x2, x2], 
                   [sig_y, sig_y + y_offset/2, sig_y + y_offset/2, sig_y],
                   'k-', linewidth=0.5)
            ax.text((x1 + x2)/2, sig_y + y_offset/2, '*', 
                   ha='center', va='bottom', fontsize=12)
            sig_y += y_offset * 2
        
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Control', 'Pregnancy', 'Postpartum'], 
                          rotation=45, ha='right')
        ax.set_title(annotation, fontsize=10)
        ax.set_ylabel('gsMap Score' if idx == 0 else '')
        
        # Make y-axis ticks more sparse
        y_ticks = ax.get_yticks()
        ax.set_yticks(y_ticks[::2])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    for idx in range(n_annotations, len(axes)):
        axes[idx].set_visible(False)
    
    fig.savefig(f'{figures_dir}/condition_comparison_boxplot.svg')
    fig.savefig(f'{figures_dir}/condition_comparison_boxplot.png', dpi=300)
    plt.close(fig)

def plot_spatial_gwas(
    output_dir, figures_dir, conditions, trait, cell_types, 
    bg_point_size=5, fg_point_size=15
):
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5), facecolor='white', sharey=True
    )
    fig.subplots_adjust(wspace=0)
    
    all_dfs = []
    for condition in conditions.keys():
        sample_names = conditions.get(condition, [])
        condition_dfs = []
        for sample in sample_names:
            file_path = (
                Path(output_dir) / sample / 'report' / trait / 'gsMap_plot' /
                f'{sample}_{trait}_gsMap_plot.csv'
            )
            if file_path.exists():
                df = pd.read_csv(file_path)
                condition_dfs.append(df)
        if condition_dfs:
            condition_df = pd.concat(condition_dfs, ignore_index=True)
            condition_df['condition'] = condition
            all_dfs.append(condition_df)

    if not all_dfs:
        print(f'No data found for trait {trait}')
        plt.close(fig)
        return
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    is_selected_type = combined_df['annotation'].isin(cell_types)
    combined_df['alpha'] = np.where(is_selected_type, 1.0, 0.4)
    combined_df['edgecolor'] = np.where(is_selected_type, 'black', 'none')
    combined_df['linewidth'] = np.where(is_selected_type, 0.5, 0.0)
    combined_df['size'] = np.where(is_selected_type, fg_point_size, bg_point_size)

    vmin = combined_df['logp'].min()
    vmax = combined_df['logp'].max()

    condition_order = ['control', 'pregnant', 'postpartum']
    for i, condition in enumerate(condition_order):
        if condition in conditions:
            ax = axes[i]
        plot_df = combined_df[combined_df['condition'] == condition]
        plot_df = plot_df.sort_values(by='size').reset_index(drop=True)
        
        ax.scatter(
            x=plot_df['sx'], y=plot_df['sy'], c=plot_df['logp'],
            alpha=plot_df['alpha'], s=plot_df['size'],
            edgecolors=plot_df['edgecolor'], linewidths=plot_df['linewidth'],
                cmap='GnBu', vmin=vmin, vmax=vmax, rasterized=True
        )

        ax.set_title(condition.capitalize())
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)

    # norm = colors.Normalize(vmin=vmin, vmax=vmax)
    # mappable = cm.ScalarMappable(norm=norm, cmap='GnBu')

    # cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
    # cbar = fig.colorbar(
    #     mappable, cax=cbar_ax, orientation='horizontal'
    # )
    # cbar.set_label(r'$-\log_{10}(P)$')
    
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = f'{figures_dir}/spatial_plot_{trait}'
    fig.savefig(f'{fig_path}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{fig_path}.svg', bbox_inches='tight')
    plt.close(fig)

#endregion

#region prep data ##############################################################

if not os.path.exists(f'{input_dir}/gsMap_resource'):
    os.makedirs(input_dir, exist_ok=True)
    run(f'wget https://yanglab.westlake.edu.cn/data/gsMap/gsMap_resource.tar.gz '
        f'-P {input_dir}')
    run(f'tar -xvzf {input_dir}/gsMap_resource.tar.gz -C {input_dir}')
    run(f'rm {input_dir}/gsMap_resource.tar.gz')

adata_curio = sc.read_h5ad(
    f'{workdir}/output/data/adata_query_curio_final.h5ad')
adata_merfish = sc.read_h5ad(
    f'{workdir}/output/data/adata_query_merfish_final.h5ad')
for adata in [adata_curio, adata_merfish]:
    for col in ['class', 'subclass']:
        adata.obs[col] = adata.obs[col].astype(str)\
            .str.extract(r'^(\d+)\s+(.*)', expand=False)[1]
common_cell_types = (
    set(adata_curio.obs[
        adata_curio.obs['subclass_keep']]['subclass'])
    & set(adata_merfish.obs[
        adata_merfish.obs['subclass_keep']]['subclass']))
del adata_curio, adata_merfish

adata = sc.read_h5ad(f'{workdir}/output/data/adata_query_curio_final.h5ad')
adata.obsm['spatial'] = adata.obs[['x_ffd', 'y_ffd']].to_numpy()

for col in ['class', 'subclass']:
    adata.obs[col] = adata.obs[col].astype(str)\
        .str.extract(r'^(\d+)\s+(.*)', expand=False)[1]
    adata.obs['type'] = adata.obs['subclass']\
        .astype(str).str.extract(r'(\w+)$', expand=False)
    adata.obs['type'] = adata.obs['type'].replace({'IMN': 'Gaba'})
    adata.obs['type'] = adata.obs['type'].replace({'Chol': 'Gaba'})

all_sample_names = adata.obs['sample'].unique()
for sample_name in all_sample_names:
    if not os.path.exists(f'{input_dir}/ST/{sample_name}.h5ad'):
        os.makedirs(f'{input_dir}/ST', exist_ok=True)
        adata_sample = adata[adata.obs['sample'] == sample_name].copy()
        adata_sample.write_h5ad(f'{input_dir}/ST/{sample_name}.h5ad')

gwas_formatted_dir = f'{input_dir}/GWAS_formatted'
os.makedirs(gwas_formatted_dir, exist_ok=True)

for f in os.listdir(f'{input_dir}/GWAS'):
    if f.endswith('.sumstats.gz'):
        basename = f.replace('.sumstats.gz', '')
        if not os.path.exists(f'{gwas_formatted_dir}/{basename}.sumstats.gz'):
            run(f'''
                gsmap format_sumstats \
                    --sumstats '{input_dir}/GWAS/{f}' \
                    --out '{gwas_formatted_dir}/{basename}'
            ''')

with open(f'{gwas_formatted_dir}/gwas_config.yaml', 'w') as f:
    for gwas_file in sorted(os.listdir(gwas_formatted_dir)):
        if gwas_file.endswith('.sumstats.gz'):
            trait = gwas_file.replace('.sumstats.gz', '')
            path = os.path.abspath(f'{gwas_formatted_dir}/{gwas_file}')
            f.write(f'{trait}: {path}\n')

conditions = {
    'control': [s for s in all_sample_names if 'CTRL' in s],
    'pregnant': [s for s in all_sample_names if 'PREG' in s],
    'postpartum': [s for s in all_sample_names if 'POSTPART' in s]
}

#endregion

#region run gsmap ##############################################################

for condition, sample_names in conditions.items():
    slice_mean_file = f'{output_dir}/{condition}_slice_mean.parquet'
    h5ad_paths = ' '.join([f'{input_dir}/ST/{name}.h5ad' for name in sample_names])
    sample_list_str = ' '.join(sample_names)
    if not os.path.exists(slice_mean_file):
        run(f'''
            gsmap create_slice_mean \
                --sample_name_list {sample_list_str} \
                --h5ad_list {h5ad_paths} \
                --slice_mean_output_file {slice_mean_file} \
                --data_layer 'counts' \
                --homolog_file '{input_dir}/gsMap_resource/homologs/mouse_human_homologs.txt'
        ''')
    for sample_name in sample_names:
        if not os.path.exists(f'{output_dir}/{sample_name}/report'):
            run(f'''
                gsmap quick_mode \
                    --workdir '{output_dir}' \
                    --homolog_file '{input_dir}/gsMap_resource/homologs/mouse_human_homologs.txt' \
                    --sample_name '{sample_name}' \
                    --gsMap_resource_dir '{input_dir}/gsMap_resource' \
                    --hdf5_path '{input_dir}/ST/{sample_name}.h5ad' \
                    --annotation 'subclass' \
                    --data_layer 'counts' \
                    --sumstats_config_file '{input_dir}/GWAS_formatted/gwas_config.yaml' \
                    --gM_slices '{slice_mean_file}'
                ''')

traits = []
for gwas_file in sorted(os.listdir(gwas_formatted_dir)):
    if gwas_file.endswith('.sumstats.gz'):
        trait = gwas_file.replace('.sumstats.gz', '')
        traits.append(trait)

for sample_name in all_sample_names:
    for trait_name in traits:
        cauchy_type_dir = f'{output_dir}/{sample_name}/cauchy_combination_type'
        os.makedirs(cauchy_type_dir, exist_ok=True)
        output_file = f'{cauchy_type_dir}/{sample_name}_{trait_name}.Cauchy.csv.gz'
        if os.path.exists(output_file):
            continue
        run(f'''
            gsmap run_cauchy_combination \
                --workdir '{output_dir}' \
                --sample_name '{sample_name}' \
                --trait_name '{trait_name}' \
                --annotation 'type' \
                --output_file '{output_file}'
        ''')

#endregion

#region analysis ###############################################################

traits = ['MDD', 'Neuroticism', 'ADHD', 'Autism', 'PTSD']

plot_trait_ranking(output_dir, figures_dir, conditions)

plot_gwas_heatmap(adata, output_dir, figures_dir, conditions, traits)

results = analyze_trait_associations(output_dir, conditions, ['MDD'])

results.to_csv(f'{figures_dir}/MDD_association_summary.csv', index=False)
print(f'\nKruskal-Wallis test results:')
kruskal_results = results[['trait', 'annotation', 'kruskal_p']]\
    .drop_duplicates().sort_values('kruskal_p')
print(kruskal_results.to_string())
print(f'\nSignificant pairwise comparisons (p_adj < 0.05):')
sig_results = results[results['p_adj'] < 0.05]\
    .sort_values('p_adj')[[
        'annotation', 'comparison', 'logfc', 'p_value', 'p_adj']]
print(sig_results.to_string())

plot_trait_boxplots(
    results, output_dir, conditions, figures_dir, ['MDD'], 
    cell_types_to_plot=[
        'SI-MPO-LPO Lhx8 Gaba', 'MPO-ADP Lhx8 Gaba', 'NDB-SI-MA-STRv Lhx8 Gaba',
        'STR Prox1 Lhx6 Gaba', 'LSX Nkx2-1 Gaba', 'LSX Prdm12 Zeb2 Gaba'
    ],
    p_threshold=0.01
)

plot_spatial_gwas(
    output_dir=output_dir,
    figures_dir=figures_dir,
    conditions=conditions,
    trait='MDD',
    cell_types=[
        'SI-MPO-LPO Lhx8 Gaba', 'MPO-ADP Lhx8 Gaba',
        'STR Prox1 Lhx6 Gaba', 
        'LSX Nkx2-1 Gaba', 'LSX Prdm12 Zeb2 Gaba'
    ],
    bg_point_size=3,
    fg_point_size=12
)

#endregion