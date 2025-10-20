import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'
os.makedirs(f'{working_dir}/figures/merfish/neuropeptides', exist_ok=True)

#region functions ##############################################################

def _insert_gap(arr, indices, axis=0):
    if isinstance(arr, pd.DataFrame):
        if axis == 1:
            arr = arr.T
        for i in sorted(indices, reverse=True):
            arr = pd.concat([
                arr.iloc[:i],
                pd.DataFrame([[np.nan]*arr.shape[1]], columns=arr.columns),
                arr.iloc[i:]
            ]).reset_index(drop=True)
        if axis == 1:
            arr = arr.T
    elif isinstance(arr, pd.Series):
        val = ''
        if arr.dtype != object and arr.dtype.kind not in ['U', 'S']:
            val = np.nan
        for i in sorted(indices, reverse=True):
            arr_list = arr.tolist()
            arr_list.insert(i, val)
            arr = pd.Series(arr_list, dtype=arr.dtype)
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

def compute_de_and_frac(adata, genes, cond1, cond2):
    pct_change = {}
    fracs = {}
    for ct in sorted(adata.obs[cell_type_col].unique()):
        sub = adata[adata.obs[cell_type_col] == ct].copy()
        
        sub = sub[sub.obs['condition'].isin([cond1, cond2])].copy()
        if len(sub) == 0:
            continue
        
        grp1 = sub[sub.obs['condition'] == cond1]
        grp2 = sub[sub.obs['condition'] == cond2]
        
        mean1 = np.asarray(grp1[:, genes].X.mean(0)).flatten()
        mean2 = np.asarray(grp2[:, genes].X.mean(0)).flatten()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            pct = (mean1 - mean2) / mean2 * 100
            pct = np.where(np.isfinite(pct), pct, 0)
        pct_change[ct] = pct
        fracs[ct] = np.asarray((sub[:, genes].X > 0).mean(0)).flatten()
    return pd.DataFrame(pct_change, index=genes).fillna(0), \
           pd.DataFrame(fracs, index=genes).fillna(0)

def plot_split_dotplot(ax, de_df1, frac_df1, de_df2, frac_df2, 
                       adata1, adata2, gene_order=None, col_order=None, 
                       min_size=50, max_size=350, min_frac=0.05, 
                       min_pct_change=15, gene_groups=None, 
                       gene_min_frac=None):
    type_info = adata1.obs[[cell_type_col, 'type']]\
        .drop_duplicates().set_index(cell_type_col)
    obs_with_prefix = adata1.obs[[cell_type_col]].copy()
    obs_with_prefix['num_prefix'] = obs_with_prefix[cell_type_col]\
        .str.extract(r'^(\d+)', expand=False)
    num_prefix_map = obs_with_prefix.groupby(cell_type_col)['num_prefix']\
        .first()
    num_prefix_map = pd.to_numeric(num_prefix_map, errors='coerce')\
        .fillna(999).astype(int)
    type_info['num_prefix'] = num_prefix_map
    
    type_order = ['Glut', 'Gaba', 'NN']
    all_cols = sorted(set(de_df1.columns) | set(de_df2.columns))
    col_with_type = pd.Series(all_cols).to_frame(name=0)\
        .set_index(0).join(type_info)
    col_with_type['type'] = pd.Categorical(
        col_with_type['type'], categories=type_order, ordered=True
    )
    
    if gene_order is None and gene_groups is not None:
        all_genes = sorted(set(de_df1.index) | set(de_df2.index))
        ordered_genes = []
        for grp_name, grp_genes in gene_groups.items():
            grp_genes_present = sorted([g for g in grp_genes 
                                       if g in all_genes])
            ordered_genes.extend(grp_genes_present)
    elif gene_order is None:
        all_genes = sorted(set(de_df1.index) | set(de_df2.index))
        combined_frac = pd.concat([
            frac_df1.reindex(index=all_genes, columns=all_cols).fillna(0),
            frac_df2.reindex(index=all_genes, columns=all_cols).fillna(0)
        ], axis=1).fillna(0)
        mean_expr = combined_frac.mean(axis=1).sort_values(ascending=False)
        ordered_genes = mean_expr.index.tolist()
    else:
        ordered_genes = gene_order
    
    if col_order is None:
        ordered_cols = []
        for type_name in type_order:
            type_cols = col_with_type[col_with_type['type'] == type_name]\
                .sort_values('num_prefix').index.tolist()
            ordered_cols.extend(type_cols)
    else:
        ordered_cols = [c for c in col_order if c in all_cols]
    
    col_boundaries = col_with_type.reindex(ordered_cols)['type']\
        .ne(col_with_type.reindex(ordered_cols)['type'].shift()).cumsum()
    col_gaps = np.where(col_boundaries.diff() > 0)[0]
    
    if gene_groups is not None:
        gene_to_group = {}
        for grp_name, grp_genes in gene_groups.items():
            for g in grp_genes:
                if g in ordered_genes:
                    gene_to_group[g] = grp_name
        gene_group_series = pd.Series([gene_to_group.get(g, 'Unknown') 
                                      for g in ordered_genes])
        row_boundaries = gene_group_series.ne(gene_group_series.shift())\
            .cumsum()
        row_gaps = np.where(row_boundaries.diff() > 0)[0]
    else:
        row_gaps = []
    
    de_df1_aligned = de_df1.reindex(index=ordered_genes, 
                                    columns=ordered_cols).fillna(0)
    frac_df1_aligned = frac_df1.reindex(index=ordered_genes, 
                                       columns=ordered_cols).fillna(0)
    de_df2_aligned = de_df2.reindex(index=ordered_genes, 
                                    columns=ordered_cols).fillna(0)
    frac_df2_aligned = frac_df2.reindex(index=ordered_genes, 
                                       columns=ordered_cols).fillna(0)
    
    plot_de1 = _insert_gap(de_df1_aligned.copy(), col_gaps, axis=1)
    plot_frac1 = _insert_gap(frac_df1_aligned.copy(), col_gaps, axis=1)
    plot_de2 = _insert_gap(de_df2_aligned.copy(), col_gaps, axis=1)
    plot_frac2 = _insert_gap(frac_df2_aligned.copy(), col_gaps, axis=1)
    
    plot_de1 = _insert_gap(plot_de1, row_gaps, axis=0)
    plot_frac1 = _insert_gap(plot_frac1, row_gaps, axis=0)
    plot_de2 = _insert_gap(plot_de2, row_gaps, axis=0)
    plot_frac2 = _insert_gap(plot_frac2, row_gaps, axis=0)
    
    cols_list = ordered_cols.copy()
    for gap_idx in sorted(col_gaps, reverse=True):
        cols_list.insert(gap_idx, '')
    cols_gapped = cols_list
    
    genes_list = ordered_genes.copy()
    for gap_idx in sorted(row_gaps, reverse=True):
        genes_list.insert(gap_idx, '')
    genes_gapped = genes_list
    
    all_vals = np.concatenate([
        de_df1.values.flatten(), de_df2.values.flatten()
    ])
    all_vals = all_vals[~np.isnan(all_vals)]
    ql, qt = np.percentile(all_vals, [5, 95])
    vmax = max(abs(ql), abs(qt))
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    
    if gene_min_frac is None:
        gene_min_frac = {}
    
    row_spacing = 2.5
    for i in range(len(plot_de1)):
        for j in range(len(plot_de1.columns)):
            if pd.isna(plot_de1.iloc[i, j]) and pd.isna(plot_de2.iloc[i, j]):
                continue
                
            row1 = i * row_spacing
            row2 = i * row_spacing + 1
            
            score1 = plot_de1.iloc[i, j]
            frac1 = plot_frac1.iloc[i, j]
            score2 = plot_de2.iloc[i, j]
            frac2 = plot_frac2.iloc[i, j]
            
            gene_name = genes_gapped[i]
            gene_thresh = gene_min_frac.get(gene_name, min_frac)
            
            if (not pd.isna(score1) and frac1 > gene_thresh and 
                    abs(score1) > min_pct_change):
                size1 = min_size + frac1 * (max_size - min_size)
                color1 = plt.cm.seismic(norm(score1))
                ax.scatter(j, row1, s=size1, c=[color1], linewidth=0.5,
                          edgecolors='black', zorder=3)
            
            if (not pd.isna(score2) and frac2 > gene_thresh and 
                    abs(score2) > min_pct_change):
                size2 = min_size + frac2 * (max_size - min_size)
                color2 = plt.cm.seismic(norm(score2))
                ax.scatter(j, row2, s=size2, c=[color2], linewidth=0.5,
                          edgecolors='black', zorder=3)
    
    total_rows = len(plot_de1) * row_spacing
    row_seg = _segments(~plot_de1.isna().all(1))
    col_seg = _segments(~plot_de1.isna().all(0))
    
    for ridx, (r0, r1) in enumerate(row_seg):
        for cidx, (c0, c1) in enumerate(col_seg):
            gene_count = 0
            for i in range(r0, r1):
                if genes_gapped[i] != '':
                    if gene_count % 2 == 0:
                        ax.add_patch(patches.Rectangle(
                            (c0 - 0.5, i * row_spacing - 0.5), 
                            c1 - c0, 2,
                            fill=True, facecolor='#f9f9f9', edgecolor='none',
                            zorder=0
                        ))
                    gene_count += 1
            
            for j in range(c0, c1):
                for i in range(r0, r1):
                    if genes_gapped[i] != '':
                        ax.plot([j, j], 
                               [i * row_spacing - 0.5, i * row_spacing + 1.5], 
                               color='lightgray', linestyle='-', linewidth=0.5,
                               zorder=1)
            
            last_gene_idx = -1
            for i in range(r0, r1):
                if genes_gapped[i] != '':
                    last_gene_idx = i
            if last_gene_idx >= 0:
                ax.add_patch(patches.Rectangle(
                    (c0 - 0.5, r0 * row_spacing - 0.5), 
                    c1 - c0, (last_gene_idx - r0) * row_spacing + 2,
                    fill=False, edgecolor='black', linewidth=0.8, zorder=2
                ))
    
    ax.set_xlim(-0.5, len(cols_gapped) - 0.5)
    ax.set_ylim(total_rows - 0.5, -0.5)
    ax.set_xticks(range(len(cols_gapped)))
    ax.set_xticklabels(cols_gapped, rotation=45, ha='right', fontsize=8)
    
    yticks = [i * row_spacing + 0.5 for i in range(len(genes_gapped))]
    ax.set_yticks(yticks)
    ax.set_yticklabels(genes_gapped, fontsize=8)
    ax.tick_params(length=0)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    return norm, ordered_genes, ordered_cols


def compute_de_and_frac_by_region(adata, genes, cond1, cond2):
    pct_change = {}
    fracs = {}
    for reg in sorted(adata.obs['parcellation_division'].unique()):
        sub = adata[adata.obs['parcellation_division'] == reg].copy()
        
        sub = sub[sub.obs['condition'].isin([cond1, cond2])].copy()
        if len(sub) == 0:
            continue
        
        grp1 = sub[sub.obs['condition'] == cond1]
        grp2 = sub[sub.obs['condition'] == cond2]
        
        if len(grp1) == 0 or len(grp2) == 0:
            continue
            
        mean1 = np.asarray(grp1[:, genes].X.mean(0)).flatten()
        mean2 = np.asarray(grp2[:, genes].X.mean(0)).flatten()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            pct = (mean1 - mean2) / mean2 * 100
            pct = np.where(np.isfinite(pct), pct, 0)
        pct_change[reg] = pct
        fracs[reg] = np.asarray((sub[:, genes].X > 0).mean(0)).flatten()
    return pd.DataFrame(pct_change, index=genes).fillna(0), \
           pd.DataFrame(fracs, index=genes).fillna(0)

def plot_square_heatmap(ax, de_df, frac_df, norm, gene_order, 
                       min_frac=0.05, square_ratio=0.8, gene_groups=None,
                       gene_min_frac=None):
    ordered_genes = [g for g in gene_order if g in de_df.index]
    
    region_mapping = {
        'fiber tracts ': 'fiber tracts',
        'ventricular systems': 'ventricular systems'
    }
    
    de_df.columns = [region_mapping.get(c, c) for c in de_df.columns]
    frac_df.columns = [region_mapping.get(c, c) for c in frac_df.columns]
    
    region_order = ['Isocortex', 'STR', 'PAL', 'OLF', 'HY', 
                   'fiber tracts', 'ventricular systems']
    ordered_cols = [r for r in region_order if r in de_df.columns]
    
    de_df_aligned = de_df.reindex(index=ordered_genes, 
                                 columns=ordered_cols).fillna(0)
    frac_df_aligned = frac_df.reindex(index=ordered_genes, 
                                     columns=ordered_cols).fillna(0)
    
    if gene_groups is not None:
        gene_to_group = {}
        for grp_name, grp_genes in gene_groups.items():
            for g in grp_genes:
                if g in ordered_genes:
                    gene_to_group[g] = grp_name
        gene_group_series = pd.Series([gene_to_group.get(g, 'Unknown') 
                                      for g in ordered_genes])
        row_boundaries = gene_group_series.ne(gene_group_series.shift())\
            .cumsum()
        row_gaps = np.where(row_boundaries.diff() > 0)[0]
    else:
        row_gaps = []
    
    plot_de = _insert_gap(de_df_aligned.copy(), [], axis=1)
    plot_frac = _insert_gap(frac_df_aligned.copy(), [], axis=1)
    plot_de = _insert_gap(plot_de, row_gaps, axis=0)
    plot_frac = _insert_gap(plot_frac, row_gaps, axis=0)
    
    genes_list = ordered_genes.copy()
    for gap_idx in sorted(row_gaps, reverse=True):
        genes_list.insert(gap_idx, '')
    
    n_cols = len(ordered_cols)
    row_spacing = 2.5
    cell_size = 2.5
    
    if gene_min_frac is None:
        gene_min_frac = {}
    
    for i in range(len(plot_de)):
        if pd.isna(plot_de.iloc[i]).all():
            continue
        
        y_pos = i * row_spacing
        gene_name = genes_list[i]
        gene_thresh = gene_min_frac.get(gene_name, min_frac)
        
        for j in range(n_cols):
            x_pos = j * cell_size
            
            rect = patches.Rectangle(
                (x_pos, y_pos), cell_size, cell_size, 
                fill=True, facecolor='white', 
                edgecolor='lightgray', linewidth=0.5)
            ax.add_patch(rect)
            
            score = plot_de.iloc[i, j]
            frac = plot_frac.iloc[i, j]
            
            if frac > gene_thresh:
                inner_size = square_ratio * np.sqrt(frac) * cell_size
                offset = (cell_size - inner_size) / 2
                
                square = patches.Rectangle(
                    (x_pos + offset, y_pos + offset),
                    inner_size, inner_size,
                    fill=True, facecolor=plt.cm.seismic(norm(score)),
                    edgecolor='none'
                )
                ax.add_patch(square)
    
    row_seg = _segments(~plot_de.isna().all(1))
    
    for r0, r1 in row_seg:
        y_start = r0 * row_spacing
        y_end = r1 * row_spacing
        ax.add_patch(patches.Rectangle(
            (0, y_start), n_cols * cell_size, y_end - y_start,
            fill=False, edgecolor='black', linewidth=0.8
        ))
    
    total_rows = len(plot_de) * row_spacing
    ax.set_xlim(0, n_cols * cell_size)
    ax.set_ylim(total_rows, 0)
    ax.set_xticks(np.arange(n_cols) * cell_size + cell_size/2)
    ax.set_xticklabels(ordered_cols, rotation=45, ha='right', fontsize=8)
    
    yticks = [i * row_spacing + cell_size/2 for i in range(len(genes_list))]
    ax.set_yticks(yticks)
    ax.set_yticklabels(genes_list, fontsize=8)
    ax.tick_params(length=0)
    
    for spine in ax.spines.values():
        spine.set_visible(False)

def plot_single_tech_dotplot(ax, de_df, frac_df, adata, gene_order=None, 
                            col_order=None, min_size=50, max_size=350, 
                            min_frac=0.05, min_pct_change=15, 
                            gene_groups=None, gene_min_frac=None):
    type_info = adata.obs[[cell_type_col, 'type']]\
        .drop_duplicates().set_index(cell_type_col)
    obs_with_prefix = adata.obs[[cell_type_col]].copy()
    obs_with_prefix['num_prefix'] = obs_with_prefix[cell_type_col]\
        .str.extract(r'^(\d+)', expand=False)
    num_prefix_map = obs_with_prefix.groupby(cell_type_col)['num_prefix']\
        .first()
    num_prefix_map = pd.to_numeric(num_prefix_map, errors='coerce')\
        .fillna(999).astype(int)
    type_info['num_prefix'] = num_prefix_map
    
    type_order = ['Glut', 'Gaba', 'NN']
    all_cols = sorted(de_df.columns)
    col_with_type = pd.Series(all_cols).to_frame(name=0)\
        .set_index(0).join(type_info)
    col_with_type['type'] = pd.Categorical(
        col_with_type['type'], categories=type_order, ordered=True
    )
    
    if gene_order is None and gene_groups is not None:
        all_genes = sorted(de_df.index)
        ordered_genes = []
        for grp_name, grp_genes in gene_groups.items():
            grp_genes_present = sorted([g for g in grp_genes 
                                       if g in all_genes])
            ordered_genes.extend(grp_genes_present)
    elif gene_order is None:
        ordered_genes = de_df.index.tolist()
    else:
        ordered_genes = gene_order
    
    if col_order is None:
        ordered_cols = []
        for type_name in type_order:
            type_cols = col_with_type[col_with_type['type'] == type_name]\
                .sort_values('num_prefix').index.tolist()
            ordered_cols.extend(type_cols)
    else:
        ordered_cols = [c for c in col_order if c in all_cols]
    
    col_boundaries = col_with_type.reindex(ordered_cols)['type']\
        .ne(col_with_type.reindex(ordered_cols)['type'].shift()).cumsum()
    col_gaps = np.where(col_boundaries.diff() > 0)[0]
    
    if gene_groups is not None:
        gene_to_group = {}
        for grp_name, grp_genes in gene_groups.items():
            for g in grp_genes:
                if g in ordered_genes:
                    gene_to_group[g] = grp_name
        gene_group_series = pd.Series([gene_to_group.get(g, 'Unknown') 
                                      for g in ordered_genes])
        row_boundaries = gene_group_series.ne(gene_group_series.shift())\
            .cumsum()
        row_gaps = np.where(row_boundaries.diff() > 0)[0]
    else:
        row_gaps = []
    
    de_df_aligned = de_df.reindex(index=ordered_genes, 
                                 columns=ordered_cols).fillna(0)
    frac_df_aligned = frac_df.reindex(index=ordered_genes, 
                                     columns=ordered_cols).fillna(0)
    
    plot_de = _insert_gap(de_df_aligned.copy(), col_gaps, axis=1)
    plot_frac = _insert_gap(frac_df_aligned.copy(), col_gaps, axis=1)
    
    plot_de = _insert_gap(plot_de, row_gaps, axis=0)
    plot_frac = _insert_gap(plot_frac, row_gaps, axis=0)
    
    cols_list = ordered_cols.copy()
    for gap_idx in sorted(col_gaps, reverse=True):
        cols_list.insert(gap_idx, '')
    cols_gapped = cols_list
    
    genes_list = ordered_genes.copy()
    for gap_idx in sorted(row_gaps, reverse=True):
        genes_list.insert(gap_idx, '')
    genes_gapped = genes_list
    
    all_vals = de_df.values.flatten()
    all_vals = all_vals[~np.isnan(all_vals)]
    ql, qt = np.percentile(all_vals, [5, 95])
    vmax = max(abs(ql), abs(qt))
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    
    if gene_min_frac is None:
        gene_min_frac = {}
    
    for i in range(len(plot_de)):
        for j in range(len(plot_de.columns)):
            if pd.isna(plot_de.iloc[i, j]):
                continue
                
            score = plot_de.iloc[i, j]
            frac = plot_frac.iloc[i, j]
            
            gene_name = genes_gapped[i]
            gene_thresh = gene_min_frac.get(gene_name, min_frac)
            
            if (not pd.isna(score) and frac > gene_thresh and 
                    abs(score) > min_pct_change):
                size = min_size + frac * (max_size - min_size)
                color = plt.cm.seismic(norm(score))
                ax.scatter(j, i, s=size, c=[color], linewidth=0.5,
                          edgecolors='black', zorder=3)
    
    total_rows = len(plot_de)
    row_seg = _segments(~plot_de.isna().all(1))
    col_seg = _segments(~plot_de.isna().all(0))
    
    for ridx, (r0, r1) in enumerate(row_seg):
        for cidx, (c0, c1) in enumerate(col_seg):
            for i in range(r0, r1):
                if genes_gapped[i] != '':
                    ax.plot([c0 - 0.5, c1 - 0.5], [i, i], color='lightgray',
                           linestyle='-', linewidth=0.5, zorder=1)
            for j in range(c0, c1):
                ax.plot([j, j], [r0 - 0.5, r1 - 0.5], color='lightgray',
                       linestyle='-', linewidth=0.5, zorder=1)
            
            ax.add_patch(patches.Rectangle(
                (c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0,
                fill=False, edgecolor='black', linewidth=0.8, zorder=2
            ))
    
    ax.set_xlim(-0.5, len(cols_gapped) - 0.5)
    ax.set_ylim(total_rows - 0.5, -0.5)
    ax.set_xticks(range(len(cols_gapped)))
    ax.set_xticklabels(cols_gapped, rotation=45, ha='right', fontsize=8)
    
    ax.set_yticks(range(len(genes_gapped)))
    ax.set_yticklabels(genes_gapped, fontsize=8)
    ax.tick_params(length=0)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    return norm, ordered_genes, ordered_cols

#endregion 

#region load data ##############################################################

cell_type_col = 'subclass'

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

sc.pp.normalize_total(adata_curio, target_sum=1e4)
sc.pp.normalize_total(adata_merfish, target_sum=1e4)
sc.pp.log1p(adata_curio)
sc.pp.log1p(adata_merfish)

gene_groups = {
    'Hormonal & Reproductive': [
        # Core HPG Axis & Reproductive Peptides
        'Kiss1r', 'Oxtr',
        # Steroid & Peptide Hormone Receptors
        'Esr1', 'Esr2', 'Pgr', 'Prlr', 'Prl',
        # Stress & Metabolic Hormones
        'Crh', 'Nr3c1', 'Nr3c2', 'Lepr',
        # Other Reproductive Signaling
        'Inhba', 'Inhbb', 'Tac2', 'Tacr3'
    ],
    'Neuropeptides & Neuromodulators': [
        # Somatostatin System
        'Sst', 'Sstr2',
        # Galanin System
        'Gal', 'Galr1',
        # Other Key Neuropeptides
        'Nts', 'Cartpt', 
        # Serotonin System
        'Htr2a',
        # Calcitonin/CGRP System
        'Calcr'
    ],
    'Neurotrophic & Growth Factors': [
        # Neurotrophins & Receptors
        'Ngf', 'Bdnf', 'Ntf3', 'Ntrk2',
        # Insulin-like Growth Factors
        'Igf1', 'Igf2',
        # Fibroblast Growth Factors
        'Fgf1', 'Fgf2', 'Fgf13', 'Fgf14',
        # TGF-beta Superfamily
        'Tgfb1', 'Tgfb2', 'Tnfsf12',
        # Other Growth Factors
        'Ptn', 'Vegfa'
    ]
}

#region plot merfish dotplot ###################################################

contrasts = [('PREG', 'CTRL'), ('POSTPART', 'PREG')]
genes = [g for grp in gene_groups.values() for g in grp]
genes_m = [g for g in genes if g in adata_merfish.var_names]

gene_min_frac = {'Oxtr': 0.01}
min_size_m, max_size_m = 10, 200
min_frac_m = 0.03
min_pct_change_m = 30

min_frac_region = 0
square_ratio_region = 0.9

n_regions = len(adata_merfish.obs['parcellation_division'].unique())
cell_size = 2.5
square_plot_width = n_regions * cell_size

fig = plt.figure(figsize=(12, 16))
gs = fig.add_gridspec(
    2, 2, width_ratios=[10, square_plot_width/10], 
    height_ratios=[1, 1], hspace=0.3, wspace=0.18,
    left=0.13, right=0.95, top=0.98, bottom=0.04)
axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(2)] 
                 for i in range(2)])

norms = []
gene_orders = []
for idx, (cond1, cond2) in enumerate(contrasts):
    ax_dot = axes[idx, 0]
    ax_square = axes[idx, 1]
    
    de_m, frac_m = compute_de_and_frac(adata_merfish, genes_m, cond1, 
                                       cond2)
    de_reg, frac_reg = compute_de_and_frac_by_region(
        adata_merfish, genes_m, cond1, cond2)
    
    norm, gene_order, col_order = plot_single_tech_dotplot(
        ax_dot, de_m, frac_m, adata_merfish, min_size=min_size_m, 
        max_size=max_size_m, min_frac=min_frac_m, 
        min_pct_change=min_pct_change_m, gene_groups=gene_groups,
        gene_min_frac=gene_min_frac
    )
    norms.append(norm)
    gene_orders.append(gene_order)
    
    plot_square_heatmap(ax_square, de_reg, frac_reg, norm, gene_order,
                       min_frac=min_frac_region, 
                       square_ratio=square_ratio_region, 
                       gene_groups=gene_groups,
                       gene_min_frac=gene_min_frac)
    
    title_map = {
        ('PREG', 'CTRL'): 'Pregnant vs Nulliparous',
        ('POSTPART', 'PREG'): 'Postpartum vs Pregnant'
    }
    ax_dot.set_title(title_map.get((cond1, cond2), f'{cond1} vs {cond2}'), 
                    fontsize=12, pad=10)

vmax = max(norms[0].vmax, norms[1].vmax)
norm_shared = plt.Normalize(vmin=-vmax, vmax=vmax)

sm = plt.cm.ScalarMappable(cmap=plt.cm.seismic, norm=norm_shared)
cbar_ax = fig.add_axes([0.04, 0.05, 0.012, 0.08])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Expression\n(% change)', fontsize=7, rotation=0, 
              labelpad=5, ha='center')
cbar.ax.yaxis.set_label_coords(0.5, -0.15)
cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.tick_params(labelsize=6)

legend_ax = fig.add_axes([0.005, 0.15, 0.08, 0.10])
legend_ax.axis('off')
legend_ax.text(0.5, 1.0, 'Fraction of cells\nexpressing gene', 
               ha='center', va='bottom', fontsize=7, 
               transform=legend_ax.transAxes)
sizes = [0.05, 0.5, 1.0]
y_positions = [0.05, 0.35, 0.65]
for i, (s, y) in enumerate(zip(sizes, y_positions)):
    dot_size = min_size_m + s * (max_size_m - min_size_m)
    legend_ax.scatter(0.65, y, s=dot_size, c='#cccccc',
                     edgecolors='black', linewidth=0.5)
    percent_text = '5%' if s == 0.05 else f'{int(s*100)}%'
    legend_ax.text(0.4, y, percent_text, va='center', ha='right',
                  fontsize=6)
legend_ax.set_xlim(0, 1)
legend_ax.set_ylim(0, 0.9)

plt.savefig(f'{working_dir}/figures/neuropeptide_merfish_dotplot.png', 
            dpi=300, bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/neuropeptide_merfish_dotplot.svg',
            format='svg', bbox_inches='tight')
plt.close()

#endregion 

#region plot dotplot comparison ################################################

gene_min_frac = {}
min_size, max_size = 10, 100
min_frac = 0.05
min_pct_change = 0

genes = []
for grp_genes in gene_groups.values():
    genes.extend(grp_genes)

genes_c = [g for g in genes if g in adata_curio.var_names]
genes_m = [g for g in genes if g in adata_merfish.var_names]

contrasts = [
    ('PREG', 'CTRL'),
    ('POSTPART', 'PREG')
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 18))
fig.subplots_adjust(left=0.20, right=0.92, top=0.98, bottom=0.04, 
                   hspace=0.3)

norms = []
for idx, (cond1, cond2) in enumerate(contrasts):
    ax = ax1 if idx == 0 else ax2
    
    de_c, frac_c = compute_de_and_frac(adata_curio, genes_c, cond1, 
                                       cond2)
    de_m, frac_m = compute_de_and_frac(adata_merfish, genes_m, cond1, 
                                       cond2)
    
    norm, _, col_order = plot_split_dotplot(
        ax, de_c, frac_c, de_m, frac_m, adata_curio, adata_merfish,
        min_size=min_size, max_size=max_size, min_frac=min_frac, 
        min_pct_change=min_pct_change, gene_groups=gene_groups,
        gene_min_frac=gene_min_frac
    )
    norms.append(norm)
    
    title_map = {
        ('PREG', 'CTRL'): 'Pregnant vs Nulliparous',
        ('POSTPART', 'PREG'): 'Postpartum vs Pregnant'
    }
    ax.set_title(title_map.get((cond1, cond2), f'{cond1} vs {cond2}'), 
                fontsize=12, pad=10)

vmax = max(norms[0].vmax, norms[1].vmax)
norm_shared = plt.Normalize(vmin=-vmax, vmax=vmax)

sm = plt.cm.ScalarMappable(cmap=plt.cm.seismic, norm=norm_shared)
cbar_ax = fig.add_axes([0.075, 0.05, 0.015, 0.08])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Expression\n(% change)', fontsize=7, rotation=0, 
              labelpad=5, ha='center')
cbar.ax.yaxis.set_label_coords(0.5, -0.15)
cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.tick_params(labelsize=6)

legend_ax = fig.add_axes([0.025, 0.15, 0.12, 0.10])
legend_ax.axis('off')
legend_ax.text(0.5, 1.0, 'Fraction of cells\nexpressing gene', 
               ha='center', va='bottom', fontsize=7, 
               transform=legend_ax.transAxes)
sizes = [0.05, 0.5, 1.0]
y_positions = [0.05, 0.35, 0.65]
for i, (s, y) in enumerate(zip(sizes, y_positions)):
    dot_size = min_size + s * (max_size - min_size)
    legend_ax.scatter(0.65, y, s=dot_size, c='#cccccc',
                     edgecolors='black', linewidth=0.5)
    percent_text = '5%' if s == 0.05 else f'{int(s*100)}%'
    legend_ax.text(0.4, y, percent_text, va='center', ha='right',
                  fontsize=6)
legend_ax.set_xlim(0, 1)
legend_ax.set_ylim(0, 0.9)

plt.savefig(f'{working_dir}/figures/neuropeptide_dotplot.png', 
            dpi=300, bbox_inches='tight')
plt.close()

#endregion 

#region spatial hemisphere plots ##############################################

gene_sample_config = {
    'Nr3c1': {
        'CTRL': {'sample': 'CTRL2', 'hemisphere': 'L'},
        'PREGL': {'sample': 'PREG2', 'hemisphere': 'L'},
        'PREGR': {'sample': 'PREG2', 'hemisphere': 'R'},
        'POSTPART': {'sample': 'POSTPART1', 'hemisphere': 'L'}
    },
    'Pgr': {
        'CTRL': {'sample': 'CTRL1', 'hemisphere': 'L'},
        'PREGL': {'sample': 'PREG1', 'hemisphere': 'R'},
        'PREGR': {'sample': 'PREG2', 'hemisphere': 'L'},
        'POSTPART': {'sample': 'POSTPART2', 'hemisphere': 'L'}
    },
    'Prlr': {
        'CTRL': {'sample': 'CTRL1', 'hemisphere': 'R'},
        'PREGL': {'sample': 'PREG1', 'hemisphere': 'R'},
        'PREGR': {'sample': 'PREG1', 'hemisphere': 'L'},
        'POSTPART': {'sample': 'POSTPART2', 'hemisphere': 'L'}
    },
    'Gal': {
        'CTRL': {'sample': 'CTRL3', 'hemisphere': 'L'},
        'PREGL': {'sample': 'PREG1', 'hemisphere': 'R'},
        'PREGR': {'sample': 'PREG1', 'hemisphere': 'L'},
        'POSTPART': {'sample': 'POSTPART2', 'hemisphere': 'L'}
    },
    'Sst': {
        'CTRL': {'sample': 'CTRL1', 'hemisphere': 'L'},
        'PREGL': {'sample': 'PREG3', 'hemisphere': 'R'},
        'PREGR': {'sample': 'PREG3', 'hemisphere': 'L'},
        'POSTPART': {'sample': 'POSTPART2', 'hemisphere': 'L'}
    },
    'Fgf1': {
        'CTRL': {'sample': 'CTRL2', 'hemisphere': 'L'},
        'PREGL': {'sample': 'PREG1', 'hemisphere': 'R'},
        'PREGR': {'sample': 'PREG1', 'hemisphere': 'L'},
        'POSTPART': {'sample': 'POSTPART2', 'hemisphere': 'L'}
    },
    'Ntrk2': {
        'CTRL': {'sample': 'CTRL1', 'hemisphere': 'L'},
        'PREGL': {'sample': 'PREG1', 'hemisphere': 'R'},
        'PREGR': {'sample': 'PREG1', 'hemisphere': 'L'},
        'POSTPART': {'sample': 'POSTPART2', 'hemisphere': 'L'}
    }
}

midline = -5.7
n_genes = len(gene_sample_config)
fig, axes = plt.subplots(n_genes, 2, figsize=(10, 3.75 * n_genes))
fig.subplots_adjust(hspace=0)
if n_genes == 1:
    axes = axes.reshape(1, -1)

for gene_idx, (gene, config) in enumerate(gene_sample_config.items()):
    
    for slice_idx, (left_cond, right_cond, title) in enumerate([
        ('CTRL', 'PREGL', 'Nulliparous      Pregnant'), 
        ('PREGR', 'POSTPART', 'Pregnant     Postpartum')]):
        
        ax = axes[gene_idx, slice_idx]
        all_x, all_y, all_expr = [], [], []
        
        for side, cond in [('left', left_cond), ('right', right_cond)]:
            sample = config[cond]['sample']
            hemi = config[cond]['hemisphere']
            
            mask = adata_merfish.obs['sample'] == sample
            sub = adata_merfish[mask]
            
            x_coords = sub.obs['x_ffd'].values
            y_coords = sub.obs['y_ffd'].values
            
            if hemi == 'L':
                hemi_mask = x_coords < midline
            else:
                hemi_mask = x_coords > midline
                
            x_hemi = x_coords[hemi_mask]
            y_hemi = y_coords[hemi_mask]
            
            expr = np.asarray(
                sub[hemi_mask, gene].layers['counts'].toarray()
            ).flatten()
            
            if side == 'left':
                if hemi == 'R':
                    x_hemi = 2 * midline - x_hemi
            else:
                if hemi == 'L':
                    x_hemi = 2 * midline - x_hemi
                    
            all_x.extend(x_hemi)
            all_y.extend(y_hemi)
            all_expr.extend(expr)
        
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        all_expr = np.array(all_expr)
        
        expr_pos = all_expr[all_expr > 0]
        vmax = np.percentile(expr_pos, 95) if len(expr_pos) > 0 else 1
        
        bg_mask = all_expr == 0
        fg_mask = all_expr > 0
        
        ax.scatter(all_x[bg_mask], all_y[bg_mask], c=all_expr[bg_mask],
                  cmap='GnBu', s=1.5, linewidths=0, rasterized=True,
                  vmin=0, vmax=vmax, alpha=0.5)
        
        ax.scatter(all_x[fg_mask], all_y[fg_mask], c=all_expr[fg_mask],
                  cmap='GnBu', s=1.5, linewidths=0, rasterized=True,
                  vmin=0, vmax=vmax)
        
        ax.axvline(midline, color='black', linestyle='--', linewidth=1,
                  alpha=1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        if gene_idx == n_genes - 1:
            ax.text(0.5, -0.05, title, transform=ax.transAxes,
                   fontsize=12, va='top', ha='center')
        
    axes[gene_idx, 0].text(-0.05, 0.5, gene, 
                          transform=axes[gene_idx, 0].transAxes,
                          fontsize=14, va='center', ha='right', rotation=0)

plt.tight_layout()

cbar_ax = fig.add_axes([0.98, 0.08, 0.012, 0.06])
sm = plt.cm.ScalarMappable(cmap='GnBu', 
                           norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Min', 'Max'])
cbar.ax.tick_params(labelsize=9)
cbar_ax.text(0.5, 1.15, 'Expression', transform=cbar_ax.transAxes,
            ha='center', va='bottom', fontsize=10)

plt.savefig(
    f'{working_dir}/figures/neuropeptide_spatial.png',
    dpi=400, bbox_inches='tight')
plt.savefig(
    f'{working_dir}/figures/neuropeptide_spatial.svg',
    format='svg', bbox_inches='tight')
plt.close()

#endregion 

#region scratch ################################################################

all_data = []
for cond1, cond2 in contrasts:
    contrast_name = f'{cond1} vs {cond2}'
    
    de_m, frac_m = compute_de_and_frac(adata_merfish, genes_m, cond1, 
                                       cond2)
    de_reg, frac_reg = compute_de_and_frac_by_region(
        adata_merfish, genes_m, cond1, cond2)
    
    for gene in de_m.index:
        gene_thresh = gene_min_frac.get(gene, min_frac_m)
        for ct in de_m.columns:
            pct = de_m.loc[gene, ct]
            frac = frac_m.loc[gene, ct]
            if frac > gene_thresh and abs(pct) > min_pct_change_m:
                all_data.append({
                    'gene': gene,
                    'comparison': contrast_name,
                    'group_type': 'cell_type',
                    'group': ct,
                    'pct_change': pct,
                    'frac_expressing': frac
                })
    
    for gene in de_reg.index:
        gene_thresh = gene_min_frac.get(gene, min_frac_region)
        for reg in de_reg.columns:
            pct = de_reg.loc[gene, reg]
            frac = frac_reg.loc[gene, reg]
            if frac > gene_thresh:
                all_data.append({
                    'gene': gene,
                    'comparison': contrast_name,
                    'group_type': 'region',
                    'group': reg,
                    'pct_change': pct,
                    'frac_expressing': frac
                })

df_all = pd.DataFrame(all_data)
df_all.to_csv(f'{working_dir}/output/neuropeptide_expression_data.csv', 
              index=False)


samples = ['CTRL1', 'CTRL2', 'CTRL3', 'PREG1', 'PREG2', 'PREG3',
           'POSTPART1', 'POSTPART2', 'POSTPART3']

all_genes = [gene for genes in gene_groups.values() for gene in genes]

for gene in all_genes:
    if gene not in adata_merfish.var_names:
        continue
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(gene, fontsize=16, y=0.995)
    
    for idx, sample in enumerate(samples):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        mask = adata_merfish.obs['sample'] == sample
        sub = adata_merfish[mask]        
        expr = np.asarray(sub[:, gene].X.toarray()).flatten()
        expr_pos = expr[expr > 0]
        vmax = np.percentile(expr_pos, 95) if len(expr_pos) > 0 else 1
        
        ax.scatter(sub.obs['x_ffd'], sub.obs['y_ffd'], c=expr, 
                  cmap='GnBu', s=0.5, linewidths=0, rasterized=True,
                  vmin=0, vmax=vmax)
        ax.set_title(sample, fontsize=10)
        ax.set_aspect('equal')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(
        f'{working_dir}/figures/merfish/neuropeptides/{gene}.png',
        dpi=300, bbox_inches='tight')
    plt.close()

#endregion


