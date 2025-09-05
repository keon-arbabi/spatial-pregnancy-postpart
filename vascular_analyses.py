import warnings
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from scipy.stats import ttest_ind
from matplotlib.ticker import MaxNLocator
from scipy.spatial import KDTree

warnings.filterwarnings('ignore')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400
sc.settings.verbosity = 0

#region functions ##############################################################

def get_density(adata: sc.AnnData, targets: list, type_col: str):
    obs = adata.obs
    coords = obs.groupby('sample')[['x', 'y']].agg(['min', 'max'])
    area = (
        (coords[('x', 'max')] - coords[('x', 'min')]) *
        (coords[('y', 'max')] - coords[('y', 'min')])
    )
    area_mm2 = area * 1e-6
    counts = obs.groupby(['sample', type_col]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=targets, fill_value=0)
    density = counts.div(area_mm2, axis=0)
    meta = obs[['sample', 'condition', 'source']].drop_duplicates()
    return density.reset_index().merge(meta, on='sample')

def calculate_summary_and_stats(density_df: pd.DataFrame):
    summary = density_df.groupby(
        ['source', 'condition', 'Cell Type']
    )['Density (cells / mm^2)'].agg(['mean', 'std', 'count']).reset_index()
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    contrasts = [('CTRL', 'PREG'), ('PREG', 'POSTPART')]
    records = []
    for ct in density_df['Cell Type'].unique():
        for ds in density_df['source'].unique():
            for c1, c2 in contrasts:
                v1 = density_df[
                    (density_df['Cell Type'] == ct) &
                    (density_df['source'] == ds) &
                    (density_df['condition'] == c1)
                ]['Density (cells / mm^2)']
                v2 = density_df[
                    (density_df['Cell Type'] == ct) &
                    (density_df['source'] == ds) &
                    (density_df['condition'] == c2)
                ]['Density (cells / mm^2)']
                if len(v1) > 1 and len(v2) > 1:
                    _, p_val = ttest_ind(v1, v2)
                    records.append({
                        'cell_type': ct, 'dataset': ds,
                        'contrast': f'{c2}_vs_{c1}',
                        't_test_P.Value': p_val
                    })
    stats_df = pd.DataFrame(records)
    return summary, stats_df

def plot_dual_axis_density(
    summary_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    cell_types: List[str],
    condition_order: List[str] = ['CTRL', 'PREG', 'POSTPART'],
    datasets_to_plot: List[str] = ['merfish', 'curio']
) -> plt.Figure:
    
    nrows = len(cell_types)
    ncols = 1
    figsize = (2.4, 2.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=False)
    if nrows == 1: axes = [axes]
    axes = axes.flatten()

    palette = {'merfish': '#4361ee', 'curio': '#4cc9f0'}
    condition_labels = {'CTRL': 'Control', 'PREG': 'Pregnancy',
                        'POSTPART': 'Postpartum'}

    for i, cell_type in enumerate(cell_types):
        ax = axes[i]
        ct_data = summary_df[summary_df['Cell Type'] == cell_type]

        do_merfish = 'merfish' in datasets_to_plot
        do_curio = 'curio' in datasets_to_plot
        
        ax_right = ax.twinx() if do_merfish and do_curio else None
        
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        if ax_right:
            ax_right.yaxis.set_major_locator(MaxNLocator(nbins=4))
        
        merfish_stats, curio_stats = {}, {}

        for dataset_name, color in palette.items():
            if dataset_name not in datasets_to_plot:
                continue

            ds_data = ct_data[
                ct_data['source'] == dataset_name
            ].set_index('condition').reindex(condition_order).reset_index()

            mean_val, std_val = ds_data['mean'].mean(), ds_data['mean'].std()
            if std_val == 0: std_val = 1.0

            if dataset_name == 'merfish':
                merfish_stats = {'mean': mean_val, 'std': std_val}
            else:
                curio_stats = {'mean': mean_val, 'std': std_val}

            ds_data['z_mean'] = (ds_data['mean'] - mean_val) / std_val
            ds_data['z_se'] = ds_data['se'] / std_val

            plot_ax = ax if dataset_name == 'merfish' or not ax_right \
                else ax_right
            plot_ax.errorbar(
                x=ds_data['condition'], y=ds_data['z_mean'],
                yerr=ds_data['z_se'], fmt='o-', color=color, alpha=0.8,
                label=dataset_name, capsize=3, markersize=5,
                linewidth=1.5
            )

            for j in range(len(condition_order) - 1):
                c1, c2 = condition_order[j], condition_order[j+1]
                sig_row = stats_df[
                    (stats_df['cell_type'] == cell_type) &
                    (stats_df['dataset'] == dataset_name) &
                    (stats_df['contrast'] == f'{c2}_vs_{c1}')
                ]
                if sig_row.empty: continue
                
                p_val = sig_row['t_test_P.Value'].iloc[0]
                if p_val < 0.05:
                    x_pos = (j + j + 1) / 2.0
                    y_pos = (ds_data['z_mean'] + ds_data['z_se']).max() + 0.2
                    plot_ax.text(x_pos, y_pos, '*', ha='center',
                                 va='bottom', fontsize=14, color='black',
                                 fontweight='bold')

        z_ticks = ax.get_yticks()
        if do_merfish:
            stats = merfish_stats
            y_labels = [f'{z*stats["std"]+stats["mean"]:.0f}' for z in z_ticks]
            ax.set_yticks(z_ticks, labels=y_labels)
        
        if do_curio:
            stats = curio_stats
            y_labels = [f'{z*stats["std"]+stats["mean"]:.1f}' for z in z_ticks]
            if ax_right:
                ax_right.set_yticks(z_ticks, labels=y_labels)
            else:
                ax.set_yticks(z_ticks, labels=y_labels)

        if not do_merfish and ax_right is None:
             ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

        ax.set_title(cell_type, fontsize=10)

        if i == len(cell_types) - 1:
            ax.set_xticks(range(len(condition_order)))
            ax.set_xticklabels([condition_labels.get(c, c) for c in
                                condition_order], rotation=45, ha='right',
                               fontsize=9)
        else:
            ax.set_xticks([])

        if i == len(cell_types) // 2:
            if do_merfish:
                ax.set_ylabel('MERFISH Density\n(cells / mm²)', fontsize=9)
            if do_curio:
                label_ax = ax_right if ax_right else ax
                label_ax.set_ylabel(
                    'Slide-tags Density\n(cells / mm²)',
                    fontsize=9, rotation=-90, va='bottom'
                )

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.25)
    return fig

def calculate_spatial_uniformity(
    adata: sc.AnnData, targets: list, type_col: str
):
    obs = adata.obs
    records = []
    for sample in obs['sample'].unique():
        sample_obs = obs[obs['sample'] == sample]
        for ct in targets:
            ct_obs = sample_obs[sample_obs[type_col] == ct]
            if len(ct_obs) < 2:
                continue
            coords = ct_obs[['x', 'y']].values
            distances, _ = KDTree(coords).query(coords, k=2)
            if distances.ndim == 1:
                continue
            nn_distances = distances[:, 1]
            m_nnd = np.mean(nn_distances)
            records.append({'sample': sample, 'Cell Type': ct, 'mNND': m_nnd})
    
    uniformity_df = pd.DataFrame(records)
    meta = obs[['sample', 'condition', 'source']].drop_duplicates()
    return uniformity_df.merge(meta, on='sample')

def calculate_mNND_summary_and_stats(uniformity_df: pd.DataFrame):
    summary = uniformity_df.groupby(
        ['source', 'condition', 'Cell Type']
    )['mNND'].agg(['mean', 'std', 'count']).reset_index()
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    contrasts = [('CTRL', 'PREG'), ('PREG', 'POSTPART')]
    records = []
    for ct in uniformity_df['Cell Type'].unique():
        for ds in uniformity_df['source'].unique():
            for c1, c2 in contrasts:
                v1 = uniformity_df[
                    (uniformity_df['Cell Type'] == ct) &
                    (uniformity_df['source'] == ds) &
                    (uniformity_df['condition'] == c1)
                ]['mNND']
                v2 = uniformity_df[
                    (uniformity_df['Cell Type'] == ct) &
                    (uniformity_df['source'] == ds) &
                    (uniformity_df['condition'] == c2)
                ]['mNND']
                if len(v1) > 1 and len(v2) > 1:
                    _, p_val = ttest_ind(v1, v2)
                    records.append({
                        'cell_type': ct, 'dataset': ds,
                        'contrast': f'{c2}_vs_{c1}',
                        't_test_P.Value': p_val
                    })
    stats_df = pd.DataFrame(records)
    return summary, stats_df

def plot_mNND(
    summary_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    cell_types: List[str],
    condition_order: List[str] = ['CTRL', 'PREG', 'POSTPART'],
    datasets_to_plot: List[str] = ['merfish', 'curio']
) -> plt.Figure:
    
    nrows = len(cell_types)
    ncols = 1
    figsize = (2.4, 2.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=False)
    if nrows == 1: axes = [axes]
    axes = axes.flatten()

    palette = {'merfish': '#4361ee', 'curio': '#4cc9f0'}
    condition_labels = {'CTRL': 'Control', 'PREG': 'Pregnancy',
                        'POSTPART': 'Postpartum'}

    for i, cell_type in enumerate(cell_types):
        ax = axes[i]
        ct_data = summary_df[summary_df['Cell Type'] == cell_type]

        do_merfish = 'merfish' in datasets_to_plot
        do_curio = 'curio' in datasets_to_plot
        
        ax_right = ax.twinx() if do_merfish and do_curio else None
        
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        if ax_right:
            ax_right.yaxis.set_major_locator(MaxNLocator(nbins=4))
        
        merfish_stats, curio_stats = {}, {}

        for dataset_name, color in palette.items():
            if dataset_name not in datasets_to_plot:
                continue

            ds_data = ct_data[
                ct_data['source'] == dataset_name
            ].set_index('condition').reindex(condition_order).reset_index()

            mean_val, std_val = ds_data['mean'].mean(), ds_data['mean'].std()
            if std_val == 0: std_val = 1.0

            if dataset_name == 'merfish':
                merfish_stats = {'mean': mean_val, 'std': std_val}
            else:
                curio_stats = {'mean': mean_val, 'std': std_val}

            ds_data['z_mean'] = (ds_data['mean'] - mean_val) / std_val
            ds_data['z_se'] = ds_data['se'] / std_val

            plot_ax = ax if dataset_name == 'merfish' or not ax_right \
                else ax_right
            plot_ax.errorbar(
                x=ds_data['condition'], y=ds_data['z_mean'],
                yerr=ds_data['z_se'], fmt='o-', color=color, alpha=0.8,
                label=dataset_name, capsize=3, markersize=5,
                linewidth=1.5
            )

            for j in range(len(condition_order) - 1):
                c1, c2 = condition_order[j], condition_order[j+1]
                sig_row = stats_df[
                    (stats_df['cell_type'] == cell_type) &
                    (stats_df['dataset'] == dataset_name) &
                    (stats_df['contrast'] == f'{c2}_vs_{c1}')
                ]
                if sig_row.empty: continue
                
                p_val = sig_row['t_test_P.Value'].iloc[0]
                if p_val < 0.05:
                    x_pos = (j + j + 1) / 2.0
                    y_pos = (ds_data['z_mean'] + ds_data['z_se']).max() + 0.2
                    plot_ax.text(x_pos, y_pos, '*', ha='center',
                                 va='bottom', fontsize=14, color='black',
                                 fontweight='bold')

        z_ticks = ax.get_yticks()
        if do_merfish:
            stats = merfish_stats
            y_labels = [f'{z*stats["std"]+stats["mean"]:.0f}' for z in z_ticks]
            ax.set_yticks(z_ticks, labels=y_labels)
        
        if do_curio:
            stats = curio_stats
            y_labels = [f'{z*stats["std"]+stats["mean"]:.1f}' for z in z_ticks]
            if ax_right:
                ax_right.set_yticks(z_ticks, labels=y_labels)
            else:
                ax.set_yticks(z_ticks, labels=y_labels)

        ax.set_title(cell_type, fontsize=10)

        if i == len(cell_types) - 1:
            ax.set_xticks(range(len(condition_order)))
            ax.set_xticklabels([condition_labels.get(c, c) for c in
                                condition_order], rotation=45, ha='right',
                               fontsize=9)
        else:
            ax.set_xticks([])

        if i == len(cell_types) // 2:
            if do_merfish:
                ax.set_ylabel('MERFISH mNND (µm)', fontsize=9)
            if do_curio:
                label_ax = ax_right if ax_right else ax
                label_ax.set_ylabel(
                    'Slide-tags mNND (µm)',
                    fontsize=9, rotation=-90, va='bottom'
                )

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.25)
    return fig

#endregion 

#region main ###################################################################

working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'
cell_type_col = 'subclass'
target_cell_types = ['Endo NN', 'Peri NN', 'VLMC NN']

adata_curio = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_curio_final.h5ad')
adata_merfish = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')
adata_merfish.var.index = adata_merfish.var['gene_symbol']

for ad in [adata_curio, adata_merfish]:
    ad.var_names_make_unique()
    for col in ['class', 'subclass']:
        ad.obs[col] = ad.obs[col].astype(str).str.extract(
            r'^(\d+)\s+(.*)', expand=False)[1]
    ad.obs = ad.obs[['sample', 'condition', 'source', 'x', 'y',
                     'x_ffd', 'y_ffd', 'class', 'subclass']]
    ad.var = ad.var[['gene_symbol']]
    ad.var.index.name = None
    g = ad.var_names
    ad.var['mt'] = g.str.match(r'^(mt-|MT-)')
    ad.var['ribo'] = g.str.match(r'^(Rps|Rpl)')
    for key in ('uns', 'varm', 'obsp', 'obsm'):
        if hasattr(ad, key):
            try:
                delattr(ad, key)
            except:
                pass

density_m = get_density(adata_merfish, target_cell_types, cell_type_col)
density_c = get_density(adata_curio, target_cell_types, cell_type_col)
combined_density = pd.concat([density_m, density_c], ignore_index=True)

melted_df = combined_density.melt(
    id_vars=['sample', 'condition', 'source'],
    value_vars=target_cell_types,
    var_name='Cell Type',
    value_name='Density (cells / mm^2)'
)

summary_data, stats_data = calculate_summary_and_stats(melted_df)
print(summary_data)
print(stats_data)

fig = plot_dual_axis_density(
    summary_data, stats_data, target_cell_types,
    datasets_to_plot=['merfish']
)
plt.savefig(f'{working_dir}/figures/vascular_density_merfish.png',
            bbox_inches='tight')
plt.show()
plt.close()

uniformity_m = calculate_spatial_uniformity(
    adata_merfish, target_cell_types, cell_type_col)
uniformity_c = calculate_spatial_uniformity(
    adata_curio, target_cell_types, cell_type_col)
combined_uniformity = pd.concat([uniformity_m, uniformity_c],
                                ignore_index=True)

summary_uni, stats_uni = calculate_mNND_summary_and_stats(combined_uniformity)
print(summary_uni)
print(stats_uni)

fig_uni = plot_mNND(
    summary_uni, stats_uni, target_cell_types,
    datasets_to_plot=['merfish']
)
plt.savefig(f'{working_dir}/figures/vascular_uniformity_merfish.png',
            bbox_inches='tight')
plt.show()
plt.close()