import os, anndata
import matplotlib.pyplot as plt

from pathlib import Path
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

# https://github.com/AllenInstitute/abc_atlas_access/blob/main/notebooks/getting_started.ipynb
# https://github.com/AllenInstitute/abc_atlas_access/blob/main/notebooks/zhuang_merfish_tutorial.ipynb

download_base = Path('projects/def-wainberg/spatial/Zhuang/abc-project-cache')
abc_cache = AbcProjectCache.from_s3_cache(download_base)
abc_cache.current_manifest

# cell meta data ###############################################################

datasets = ['Zhuang-ABCA-1', 'Zhuang-ABCA-2']

cell = {}
for d in datasets :
    cell[d] = abc_cache.get_metadata_dataframe(
        directory=d,
        file_name='cell_metadata',
        dtype={"cell_label": str}
    )
    cell[d].set_index('cell_label', inplace=True)
    sdf = cell[d].groupby('brain_section_label')
    print(d,":","Number of cells = ", len(cell[d]), ", ", 
          "Number of sections =", len(sdf))
 
cluster_details = abc_cache.get_metadata_dataframe(
    directory='WMB-taxonomy',
    file_name='cluster_to_cluster_annotation_membership_pivoted',
    keep_default_na=False
)
cluster_details.set_index('cluster_alias', inplace=True)

cluster_colors = abc_cache.get_metadata_dataframe(
    directory='WMB-taxonomy',
    file_name='cluster_to_cluster_annotation_membership_color',
)
cluster_colors.set_index('cluster_alias', inplace=True)

cell_extended = {}
for d in datasets :
    cell_extended[d] = cell[d].join(cluster_details, on='cluster_alias')
    cell_extended[d] = cell_extended[d].join(cluster_colors, on='cluster_alias')
    
ccf_coordinates = {}
for d in datasets :
    ccf_coordinates[d] = abc_cache.get_metadata_dataframe(
        directory=f"{d}-CCF", file_name='ccf_coordinates')
    ccf_coordinates[d].set_index('cell_label', inplace=True)
    ccf_coordinates[d].rename(columns={
        'x': 'x_ccf',
        'y': 'y_ccf',
        'z': 'z_ccf'}, inplace=True)
    cell_extended[d] = cell_extended[d].join(ccf_coordinates[d], how='inner')

parcellation_annotation = abc_cache.get_metadata_dataframe(
    directory="Allen-CCF-2020", 
    file_name='parcellation_to_parcellation_term_membership_acronym')
parcellation_annotation.set_index('parcellation_index', inplace=True)
parcellation_annotation.columns = ['parcellation_%s'% x 
                                   for x in parcellation_annotation.columns]

parcellation_color = abc_cache.get_metadata_dataframe(
    directory="Allen-CCF-2020",
    file_name='parcellation_to_parcellation_term_membership_color')
parcellation_color.set_index('parcellation_index', inplace=True)
parcellation_color.columns = ['parcellation_%s'% x
                              for x in parcellation_color.columns]
for d in datasets :
    cell_extended[d] = cell_extended[d].join(
        parcellation_annotation, on='parcellation_index')
    cell_extended[d] = cell_extended[d].join(
        parcellation_color, on='parcellation_index')   

for d in datasets:
    cell_extended[d].to_csv(
        'projects/def-wainberg/spatial/Zhuang/direct-downloads/'
        f'{d}-metadata.csv')
    
# plot slices ##################################################################

figure_dir = 'projects/def-wainberg/karbabi/spatial-pregnancy-postpart/figures'
os.makedirs(f'{figure_dir}/Zhuang_ref', exist_ok=True)

def subplot_section(ax, xx, yy, cc=None, val=None, cmap=None):
    if cmap is not None:
        ax.scatter(xx, yy, s=0.5, c=val, marker='.', cmap=cmap)
    elif cc is not None:
        ax.scatter(xx, yy, s=0.5, color=cc, marker='.')
    ax.set_ylim(11, 0)
    ax.set_xlim(0, 11)
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])

def plot_sections(cell_extended, dataset, sections, 
                  cc=None, val=None, cmap=None):
    # Parse the floating point numbers and sort the sections
    sorted_sections = sorted(sections, key=lambda x: float(x.split('.')[-1]))
    n_sections = len(sorted_sections)
    n_plots = (n_sections + 7) // 8  
    plots = []

    for plot_idx in range(n_plots):
        start_idx = plot_idx * 8
        end_idx = min(start_idx + 8, n_sections)
        selected_sections = sorted_sections[start_idx:end_idx]

        fig, axes = plt.subplots(4, 2, figsize=(10, 10))
        axes = axes.flatten()
        
        for i, section_label in enumerate(selected_sections):
            pred = cell_extended[dataset]['brain_section_label'] == \
                section_label
            section = cell_extended[dataset][pred]
            if cmap is not None:
                subplot_section(axes[i], section['x'], section['y'], 
                                val=section[val], cmap=cmap)
            elif cc is not None:
                subplot_section(axes[i], section['x'], section['y'], 
                                section[cc])
            axes[i].set_title(f"{section_label}")
        
        plt.tight_layout()
        plots.append(fig)
    
    return plots

for dataset in cell_extended.keys():
    sections = cell_extended[dataset]['brain_section_label'].unique().tolist()
    plots = plot_sections(cell_extended, dataset, sections, 'class_color')
    for idx, fig in enumerate(plots, 1):
        fig.suptitle(f'{dataset}', fontsize=14)
        fig.savefig(f'{figure_dir}/reference/zhuang_ref/{dataset}_{idx}.png',
                    dpi=300)  
        plt.close(fig)  
