import os, anndata
import matplotlib.pyplot as plt
from pathlib import Path
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

# https://github.com/AllenInstitute/abc_atlas_access/blob/main/notebooks/getting_started.ipynb
# https://github.com/AllenInstitute/abc_atlas_access/blob/main/notebooks/zhuang_merfish_tutorial.ipynb

# imputed update ###############################################################

download_base = Path('projects/def-wainberg/single-cell/ABC')
abc_cache = AbcProjectCache.from_cache_dir(download_base)
abc_cache.load_manifest('releases/20240831/manifest.json')

abc_cache.current_manifest
abc_cache.cache.manifest_file_names
abc_cache.cache.manifest_file_names.append('releases/20240831/manifest.json')
abc_cache.load_manifest('releases/20240831/manifest.json')

cell = abc_cache.get_metadata_dataframe(
    directory='MERFISH-C57BL6J-638850',
    file_name='cell_metadata_with_cluster_annotation',
    dtype={"cell_label": str,
           "neurotransmitter": str}
)
cell.set_index('cell_label', inplace=True)

imputed_h5ad_path = abc_cache.get_data_path('MERFISH-C57BL6J-638850-imputed')


adata = anndata.read_h5ad(imputed_h5ad_path, backed='r')
gene_list = ['Calb2', 'Baiap3', 'Lypd1']

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
        fig.savefig(f'{figure_dir}/zhuang_ref/{dataset}_{idx}.png', dpi=300)  
        plt.close(fig)  
