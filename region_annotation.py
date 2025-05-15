import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

adata_merfish = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')
adata_ref = sc.read_h5ad(
    f'{working_dir}/output/merfish/adata_comb_cast_stack.h5ad')
adata_ref = adata_ref[adata_ref.obs['source'] == 'Zeng-ABCA-Reference']

k_select = 'k100_cluster'

for region_type in ['parcellation_division', 'parcellation_structure']:
    k_to_region = {}
    for cluster in adata_ref.obs[k_select].unique():
        mask = adata_ref.obs[k_select] == cluster
        cells = adata_ref.obs[mask]
        region_counts = cells[region_type].value_counts()
        if len(region_counts) > 0:
            majority_region = region_counts.index[0]
            k_to_region[cluster] = majority_region
        else:
            k_to_region[cluster] = "Unknown"

    region_to_color = {}
    for region in adata_ref.obs[region_type].unique():
        mask = adata_ref.obs[region_type] == region
        cells = adata_ref.obs[mask]
        if len(cells) > 0:
            color = cells[f'{region_type}_color'].iloc[0]
            region_to_color[region] = color
    
    if "Unknown" not in region_to_color:
        region_to_color["Unknown"] = "#FFFFFF"

    adata_merfish.obs[region_type] = (
        adata_merfish.obs[k_select]
        .map(k_to_region)
        .fillna("Unknown")
        .astype('category')
    )
    
    adata_merfish.obs[f'{region_type}_color'] = (
        adata_merfish.obs[region_type]
        .astype(str)
        .map(region_to_color)
        .fillna("#FFFFFF")
        .astype('category')
    )

    samples = list(adata_ref.obs['sample'].unique()) + \
        list(adata_merfish.obs['sample'].unique())
    n_cols = min(3, len(samples))
    n_rows = (len(samples) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.flatten() if len(samples) > 1 else [axes]
    
    all_regions = sorted(set(adata_ref.obs[region_type].unique()) | 
                        set(adata_merfish.obs[region_type].unique()))
    
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab20', len(all_regions))
    region_colors = {r: cmap(i) for i, r in enumerate(all_regions)}
    
    for i, sample in enumerate(samples):
        is_ref = sample in adata_ref.obs['sample'].unique()
        data = adata_ref if is_ref else adata_merfish
        data = data[data.obs['sample'] == sample]
        
        for r in data.obs[region_type].unique():
            mask = data.obs[region_type] == r
            axes[i].scatter(data[mask].obs['x_ffd'], data[mask].obs['y_ffd'],
                          s=1, c=region_colors.get(r), alpha=0.7)
        
        axes[i].set_title(f"{'Ref' if is_ref else 'MERFISH'}: {sample}")
        axes[i].set_aspect('equal')
        axes[i].axis('off')
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    handles = [mpatches.Patch(color=region_colors.get(r), label=r) 
              for r in all_regions]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(0.5, 0.02), 
            ncol=5, fontsize='xx-small')
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    plt.savefig(
        f'{working_dir}/figures/merfish/{region_type}_comparison.png', dpi=300)
    
# adata_merfish.write_h5ad(
#     f'{working_dir}/output/data/adata_query_merfish_final.h5ad')