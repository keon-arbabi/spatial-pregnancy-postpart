import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as hc
import matplotlib as mpl
from scipy.stats import zscore
import re

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

dotplot_cmap = "seismic"
condition_colors = {
    'CTRL': '#7209b7',
    'PREG': '#b5179e',
    'POSTPART': '#f72585'
}
modality_colors = {
    'merfish': '#4361ee',
    'curio': '#4cc9f0'
}

working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'

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

adata_curio = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_curio_final.h5ad')

adata_merfish = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')
adata_merfish.var.index = adata_merfish.var['gene_symbol']

# adata_curio.X = adata_curio.layers['log1p'].copy()
# adata_merfish.X = adata_merfish.layers['volume_log1p'].copy()

common_subclasses_numbered = (
    set(adata_curio.obs['subclass'])
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

for level in color_mappings:
    color_mappings[level] = {
        k.split(' ', 1)[1]: v for k, v in color_mappings[level].items()
    }

common_subclasses = (
    set(adata_curio.obs['subclass'])
    & set(adata_merfish.obs[adata_merfish.obs['subclass_keep']]['subclass']))

#region marker dotplot #########################################################

curio_markers = [
    # ── Excitatory (Glut) ───────────────────────────────────
    'Slc17a7',  # pan-glutamatergic
    'Cux2',     # L2/3
    'Rorb',     # L4/5
    'Tshz2',    # L5
    'Syt6',     # L6
    
    # ── Inhibitory (Gaba) ───────────────────────────────────
    'Gad2',     # pan-GABA
    'Vip',      # CGE-derived
    'Lamp5',    # CGE-derived
    'Lhx6',     # MGE lineage factor
    'Pvalb',    # MGE interneurons
    'Sst',      # MGE interneurons
    
    # ── Striatal MSNs ───────────────────────────────────────
    'Drd1',     # D1 MSNs
    'Drd2',     # D2 MSNs
    
    # ── Non-neuronal (NN) ───────────────────────────────────
    'Slc1a3',   # astro
    'Foxj1',    # ependymal
    'Pdgfra',   # OPC
    'Cspg4',    # OPC
    'Olig1',    # oligodendrocytes
    'Igf2',     # VLMC
    'Vtn',      # pericytes
    'Flt1',     # endothelial
    'Cldn5',    # endothelial
    'Ctss',     # microglia
    'C1qa'      # microglia
]

merfish_markers = [
    # ── Excitatory (Glut) ───────────────────────────────────
    'Slc17a7',  # pan-glutamatergic
    'Cux2',     # L2/3
    'Fezf2',    # L5
    'Sox5',     # L5/6 factor
    
    # ── Inhibitory (Gaba) ───────────────────────────────────
    'Gad2',     # pan-GABA
    'Reln',     # often CGE-related
    'Calb2',    # CGE (e.g. Vip/Lamp5 clusters)
    'Lhx6',     # MGE factor
    'Pvalb',    # MGE (PV)
    'Sst',      # MGE (SST)

    # ── Striatal MSNs ───────────────────────────────────────
    'Foxp2',
    'Gpr6',
    
    # ── Non-neuronal (NN) ───────────────────────────────────
    'Slc1a3',   # astro
    'Foxj1',    # ependymal
    'Pdgfra',   # OPC
    'Cspg4',    # OPC
    'Olig1',    # oligodendrocytes
    'Igf2',     # VLMC
    'Pdgfrb',   # pericytes
    'Anpep',    # pericytes
    'Flt1',     # endothelial
    'Pecam1',   # endothelial
    'Ctss',     # microglia
    'Cx3cr1',   # microglia
    'P2ry12'    # microglia
]

# Set up figure
f = plt.figure(figsize=(9, 18))
gs = plt.GridSpec(nrows=8, ncols=1, 
                 height_ratios=[5, 6, 2, 6, 25, 25, 6, 6],  
                 hspace=0.1)

# calculate clustering
clust_avg_curio = []
clust_ids = sorted(list(common_subclasses))
for i in clust_ids:
    clust_avg_curio.append(
        adata_curio[adata_curio.obs['subclass'] == i].X.mean(0))
clust_avg_curio = np.vstack(clust_avg_curio)

D = pdist(clust_avg_curio, 'correlation')
Z = hc.linkage(D, 'complete', optimal_ordering=False)

# plot dendrogram
ax0 = plt.subplot(gs[0])
dn = hc.dendrogram(Z, ax=ax0, labels=clust_ids, leaf_font_size=10, 
                   color_threshold=0, above_threshold_color='k')
ax0.axis('off')

# restore original ordering based on numbered subclass names
lbl_order = []
for subclass in sorted(list(common_subclasses_numbered)):
    clean_name = re.sub(r'^\d+\s+', '', subclass)
    if clean_name in common_subclasses:
        lbl_order.append(clean_name)

# add first blank subplot for spacing
ax_blank1 = plt.subplot(gs[1])
ax_blank1.axis('off')

# get counts
curio_counts = pd.Series({
    sub: len(adata_curio[adata_curio.obs['subclass'] == sub]) 
    for sub in lbl_order})
merfish_counts = pd.Series({
    sub: len(adata_merfish[adata_merfish.obs['subclass'] == sub]) 
    for sub in lbl_order})

# plot subclass colors with discrete color handling
ax1 = plt.subplot(gs[2])
color_list = [color_mappings['subclass'][c] for c in lbl_order]
curr_cols = mpl.colors.ListedColormap(color_list, N=len(lbl_order))
color_array = np.expand_dims(np.arange(len(lbl_order)), 1).T

ax1.imshow(color_array,
           cmap=curr_cols,
           aspect='auto',
           interpolation='nearest',
           extent=[-0.5, len(lbl_order)-0.5, 0.5, 1],
           vmin=-0.5,
           vmax=len(lbl_order)-0.5)

ax1.set_xticks(np.arange(len(lbl_order)))
ax1.set_xticklabels([f"{curio_counts[c]:,}" for c in lbl_order],
                    rotation=90, ha='right', va='top')
ax1.tick_params(axis='x', length=0, pad=15)

ax1_2 = ax1.twiny()
ax1_2.set_xlim(ax1.get_xlim())
ax1_2.set_xticks(np.arange(len(lbl_order)))
ax1_2.set_xticklabels([f"{merfish_counts[c]:,}" for c in lbl_order],
                      rotation=90, ha='right', va='top')
ax1_2.tick_params(axis='x', length=0, pad=45)

ax1.set_yticks([])
ax1_2.set_yticks([])

for spine in ax1.spines.values():
    spine.set_visible(False)
for spine in ax1_2.spines.values():
    spine.set_visible(False)

# add second blank subplot between colorbar and dotplot
ax_blank2 = plt.subplot(gs[3])
ax_blank2.axis('off')

# calculate dotplot values for curio
ax2 = plt.subplot(gs[4])
dotplot_vals_curio = np.zeros((len(curio_markers), len(lbl_order)))
dotplot_frac_curio = np.zeros((len(curio_markers), len(lbl_order)))

# reverse marker order before calculations
curio_markers_rev = curio_markers[::-1]
for n, i in enumerate(lbl_order):
    mask = adata_curio.obs['subclass'] == i
    curr_data = adata_curio[mask][:,curio_markers_rev].X.toarray()
    dotplot_vals_curio[:,n] = np.mean(curr_data, 0)
    dotplot_frac_curio[:,n] = np.sum(curr_data > 0, 0) / np.sum(mask)

# z-score the expression values
for n in range(len(curio_markers_rev)):
    dotplot_vals_curio[n,:] = zscore(dotplot_vals_curio[n,:])

# plot curio dotplot
dotscale = 35
for i in range(dotplot_vals_curio.shape[1]):
    s = dotscale * (0.15 + 2.0 * dotplot_frac_curio[:,i])
    scatter = ax2.scatter(i * np.ones(dotplot_vals_curio.shape[0]),
                         np.arange(dotplot_vals_curio.shape[0]), 
                         c=dotplot_vals_curio[:,i],
                         s=s, cmap=dotplot_cmap, vmin=-2, vmax=2)

ax2.set_yticks(np.arange(len(curio_markers_rev)))
ax2.set_yticklabels(curio_markers_rev)
ax2.set_xlim([-0.5, dotplot_vals_curio.shape[1]-0.5])
ax2.set_xticks([])
sns.despine(ax=ax2, bottom=True)

# calculate dotplot values for merfish
ax3 = plt.subplot(gs[5])
dotplot_vals_merfish = np.zeros((len(merfish_markers), len(lbl_order)))
dotplot_frac_merfish = np.zeros((len(merfish_markers), len(lbl_order)))

# reverse marker order before calculations
merfish_markers_rev = merfish_markers[::-1]
for n, i in enumerate(lbl_order):
    mask = adata_merfish.obs['subclass'] == i
    curr_data = adata_merfish[mask][:,merfish_markers_rev].X.toarray()
    dotplot_vals_merfish[:,n] = np.mean(curr_data, 0)
    dotplot_frac_merfish[:,n] = np.sum(curr_data > 0, 0) / np.sum(mask)

# z-score the expression values
for n in range(len(merfish_markers_rev)):
    dotplot_vals_merfish[n,:] = zscore(dotplot_vals_merfish[n,:])

# plot merfish dotplot
for i in range(dotplot_vals_merfish.shape[1]):
    s = dotscale * (0.15 + 2.0 * dotplot_frac_merfish[:,i])
    scatter = ax3.scatter(i * np.ones(dotplot_vals_merfish.shape[0]),
                         np.arange(dotplot_vals_merfish.shape[0]), 
                         c=dotplot_vals_merfish[:,i],
                         s=s,
                         cmap=dotplot_cmap, vmin=-2, vmax=2)

ax3.set_yticks(np.arange(len(merfish_markers_rev)))
ax3.set_yticklabels(merfish_markers_rev)
ax3.set_xlim([-0.5, dotplot_vals_merfish.shape[1]-0.5])
ax3.set_xticks([])
sns.despine(ax=ax3, bottom=True)

# calculate condition proportions
n_bins = 200
conditions = ['CTRL', 'PREG', 'POSTPART']
total_cells = {cond: np.sum(adata_merfish.obs['condition'] == cond) 
               for cond in conditions}

frac_per_condition = np.zeros((len(lbl_order), n_bins))
for n, subclass in enumerate(lbl_order):
    curr_cells = adata_merfish[adata_merfish.obs['subclass'] == subclass]
    
    props = {}
    for cond in conditions:
        count = np.sum(curr_cells.obs['condition'] == cond)
        props[cond] = (count / total_cells[cond]) / curr_cells.shape[0]
    
    total = sum(props.values())
    for cond in conditions:
        if total > 0:
            props[cond] /= total
    
    start_idx = 0
    for i, cond in enumerate(conditions):
        prop = props[cond]
        n_cond_bins = int(round(n_bins * prop))
        val = i
        end_idx = start_idx + n_cond_bins
        if i == len(conditions) - 1:
            end_idx = n_bins
        frac_per_condition[n, start_idx:end_idx] = val
        start_idx = end_idx

# plot condition proportions
# For the condition proportions section
ax4 = plt.subplot(gs[6])
ordered_colors = [condition_colors[c] for c in conditions]
condition_cmap = mpl.colors.ListedColormap(ordered_colors, N=len(conditions))
ax4.imshow(frac_per_condition.T, aspect='auto', interpolation='nearest',  
           cmap=condition_cmap)
ax4.set_yticks([])
ax4.set_xticks([])
sns.despine(ax=ax4, left=True)
ax4.axhline(n_bins/3, color='w', linestyle='--')
ax4.axhline(2*n_bins/3, color='w', linestyle='--')

# calculate modality proportions
frac_per_modality = np.zeros((len(lbl_order), n_bins))
total_curio = len(adata_curio)
total_merfish = len(adata_merfish)

for n, subclass in enumerate(lbl_order):
    curio_count = np.sum(adata_curio.obs['subclass'] == subclass)
    merfish_count = np.sum(adata_merfish.obs['subclass'] == subclass)
    
    curio_prop = (curio_count / total_curio)
    merfish_prop = (merfish_count / total_merfish)
    
    total = curio_prop + merfish_prop
    curio_prop /= total
    merfish_prop /= total
    
    n_curio_bins = int(round(n_bins * curio_prop))
    frac_per_modality[n, :n_curio_bins] = 1
    frac_per_modality[n, n_curio_bins:] = 0

# plot modality proportions
ax5 = plt.subplot(gs[7])
ordered_modality_colors = [modality_colors['merfish'], modality_colors['curio']]
modality_cmap = mpl.colors.ListedColormap(ordered_modality_colors, N=2)   
ax5.imshow(frac_per_modality.T, aspect='auto', interpolation='nearest',
           cmap=modality_cmap, vmin=-0.5, vmax=1.5) 
ax5.set_yticks([])
ax5.set_xticks(np.arange(len(lbl_order)))
ax5.set_xticklabels(lbl_order, rotation=90)
sns.despine(ax=ax5, left=True)
ax5.axhline(n_bins/2, color='w', linestyle='--')

# add unified legends for expression
norm = plt.Normalize(vmin=-2, vmax=2)
sm = plt.cm.ScalarMappable(cmap=dotplot_cmap, norm=norm)
cax = f.add_axes([0.97, 0.4, 0.02, 0.1])
cbar = plt.colorbar(sm, cax=cax)
cbar.outline.set_visible(False)
cbar.ax.set_title('z-scored expression', pad=10)

# size legend
legend_elements = []
for size in [0.15, 1.0, 2.15]:
    legend_elements.append(plt.scatter([],[], 
                                     c='gray',
                                     s=dotscale * size, 
                                     label=f'{int(size*100)}%'))

size_legend = f.legend(handles=legend_elements,
                      title='fraction of cells\nexpressing gene',
                      loc='center right',
                      bbox_to_anchor=(1.05, 0.25),
                      frameon=False,
                      title_fontsize=8)
size_legend._legend_box.align = 'right'

plt.savefig(f'{working_dir}/figures/marker_plot.png', 
            dpi=300, bbox_inches='tight',
            bbox_extra_artists=[size_legend])
plt.savefig(f'{working_dir}/figures/marker_plot.svg', 
            format='svg', bbox_inches='tight',
            bbox_extra_artists=[size_legend])

#endregion 

#region spatial gallery ########################################################

celltype_categories = {
    'Pallium Glut': [
        # Cortical / hippocampal excitatory neurons
        'L2/3 IT CTX Glut', 'L4/5 IT CTX Glut', 'L5 IT CTX Glut',
        'L5 ET CTX Glut', 'L5 NP CTX Glut', 'L6 IT CTX Glut',
        'L6 CT CTX Glut', 'L6b CTX Glut', 'L2/3 IT PIR-ENTl Glut',
        'L2/3 IT RSP Glut', 'L5/6 IT TPE-ENT Glut',
        'CLA-EPd-CTX Car3 Glut', 'IT EP-CLA Glut', 'CA2-FC-IG Glut',
        'L6b EPd Glut'
        # The following were previously in Pallium but should move to HY-EA:
        # 'TRS-BAC Sln Glut', 'PVH-SO-PVa Otp Glut', 'LHA Barhl2 Glut'
    ],
    'Pallium Gaba': [
        # Cortical interneurons
        'Pvalb Gaba', 'Sst Gaba', 'Vip Gaba', 'Lamp5 Gaba',
        'Sncg Gaba', 'Lamp5 Lhx6 Gaba', 'Pvalb chandelier Gaba'
    ],
    'Subpallium Gaba': [
        # Striatal / pallidal / basal ganglia inhibitory
        'STR D1 Gaba', 'STR D2 Gaba', 'STR D1 Sema5a Gaba',
        'STR Prox1 Lhx6 Gaba', 'STR Lhx8 Gaba', 'STR-PAL Chst9 Gaba',
        'PAL-STR Gaba-Chol', 'GPe-SI Sox6 Cyp26b1 Gaba', 'OT D3 Folh1 Gaba',
        'OB-STR-CTX Inh IMN', 'LSX Otx2 Gaba', 'LSX Prdm12 Zeb2 Gaba',
        'LSX Nkx2-1 Gaba', 'LSX Sall3 Pax6 Gaba', 'LSX Sall3 Lmo1 Gaba',
        'LSX Prdm12 Slit2 Gaba', 'OB Meis2 Thsd7b Gaba', 'OB Dopa-Gaba',
        'OB-out Frmd7 Gaba', 'OB-in Frmd7 Gaba', 'Sst Chodl Gaba'
    ],
    'HY-EA': [
        # Hypothalamus–extended amygdala (GABA + Glut)
        'SI-MPO-LPO Lhx8 Gaba', 'NDB-SI-MA-STRv Lhx8 Gaba',
        'MPO-ADP Lhx8 Gaba', 'CEA-AAA-BST Six3 Sp9 Gaba',
        'CEA-BST Ebf1 Pdyn Gaba', 'CEA-BST Rai14 Pdyn Crh Gaba',
        'CEA-BST Six3 Cyp26b1 Gaba', 'MEA-BST Lhx6 Sp9 Gaba',
        'MEA-BST Lhx6 Nfib Gaba', 'BST Tac2 Gaba',
        'BST-SI-AAA Six3 Slc22a3 Gaba', 'BST-MPN Six3 Nrgn Gaba',
        'PVR Six3 Sox3 Gaba', 'ACB-BST-FS D1 Gaba',
        'PVpo-VMPO-MPN Hmx2 Gaba', 'NDB-SI-ant Prdm12 Gaba',
        'AHN Onecut3 Gaba', 'AVPV-MEPO-SFO Tbr1 Glut',
        'ADP-MPO Trp73 Glut', 'SI-MA-LPO-LHA Skor1 Glut',
        'MPN-MPO-PVpo Hmx2 Glut', 'COAa-PAA-MEA Barhl2 Glut',
        'MEA-BST Otp Zic2 Glut', 'LHA-AHN-PVH Otp Trh Glut',
        'MS-SF Bsx Glut', 'IA Mgp Gaba', 'RHP-COA Ndnf Gaba',
        'CEA-BST Gal Avp Gaba', 'MEA-BST Lhx6 Nr2e1 Gaba',
        'MEA-BST Sox6 Gaba', 'DMH Hmx2 Gaba', 'ZI Pax6 Gaba',
        'LGv-ZI Otx2 Gaba', 'SI-MA-ACB Ebf1 Bnc2 Gaba',
        'MPN-MPO-LPO Lhx6 Zfhx3 Gaba',
        'TRS-BAC Sln Glut', 'PVH-SO-PVa Otp Glut', 'LHA Barhl2 Glut'
    ],
    'Non-Neuronal': [
        # Glia, vascular, ependymal, etc.
        'Microglia NN', 'Oligo NN', 'Astro-TE NN', 'Endo NN', 'OPC NN',
        'Astro-NT NN', 'CHOR NN', 'VLMC NN', 'Peri NN', 'SMC NN',
        'ABC NN', 'Astroependymal NN', 'Ependymal NN', 'BAM NN',
        'Tanycyte NN', 'Lymphoid NN'
    ]
}

celltype_categories = {
    k: [x for x in v if x in common_subclasses]
    for k, v in celltype_categories.items()
    if any(x in common_subclasses for x in v)
}

samples = {
    'curio': {'CTRL': 'CTRL_2', 'PREG': 'PREG_1', 'POSTPART': 'POSTPART_1'},
    'merfish': {'CTRL': 'CTRL3', 'PREG': 'PREG1', 'POSTPART': 'POSTPART2'}
}
point_sizes = {
    'curio': {'background': 9, 'foreground': 12},  
    'merfish': {'background': 1, 'foreground': 2}
}
conditions = ['CTRL', 'PREG', 'POSTPART']

# function to create plots for a specific dataset
def plot_spatial_gallery(adata, dataset_name, celltype_categories):
    fig = plt.figure(figsize=(12, 22), constrained_layout=False)
    gs = gridspec.GridSpec(
        6, 3, wspace=0.05, hspace=0, left=0, right=1, bottom=0, top=1)
    
    for row, (category, subtypes) in enumerate(celltype_categories.items()):
        for col, condition in enumerate(conditions):
            sample = samples[dataset_name][condition]
            adata_subset = adata[adata.obs['sample'] == sample]
            
            ax = plt.subplot(gs[row, col])        
            ax.scatter(
                adata_subset.obs['x_ffd'], 
                adata_subset.obs['y_ffd'],
                c='lightgray', 
                s=point_sizes[dataset_name]['background'], 
                linewidths=0,
                rasterized=True
            )
            category_mask = adata_subset.obs['subclass'].isin(subtypes)
            if np.sum(category_mask) > 0:
                cell_colors = [
                    color_mappings['subclass'][c] 
                    for c in adata_subset.obs['subclass'][category_mask]
                ]
                ax.scatter(
                    adata_subset.obs['x_ffd'][category_mask], 
                    adata_subset.obs['y_ffd'][category_mask],
                    c=cell_colors, 
                    s=point_sizes[dataset_name]['foreground'],  
                    linewidths=0,
                    rasterized=True
                )
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.margins(0, 0)
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.savefig(
        f'{working_dir}/figures/spatial_gallery_{dataset_name}.png', 
        dpi=400, bbox_inches='tight', pad_inches=0)
    plt.savefig(
        f'{working_dir}/figures/spatial_gallery_{dataset_name}.svg', 
        format='svg', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

plot_spatial_gallery(adata_curio[~adata_curio.obs['subclass_corrected']], 
                     'curio', celltype_categories)

plot_spatial_gallery(adata_merfish, 'merfish', celltype_categories)

#endregion 

#region umaps and spatial exemplar #############################################

level = 'subclass'

# curio umap
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(
    adata_curio.obsm['X_umap'][:, 0], adata_curio.obsm['X_umap'][:, 1],
    c=[color_mappings[level][c] for c in adata_curio.obs[level]],
    s=4, linewidths=0, rasterized=True)
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([])
ax.set_yticks([])
ax.set_axis_off()
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig(f'{working_dir}/figures/umap_{level}_curio.png', dpi=400)
plt.savefig(f'{working_dir}/figures/umap_{level}_curio.svg', format='svg')
plt.close()

# merfish umap
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(
    adata_merfish.obsm['X_umap'][:, 0], adata_merfish.obsm['X_umap'][:, 1],
    c=[color_mappings[level][c] for c in adata_merfish.obs[level]],
    s=1, linewidths=0, rasterized=True)
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([])
ax.set_yticks([])
ax.set_axis_off()
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig(f'{working_dir}/figures/umap_{level}_merfish.png', dpi=400)
plt.savefig(f'{working_dir}/figures/umap_{level}_merfish.svg', format='svg')
plt.close()

# curio spatial exemplar 
sample = 'CTRL_2'
plot_color = adata_curio[(adata_curio.obs['sample'] == sample)].obs
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    plot_color['x_ffd'], plot_color['y_ffd'],
    c=[color_mappings[level][c] for c in plot_color[level]], 
    s=40, linewidths=0, rasterized=True)
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
plt.savefig(f'{working_dir}/figures/spatial_example_{level}_curio.png',
            dpi=300, bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/spatial_example_{level}_curio.svg',
            format='svg', bbox_inches='tight')
plt.close()

# merfish spatial exemplar
sample = 'PREG1'  
plot_color = adata_merfish[(adata_merfish.obs['sample'] == sample)].obs
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    plot_color['x_ffd'], plot_color['y_ffd'],
    c=[color_mappings[level][c] for c in plot_color[level]], 
    s=3, linewidths=0, rasterized=True)
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
plt.savefig(f'{working_dir}/figures/spatial_example_{level}_merfish.png',
            dpi=300, bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/spatial_example_{level}_merfish.svg',
            format='svg', bbox_inches='tight')
plt.close()

#endregion 

#region link plot ##############################################################

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'

proj_dir = f'{working_dir}/output/curio/CAST-PROJECT'
proj_dir += '/Zeng-ABCA-Reference_to_CTRL_2'
harmony_file = f'{proj_dir}/X_harmony_Zeng-ABCA-Reference_to_CTRL_2.h5ad'
adata_proj = sc.read_h5ad(harmony_file)

ref_mask = adata_proj.obs['batch'] == 'Zeng-ABCA-Reference'
query_mask = adata_proj.obs['batch'] == 'CTRL_2'
coords_source = np.array(adata_proj[ref_mask].obs[['x_ffd', 'y_ffd']])
coords_target = np.array(adata_proj[query_mask].obs[['x_ffd', 'y_ffd']])

list_ts = torch.load(f'{proj_dir}/list_ts_CTRL_2.pt', weights_only=False)
project_ind = list_ts[0][:, 0:1] 
cells_joined = pd.read_csv(
    'projects/rrg-wainberg/single-cell/ABC/metadata/'
    'MERFISH-C57BL6J-638850/20231215/views/cells_joined.csv')
color_map = {k.replace('_', '/'): v for k, v in 
            zip(cells_joined['subclass'].str.replace('/', '_'),
                cells_joined['subclass_color'])}
ref_colors = [color_map.get(c, '#999999') 
             for c in adata_proj[ref_mask].obs['subclass']]

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.set_box_aspect([1, 1, 0.6])
ax.view_init(elev=25)

padding = 0.05 * max(coords_source.ptp(0).max(), coords_target.ptp(0).max())
ax.set_xlim(min(coords_source[:,0].min(), coords_target[:,0].min()) - padding,
           max(coords_source[:,0].max(), coords_target[:,0].max()) + padding)
ax.set_ylim(min(coords_source[:,1].min(), coords_target[:,1].min()) - padding,
           max(coords_source[:,1].max(), coords_target[:,1].max()) + padding)
ax.set_zlim(-0.1, 1.1)

sample_n = 300
indices = np.random.choice(len(coords_target), size=sample_n, replace=False)
t1 = coords_source[project_ind[indices, 0]].T
t2 = coords_target[indices].T
segs = [[(*t2[:, i], 0), (*t1[:, i], 1)] for i in range(sample_n)]
lc = Line3DCollection(segs, colors='#999999', lw=0.5, alpha=0.3)
lc.set_rasterized(True)
ax.add_collection(lc)

ax.scatter(coords_target[:,0], coords_target[:,1], 0, 
          s=5, c='#999999', alpha=0.6, rasterized=True)
ax.scatter(coords_source[:,0], coords_source[:,1], 1, 
          s=5, c=ref_colors, alpha=0.6, rasterized=True)

ax.axis('off')
plt.savefig(
    f'{working_dir}/figures/linkplot.png', bbox_inches='tight', dpi=300)
plt.savefig(
    f'{working_dir}/figures/linkplot.svg', format='svg', bbox_inches='tight')
plt.close()


import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'

adata_merfish = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')

cells_joined = pd.read_csv(
    'projects/rrg-wainberg/single-cell/ABC/metadata/'
    'MERFISH-C57BL6J-638850/20231215/views/cells_joined.csv')

color_map = {k.replace('_', '/'): v for k, v in 
            zip(cells_joined['subclass'].str.replace('/', '_'),
                cells_joined['subclass_color'])}

sample = 'PREG1'
adata_sample = adata_merfish[adata_merfish.obs['sample'] == sample].copy()

for col in ['subclass']:
    adata_sample.obs[col] = adata_sample.obs[col].astype(str)\
        .str.extract(r'^(\d+)\s+(.*)', expand=False)[1]

colors = [color_map.get(c, '#999999') 
         for c in adata_sample.obs['subclass']]

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.set_box_aspect([1, 1, 0.6])
ax.view_init(elev=25)

coords = np.array(adata_sample.obs[['x_ffd', 'y_ffd']])
padding = 0.05 * coords.ptp(0).max()
ax.set_xlim(coords[:,0].min() - padding, coords[:,0].max() + padding)
ax.set_ylim(coords[:,1].min() - padding, coords[:,1].max() + padding)
ax.set_zlim(-0.1, 1.1)

ax.scatter(coords[:,0], coords[:,1], 0, s=0.5, c=colors, alpha=0.6, 
          rasterized=True)

ax.axis('off')
plt.savefig(f'{working_dir}/figures/merfish_angled.png', 
           bbox_inches='tight', dpi=300)
plt.savefig(f'{working_dir}/figures/merfish_angled.svg', 
           format='svg', bbox_inches='tight')
plt.close() 

#endregion 

# create standalone legend for common subclasses
f_legend = plt.figure(figsize=(3, 8))
legend_elements = [plt.Line2D(
    [0], [0], marker='o', color='w',
    markerfacecolor=color_mappings['subclass'][subclass],
    label=subclass, markersize=8)
    for subclass in sorted(list(common_subclasses))]

f_legend.legend(handles=legend_elements, 
                loc='center',
                frameon=False)
plt.axis('off')
plt.savefig(f'{working_dir}/figures/common_subclasses_legend.svg',
            format='svg', bbox_inches='tight')
