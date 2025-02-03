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

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

working_dir = 'project/spatial-pregnancy-postpart'

# load metadata and color mappings
cells_joined = pd.read_csv(
    'project/single-cell/ABC/metadata/MERFISH-C57BL6J-638850/20231215/'
    'views/cells_joined.csv')
color_mappings = {
    'class': dict(zip(
        cells_joined['class'].str.replace('/', '_'), 
        cells_joined['class_color'])),
    'subclass': {k.replace('_', '/'): v for k,v in dict(zip(
        cells_joined['subclass'].str.replace('/', '_'), 
        cells_joined['subclass_color'])).items()}
}

# load data
adata_curio = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_curio_final.h5ad')
adata_merfish = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')
adata_merfish.var.index = adata_merfish.var['gene_symbol']

# get common subclasses
common_subclasses = (
    set(adata_curio.obs[adata_curio.obs['subclass_keep']]['subclass'])
    & set(adata_merfish.obs[adata_merfish.obs['subclass_keep']]['subclass']))

# filter curio data
adata_curio = adata_curio[
    adata_curio.obs['subclass'].isin(common_subclasses),
    adata_curio.var['protein_coding']].copy()
adata_curio.X = adata_curio.layers['log1p'].copy()

# define marker genes
curio_markers = [
    # excitatory neurons (glut)
    'Slc17a7',   # pan-glutamatergic marker
    'Cux2',      # l2/3 it marker
    'Rorb',      # l4/5 it marker
    'Tshz2',     # l5 marker
    'Syt6',      # l6 marker
    
    # inhibitory neurons (gaba)
    'Gad2',      # pan-gabaergic marker
    'Lhx6',      # mge-derived interneurons
    'Lamp5',     # lamp5 interneurons
    'Vip',       # vip interneurons
    'Pvalb',     # parvalbumin interneurons
    'Sst',       # somatostatin interneurons
    
    # striatal neurons
    'Drd1',      # d1 
    'Drd2',      # d2 
    
    # non-neuronal cells (nn)
    'Slc1a3',    # astro
    'Pdgfra',    # opcs
    'Cspg4',     # opcs
    'Mbp',       # oligo
    'Plp1',      # oligo
    'Igf2',      # vlmc
    'Vtn',       # peri
    'Flt1',      # endo
    'Cldn5',     # endo
    'Ctss',      # microglia
    'C1qa'       # microglia
]

merfish_markers = [
    # excitatory neurons (glut)
    'Slc17a7',   # pan-glutamatergic
    'Cux2',      # l2/3 it
    'Sox5',      # deep layer marker
    'Tbr1',      # deep layer marker
    'Fezf2',     # deep layer marker
    
    # inhibitory neurons (gaba)
    'Gad2',      # pan-gabaergic
    'Lhx6',      # mge-derived
    'Pvalb',     # pv interneurons
    'Sst',       # sst interneurons
    'Calb2',     # lamp5/vip-related
    'Reln',      # interneuron marker
    
    # non-neuronal cells (nn)
    'Slc1a3',    # astrocytes
    'Pdgfra',    # opcs
    'Cspg4',     # opcs
    'Mbp',       # oligodendrocytes
    'Igf2',      # vlmc
    'Pdgfrb',    # pericytes
    'Anpep',     # pericytes
    'Flt1',      # endothelial
    'Pecam1',    # endothelial
    'Ctss',      # microglia
    'Cx3cr1',    # microglia
    'P2ry12',    # microglia
]

# calculate clustering
clust_avg_curio = []
clust_ids = sorted(list(common_subclasses))
for i in clust_ids:
    clust_avg_curio.append(
        adata_curio[adata_curio.obs['subclass'] == i].X.mean(0))
clust_avg_curio = np.vstack(clust_avg_curio)

D = pdist(clust_avg_curio, 'correlation')
Z = hc.linkage(D, 'complete', optimal_ordering=False)

# set up figure
f = plt.figure(figsize=(10,14))
gs = plt.GridSpec(nrows=8, ncols=1, 
                 height_ratios=[5,6,2,6,25,25,6,6],  
                 hspace=0.1)

# plot dendrogram
ax0 = plt.subplot(gs[0])
dn = hc.dendrogram(Z, ax=ax0, labels=clust_ids, leaf_font_size=10, 
                   color_threshold=0, above_threshold_color='k')
ax0.axis('off')
lbl_order = sorted(clust_ids)

# add first blank subplot for spacing
ax_blank1 = plt.subplot(gs[1])
ax_blank1.axis('off')

# get counts and plot colorbar
curio_counts = pd.Series({
    sub: len(adata_curio[adata_curio.obs['subclass'] == sub]) 
    for sub in lbl_order})
merfish_counts = pd.Series({
    sub: len(adata_merfish[adata_merfish.obs['subclass'] == sub]) 
    for sub in lbl_order})

# plot subclass colors
ax1 = plt.subplot(gs[2])
curr_cols = mpl.colors.ListedColormap([color_mappings['subclass'][c] 
                                      for c in lbl_order])
ax1.imshow(np.expand_dims(np.arange(len(lbl_order)), 1).T,
           cmap=curr_cols, aspect='auto', interpolation='none',
           extent=[-0.5, len(lbl_order)-0.5, 0.5, 1])

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
    s = dotscale * (0.05 + 2.0 * dotplot_frac_curio[:,i])
    scatter = ax2.scatter(i * np.ones(dotplot_vals_curio.shape[0]),
                         np.arange(dotplot_vals_curio.shape[0]), 
                         c=dotplot_vals_curio[:,i],
                         s=s,
                         cmap='seismic', vmin=-2, vmax=2)

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
    s = dotscale * (0.05 + 2.0 * dotplot_frac_merfish[:,i])
    scatter = ax3.scatter(i * np.ones(dotplot_vals_merfish.shape[0]),
                         np.arange(dotplot_vals_merfish.shape[0]), 
                         c=dotplot_vals_merfish[:,i],
                         s=s,
                         cmap='seismic', vmin=-2, vmax=2)

ax3.set_yticks(np.arange(len(merfish_markers_rev)))
ax3.set_yticklabels(merfish_markers_rev)
ax3.set_xlim([-0.5, dotplot_vals_merfish.shape[1]-0.5])
ax3.set_xticks([])
sns.despine(ax=ax3, bottom=True)

# calculate condition proportions
n_bins = 200
conditions = ['PREG', 'CTRL', 'POSTPART']
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
        props[cond] /= total
    
    start_idx = 0
    for cond, prop in props.items():
        n_cond_bins = int(round(n_bins * prop))
        val = 2 if cond == 'PREG' else (1 if cond == 'CTRL' else 0)
        end_idx = start_idx + n_cond_bins
        frac_per_condition[n, start_idx:end_idx] = val
        start_idx = end_idx

# plot condition proportions
ax4 = plt.subplot(gs[6])
condition_colors = ['#f72585', '#b5179e', '#7209b7']
condition_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'custom_condition', condition_colors)
ax4.imshow(frac_per_condition.T, aspect='auto', interpolation='none',
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
modality_colors = ['#4361ee', '#4cc9f0']
modality_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'custom_modality', modality_colors)
ax5.imshow(frac_per_modality.T, aspect='auto', interpolation='none',
           cmap=modality_cmap)
ax5.set_yticks([])
ax5.set_xticks(np.arange(len(lbl_order)))
ax5.set_xticklabels(lbl_order, rotation=90)
sns.despine(ax=ax5, left=True)
ax5.axhline(n_bins/2, color='w', linestyle='--')

# add unified legends for expression
norm = plt.Normalize(vmin=-2, vmax=2)
sm = plt.cm.ScalarMappable(cmap='seismic', norm=norm)
cax = f.add_axes([0.97, 0.4, 0.02, 0.1])
cbar = plt.colorbar(sm, cax=cax)
cbar.outline.set_visible(False)
cbar.ax.set_title('z-scored expression', pad=10)

# size legend
legend_elements = []
for size in [0.05, 1.0, 2.05]: 
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
size_legend._legend_box.align = "right"

plt.savefig(f'{working_dir}/figures/cell_types_overview.png', 
            dpi=300, bbox_inches='tight',
            bbox_extra_artists=[size_legend])
plt.savefig(f'{working_dir}/figures/cell_types_overview.svg', 
            format='svg', bbox_inches='tight',
            bbox_extra_artists=[size_legend])









































sc.tl.rank_genes_groups(adata_merfish, 'subclass')

sc.pl.rank_genes_groups_dotplot(
    adata_merfish, n_genes=5, 
    values_to_plot='logfoldchanges', standard_scale=None,
    vmin=-4, vmax=4, min_logfoldchange=3, cmap='bwr', 
    swap_axes=True, dendrogram=False)
plt.savefig(
    f'{working_dir}/figures/tmp.png',
    dpi=300, bbox_inches='tight')
















sc_curio = SingleCell(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')\
    .with_uns(QCed=True)\
    .filter_obs(pl.col(f'subclass_keep'))\
    .filter_var(pl.col('protein_coding'))

markers_df = sc_curio\
    .find_markers(
        cell_type_column='subclass',
        min_detection_rate=0.20,
        min_fold_change=1.2)\
    .unique('gene', keep='none')\
    .group_by('cell_type')\
    .head(5)\
    .sort('cell_type', descending=True)

markers_list = [
   'Nr4a2', 'Gria4', 'Satb2', 'Rims1', 'Rorb', 'Cux2', 'Cntn5', 'Gria2', 'Grik3', 
   'Sdk2', 'Ptprd', 'Tshz2', 'Vip', 'Grik2', 'Fgf12', 'Grin3a', 'Adamts5', 'Cntnap4',
   'Nos1', 'Elavl2', 'Drd3', 'Drd1', 'Drd2', 'Sema5b', 'Tshz1', 'Gria1', 'Cadm1', 
   'Cep112', 'Zfhx4', 'Agt', 'Gli2', 'Pdgfra', 'Plp1', 'Prdm6', 'Abcc9', 'Flt1', 
   'Csf3r'
]

from matplotlib.colors import LogNorm
sc_curio.plot_markers(
    genes=markers_df['gene'],
    cell_type_column='subclass',
    filename=f'{working_dir}/figures/curio/marker_dot_subclass_pareto.png',
    colormap='coolwarm')





sc_merfish = SingleCell(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad', 
    X_key='layers/volume_log1p')\
    .with_uns(QCed=True)\
    .set_var_names('gene_symbol')\
    .filter_obs(pl.col('keep_subclass'))

markers_merfish = sc_merfish.find_markers(
    cell_type_column='class', all_genes=False)

print(set(sc_curio.var['gene_symbol'].unique()).intersection(
    sc_merfish.var['gene_symbol'].unique()))


common_subclasses = set(sc_curio.obs['subclass'].unique()).union(
    sc_merfish.obs['subclass'].unique())

subclass_annotations = pl.read_csv(
    'project/single-cell/ABC/metadata/WMB-10X/'
    '20241115/subclass_annotations.csv')\
    .filter(pl.col('subclass_id_label').is_in(common_subclasses))











markers_allen = pd.read_csv(
   'project/single-cell/ABC/metadata/WMB-10X/'
   '20241115/subclass_annotations.csv')\
    .query("subclass_id_label in @adata_query_filt.obs['subclass']")

marker_map = dict(zip(
    markers_allen['subclass_id_label'], 
    markers_allen['subclass.markers.combo'].str.split(',')))

sc.pl.stacked_violin(
    adata_query_filt,
    marker_map,  
    groupby='subclass',             
    swap_axes=False,
    dendrogram=False)
plt.savefig(f'{working_dir}/figures/marker_stacked_violin.png', dpi=300,
            bbox_inches='tight')













cells_joined = pd.read_csv(
  'project/single-cell/ABC/metadata/MERFISH-C57BL6J-638850/20231215/'
  'views/cells_joined.csv')
color_mappings = {
   'class': dict(zip(cells_joined['class'].str.replace('/', '_'), 
                     cells_joined['class_color'])),
   'subclass': {k.replace('_', '/'): v for k,v in dict(zip(
       cells_joined['subclass'].str.replace('/', '_'), 
       cells_joined['subclass_color'])).items()}
}

data = 'merfish'
level = 'subclass'

adata = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_{data}_final.h5ad')

# spatial exemplar 

sample = 'PREG1'
plot_color = adata[(adata.obs['sample'] == sample)].obs
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    plot_color['x_ffd'], plot_color['y_ffd'],
    c=[color_mappings[level][c] for c in plot_color[level]], 
    s=1, linewidths=0)
unique_classes = sorted(plot_color[level].unique(),
                      key=lambda x: int(x.split()[0]))
legend_elements = [plt.Line2D(
    [0], [0], marker='o', color='w',
    markerfacecolor=color_mappings[level][class_],
    label=class_, markersize=8)
    for class_ in unique_classes]
ax.legend(handles=legend_elements, loc='center left',
         bbox_to_anchor=(1, 0.5), frameon=False)
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
plt.savefig(f'{working_dir}/figures/{data}/spatial_example_{level}.png',
            dpi=300, bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/{data}/spatial_example_{level}.svg',
            format='svg', bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/{data}/spatial_example_{level}.pdf',
            bbox_inches='tight')

# umap

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
  adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1],
  c=[color_mappings[level][c] for c in adata.obs[level]],
  s=0.2, linewidths=0)
ax.set_aspect('equal')
ax.spines[['top', 'right', 'bottom', 'left']].set_visible(True)
ax.spines[['top', 'right', 'bottom', 'left']].set_linewidth(2)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(f'{working_dir}/figures/{data}/umap_{level}.png', dpi=300,
          bbox_inches='tight')

# marker heatmap

marker_dict = {
    '01 IT-ET Glut': ['Slc17a7', 'Satb2', 'Grin2a'],
    '02 NP-CT-L6b Glut': ['Dpp10', 'Hs3st4'],
    '05 OB-IMN GABA': ['Sox2ot', 'Dlx6os1', 'Meis2'],
    '06 CTX-CGE GABA': ['Adarb2', 'Grip1', 'Kcnip1'], 
    '07 CTX-MGE GABA': ['Nxph1', 'Lhx6', 'Sox6'],
    '08 CNU-MGE GABA': ['Galntl6'],
    '09 CNU-LGE GABA': ['Pde10a', 'Rarb', 'Drd1', 'Kcnip2'],
    '10 LSX GABA': ['Trpc4', 'Myo5b'],
    '11 CNU-HYa GABA': ['Unc5d', 'Nhs'],
    '12 HY GABA': ['Efna5', 'Nell1'],
    '13 CNU-HYa Glut': ['Slc17a6', 'Ntng1'],
    '14 HY Glut': ['Chrm3', 'Tafa1'],  
    '30 Astro-Epen': ['Slc1a2', 'Gpc5', 'Aqp4', 'Mertk'],
    '31 OPC-Oligo': ['Plp1', 'Pdgfra', 'Mbp', 'Plcl1'],
    '33 Vascular': ['Flt1', 'Cldn5', 'Bsg'],
    '34 Immune': ['Plxdc2', 'Hexb', 'C1qa', 'Ly86']
}

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

sc.tl.rank_genes_groups(adata, 'class')
sc.pl.rank_genes_groups_dotplot(
    adata,
    n_genes=10,
    values_to_plot="logfoldchanges", cmap='bwr',
    vmin=-4,
    vmax=4,
    min_logfoldchange=3,
    colorbar_title='log fold change',
    swap_axes=True,
    dendrogram=False
)
plt.savefig(f'{working_dir}/figures/{data}/tmp.png',
            dpi=300, bbox_inches='tight')



sc.pl.rank_genes_groups_dotplot(
    adata, var_names=marker_dict, values_to_plot='logfoldchanges', 
    vmin=-4, vmax=4, min_logfoldchange=3,
    cmap='bwr', swap_axes=True, dendrogram=False)
plt.savefig(f'{working_dir}/figures/{data}/marker_dotplot_de_{level}.png',
            dpi=300, bbox_inches='tight')



sc.pl.matrixplot(
    adata, marker_dict, groupby='class', layer='log1p', standard_scale='var',
    gene_symbols='gene_symbol', dendrogram=False, cmap='Blues',
    swap_axes=True)
plt.savefig(f'{working_dir}/figures/{data}/marker_heatmap_{level}.png',
            dpi=300, bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/{data}/marker_heatmap_{level}.svg',
            format='svg', bbox_inches='tight')

sc.pl.stacked_violin(
    adata, marker_dict, groupby='class', layer='log1p',
    swap_axes=True, dendrogram=False)
plt.savefig(f'{working_dir}/figures/{data}/marker_stacked_violin_{level}.png',
            dpi=300, bbox_inches='tight')

sc.pl.dotplot(
    adata, marker_dict, groupby='class', layer='log1p', 
    standard_scale='var', gene_symbols='gene_symbol', dendrogram=False, 
    cmap='Blues', swap_axes=True, var_group_labels=None)
plt.savefig(f'{working_dir}/figures/{data}/marker_dotplot_{level}.png',
            dpi=300, bbox_inches='tight')












sc.tl.rank_genes_groups(adata, groupby='class')
for group in sorted(adata.obs['class'].unique()):
    print(f'{group}:')
    print(sc.get.rank_genes_groups_df(
        adata, group=group, gene_symbols='gene_symbol').head(10))







subclass_annotations = pd.read_csv(
    'project/single-cell/ABC/metadata/WMB-10X/'
    '20241115/subclass_annotations.csv')\
    .query("subclass_id_label in @adata.obs['subclass'].unique()")

classes, markers, tf_markers = {}, {}, {}
for _, row in subclass_annotations.iterrows():
    cid = row['class_id_label']
    if cid not in classes:
        classes[cid] = []
        markers[cid] = []
        tf_markers[cid] = []
    classes[cid].append(row['subclass_id'])
    if pd.notna(row['subclass.markers.combo']):
        markers[cid].extend(row['subclass.markers.combo'].split(','))
    if pd.notna(row['subclass.tf.markers.combo']):    
        tf_markers[cid].extend(row['subclass.tf.markers.combo'].split(','))

marker_dict = {}
for cid in classes:
    m = set(markers[cid]) | set(tf_markers[cid])
    marker_dict[cid] = [x for x in m if markers[cid].count(x) > 
                        len(classes[cid])/5]

shared = set()
for c1 in classes:
    for c2 in classes:
        if c1 != c2:
            common = set(marker_dict[c1]) & set(marker_dict[c2])
            shared.update([x for x in common 
                         if marker_dict[c1].count(x) < 
                         2*marker_dict[c2].count(x)])

marker_dict = {k: list(set(v) - shared) for k,v in marker_dict.items()}








marker_dict = {
    'Neuron':['Dpp6'],
    'Excitatory Neuron':['Slc17a7'],
    'Inhibitory Neuron':['Slc32a1', 'Gad1', 'Gad2'],
    'MSN':['Drd2', 'Adora2a'],
    'Oligodendrocyte':['Olig1', 'Olig2', 'Opalin', 'Mog', 'Cldn11'],
    'OPC':['Pdgfra', 'Vcan'],
    'Astrocyte':['Aqp4', 'Gfap', 'Aldh1l1'],
    'Microglia':['Cx3cr1', 'Csf1r','C1qa','C1qb','Hexb'],
    'Endothelial':['Flt1','Cldn5','Apold1','Ly6c1'],
    'Pericyte':['Kcnj8', 'Vtn', 'Ifitm1'],
    'VSMC':['Acta2'],
    'VLMC':['Slc6a13'],
    'Ependymal':['Ccdc153', 'Rarres2', 'Tmem212'],
    'Neuroblast':['Stmn2', 'Dlx6os1', 'Igfbpl1', 'Dcx', 'Cd24a', 'Tubb3', 'Sox11', 'Dlx1'],
    'NSC':['Pclaf', 'H2afx', 'Rrm2', 'Insm1', 'Egfr', 'Mki67', 'Mcm2', 'Cdk1'],
    'Macrophage':['Mrc1', 'Pf4', 'Lyz2'],
    'T cell':['Cd3e','Nkg7','Ccl5','Ms4a4b','Cd3g'],
    'B cell':['Cd79a','Cd19','Ighm','Ighd'],
    'Neutrophil':['S100a9','Itgam','Cxcr2'],
    'Mast cell':['Hdc', 'Cma1'],
    'Dendritic cell':['Cd209a'],
}

sc.pl.matrixplot(
    adata_query,
    marker_dict,
    'class',
    dendrogram=True,
    cmap='Blues',
    standard_scale='var',
    colorbar_title='column scaled\nexpression',
)


















import sys
import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('project/utils')
from single_cell import SingleCell, options
from utils import debug
options(num_threads=-1, seed=42)
debug(third_party=True)

working_dir = 'project/spatial-pregnancy-postpart'

adata_ref = ad.read_h5ad(
    'project/single-cell/ABC/anndata/zeng_combined_10Xv3_sub.h5ad')
adata_ref.var['gene_symbol'] = adata_ref.var.index
adata_ref.var.set_index('gene_identifier', inplace=True)
sc.pp.normalize_total(adata_ref)
sc.pp.log1p(adata_ref)

adata_merfish = ad.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')
adata_merfish.X = adata_merfish.layers['volume_log1p']

adata_curio = ad.read_h5ad(
    f'{working_dir}/output/data/adata_query_curio_final.h5ad')
adata_curio.X = adata_curio.layers['log1p']
adata_curio.var.index = adata_curio.var['gene_id']
adata_curio.var.index = adata_curio.var.index.fillna('Unknown')
adata_curio.var_names_make_unique()

subclass_counts = [
    adata.obs['subclass'].value_counts() >= 10 
    for adata in [adata_merfish, adata_curio, adata_ref]]
common_cell_types = set.intersection(*[
    set(counts[counts].index) for counts in subclass_counts])

adata_ref_filt = adata_ref[
    adata_ref.obs['subclass'].isin(common_cell_types)].copy()
adata_merfish = adata_merfish[
    adata_merfish.obs['subclass'].isin(common_cell_types)]
adata_curio = adata_curio[
    adata_curio.obs['subclass'].isin(common_cell_types)]

common_genes = set(adata_merfish.var_names).intersection(adata_curio.var_names)
adata_ref_filt = adata_ref_filt[:,adata_ref_filt.var_names.isin(common_genes)]

bonafide_genes = pd.read_csv(
    'project/single-cell/ABC/metadata/WMB-10X/20241115/subclass_annotations.csv')
bonafide_genes['markers'] = bonafide_genes[
    'subclass.markers.combo'].str.split(',')
all_genes = []
for markers in bonafide_genes['markers']:
   all_genes.extend(markers)
duplicate_genes = [
    gene for gene in set(all_genes) if all_genes.count(gene) > 1]
bonafide_genes['markers'] = bonafide_genes['markers'].apply(
    lambda x: [gene for gene in x if gene not in duplicate_genes])
bonafide_genes = dict(zip(
    bonafide_genes['subclass_id'], bonafide_genes['markers']))

n=2

sc.tl.rank_genes_groups(adata_ref_filt, 'subclass', method='wilcoxon')
subclass_rank_genes = pd.concat([pd.DataFrame({
    'group': g,
    'symbol': adata_ref_filt.var.loc[
        adata_ref_filt.uns['rank_genes_groups']['names'][g], 'gene_symbol'],
    'score': adata_ref_filt.uns['rank_genes_groups']['scores'][g]
}) for g in adata_ref_filt.uns['rank_genes_groups']['names'].dtype.names])

subclass_genes, used_genes = [], set()
for group in adata_ref_filt.obs['subclass'].unique():
    if group in bonafide_genes:
        genes = [g for g in bonafide_genes[group] 
                if g in adata_ref_filt.var['gene_symbol'].values 
                and g not in used_genes][:n]
        if len(genes) < n:
            genes += [g for g in subclass_rank_genes[
                subclass_rank_genes['group'] == group
                ].sort_values('score', ascending=False)['symbol'] 
                if g not in used_genes][:n-len(genes)]
    else:
        genes = [g for g in subclass_rank_genes[
            subclass_rank_genes['group'] == group
            ].sort_values('score', ascending=False)['symbol'] 
            if g not in used_genes][:n]
    subclass_genes.extend(genes)
    used_genes.update(genes)

n=4
sc.tl.rank_genes_groups(adata_ref_filt, 'class', method='wilcoxon')
class_rank_genes = pd.concat([pd.DataFrame({
    'group': g,
    'symbol': adata_ref_filt.var.loc[
        adata_ref_filt.uns['rank_genes_groups']['names'][g], 'gene_symbol'],
    'score': adata_ref_filt.uns['rank_genes_groups']['scores'][g]
}) for g in adata_ref_filt.uns['rank_genes_groups']['names'].dtype.names])

class_genes, used_genes = [], set()
for group in class_rank_genes['group'].unique():
    genes = [g for g in class_rank_genes[class_rank_genes['group'] == group
            ].sort_values('score', ascending=False)['symbol'] 
            if g not in used_genes][:n]
    class_genes.extend(genes)
    used_genes.update(genes)

for adata in [adata_ref_filt, adata_curio, adata_merfish]:
    class_cats = adata.obs['class'].cat.categories
    subclass_cats = adata.obs['subclass'].cat.categories
    adata.uns['class_colors'] = [
        adata.obs.loc[adata.obs['class'] == cat, 'class_color'].iloc[0]
        for cat in class_cats
    ]
    adata.uns['subclass_colors'] = [
        adata.obs.loc[adata.obs['subclass'] == cat, 'subclass_color'].iloc[0]
        for cat in subclass_cats
    ]

plot_params = dict(
    log=False, standard_scale='var', dendrogram=False, swap_axes=False,
    show_gene_labels=True, gene_symbols='gene_symbol', cmap='viridis',
    figsize=(8, 16)
)

datasets = [adata_ref_filt, adata_curio, adata_merfish]
names = ['reference', 'curio', 'merfish']

for adata, name in zip(datasets, names):
    plot_params.update({'groupby': 'subclass'})
    sc.pl.heatmap(adata, subclass_genes, **plot_params)
    plt.savefig(f'{working_dir}/figures/marker_heatmaps_subclass_{name}.pdf',
                bbox_inches='tight')
    plt.close()
    
    plot_params.update({'groupby': 'class'})
    sc.pl.heatmap(adata, class_genes, **plot_params)
    plt.savefig(f'{working_dir}/figures/marker_heatmaps_class_{name}.pdf',
                bbox_inches='tight')
    plt.close()






class_genes = ['Sncg', 'Sst', 'Pvalb', 'Neurod2', 'Tbr1', 'Satb2', 'Neurod6', 
               'Fezf2', 'Sst', 'Dlx2', 'Sp8', 'Sp9', 'Kit', 'Mdga1', 'Cd4', 
               'Csf3r', 'Cx3cr1', 'Tmem119', 'Esr1', 'Nts', 'Gal', 'Calb2', 
               'Unc5d', 'Serpina3n', 'Plp1', 'Slc17a6', 'Arx', 'Lhx6', 'Lrig1', 
               'Aqp4', 'Sox9', 'Esam', 'Cdh5', 'Ly6c1']

available_genes = [g for g in class_genes 
                  if g in adata_ref_filt.var['gene_symbol'].values]

for adata, name in zip(datasets, names):
    plot_params.update({'groupby': 'class'})
    adata_plot = adata.copy()
    adata_plot.uns['class_colors'] = [
        adata.obs.loc[adata.obs['class'] == cat, 'class_color'].iloc[0]
        for cat in adata.obs['class'].cat.categories
    ]
    sc.pl.heatmap(adata_plot, available_genes, **plot_params)
    plt.savefig(f'{working_dir}/figures/marker_heatmaps_{name}_class.pdf',
                bbox_inches='tight')
    plt.close()









