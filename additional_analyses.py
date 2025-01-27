import sys
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('project/utils')
from single_cell import SingleCell, options
options(num_threads=-1, seed=42)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

working_dir = 'project/spatial-pregnancy-postpart'

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

adata_curio = SingleCell(
    f'{working_dir}/output/curio/adata_comb_cast_stack.h5ad')
adata_merfish = SingleCell(
    f'{working_dir}/output/merfish/adata_comb_cast_stack.h5ad')




























data = 'curio'
level = 'class'

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


sc.tl.rank_genes_groups(adata, groupby='class', method='wilcoxon')
for group in sorted(adata.obs['class'].unique()):
    print(f'{group}:')
    print(sc.get.rank_genes_groups_df(
        adata, group=group, gene_symbols='gene_symbol').head(50))

marker_dict = {
    '01 IT-ET Glut': ['Slc17a7', 'Satb2', 'Grin2a'],
    '02 NP-CT-L6b Glut': ['Dpp10', 'Hs3st4'],
    '05 OB-IMN GABA': ['Sox2ot', 'Dlx6os1', 'Meis2'],
    '06 CTX-CGE GABA': ['Adarb2', 'Grip1', 'Kcnip1'], 
    '07 CTX-MGE GABA': ['Nxph1', 'Lhx6', 'Sox6'],
    '08 CNU-MGE GABA': ['Galntl6'],
    '09 CNU-LGE GABA': ['Pde10a', 'Rarb', 'Drd1', 'Kcnip2'],
    '10 LSX GABA': ['Trpc4', 'Myo5b'],
    '11 CNU-HYa GABA': ['Unc5d'],
    '12 HY GABA': ['Efna5', 'Nell1'],
    '13 CNU-HYa Glut': ['Slc17a6', 'Ntng1'],
    '14 HY Glut': ['Chrm3', 'Tafa1'],  
    '30 Astro-Epen': ['Slc1a2', 'Gpc5', 'Aqp4'],
    '31 OPC-Oligo': ['Plp1', 'Pdgfra', 'Mbp'],
    '33 Vascular': ['Flt1', 'Cldn5', 'Bsg'],
    '34 Immune': ['Plxdc2', 'Hexb', 'C1qa']
}

sc.pl.matrixplot(
    adata, marker_dict, 'class', dendrogram=False, cmap='Reds',
    standard_scale='var', gene_symbols='gene_symbol',
    var_group_labels=None)
plt.savefig(f'{working_dir}/figures/{data}/marker_heatmap_{level}.png',
            dpi=300, bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/{data}/marker_heatmap_{level}.svg',
            format='svg', bbox_inches='tight')

sc.pl.dotplot(adata, marker_dict, 'class', dendrogram=False)
plt.savefig(f'{working_dir}/figures/{data}/marker_dotplot_{level}.png',
            dpi=300, bbox_inches='tight')




















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









