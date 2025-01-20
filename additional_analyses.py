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









