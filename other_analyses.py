import sys, scipy
import polars as pl
import matplotlib.pyplot as plt
sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')

from single_cell import SingleCell, options
options(num_threads=-1, seed=42)

working_dir = '/home/karbabi/projects/def-wainberg/karbabi/' \
    'spatial-pregnancy-postpart'

sc_ref = SingleCell(
    'projects/def-wainberg/single-cell/ABC/h5ad/combined_10Xv3.h5ad')

sc_query = SingleCell(
    f'{working_dir}/output/data/adata_query_curio_final.h5ad')\
    .qc(allow_float=True)

sc_query = sc_query\
    .hvg(allow_float=True)\
    .normalize(allow_float=True)\
    .PCA()\
    .neighbors()\
    .embed()

sc_query.plot_embedding(
    color_column='class', 
    filename=f'{working_dir}/figures/curio/umap_class.png',
    cells_to_plot_column=None,
    legend_kwargs={
        'fontsize': 7,
        'loc': 'center left'},
    savefig_kwargs={'dpi': 600})

sc_query.save(f'{working_dir}/output/data/adata_query_curio_final_filt.h5ad',
              overwrite=True)
















mmc = pl.read_csv(
    f'{working_dir}/output/curio/data/curio_mmc_corr_annotations.csv')
sc_ref = SingleCell(
    f'{working_dir}/output/data/adata_ref_zeng_imputed.h5ad')\
    .with_uns(QCed=True)\
    .with_uns(normalized=True)\
    .filter_obs(pl.col.brain_section_label_x.is_in([
        'C57BL6J-638850.49', 'C57BL6J-638850.48', 
        'C57BL6J-638850.47', 'C57BL6J-638850.46']))



sc_query = SingleCell(
    f'{working_dir}/output/data/adata_query_curio_positioned_final.h5ad')\
    .with_uns(QCed=True)\
    .with_uns(normalized=False)\
    .filter_var(pl.col.gene.is_in(sc_ref.var['gene_symbol']))\
    .join_obs(mmc, left_on='_index', right_on='cell_id')


mdr = 0.05
mlfc = 1.2

query_cast_markers = sc_query.find_markers(
    cell_type_column='subclass',
    all_genes=True)\
    .cast({'cell_type': pl.String})\
    .filter((pl.col.detection_rate > mdr) & (pl.col.fold_change > mlfc))

query_mmc_markers = sc_query.find_markers(
    cell_type_column='subclass_right',
    all_genes=True)\
    .cast({'cell_type': pl.String})\
    .filter((pl.col.detection_rate > mdr) & (pl.col.fold_change > mlfc))

ref_markers = sc_ref.find_markers(
    cell_type_column='subclass',
    all_genes=True)\
    .cast({'cell_type': pl.String})\
    .filter((pl.col.detection_rate > mdr) & (pl.col.fold_change > mlfc))

matched_markers = ref_markers.join(
    query_cast_markers, 
    on=['cell_type', 'gene'], 
    how='inner')

print('corr ref vs query cast')
print(matched_markers.shape[0])
print(scipy.stats.spearmanr(
    matched_markers['fold_change'], 
    matched_markers['fold_change_right']))

# corr ref vs query cast
# 13879
# SignificanceResult(statistic=0.2965941996495267, pvalue=7.723594458794918e-280)

matched_markers = ref_markers.join(
    query_mmc_markers, 
    on=['cell_type', 'gene'], 
    how='inner')

print('corr ref vs query mmc')
print(matched_markers.shape[0])
print(scipy.stats.spearmanr(
    matched_markers['fold_change'], 
    matched_markers['fold_change_right']))

# corr ref vs query mmc
# 15600
# SignificanceResult(statistic=0.3865836709477447, pvalue=0.0)