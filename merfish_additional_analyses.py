import sys
import scanpy
import scanorama
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('project/utils')

from single_cell import SingleCell, options
options(num_threads=-1, seed=42)

working_dir = 'project/spatial-pregnancy-postpart'

sc_query = SingleCell(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad',
    X_key='layers/volume_log1p')

sc_ref = SingleCell(
    f'{working_dir}/output/data/adata_ref_zeng_imputed.h5ad')

sc_query, sc_ref = scanorama.correct_scanpy([
    sc_query.to_anndata(), sc_ref.to_anndata()])
sc_query = SingleCell(sc_query); sc_ref = SingleCell(sc_ref)

sc_query, sc_ref = sc_query.PCA(
    sc_ref, allow_float=True, hvg_column=None)





adata_query = scanpy.read(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')

