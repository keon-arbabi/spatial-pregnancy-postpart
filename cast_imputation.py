import os
import gc 
import sys
import scanorama
import torch
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from utils import debug
        
warnings.filterwarnings('ignore')
debug(third_party=True)

cast_path = os.path.abspath('projects/rrg-wainberg/karbabi/CAST-keon')
sys.path.insert(0, cast_path)
if 'CAST' in sys.modules:
    del sys.modules['CAST']
import CAST
from CAST.CAST_Projection_Impute import space_project as space_project_impute
print(CAST.__file__)

#region functions ##############################################################

def plot_gene_validation(
    imputed_adatas, adata_curio, gene, cell_type_col, cmap='magma_r'):

    sample_names = list(imputed_adatas.keys())

    all_measured_log1p = []
    all_imputed_log1p = []
    merfish_data_cache = {}

    for sample in sample_names:
        sdata_ref = imputed_adatas[sample]
        merfish_cells = sdata_ref[sdata_ref.obs['source'] == 'merfish']

        measured = (
            merfish_cells[:, gene]
            .layers['merfish_log1p']
            .toarray()
            .flatten()
        )
        imputed = (
            merfish_cells[:, gene].layers['curio_log1p'].toarray().flatten()
        )
        merfish_data_cache[sample] = {
            'measured': measured,
            'imputed': imputed,
            'cells': merfish_cells
        }

        all_measured_log1p.extend(measured[~np.isnan(measured)])
        all_imputed_log1p.extend(imputed[~np.isnan(imputed)])

    curio_expr = adata_curio[:, gene].layers['log1p'].toarray().flatten()
    all_imputed_log1p.extend(curio_expr[~np.isnan(curio_expr)])

    all_measured_log1p = np.array(all_measured_log1p)
    all_imputed_log1p = np.array(all_imputed_log1p)

    global_mean_m = np.nanmean(all_measured_log1p)
    global_std_m = np.nanstd(all_measured_log1p)
    if global_std_m == 0 or np.isnan(global_std_m): global_std_m = 1

    global_mean_i = np.nanmean(all_imputed_log1p)
    global_std_i = np.nanstd(all_imputed_log1p)
    if global_std_i == 0 or np.isnan(global_std_i): global_std_i = 1

    def safe_zscore_global(data, mean, std):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            z_scores = (data - mean) / std
        return z_scores

    def safe_zscore_local(data):
        mean = np.nanmean(data)
        std = np.nanstd(data)
        if std == 0 or np.isnan(std): std = 1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            z_scores = (data - mean) / std
        return z_scores

    plot_data = []
    violin_data_m = {'condition': [], 'z_score': [], 'sample': []}
    violin_data_i = {'condition': [], 'z_score': [], 'sample': []}
    all_measured_z_flat = []
    all_imputed_z_flat = []
    all_cell_data = {'measured_z': [], 'imputed_z': []}
    subclass_means = {}

    for sample in sample_names:
        if sample not in merfish_data_cache:
            continue

        cached_data = merfish_data_cache[sample]
        measured = cached_data['measured']
        imputed = cached_data['imputed']
        merfish_cells = cached_data['cells']

        if 'condition' not in merfish_cells.obs.columns:
            condition = 'unknown'
        else:
            condition = merfish_cells.obs['condition'].iloc[0]

        measured_z_global = safe_zscore_global(
            measured, global_mean_m, global_std_m)
        imputed_z_global = safe_zscore_global(
            imputed, global_mean_i, global_std_i)

        measured_z_local = safe_zscore_local(measured)
        imputed_z_local = safe_zscore_local(imputed)

        m_nonzero_mask = measured > 0
        m_nonzero = measured[m_nonzero_mask]
        m_z_nonzero = safe_zscore_global(
            m_nonzero, global_mean_m, global_std_m)
        m_valid_idx = np.isfinite(m_z_nonzero)
        violin_data_m['z_score'].extend(m_z_nonzero[m_valid_idx])
        violin_data_m['condition'].extend([condition] * m_valid_idx.sum())
        violin_data_m['sample'].extend([sample] * m_valid_idx.sum())

        i_nonzero_mask = imputed > 0
        i_nonzero = imputed[i_nonzero_mask]
        i_z_nonzero = safe_zscore_global(
            i_nonzero, global_mean_i, global_std_i)
        i_valid_idx = np.isfinite(i_z_nonzero)
        violin_data_i['z_score'].extend(i_z_nonzero[i_valid_idx])
        violin_data_i['condition'].extend([condition] * i_valid_idx.sum())
        violin_data_i['sample'].extend([sample] * i_valid_idx.sum())

        plot_data.append({
            'sample': sample,
            'coords': merfish_cells.obs[['x_ffd', 'y_ffd']].values,
            'measured': measured,
            'imputed': imputed,
            'measured_z_global': measured_z_global,
            'imputed_z_global': imputed_z_global,
            'measured_z_local': measured_z_local,
            'imputed_z_local': imputed_z_local,
            'subclass': merfish_cells.obs[cell_type_col].values,
        })

        valid_cell_idx = np.isfinite(measured_z_local) & \
            np.isfinite(imputed_z_local)
        all_cell_data['measured_z'].extend(measured_z_local[valid_cell_idx])
        all_cell_data['imputed_z'].extend(imputed_z_local[valid_cell_idx])

        all_measured_z_flat.extend(
            measured_z_global[np.isfinite(measured_z_global)]
        )
        all_imputed_z_flat.extend(
            imputed_z_global[np.isfinite(imputed_z_global)])

        subclasses = merfish_cells.obs[cell_type_col]
        for sc in subclasses.unique():
            if sc not in subclass_means:
                subclass_means[sc] = {'measured': [], 'imputed': []}
            sc_mask = subclasses == sc
            subclass_means[sc]['measured'].append(
                np.nanmean(measured[sc_mask])
            )
            subclass_means[sc]['imputed'].append(
                np.nanmean(imputed[sc_mask])
            )

    curio_z_global = safe_zscore_global(
        curio_expr, global_mean_i, global_std_i)
    curio_z_local = safe_zscore_local(curio_expr)
    all_imputed_z_flat.extend(curio_z_global[np.isfinite(curio_z_global)])

    def get_vmin_vmax(data_flat):
        valid_data = [x for x in data_flat if np.isfinite(x)]
        if not valid_data:
            return -1, 1
        vmin = np.min(valid_data)
        vmax = np.max(valid_data)
        if vmin == vmax:
            vmin -= 0.1
            vmax += 0.1
        return vmin, vmax

    vmin_m, vmax_m = get_vmin_vmax(all_measured_z_flat)
    vmin_i, vmax_i = get_vmin_vmax(all_imputed_z_flat)

    fig, axes = plt.subplots(9, 3, figsize=(10, 27))

    for i in range(9):
        if i < len(plot_data):
            data = plot_data[i]
            coords = data['coords']
            axes[i, 0].scatter(
                coords[:, 0],
                coords[:, 1],
                s=0.1,
                c=data['measured_z_global'],
                cmap=cmap,
                vmin=vmin_m,
                vmax=vmax_m,
                rasterized=True,
            )
            axes[i, 0].set_title(f'{data['sample']} [measured]', fontsize=8)
            axes[i, 1].scatter(
                coords[:, 0],
                coords[:, 1],
                s=0.1,
                c=data['imputed_z_global'],
                cmap=cmap,
                vmin=vmin_i,
                vmax=vmax_i,
                rasterized=True,
            )
            axes[i, 1].set_title(f'{data['sample']} [imputed]', fontsize=8)
        else:
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')

        for j in range(2):
            axes[i, j].axis('equal')
            axes[i, j].axis('off')

    ax_curio = axes[0, 2]
    curio_coords = adata_curio.obs[['x_ffd', 'y_ffd']].values
    ax_curio.scatter(
        curio_coords[:, 0],
        curio_coords[:, 1],
        s=0.1,
        c=curio_z_global,
        cmap=cmap,
        vmin=vmin_i,
        vmax=vmax_i,
        rasterized=True,
    )
    ax_curio.set_title('curio [source]', fontsize=8)
    ax_curio.axis('equal')
    ax_curio.axis('off')

    ax_cell = axes[1, 2]
    cell_df = pd.DataFrame(all_cell_data)
    cell_corr, _ = pearsonr(
        cell_df['measured_z'], cell_df['imputed_z']
    )
    sns.regplot(
        data=cell_df,
        x='measured_z',
        y='imputed_z',
        ax=ax_cell,
        scatter_kws={'s': 1, 'alpha': 0.1, 'rasterized': True,
                     'edgecolor': 'none'},
        line_kws={'color': 'red', 'linewidth': 1}
    )
    ax_cell.set_title(f'cell corr: {cell_corr:.2f}', fontsize=8)
    ax_cell.set_xlabel('measured z', fontsize=7)
    ax_cell.set_ylabel('imputed z', fontsize=7)
    ax_cell.tick_params(axis='both', which='major', labelsize=6)

    ax_subclass = axes[2, 2]
    subclass_df = pd.DataFrame([
        {'subclass': sc,
         'measured_mean': np.nanmean(means['measured']),
         'imputed_mean': np.nanmean(means['imputed'])}
        for sc, means in subclass_means.items()
        if means['measured'] and means['imputed']
    ])
    subclass_corr, _ = pearsonr(
        subclass_df['measured_mean'], subclass_df['imputed_mean']
    )
    sns.regplot(
        data=subclass_df,
        x='measured_mean',
        y='imputed_mean',
        ax=ax_subclass,
        scatter_kws={'s': 10},
        line_kws={'color': 'red', 'linewidth': 1}
    )
    ax_subclass.set_title(f'subclass corr: {subclass_corr:.2f}', fontsize=8)
    ax_subclass.set_xlabel('mean measured log1p', fontsize=7)
    ax_subclass.set_ylabel('mean imputed log1p', fontsize=7)
    ax_subclass.tick_params(axis='both', which='major', labelsize=6)

    condition_order = ['CTRL', 'PREG', 'POSTPART']

    ax_violin_m = axes[3, 2]
    violin_df_m = pd.DataFrame(violin_data_m)
    ordered_conditions_m = [
        c for c in condition_order if c in violin_df_m['condition'].unique()
    ]
    violins_m = sns.violinplot(
        data=violin_df_m, x='condition', y='z_score', ax=ax_violin_m,
        cut=0, inner=None, palette='viridis', order=ordered_conditions_m
    )
    for i, condition in enumerate(ordered_conditions_m):
        condition_data = violin_df_m[violin_df_m['condition'] == condition]
        q1, q2, q3 = np.percentile(condition_data['z_score'], [25, 50, 75])
        violins_m.hlines(q2, i-0.2, i+0.2, colors='white', linestyles='-')
        violins_m.hlines(q1, i-0.2, i+0.2, colors='white', linestyles=':')
        violins_m.hlines(q3, i-0.2, i+0.2, colors='white', linestyles=':')
    ax_violin_m.set_title('measured (non-zero)', fontsize=8)
    ax_violin_m.set_xlabel('condition', fontsize=7)
    ax_violin_m.set_ylabel('z-score (across samples)', fontsize=7)
    ax_violin_m.tick_params(axis='x', rotation=45, labelsize=6)
    ax_violin_m.tick_params(axis='y', labelsize=6)

    ax_violin_i = axes[4, 2]
    violin_df_i = pd.DataFrame(violin_data_i)
    ordered_conditions_i = [
        c for c in condition_order if c in violin_df_i['condition'].unique()
    ]
    violins_i = sns.violinplot(
        data=violin_df_i, x='condition', y='z_score', ax=ax_violin_i,
        cut=0, inner=None, palette='viridis', order=ordered_conditions_i
    )
    for i, condition in enumerate(ordered_conditions_i):
        condition_data = violin_df_i[violin_df_i['condition'] == condition]
        q1, q2, q3 = np.percentile(condition_data['z_score'], [25, 50, 75])
        violins_i.hlines(q2, i-0.2, i+0.2, colors='white', linestyles='-')
        violins_i.hlines(q1, i-0.2, i+0.2, colors='white', linestyles=':')
        violins_i.hlines(q3, i-0.2, i+0.2, colors='white', linestyles=':')
    ax_violin_i.set_title('imputed (non-zero)', fontsize=8)
    ax_violin_i.set_xlabel('condition', fontsize=7)
    ax_violin_i.set_ylabel('z-score (across samples)', fontsize=7)
    ax_violin_i.tick_params(axis='x', rotation=45, labelsize=6)
    ax_violin_i.tick_params(axis='y', labelsize=6)

    for i in range(5, 9):
        axes[i, 2].axis('off')

    fig.suptitle(f'validation: {gene}', fontsize=14, y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()

    return fig

#endregion 

#region cast imputation ########################################################

working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'
output_dir = f'{working_dir}/output/merfish/CAST-IMPUTATION'
os.makedirs(output_dir, exist_ok=True)

adata_merfish = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')
adata_merfish.var.index = adata_merfish.var['gene_symbol']
adata_merfish.obs = adata_merfish.obs[
    ['sample', 'condition', 'source', 'x_ffd', 'y_ffd', 'class', 'subclass']]
adata_merfish.obs['unknown'] = 'unknown'
adata_merfish.var = adata_merfish.var[['gene_symbol']]
adata_merfish.var.index.name = None
adata_merfish.layers['log1p'] = adata_merfish.layers['volume_log1p'].copy()
adata_merfish.X = adata_merfish.layers['log1p'].copy()

del adata_merfish.uns, adata_merfish.obsm, adata_merfish.varm, adata_merfish.obsp
del adata_merfish.layers['volume_log1p']

adata_curio = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_curio_final.h5ad')
adata_curio.obs = adata_curio.obs[
    ['sample', 'condition', 'source', 'x_ffd', 'y_ffd', 'class', 'subclass']]
adata_curio.obs['unknown'] = 'unknown'
adata_curio = adata_curio[:,adata_curio.var['protein_coding']]
adata_curio.var = adata_curio.var[['gene_symbol']]
adata_curio.var.index.name = None
adata_curio.X = adata_curio.layers['log1p'].copy()
print(adata_curio.shape[1])

sc.pp.filter_genes(adata_curio, min_cells=100)
sc.pp.filter_genes(adata_curio, min_counts=200)
sc.pp.highly_variable_genes(
    adata_curio, n_top_genes=8000, flavor='seurat_v3', batch_key='sample')
adata_curio = adata_curio[:, adata_curio.var['highly_variable'] | \
    adata_curio.var.index.isin(set(adata_merfish.var['gene_symbol']))]
print(adata_curio.shape[1])

del adata_curio.uns, adata_curio.obsm, adata_curio.varm, adata_curio.obsp

print(adata_curio.layers['log1p'][0:10,0:5].todense())
print(adata_merfish.layers['log1p'][0:10,0:5].todense())

cells_joined = pd.read_csv(
  'projects/rrg-wainberg/single-cell/ABC/metadata/MERFISH-C57BL6J-638850/'
  '20231215/views/cells_joined.csv')
color_dict = {
   'class': dict(zip(cells_joined['class'].str.replace('/', '_'), 
                     cells_joined['class_color'])),
   'subclass': {k.replace('_', '/'): v for k,v in dict(zip(
       cells_joined['subclass'].str.replace('/', '_'), 
       cells_joined['subclass_color'])).items()}
}
color_dict = color_dict['class']
color_dict['unknown'] = 'black'

batch_key = 'source'
level = 'class' 
merfish_samples = adata_merfish.obs['sample'].unique()
imputed_adatas, list_ts_results = {}, {}

for sample, condition in adata_merfish.obs[[
    'sample', 'condition']].drop_duplicates().values:
    
    print(f'\nprocessing MERFISH sample: {sample}')
    sample_dir = f'{output_dir}/sample_{sample}'
    os.makedirs(sample_dir, exist_ok=True)
    
    list_ts_file = f'{sample_dir}/list_ts.pt'
    sdata_ref_file = f'{sample_dir}/sdata_ref.h5ad'
    harmony_file = f'{sample_dir}/harmony_integration.h5ad'

    if os.path.exists(list_ts_file) and os.path.exists(sdata_ref_file):
        print(f'  loading cached results for {sample}')
        sdata_ref = sc.read_h5ad(sdata_ref_file)
        list_ts = torch.load(list_ts_file, weights_only=False)
        imputed_adatas[sample] = sdata_ref
        list_ts_results[sample] = list_ts
        continue
    
    if os.path.exists(harmony_file):
        print(f'  loading precomputed harmony integration')
        adata_comb_outer = sc.read_h5ad(harmony_file)
    else:
        print(f'  creating combined dataset...')
        adata_curio_ref = adata_curio.copy()
        adata_merfish_sample = adata_merfish[
            adata_merfish.obs['sample'] == sample].copy()
    
        print(f'  curio reference: {adata_curio_ref.shape[0]} cells')
        print(f'  merfish {sample}: {adata_merfish_sample.shape[0]} cells')

        adata_comb_inner = ad.concat(
            [adata_curio_ref, adata_merfish_sample], axis=0, join='inner')
        adata_comb_inner = adata_comb_inner[
            :, adata_comb_inner.var_names.sort_values()]
            
        adata_curio_s, adata_merfish_s = scanorama.correct_scanpy(
            [adata_curio_ref, adata_merfish_sample])
        adata_comb_inner.layers['X_scanorama'] = ad.concat(
            [adata_curio_s, adata_merfish_s], axis=0, join='inner').X.copy()
        del adata_curio_s, adata_merfish_s; gc.collect()

        print('  running harmony integration...')        
        adata_comb = CAST.Harmony_integration(
            sdata_inte=adata_comb_inner,
            scaled_layer='X_scanorama',
            use_highly_variable_t=False,
            batch_key='source',
            n_components=50,
            umap_n_neighbors=15,
            umap_n_pcs=30,
            min_dist=0.1,
            spread_t=1.0,
            source_sample_ctype_col=level,
            color_dict=color_dict,
            output_path=sample_dir,
            ifplot=False,
            ifcombat=False)
        
        adata_comb_outer = ad.concat(
            [adata_curio_ref, adata_merfish_sample], axis=0, join='outer')
        adata_comb_outer.obsm['X_pca_harmony'] = \
            adata_comb_inner.obsm['X_pca_harmony'].copy()
        
        adata_comb_outer.write_h5ad(harmony_file)
        gc.collect()
    
    print('  running expression imputation...')
    idx_source = adata_comb_outer.obs['source'] == 'curio'
    idx_target = adata_comb_outer.obs['source'] == 'merfish'
    
    pc_feature_name = 'X_pca_harmony'
    source_cell_pc_feature = adata_comb_outer[idx_source, :]\
        .obsm[pc_feature_name]
    target_cell_pc_feature = adata_comb_outer[idx_target, :]\
        .obsm[pc_feature_name]

    sdata_ref, list_ts = space_project_impute(
        sdata_inte=adata_comb_outer,
        idx_source = idx_source,
        idx_target = idx_target,
        source_sample='curio',
        target_sample='merfish',
        coords_source=np.array(adata_comb_outer[idx_source]
            .obs[['x_ffd', 'y_ffd']]),
        coords_target=np.array(adata_comb_outer[idx_target]
            .obs[['x_ffd', 'y_ffd']]),
        source_cell_pc_feature=source_cell_pc_feature,
        target_cell_pc_feature=target_cell_pc_feature,
        k2=5, 
        raw_layer='counts',
        source_sample_ctype_col=None, 
        output_path=sample_dir,
        umap_feature='X_umap',
        ave_dist_fold=15, 
        alignment_shift_adjustment=0,
        adjust_shift=False,
        metric_t='cosine',
        working_memory_t=1000,
        ifplot=False
        )
    
    print('  saving results...')
    imputed_adatas[sample] = sdata_ref
    list_ts_results[sample] = list_ts
    sdata_ref.write_h5ad(sdata_ref_file)
    torch.save(list_ts, list_ts_file)
    del adata_comb_outer, sdata_ref; gc.collect()

common_genes = sorted(set(adata_curio.var_names)\
    .intersection(set(adata_merfish.var_names)))
os.makedirs(f'{working_dir}/figures/merfish/imputation', exist_ok=True)

gene = 'Fgf1'

for gene in common_genes:
    fig = plot_gene_validation(
            imputed_adatas=imputed_adatas, 
            adata_curio=adata_curio, 
            gene=gene,
            cell_type_col='subclass',
            cmap='magma_r')
    plt.savefig(f'{working_dir}/figures/merfish/imputation/{gene}.png', dpi=200)
    plt.close(fig)

imputed_adata = sc.concat(imputed_adatas, axis=0)
imputed_adata.write_h5ad(
    f'{working_dir}/output/data/adata_imputed_merfish_cast.h5ad')

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

cast_path = os.path.abspath('projects/rrg-wainberg/karbabi/CAST-keon')
sys.path.insert(0, cast_path)
from CAST.CAST_Projection_Impute import average_dist

working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'
ave_dist_fold = 15
alignment_shift = 0

merfish_coords = adata_merfish[
    adata_merfish.obs['sample'] == 'CTRL1'].obs[['x_ffd', 'y_ffd']].values
curio_coords = adata_curio.obs[['x_ffd', 'y_ffd']].values

merfish_avg_dist, _, _, _ = average_dist(merfish_coords, working_memory_t=1000)
curio_avg_dist, _, _, _ = average_dist(curio_coords, working_memory_t=1000)

merfish_radius = ave_dist_fold * merfish_avg_dist + alignment_shift
curio_radius = ave_dist_fold * curio_avg_dist + alignment_shift

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.scatter(merfish_coords[:, 0], merfish_coords[:, 1], s=1, color='grey')
random_merfish_idx = np.random.randint(0, len(merfish_coords))
random_merfish = merfish_coords[random_merfish_idx]
circle1 = Circle(random_merfish, merfish_radius, fill=False, color='red')
ax1.scatter(random_merfish[0], random_merfish[1], s=20, color='red')
ax1.add_artist(circle1)
ax1.set_title(f'MERFISH (r={merfish_radius:.4f})')
ax1.axis('equal')
ax1.axis('off')

ax2.scatter(curio_coords[:, 0], curio_coords[:, 1], s=1, color='grey')
random_curio_idx = np.random.randint(0, len(curio_coords))
random_curio = curio_coords[random_curio_idx]
circle2 = Circle(random_curio, curio_radius, fill=False, color='red')
ax2.scatter(random_curio[0], random_curio[1], s=20, color='red')
ax2.add_artist(circle2)
ax2.set_title(f'Curio (r={curio_radius:.4f})')
ax2.axis('equal')
ax2.axis('off')

plt.tight_layout()
plt.savefig(
    f'{working_dir}/figures/merfish/imputation_radius_comparison.png', 
    dpi=300)

#endregion

#region tangram imputation ######################################################
import os
import torch
import scanorama
import scanpy as sc
import tangram as tg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'
data_dir = f'{working_dir}/output/merfish/TANGRAM-IMPUTATION'
os.makedirs(data_dir, exist_ok=True)

def segment_spatial_data(
    adata_main, adata_aux, sample_name_main, n_splits_approx):

    sample_adata_main = adata_main[
        adata_main.obs['sample'] == sample_name_main
    ].copy()
    
    coords_main = sample_adata_main.obsm['spatial']
    coords_aux = adata_aux.obsm['spatial']

    xmin, ymin = coords_main.min(axis=0)
    xmax, ymax = coords_main.max(axis=0)

    epsilon = 1e-6
    x_lines = np.linspace(xmin, xmax + epsilon, n_splits_approx + 1)
    y_lines = np.linspace(ymin, ymax + epsilon, n_splits_approx + 1)

    segmented_adatas_main = []
    segmented_adatas_aux = []
    segment_count = 0
    for i in range(len(x_lines) - 1):
        for j in range(len(y_lines) - 1):
            xmin_seg, xmax_seg = x_lines[i], x_lines[i+1]
            ymin_seg, ymax_seg = y_lines[j], y_lines[j+1]
            
            mask_main = (
                (coords_main[:, 0] >= xmin_seg) & (coords_main[:, 0] < xmax_seg) &
                (coords_main[:, 1] >= ymin_seg) & (coords_main[:, 1] < ymax_seg)
            )
            adata_segment_main = sample_adata_main[mask_main].copy()

            if adata_segment_main.n_obs > 0:
                mask_aux = (
                    (coords_aux[:, 0] >= xmin_seg) & (coords_aux[:, 0] < xmax_seg) &
                    (coords_aux[:, 1] >= ymin_seg) & (coords_aux[:, 1] < ymax_seg)
                )
                adata_segment_aux = adata_aux[mask_aux].copy()
                
                if adata_segment_aux.n_obs > 0:
                    segment_count += 1
                    m_obs = adata_segment_main.n_obs
                    a_obs = adata_segment_aux.n_obs
                    print(f'Segment {segment_count}: main={m_obs}, aux={a_obs}')
                    segmented_adatas_main.append(adata_segment_main)
                    segmented_adatas_aux.append(adata_segment_aux)
    return segmented_adatas_main, segmented_adatas_aux

adata_merfish = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')
adata_merfish.var.index = adata_merfish.var['gene_symbol']
adata_merfish.obs = adata_merfish.obs[
    ['sample', 'condition', 'source', 'x_ffd', 'y_ffd', 'class', 'subclass']]
adata_merfish.var = adata_merfish.var[['gene_symbol']]
adata_merfish.var.index.name = None
adata_merfish.X = adata_merfish.layers['volume_log1p'].copy()
adata_merfish.obsm['spatial'] = adata_merfish.obs[['x_ffd', 'y_ffd']].values
del adata_merfish.uns, adata_merfish.varm, adata_merfish.obsp

adata_curio = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_curio_final.h5ad')
adata_curio.obs = adata_curio.obs[
    ['sample', 'condition', 'source', 'x_ffd', 'y_ffd', 'class', 'subclass']]
adata_curio = adata_curio[:,adata_curio.var['protein_coding']]
adata_curio.var = adata_curio.var[['gene_symbol']]
adata_curio.var.index.name = None
adata_curio.X = adata_curio.layers['log1p'].copy()
adata_curio.obsm['spatial'] = adata_curio.obs[['x_ffd', 'y_ffd']].values
del adata_curio.uns, adata_curio.varm, adata_curio.obsp

adatas_merfish_segmented, adatas_curio_segmented = segment_spatial_data(
    adata_merfish, adata_curio, 'CTRL1', 3
)

ad_maps_list = []
ad_ges_list = []
for adata_m_seg, adata_c_seg in zip(
    adatas_merfish_segmented, adatas_curio_segmented):
    print(adata_m_seg.obs['sample'].unique())
    tg.pp_adatas(adata_curio, adata_m_seg, genes=None)
    ad_map_segment = tg.map_cells_to_space(
        adata_sc=adata_curio,
        adata_sp=adata_m_seg,
        mode='cells',
        cluster_label='subclass', #not used
        density_prior='uniform',
        num_epochs=300,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    ad_maps_list.append(ad_map_segment)
    
    ad_ge_segment = tg.project_genes(
        adata_map=ad_map_segment, adata_sc=adata_curio
    )
    ad_ges_list.append(ad_ge_segment)

ad_ge_concat = sc.concat(ad_ges_list, axis=0)

tg.plot_genes_sc(
    genes=['fgf1'],
    adata_measured=adata_merfish[adata_merfish.obs['sample'] == 'CTRL1'],
    adata_predicted=ad_ge_concat[ad_ge_concat.obs['sample'] == 'CTRL1'],
    perc=0.05,
    spot_size=0.05,
)

tg.plot_training_scores(ad_ge_concat, bins=20, alpha=0.5)
plt.savefig(
    f'{working_dir}/figures/merfish/imputation_tangram_training_scores.png',
    dpi=300)

adata_merfish_ctrl1 = adata_merfish[
    adata_merfish.obs['sample'] == 'CTRL1'
].copy()

#endregion