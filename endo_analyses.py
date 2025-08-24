import os
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from matplotlib.gridspec import GridSpec
from scipy.stats import zscore

warnings.filterwarnings('ignore')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400
sc.settings.verbosity = 0

WORKING_DIR = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'
CELL_TYPE_COL = 'subclass'
TARGET_CELL_TYPE = 'Endo NN'

adata_curio = sc.read_h5ad(
    f'{WORKING_DIR}/output/data/adata_query_curio_final.h5ad')
adata_merfish = sc.read_h5ad(
    f'{WORKING_DIR}/output/data/adata_query_merfish_final.h5ad')
adata_merfish.var.index = adata_merfish.var['gene_symbol']

for ad in [adata_curio, adata_merfish]:
    ad.var_names_make_unique()
    for col in ['class', 'subclass']:
        ad.obs[col] = ad.obs[col].astype(str)\
            .str.extract(r'^(\d+)\s+(.*)', expand=False)[1]
    ad.obs = ad.obs[['sample', 'condition', 'source', 'x_ffd', 'y_ffd',
                     'class', 'subclass']]
    ad.var = ad.var[['gene_symbol']]
    ad.var.index.name = None
    g = ad.var_names
    ad.var['mt'] = g.str.match(r'^(mt-|MT-)')
    ad.var['ribo'] = g.str.match(r'^(Rps|Rpl)')
    for key in ('uns', 'varm', 'obsp'):
        if hasattr(ad, key):
            try:
                delattr(ad, key)
            except:
                pass

adata_curio = adata_curio[adata_curio.obs[CELL_TYPE_COL] == TARGET_CELL_TYPE].copy()
adata_merfish = adata_merfish[adata_merfish.obs[CELL_TYPE_COL] == TARGET_CELL_TYPE].copy()

def score_and_qc(ad, mt_max=10.0, frac_drop=0.10, k_mad=4.0):
    ad = ad.copy()
    ad.var['mt'] = ad.var_names.str.match(r'^(mt-|MT-)')
    sc.pp.calculate_qc_metrics(
        ad, qc_vars=['mt'], percent_top=None, inplace=True
    )
    gsets = {
        'EC_score': [
            'Pecam1', 'Cldn5', 'Kdr', 'Klf2', 'Klf4', 'Mfsd2a',
            'Slco1a4', 'Abcb1a', 'Slc2a1', 'Cdh5', 'Esam', 'Tek', 'Tie1'],
        'Neuron_score': [
            'Snap25', 'Rbfox3', 'Tubb3', 'Syt1', 'Slc17a7', 'Gad1', 'Gad2'],
        'Pericyte_score': [
            'Rgs5', 'Pdgfrb', 'Kcnj8', 'Abcc9', 'Acta2'],
        'Oligo_score': [
            'Plp1', 'Mbp', 'Mog', 'Sox10', 'Cnp'],
    }
    for k, v in gsets.items():
        genes = [g for g in v if g in ad.var_names]
        if genes: sc.tl.score_genes(ad, genes, score_name=k)
        else: ad.obs[k] = 0.0

    obs, grp = ad.obs, ad.obs.groupby('sample')

    def get_mad_z(s):
        med = grp[s].transform('median')
        mad = (obs[s] - med).abs().groupby(obs['sample']).transform('median')
        return (obs[s] - med).abs() / mad.replace(0, 1.0)

    mt_thr = grp['pct_counts_mt'].transform(lambda x: np.percentile(x, 99))
    qc1_mask = (get_mad_z('n_genes_by_counts') <= k_mad) & \
               (get_mad_z('total_counts') <= k_mad) & \
               (obs['pct_counts_mt'] <= np.minimum(mt_max, mt_thr))
    ad = ad[qc1_mask].copy()

    grp = ad.obs.groupby('sample')
    others = ['Neuron_score', 'Pericyte_score', 'Oligo_score']
    is_mer = 'merfish' in ad.obs['source'].unique()
    adv = (grp['EC_score'].transform(lambda s: s.rank(pct=True)) -
           grp[others].transform(lambda s: s.rank(pct=True)).max(1)
          ) if is_mer else (ad.obs['EC_score'] - ad.obs[others].max(1))

    thr = adv.groupby(ad.obs['sample']).transform('quantile', frac_drop)
    ad = ad[adv >= thr].copy()
    return ad

adata_curio = score_and_qc(adata_curio)

PROGS = {
    'BBB_prog': ['Mfsd2a', 'Slco1a4', 'Abcb1a', 'Slc2a1', 'Cldn5', 'Ocln', 'Tjp1', 'Lsr', 'Slc7a5'],
    'Tip_prog': ['Apln', 'Esm1', 'Kdr', 'Angpt2', 'Dll4', 'Flt4', 'Ackr3'],
    'Arterial_prog': ['Gja5', 'Efnb2', 'Sema3g', 'Dll4', 'Hey1', 'Gkn3'],
    'Venous_prog': ['Nr2f2', 'Ephb4', 'Vwf', 'Slc38a5', 'Nrp2'],
    'Shear_prog': ['Klf2', 'Klf4', 'Nos3'],
    'IFN_prog': ['Irf7', 'Isg15', 'Ifit1', 'Stat1', 'Mx1', 'Ifit3', 'Cxcl10'],
}

sc.pp.normalize_total(adata_curio, target_sum=1e4)
sc.pp.log1p(adata_curio)

for k, v in PROGS.items():
    v = [g for g in v if g in adata_curio.var_names]
    if len(v) >= 2:
        sc.tl.score_genes(adata_curio, v, score_name=k, use_raw=False)
    else:
        adata_curio.obs[k] = 0.0
prog_cols = list(PROGS.keys())

sc.pp.highly_variable_genes(adata_curio, batch_key='sample')
sc.pp.normalize_total(adata_curio)
sc.pp.log1p(adata_curio)
sc.tl.pca(adata_curio)
sc.pp.neighbors(adata_curio)
sc.tl.umap(adata_curio)

for r in [0.4, 0.6, 0.8, 1.0]:
    sc.tl.leiden(adata_curio, resolution=r, key_added=f'leiden_r{r}')

r = 0.8
adata_curio.obs['leiden'] = adata_curio.obs[f'leiden_r{r}']\
    .astype('category')

med = adata_curio.obs.groupby('leiden')[prog_cols].median()
lab = med.idxmax(axis=1)
adata_curio.obs['state'] = adata_curio.obs['leiden'].map(lab)\
    .astype('category')



ps = (adata_curio.obs.groupby(['sample', 'condition'])[prog_cols]
      .median().reset_index())
abs_m = ps.groupby('condition')[prog_cols].median().reindex(
    ['CTRL', 'PREG', 'POSTPART'])
dlt = abs_m.sub(abs_m.loc['CTRL'], axis=1).drop('CTRL')

props = pd.crosstab(
    adata_curio.obs['condition'], adata_curio.obs['leiden'], normalize='index'
).reindex(['CTRL', 'PREG', 'POSTPART'])

# Main figure and outer grid for vertical sections
fig = plt.figure(figsize=(16, 20))
gs_outer = GridSpec(3, 1, figure=fig, hspace=0.6,
                    height_ratios=[2, 1.2, 2.2])

# --- Top Section: UMAPs and Delta Heatmap ---
gs_top = gs_outer[0].subgridspec(2, 5, wspace=0.6, hspace=0.7)

ax0 = fig.add_subplot(gs_top[0, 0])
sc.pl.umap(adata_curio, color='leiden',
           ax=ax0, show=False, title='Leiden Clusters')

leiden_colors = adata_curio.uns['leiden_colors']
ax_bar = fig.add_subplot(gs_top[0, 1])
props.plot(kind='bar', stacked=True, ax=ax_bar, color=leiden_colors, legend=None)
ax_bar.set_title('Cluster Proportions')
ax_bar.set_ylabel('Proportion')
ax_bar.set_xlabel('')
ax_bar.tick_params(axis='x', rotation=45)

axes_row0 = [fig.add_subplot(gs_top[0, 2]), fig.add_subplot(gs_top[0, 3])]
axes_row1 = [fig.add_subplot(gs_top[1, 0]), fig.add_subplot(gs_top[1, 1]),
             fig.add_subplot(gs_top[1, 2]), fig.add_subplot(gs_top[1, 3])]
for ax, col in zip(axes_row0 + axes_row1, prog_cols):
    sc.pl.umap(adata_curio, color=col, ax=ax, show=False, cmap='GnBu')

axh = fig.add_subplot(gs_top[:, 4])
v = np.abs(dlt.values).max()
sns.heatmap(dlt, ax=axh, cmap='seismic', center=0, vmin=-v, vmax=v,
            annot=True, fmt='.2f', cbar_kws={'shrink': 0.4})
axh.set_title('Δ Median Program Score vs CTRL')
axh.tick_params(axis='y', rotation=0)

# --- Middle Section: Gene Expression Heatmap ---
gs_mid = gs_outer[1].subgridspec(1, 1)
axh2 = fig.add_subplot(gs_mid[0])
ordered_genes = [g for prog in PROGS.values() for g in prog]
genes_to_plot = [g for g in ordered_genes if g in adata_curio.var_names]
df_expr = pd.DataFrame(adata_curio[:, genes_to_plot].X.toarray(),
    index=adata_curio.obs.index, columns=genes_to_plot)
df_expr['condition'] = adata_curio.obs['condition']
mean_exp = df_expr.groupby('condition').mean().reindex(['CTRL', 'PREG', 'POSTPART'])
zscored_exp = mean_exp.apply(zscore)
sns.heatmap(zscored_exp, ax=axh2, cmap='seismic', center=0,
            vmin=-2, vmax=2,
            cbar_kws={'label': 'Z-scored Expression', 'shrink': 0.8})
axh2.tick_params(axis='x', rotation=90)
axh2.tick_params(axis='y', rotation=0)
axh2.set_title('')
axh2.set_xlabel('')

prog_ticks, current_pos = [], 0
for prog_name, gene_list in PROGS.items():
    num_genes_in_prog = len([g for g in gene_list if g in genes_to_plot])
    if num_genes_in_prog > 0:
        tick_pos = current_pos + (num_genes_in_prog / 2.0)
        prog_ticks.append(tick_pos)
        current_pos += num_genes_in_prog
y_pos = -0.7
for tick, label in zip(prog_ticks, PROGS.keys()):
    axh2.text(tick, y_pos, label, ha='center', va='bottom', fontsize=12)

gene_group_boundaries = np.cumsum(
    [len([g for g in v if g in genes_to_plot]) for v in PROGS.values()]
)[:-1]
for pos in gene_group_boundaries:
    axh2.axvline(x=pos, color='white', linewidth=2)

# --- Bottom Section: Large Spatial Plots ---
gs_bot = gs_outer[2].subgridspec(1, 3, wspace=0.2)
spatial_axes = [fig.add_subplot(gs_bot[0, i]) for i in range(3)]
conditions = ['CTRL', 'PREG', 'POSTPART']
vmin, vmax = adata_curio.obs['Tip_prog'].min(), adata_curio.obs['Tip_prog'].max()
norm = plt.Normalize(vmin=vmin, vmax=vmax)
cmap = 'GnBu'

for ax, cond in zip(spatial_axes, conditions):
    data = adata_curio.obs[adata_curio.obs['condition'] == cond]
    sns.scatterplot(data=data, x='x_ffd', y='y_ffd', hue='Tip_prog',
                    palette=cmap, norm=norm, s=25, linewidth=0,
                    ax=ax, legend=False)
    ax.set_title(cond)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False)

cbar_ax = fig.add_axes([0.4, 0.05, 0.2, 0.01])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
fig.colorbar(sm, cax=cbar_ax, orientation='horizontal',
             label='Tip_prog Score')

plt.savefig(f'{WORKING_DIR}/figures/endo_curio_PCA_panel.png',
            bbox_inches='tight', dpi=300)
plt.close()




import os
import scvi
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as sp

os.environ['SCIPY_ARRAY_API'] = '1'
import scarches as sca

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

scvi.settings.seed = 0
sc.settings.verbosity = 2

condition_colors = {
    'CTRL': '#7209b7',
    'PREG': '#b5179e',
    'POSTPART': '#f72585'
}
modality_colors = {'merfish': '#4361ee','curio': '#4cc9f0'}

#region load data #############################################################

working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'
cell_type_col = 'subclass'

adata_curio = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_curio_final.h5ad')
adata_merfish = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')
adata_merfish.var.index = adata_merfish.var['gene_symbol']

for ad in [adata_curio, adata_merfish]:
    ad.var_names_make_unique()
    for col in ['class','subclass']:
        ad.obs[col] = ad.obs[col].astype(str)\
            .str.extract(r'^(\d+)\s+(.*)', expand=False)[1]
    ad.obs = ad.obs[['sample','condition','source','x_ffd','y_ffd',
                     'class','subclass']]
    ad.var = ad.var[['gene_symbol']]
    ad.var.index.name = None
    g = ad.var_names
    ad.var['mt'] = g.str.match(r'^(mt-|MT-)')
    ad.var['ribo'] = g.str.match(r'^(Rps|Rpl)')
    for key in ('uns','varm','obsp'):
        if hasattr(ad, key):
            try: delattr(ad, key)
            except: pass

adata_curio = adata_curio[adata_curio.obs['subclass']=='Endo NN'].copy()
adata_merfish = adata_merfish[adata_merfish.obs['subclass']=='Endo NN'].copy()


def score_and_qc(ad, mt_max=10.0, frac_drop=0.10, k_mad=4.0):
    ad = ad.copy()
    ad.var['mt'] = ad.var_names.str.match(r'^(mt-|MT-)')
    sc.pp.calculate_qc_metrics(
        ad, qc_vars=['mt'], percent_top=None, inplace=True
    )
    gsets = {
        'EC_score': [
            'Pecam1', 'Cldn5', 'Kdr', 'Klf2', 'Klf4', 'Mfsd2a',
            'Slco1a4', 'Abcb1a', 'Slc2a1', 'Cdh5', 'Esam', 'Tek', 'Tie1'],
        'Neuron_score': [
            'Snap25', 'Rbfox3', 'Tubb3', 'Syt1', 'Slc17a7', 'Gad1', 'Gad2'],
        'Pericyte_score': [
            'Rgs5', 'Pdgfrb', 'Kcnj8', 'Abcc9', 'Acta2'],
        'Oligo_score': [
            'Plp1', 'Mbp', 'Mog', 'Sox10', 'Cnp'],
    }
    for k, v in gsets.items():
        genes = [g for g in v if g in ad.var_names]
        if genes: sc.tl.score_genes(ad, genes, score_name=k)
        else: ad.obs[k] = 0.0

    print(f'QC start: {ad.n_obs}')
    obs, grp = ad.obs, ad.obs.groupby('sample')

    def get_mad_z(s):
        med = grp[s].transform('median')
        mad = (obs[s] - med).abs().groupby(obs['sample']).transform('median')
        return (obs[s] - med).abs() / mad.replace(0, 1.0)

    mt_thr = grp['pct_counts_mt'].transform(lambda x: np.percentile(x, 99))
    qc1_mask = (get_mad_z('n_genes_by_counts') <= k_mad) & \
               (get_mad_z('total_counts') <= k_mad) & \
               (obs['pct_counts_mt'] <= np.minimum(mt_max, mt_thr))

    n0 = ad.n_obs
    ad = ad[qc1_mask].copy()
    print(f'Metric QC keep: {ad.n_obs}/{n0} ({100*ad.n_obs/n0:.1f}%)')
    n1 = ad.n_obs

    grp = ad.obs.groupby('sample')
    others = ['Neuron_score', 'Pericyte_score', 'Oligo_score']
    is_mer = 'merfish' in ad.obs['source'].unique()
    adv = (grp['EC_score'].transform(lambda s: s.rank(pct=True)) -
           grp[others].transform(lambda s: s.rank(pct=True)).max(1)
          ) if is_mer else (ad.obs['EC_score'] - ad.obs[others].max(1))

    thr = adv.groupby(ad.obs['sample']).transform('quantile', frac_drop)
    ad = ad[adv >= thr].copy()
    print(f'EC keep: {ad.n_obs}/{n1} ({100*ad.n_obs/n1:.1f}%)')
    return ad

adata_curio = score_and_qc(adata_curio)
adata_merfish = score_and_qc(adata_merfish)


PROGS = {
    'BBB_prog': ['Mfsd2a','Slco1a4','Abcb1a','Slc2a1','Cldn5','Ocln'],
    'Tip_prog': ['Apln','Esm1','Kdr','Angpt2'],
    'Arterial_prog': ['Gja5','Efnb2','Sema3g'],
    'Venous_prog': ['Nr2f2','Ephb4','Vwf'],
    'Shear_prog': ['Klf2','Klf4'],
    'IFN_prog': ['Irf7','Isg15','Ifit1','Stat1'],
}

curio_prog = adata_curio.copy()
sc.pp.normalize_total(curio_prog, target_sum=1e4)
sc.pp.log1p(curio_prog)
for k,v in PROGS.items():
    v = [g for g in v if g in curio_prog.var_names]
    print(f'{k} genes: {len(v)}')
    if len(v) >= 2:
        sc.tl.score_genes(curio_prog, v, score_name=k, use_raw=False)
        adata_curio.obs[k] = curio_prog.obs[k]
    else:
        adata_curio.obs[k] = 0.0
prog_cols = list(PROGS.keys())


def normalize_hvg(ad, n_top=3000):
    tmp = ad.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(
        tmp, flavor='seurat_v3', n_top_genes=n_top, batch_key='sample'
    )
    ribo = ad.var_names.str.match(r'^(Rps|Rpl)')
    mask = tmp.var['highly_variable'] & ~ad.var['mt'] & ~ribo
    if int(mask.sum()) == 0:
        mask = tmp.var['highly_variable']
    kept = int(mask.sum())
    total = int(tmp.var['highly_variable'].sum())
    print(f'HVG kept: {kept}/{total}')
    return ad[:, mask].copy()

adata_curio = normalize_hvg(adata_curio)
adata_merfish = normalize_hvg(adata_merfish)


model_dir = f'{working_dir}/output/curio/scvi_endothelial'
os.makedirs(model_dir, exist_ok=True)

scvi.model.SCVI.setup_anndata(adata_curio)
path = f'{model_dir}/model.pt'
if not os.path.exists(path):
    model = scvi.model.SCVI(adata_curio, n_latent=20)
    model.train(max_epochs=300, early_stopping=True)
    model.save(model_dir, overwrite=True)
    print('scVI trained and saved')
else:
    model = scvi.model.SCVI.load(model_dir, adata=adata_curio)
    print('scVI loaded')

adata_curio.obsm['X_scvi'] = model.get_latent_representation()
sc.pp.neighbors(adata_curio, use_rep='X_scvi', n_neighbors=20)
sc.tl.umap(adata_curio, min_dist=0.3)

for r in [0.4,0.6,0.8,1.0]:
    sc.tl.leiden(adata_curio, resolution=r, key_added=f'leiden_r{r}')
    print(f'{r=}:', adata_curio.obs[f'leiden_r{r}']\
          .value_counts().to_dict())
r = 0.8
adata_curio.obs['leiden'] = adata_curio.obs[f'leiden_r{r}']\
    .astype('category')

med = adata_curio.obs.groupby('leiden')[prog_cols].median()
lab = med.idxmax(axis=1)
adata_curio.obs['state'] = adata_curio.obs['leiden'].map(lab)\
    .astype('category')

ps = (adata_curio.obs.groupby(['sample','condition'])[prog_cols]
      .median().reset_index())
abs_m = ps.groupby('condition')[prog_cols].median().reindex(
    ['CTRL','PREG','POSTPART'])
dlt = abs_m.sub(abs_m.loc['CTRL'], axis=1)


from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(16,6))
gs = GridSpec(2,5, figure=fig, hspace=0.25, wspace=0.25,
              width_ratios=[1,1,1,1,1.8])

ax0 = fig.add_subplot(gs[0,0])
sc.pl.umap(adata_curio, color='leiden', ax=ax0, show=False)

axes = [fig.add_subplot(gs[0,1]), fig.add_subplot(gs[0,2]),
        fig.add_subplot(gs[0,3]), fig.add_subplot(gs[1,0]),
        fig.add_subplot(gs[1,1]), fig.add_subplot(gs[1,2])]
for ax, col in zip(axes, prog_cols):
    sc.pl.umap(adata_curio, color=col, ax=ax, show=False)

axh = fig.add_subplot(gs[:,4])
v = np.abs(dlt.values).max()
sns.heatmap(dlt, ax=axh, cmap='vlag', center=0, vmin=-v, vmax=v,
            annot=True, fmt='.2f', cbar_kws={'shrink':0.8})
axh.set_title('Δ median program score vs CTRL')

out_png = f'{working_dir}/figures/endo_curio_final_panel.png'
plt.savefig(out_png, bbox_inches='tight')
plt.close()

#endregion










import os
import scvi
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as sp

os.environ['SCIPY_ARRAY_API'] = '1'
import scarches as sca

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

scvi.settings.seed = 0
sc.settings.verbosity = 2

condition_colors = {
    'CTRL': '#7209b7',
    'PREG': '#b5179e',
    'POSTPART': '#f72585'
}
modality_colors = {
    'merfish': '#4361ee',
    'curio': '#4cc9f0'
}

#region load data #############################################################

working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'
cell_type_col = 'subclass'

adata_curio = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_curio_final.h5ad')

adata_merfish = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')
adata_merfish.var.index = adata_merfish.var['gene_symbol']

for ad in [adata_curio, adata_merfish]:
    ad.var_names_make_unique()
    for col in ['class', 'subclass']:
        ad.obs[col] = ad.obs[col]\
            .astype(str).str.extract(r'^(\d+)\s+(.*)', expand=False)[1]
    ad.obs = ad.obs[['sample','condition','source','x_ffd','y_ffd',
                     'class','subclass']]
    ad.var = ad.var[['gene_symbol']]
    ad.var.index.name = None
    g = ad.var_names
    ad.var['mt']   = g.str.match(r'^(mt-|MT-)')
    ad.var['ribo'] = g.str.match(r'^(Rps|Rpl)')
    for key in ('uns','varm','obsp'):
        if hasattr(ad, key):
            try: delattr(ad, key)
            except: pass




adata_curio = adata_curio[adata_curio.obs['subclass'] == 'Endo NN'].copy()
adata_merfish = adata_merfish[adata_merfish.obs['subclass'] == 'Endo NN'].copy()

def score_and_qc(adata, mt_max=15.0, margin=0.15, m_delta=0.35):
    adata = adata.copy()
    adata.var['mt'] = adata.var_names.str.match(r'^(mt-|MT-)')
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=['mt'], percent_top=None, inplace=True)
    gene_sets = {
        'EC_score': [
            'Pecam1','Cldn5','Kdr','Klf2','Klf4','Mfsd2a','Slco1a4','Abcb1a',
            'Slc2a1','Cdh5','Esam','Tek','Tie1'],
        'Neuron_score': [
            'Snap25','Rbfox3','Tubb3','Syt1','Slc17a7','Gad1','Gad2'],
        'Pericyte_score': [
            'Rgs5','Pdgfrb','Kcnj8','Abcc9','Acta2'],
        'Oligo_score': [
            'Plp1','Mbp','Mog','Sox10','Cnp'],
    }
    for name, genes in gene_sets.items():
        valid = [g for g in genes if g in adata.var_names]
        print(f'{name} genes: {len(valid)}/{len(genes)}')
        if len(valid) >= 1:
            sc.tl.score_genes(
                adata, valid, score_name=name, use_raw=False)
        else:
            adata.obs[name] = 0.0
    print(f'QC start: {adata.n_obs}')
    obs = adata.obs
    grp = obs.groupby('sample')
    g_lo = grp['n_genes_by_counts'].transform(np.percentile, 2)
    g_hi = grp['n_genes_by_counts'].transform(np.percentile, 99.8)
    c_lo = grp['total_counts'].transform(np.percentile, 2)
    c_hi = grp['total_counts'].transform(np.percentile, 99.8)
    qc1 = obs['n_genes_by_counts'].between(g_lo, g_hi) & \
          obs['total_counts'].between(c_lo, c_hi)
    if adata.var['mt'].sum() >= 5:
        mt99 = grp['pct_counts_mt'].transform(np.percentile, 99)
        mt_thr = np.minimum(mt_max, mt99)
        qc1 &= obs['pct_counts_mt'] <= mt_thr
    adata = adata[qc1].copy()
    print(f'QC1 keep: {adata.n_obs}/{len(obs)} '
          f'({100*adata.n_obs/len(obs):.1f}%)')
    n1 = adata.n_obs
    is_merfish = adata.obs['source'].astype(str).str.contains(
        'merfish', case=False).all()
    cols = ['Neuron_score','Pericyte_score','Oligo_score']
    if 'EC_score' in adata.obs:
        if is_merfish:
            r = lambda s: s.rank(pct=True)
            ecp = adata.obs.groupby('sample')['EC_score'].transform(r)
            oth = adata.obs.groupby('sample')[cols].transform(r).max(1)
            q2 = ecp >= oth - m_delta
            print(f'EC gate: merfish m_delta={m_delta:.2f}')
        else:
            oth = adata.obs[cols].max(1)
            q2 = adata.obs['EC_score'] >= oth - margin
            print(f'EC gate: margin={margin:.2f}')
        adata = adata[q2].copy()
        print(f'EC keep: {adata.n_obs}/{n1} '
              f'({100*adata.n_obs/n1:.1f}%)')
    else:
        print('Skip EC gate (few EC genes)')
    return adata

def normalize_hvg(adata):
    tmp = adata.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(
        tmp, flavor='seurat_v3', n_top_genes=3000, batch_key='sample'
    )
    ribo = adata.var_names.str.match(r'^(Rps|Rpl)')
    mask = tmp.var['highly_variable'] & ~adata.var['mt'] & ~ribo
    print(f'HVG kept: {int(mask.sum())}/'
          f'{int(tmp.var['highly_variable'].sum())}')
    return adata[:, mask].copy()

adata_curio = score_and_qc(adata_curio, mt_max=15.0)
adata_merfish = score_and_qc(adata_merfish, mt_max=15.0)

adata_curio = normalize_hvg(adata_curio)
adata_merfish = normalize_hvg(adata_merfish)



model_dir = f'{working_dir}/output/curio/scvi_endothelial'
os.makedirs(model_dir, exist_ok=True)

scvi.model.SCVI.setup_anndata(adata_curio, batch_key='sample')

model_path = f'{model_dir}/model.pt'
if not os.path.exists(model_path):
    model = scvi.model.SCVI(adata_curio, n_latent=20)
    model.train(max_epochs=300, early_stopping=True)
    model.save(model_dir, overwrite=True)
    print('scVI trained and saved')
else:
    model = scvi.model.SCVI.load(model_dir, adata=adata_curio)
    print('scVI loaded')

adata_curio.obsm['X_scvi'] = model.get_latent_representation()
sc.pp.neighbors(adata_curio, use_rep='X_scvi', n_neighbors=20)
sc.tl.umap(adata_curio, min_dist=0.3)

for res in [0.4, 0.6, 0.8, 1.0]:
    sc.tl.leiden(adata_curio, resolution=res, key_added=f'leiden_r{res}')
    print(f'{res=}:', adata_curio.obs[f'leiden_r{res}']
          .value_counts().to_dict())

res = 0.6
adata_curio.obs['leiden'] = adata_curio.obs[f'leiden_r{res}'].astype('category')

sc.pl.umap(
    adata_curio, wspace=0.3,
    color=['leiden', 'pct_counts_mt', 'EC_score', 'Neuron_score',
           'Pericyte_score', 'sample', 'condition'])
plt.savefig(
    f'{working_dir}/figures/endo_umap_scvi.png', bbox_inches='tight')
plt.close()

de_res = model.differential_expression(
    groupby='leiden', batch_correction=True, mode='change',
    delta=0.1, all_stats=True
)

for group in de_res['group1'].unique():
    group_df = de_res[de_res['group1'] == group]
    top_hits = group_df.sort_values(['proba_de', 'bayes_factor'], ascending=[False, True]).head(10)
    print(f'\nTop 10 DE genes for group1 = {group}:')
    print(top_hits[['proba_de', 'bayes_factor', 'comparison']])




import os
import scvi
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as sp

os.environ['SCIPY_ARRAY_API'] = '1'
import scarches as sca

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

condition_colors = {
    'CTRL': '#7209b7',
    'PREG': '#b5179e',
    'POSTPART': '#f72585'
}
modality_colors = {
    'merfish': '#4361ee',
    'curio': '#4cc9f0'
}

#region load data #############################################################

working_dir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'
cell_type_col = 'subclass'

adata_curio = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_curio_final.h5ad')

adata_merfish = sc.read_h5ad(
    f'{working_dir}/output/data/adata_query_merfish_final.h5ad')
adata_merfish.var.index = adata_merfish.var['gene_symbol']

for adata in [adata_curio, adata_merfish]:
    for col in ['class', 'subclass']:
        adata.obs[col] = adata.obs[col].astype(str)\
            .str.extract(r'^(\d+)\s+(.*)', expand=False)[1]

adata_curio.obs = adata_curio.obs[
    ['sample', 'condition', 'source', 'x_ffd', 'y_ffd', 'class', 'subclass']]
adata_curio.var = adata_curio.var[['gene_symbol']]
adata_curio.var.index.name = None
del adata_curio.uns, adata_curio.varm, adata_curio.obsp

adata_merfish.var.index = adata_merfish.var['gene_symbol']
adata_merfish.obs = adata_merfish.obs[
    ['sample', 'condition', 'source', 'x_ffd', 'y_ffd', 'class', 'subclass']]
adata_merfish.var = adata_merfish.var[['gene_symbol']]
adata_merfish.var.index.name = None
del adata_merfish.uns, adata_merfish.varm, adata_merfish.obsp

#endregion

#region endo analyses #########################################################

adata_curio = adata_curio[
    adata_curio.obs['subclass'] == 'Endo NN'
].copy()
adata_merfish = adata_merfish[
    adata_merfish.obs['subclass'] == 'Endo NN'
].copy()

adata_curio.layers['counts'] = adata_curio.X.copy()
sc.pp.normalize_total(adata_curio, target_sum=1e4)
sc.pp.log1p(adata_curio)

curio_model_dir = 'spatial-pregnancy-postpart/output/curio/scvi'
os.makedirs(curio_model_dir, exist_ok=True)

if not os.path.exists(os.path.join(curio_model_dir, 'model.pt')):
    scvi.model.SCVI.setup_anndata(adata_curio, layer='counts')
    ref_model = scvi.model.SCVI(adata_curio)
    ref_model.train()
    ref_model.save(curio_model_dir, overwrite=True)
else:
    ref_model = scvi.model.SCVI.load(curio_model_dir, adata=adata_curio)

adata_curio.obsm['X_scvi'] = ref_model.get_latent_representation()
sc.pp.neighbors(adata_curio, use_rep='X_scvi')
sc.tl.umap(adata_curio)
sc.tl.leiden(adata_curio, resolution=0.2)


adata_merfish.layers['counts'] = adata_merfish.X.copy()
sc.pp.normalize_total(adata_merfish, target_sum=1e4)
sc.pp.log1p(adata_merfish)

merfish_model_dir = 'spatial-pregnancy-postpart/output/merfish/scvi'
os.makedirs(merfish_model_dir, exist_ok=True)

if not os.path.exists(os.path.join(merfish_model_dir, 'model.pt')):
    scvi.model.SCVI.setup_anndata(adata_merfish, layer='counts')
    ref_model = scvi.model.SCVI(adata_merfish)
    ref_model.train()
    ref_model.save(merfish_model_dir, overwrite=True)
else:
    ref_model = scvi.model.SCVI.load(merfish_model_dir, adata=adata_merfish)

adata_merfish.obsm['X_scvi'] = ref_model.get_latent_representation()
sc.pp.neighbors(adata_merfish, use_rep='X_scvi')
sc.tl.umap(adata_merfish)




sc.tl.rank_genes_groups(adata_curio, f'leiden_res_{res}', use_raw=False)

for res in resolutions:
    res_key = f'leiden_res_{res}'
    score_cols = []
    markers_df = sc.get.rank_genes_groups_df(adata_curio, group=None)

    for cluster_id in adata_curio.obs[res_key].cat.categories:
        score_col = f'score_{res_key}_c{cluster_id}'
        score_cols.append(score_col)
        
        ref_markers = markers_df[
            markers_df['group'] == cluster_id
        ]['names'].tolist()
        
        query_markers = list(set(ref_markers) & set(adata_merfish.var_names))
        
        if len(query_markers) > 0:
            sc.tl.score_genes(adata_merfish, gene_list=query_markers, score_name=score_col)
        else:
            adata_merfish.obs[score_col] = 0

    adata_merfish.obs[res_key] = adata_merfish.obs[score_cols].idxmax(axis=1)\
        .str.replace(f'score_{res_key}_c', '')

cluster_keys = [f'leiden_res_{res}' for res in resolutions]

sc.pl.umap(
    adata_curio,
    color=['condition'] + cluster_keys,
    ncols=3
)
plt.savefig(
    f'{working_dir}/figures/endo_umap_reference.png', bbox_inches='tight')
plt.close()

for res in resolutions:
    res_key = f'leiden_res_{res}'
    
    ct = pd.crosstab(adata_curio.obs[res_key], adata_curio.obs['condition'])
    
    conditions_ordered = ['CTRL', 'PREG', 'POSTPART']
    ct = ct[conditions_ordered]

    ct_prop = ct.div(ct.sum(axis=1), axis=0)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ct_prop.plot(
        kind='bar', 
        stacked=False, 
        ax=ax, 
        color=[condition_colors[c] for c in ct_prop.columns]
    )
    
    plt.title(f'Proportion of Conditions in Leiden Clusters (res={res})')
    plt.xlabel('Leiden Cluster')
    plt.ylabel('Proportion')
    plt.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(
        f'{working_dir}/figures/endo_cluster_condition_proportion_res_{res}.png', 
        bbox_inches='tight'
    )
    plt.close()











sc.pp.neighbors(adata_merfish)
sc.tl.umap(adata_merfish)
sc.pl.umap(
    adata_merfish,
    color=['condition'] + cluster_keys,
    ncols=3
)
plt.savefig(
    f'{working_dir}/figures/endo_umap_projected_signature.png', bbox_inches='tight')
plt.close()


del adata_curio.uns
sc_ref = SingleCell(adata_curio)

print_df(sc_ref.with_uns(QCed = True).find_markers('leiden_res_0.2'))





















from single_cell import SingleCell, PseudoBulk, DE

sc_curio = SingleCell(adata_curio)\
    .qc(allow_float=True)

pb_curio = sc_curio\
    .pseudobulk('sample', 'subclass')

de_curio = pb_curio\
    .qc('condition')\
    .library_size()\
    .DE('~ 0 + condition', 
        contrasts={
            'PREG_vs_CTRL': 'conditionPREG - conditionCTRL',
            'POSTPART_vs_PREG': 'conditionPOSTPART - conditionPREG'},
        categorical_columns='condition',
        group='condition')

de_curio.table.write_csv('tmp.csv')










model_dir = 'spatial-pregnancy-postpart/output/curio'
ref_model_dir = f'{model_dir}/scvi/ref'
query_model_dir = f'{model_dir}/scvi/query'
resolutions = [0.1, 0.2, 0.3, 0.5, 1.0]
os.makedirs(ref_model_dir, exist_ok=True)
os.makedirs(query_model_dir, exist_ok=True)

adata_curio = adata_curio[
    adata_curio.obs['subclass'] == 'Endo NN'
].copy()
adata_merfish = adata_merfish[
    adata_merfish.obs['subclass'] == 'Endo NN'
].copy()

if not os.path.exists(os.path.join(ref_model_dir, 'model.pt')):
    scvi.model.SCVI.setup_anndata(adata_curio)
    ref_model = scvi.model.SCVI(adata_curio)
    ref_model.train()
    ref_model.save(ref_model_dir, overwrite=True)
else:
    ref_model = scvi.model.SCVI.load(ref_model_dir, adata=adata_curio)

scvi.model.SCVI.prepare_query_anndata(adata_merfish, ref_model_dir)

if not os.path.exists(os.path.join(query_model_dir, 'model.pt')):
    query_model = sca.models.SCVI.load_query_data(
        adata_merfish, ref_model_dir)
    query_model.train(max_epochs=200)
    query_model.save(query_model_dir, overwrite=True)
else:
    query_model = sca.models.SCVI.load(
        query_model_dir, adata=adata_merfish)

adata_merfish.obsm['X_scvi_projected'] = query_model.get_latent_representation()

for res in resolutions:
    knn = KNeighborsClassifier()
    knn.fit(
        adata_curio.obsm['X_scvi'],
        adata_curio.obs[f'leiden_res_{res}']
    )
    adata_merfish.obs[f'leiden_res_{res}'] = knn.predict(
        adata_merfish.obsm['X_scvi_projected']
    )

cluster_keys = [f'leiden_res_{res}' for res in resolutions]

sc.pl.umap(
    adata_curio,
    color=['source'] + cluster_keys,
    ncols=3
)
plt.savefig(
    f'{working_dir}/figures/endo_umap_reference.svg', bbox_inches='tight')
plt.close()

sc.tl.umap(adata_merfish, min_dist=0.1)
sc.pl.umap(
    adata_merfish,
    color=['source'] + cluster_keys,
    ncols=3
)
plt.savefig(
    f'{working_dir}/figures/endo_umap_projected.svg', bbox_inches='tight')
plt.close()























adata_full = adata_curio.concatenate(adata_merfish)
adata_full.obsm['X_scvi'] = query_model.get_latent_representation(adata_full)

sc.pp.neighbors(adata_full, use_rep='X_scvi')
sc.tl.umap(adata_full)

for res in resolutions:
    sc.tl.leiden(adata_full, resolution=res, key_added=f'leiden_res_{res}')

cluster_keys = ['batch'] + [f'leiden_res_{res}' for res in resolutions]

sc.pl.umap(
    adata_full,
    color=cluster_keys,
    ncols=3
)
plt.savefig(
    f'{working_dir}/figures/endo_umap.svg', bbox_inches='tight')
plt.close()

















common_genes = adata_curio.var_names.intersection(adata_merfish.var_names)
adata_c = adata_curio[:, common_genes].copy()
adata_m = adata_merfish[:, common_genes].copy()

adata = adata_c.concatenate(adata_m)

adata.layers['counts'] = adata.X.copy()
scvi.model.SCVI.setup_anndata(adata, layer='counts', batch_key='source')

model = scvi.model.SCVI(adata, n_latent=30)
model.train()

# 3. Find and visualize shared subclusters
adata.obsm['X_scvi'] = model.get_latent_representation()
sc.pp.neighbors(adata, use_rep='X_scvi')
sc.tl.leiden(adata, resolution=1.5, key_added='subclusters')
sc.tl.umap(adata)

sc.pl.umap(
    adata,
    color=['source', 'condition', 'subclusters'],
    title=['Technology', 'Condition', 'Shared Subclusters']
)

# 4. Quantify cluster composition
tech_dist = pd.crosstab(adata.obs.subclusters, adata.obs.source)
print('Technology distribution in clusters:')
print(tech_dist)

condition_dist = pd.crosstab(adata.obs.subclusters, adata.obs.condition)
print('\nCondition distribution in clusters:')
print(condition_dist)

# 5. Find marker genes for new subclusters
de_df = model.differential_expression(groupby='subclusters')




















common_genes = adata_curio.var_names.intersection(adata_merfish.var_names)
adata_curio_subset = adata_curio[:, common_genes].copy()
adata_merfish_subset = adata_merfish[:, common_genes].copy()

adata_combined = adata_curio_subset.concatenate(
    adata_merfish_subset, batch_key='source')

sc.pp.normalize_total(adata_combined, target_sum=1e4)
sc.pp.log1p(adata_combined)
sc.pp.pca(adata_combined)

sc.external.pp.bbknn(adata_combined, batch_key='source')

sc.tl.leiden(adata_combined, resolution=1.0)
sc.tl.umap(adata_combined)
sc.pl.umap(adata_combined, color=['source', 'leiden'])
plt.savefig(
    f'{working_dir}/figures/endo_umap.svg', bbox_inches='tight')
plt.close()


#endregion
