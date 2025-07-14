import os
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scanpy as sc
from utils import run
from pathlib import Path

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 500

workdir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'
input_dir = f'{workdir}/gsmap/input'
output_dir = f'{workdir}/gsmap/output'
figures_dir = f'{workdir}/figures/gsmap'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

#region prep data ##############################################################

if not os.path.exists(f'{input_dir}/gsMap_resource'):
    os.makedirs(input_dir, exist_ok=True)
    run(f'wget https://yanglab.westlake.edu.cn/data/gsMap/gsMap_resource.tar.gz '
        f'-P {input_dir}')
    run(f'tar -xvzf {input_dir}/gsMap_resource.tar.gz -C {input_dir}')
    run(f'rm {input_dir}/gsMap_resource.tar.gz')

adata = sc.read_h5ad(f'{workdir}/output/data/adata_query_curio_final.h5ad')
adata.obsm['spatial'] = adata.obs[['x_ffd', 'y_ffd']].to_numpy()

for col in ['class', 'subclass']:
    adata.obs[col] = adata.obs[col].astype(str)\
        .str.extract(r'^(\d+)\s+(.*)', expand=False)[1]

all_sample_names = adata.obs['sample'].unique()
for sample_name in all_sample_names:
    if not os.path.exists(f'{input_dir}/ST/{sample_name}.h5ad'):
        adata_sample = adata[adata.obs['sample'] == sample_name].copy()
        adata_sample.write_h5ad(f'{input_dir}/ST/{sample_name}.h5ad')

gwas_formatted_dir = f'{input_dir}/GWAS_formatted'
os.makedirs(gwas_formatted_dir, exist_ok=True)

for f in os.listdir(f'{input_dir}/GWAS'):
    if f.endswith('.sumstats.gz'):
        basename = f.replace('.sumstats.gz', '')
        if not os.path.exists(f'{gwas_formatted_dir}/{basename}.sumstats.gz'):
            run(f'''
                gsmap format_sumstats \
                    --sumstats '{input_dir}/GWAS/{f}' \
                    --out '{gwas_formatted_dir}/{basename}'
            ''')

with open(f'{gwas_formatted_dir}/gwas_config.yaml', 'w') as f:
    for gwas_file in sorted(os.listdir(gwas_formatted_dir)):
        if gwas_file.endswith('.sumstats.gz'):
            trait = gwas_file.replace('.sumstats.gz', '')
            path = os.path.abspath(f'{gwas_formatted_dir}/{gwas_file}')
            f.write(f'{trait}: {path}\n')

#endregion

#region mouse embryo ##########################################################

conditions = {
    "control": [s for s in all_sample_names if "CTRL" in s],
    "pregnant": [s for s in all_sample_names if "PREG" in s],
    "postpartum": [s for s in all_sample_names if "POSTPART" in s]
}

for condition, sample_names in conditions.items():
    slice_mean_file = f'{output_dir}/{condition}_slice_mean.parquet'   
    h5ad_paths = ' '.join([f"{input_dir}/ST/{name}.h5ad" for name in sample_names])
    sample_list_str = ' '.join(sample_names)

    if not os.path.exists(slice_mean_file):
        run(f'''
            gsmap create_slice_mean \
                --sample_name_list {sample_list_str} \
                --h5ad_list {h5ad_paths} \
                --slice_mean_output_file {slice_mean_file} \
                --data_layer 'counts' \
                --homolog_file '{input_dir}/gsMap_resource/homologs/mouse_human_homologs.txt'
        ''')

    for sample_name in sample_names:
        if not os.path.exists(f'{output_dir}/{sample_name}/report'):
            run(f'''
                gsmap quick_mode \
                    --workdir '{output_dir}' \
                    --homolog_file '{input_dir}/gsMap_resource/homologs/mouse_human_homologs.txt' \
                    --sample_name '{sample_name}' \
                    --gsMap_resource_dir '{input_dir}/gsMap_resource' \
                    --hdf5_path '{input_dir}/ST/{sample_name}.h5ad' \
                    --annotation 'subclass' \
                    --data_layer 'counts' \
                    --sumstats_config_file '{input_dir}/GWAS_formatted/gwas_config.yaml' \
                    --gM_slices '{slice_mean_file}' 
                ''')













color_dict = {
    "Adipose tissue": "#6e2ee6",
    "Adrenal gland": "#c00250",
    "Bone": "#113913",
    "Brain": "#ef833a",
    "Cartilage": "#35596b",
    "Cartilage primordium": "#3bb54c",
    "Cavity": "#dddddf",
    "Choroid plexus": "#bd39dc",
    "Connective tissue": "#0bd4b4",
    "Dorsal root ganglion": "#b84b11",
    "Epidermis": "#046df4",
    "GI tract": "#5c5ba9",
    "Heart": "#d3245a",
    "Inner ear": "#03fff3",
    "Jaw and tooth": "#f061fa",
    "Kidney": "#61cfe5",
    "Liver": "#ca23b2",
    "Lung": "#7dc136",
    "Meninges": "#e0ca44",
    "Mucosal epithelium": "#307dd3",
    "Muscle": "#ae1041",
    "Smooth muscle": "#fc5252",
    "Spinal cord": "#fad5bb",
    "Submandibular gland": "#ab33e6",
    "Sympathetic nerve": "#cd5a0c"
}

sample_name = 'E16.5_E1S1.MOSTA'
trait_name = 'Fibromyalgia'

gsmap_plot_file = (
    f'{output_dir}/{sample_name}/report/{trait_name}/gsMap_plot/'
    f'{sample_name}_{trait_name}_gsMap_plot.csv'
)
cauchy_results_file = (
    f'{output_dir}/{sample_name}/cauchy_combination/'
    f'{sample_name}_{trait_name}.Cauchy.csv.gz'
)

if os.path.exists(gsmap_plot_file) and os.path.exists(cauchy_results_file):
    df = pl.read_csv(gsmap_plot_file).to_pandas()
    plot_df = pl.read_csv(cauchy_results_file)\
        .sort('p_cauchy')\
        .with_columns((-pl.col('p_cauchy').log10())
                      .alias('-log10(p_cauchy)'))

    fig = plt.figure(figsize=(20, 8))
    fig.set_facecolor('white')
    gs_main = gridspec.GridSpec(
        1, 2, width_ratios=[2, 0.6], wspace=0.1, figure=fig
    )
    gs_spatial = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_main[0], wspace=0.01
    )
    ax1 = fig.add_subplot(gs_spatial[0])
    ax2 = fig.add_subplot(gs_spatial[1])
    ax3 = fig.add_subplot(gs_main[1])

    annos = sorted(df['annotation'].unique())
    color_map = {anno: color_dict[anno] for anno in annos}
    sns.scatterplot(
        data=df, x='sx', y='sy', hue='annotation', hue_order=annos,
        palette=color_map, s=5, linewidth=0, ax=ax1, legend=False,
        rasterized=True
    )
    ax1.set_facecolor('black')
    ax1.set_aspect('equal', adjustable='box')
    ax1.axis('off')
    ax1.set_title('E16.5', color='black', fontsize=16)

    sns.scatterplot(
        data=df, x='sx', y='sy', hue='logp', palette='magma',
        s=5, linewidth=0, ax=ax2, legend=False,
        rasterized=True
    )
    ax2.set_facecolor('black')
    ax2.set_aspect('equal', adjustable='box')
    ax2.axis('off')
    ax2.set_title(trait_name, color='black', fontsize=16)

    norm = plt.Normalize(df['logp'].min(), df['logp'].max())
    sm = plt.cm.ScalarMappable(cmap="magma", norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.33, 0.08, 0.15, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'$-\log_{10}(\mathrm{P\ value})$', color='black', size=12)
    cbar.ax.tick_params(colors='black')
    cbar.outline.set_edgecolor('black')

    plot_df_pd = plot_df.to_pandas()
    sns.barplot(
        data=plot_df_pd, x='-log10(p_cauchy)', y='annotation',
        palette=color_map, ax=ax3, hue='annotation', dodge=False,
        legend=False, order=plot_df_pd['annotation'],
        rasterized=True
    )
    ax3.set_title(f'gsMap Cauchy P-values for {trait_name}')
    ax3.set_xlabel('-log10(p-value)')
    ax3.set_ylabel('')

    path_svg = (f'{figures_dir}/'
                f'{sample_name}_{trait_name}_spatial_combined.svg')
    path_png = (f'{figures_dir}/'
                f'{sample_name}_{trait_name}_spatial_combined.png')
    plt.savefig(path_svg, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(path_png, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

#endregion

#region mouse brain ###########################################################

run(f'''
    gsmap quick_mode \
    --workdir '{output_dir}' \
    --homolog_file '{input_dir}/gsMap_resource/homologs/mouse_human_homologs.txt' \
    --sample_name 'Zhuang-ABCA-1-raw' \
    --gsMap_resource_dir '{input_dir}/gsMap_resource' \
    --hdf5_path '{input_dir}/gsMap_example_data/ST/Zhuang-ABCA-1-raw.h5ad' \
    --annotation 'parcellation_division' \
    --data_layer 'count' \
    --sumstats_file '{formatted_gwas_file}' \
    --trait_name 'Fibromyalgia'
''')

run(f'''
    gsmap quick_mode \
    --workdir '{output_dir}' \
    --homolog_file '{input_dir}/gsMap_resource/homologs/mouse_human_homologs.txt' \
    --sample_name 'Zhuang-ABCA-3-raw' \
    --gsMap_resource_dir '{input_dir}/gsMap_resource' \
    --hdf5_path '{input_dir}/gsMap_example_data/ST/Zhuang-ABCA-3-raw.h5ad' \
    --annotation 'parcellation_division' \
    --data_layer 'count' \
    --sumstats_file '{formatted_gwas_file}' \
    --trait_name 'Fibromyalgia'
''')


adata_1 = sc.read_h5ad(f'{input_dir}/gsMap_example_data/ST/Zhuang-ABCA-1-raw.h5ad')
adata_3 = sc.read_h5ad(f'{input_dir}/gsMap_example_data/ST/Zhuang-ABCA-3-raw.h5ad')

adata = sc.concat([adata_1, adata_3])

color_df = adata.obs[['class', 'class_color']].drop_duplicates()
brain_color_map = {
    row['class']: row['class_color'] for _, row in color_df.iterrows()
}

parc_color_df = adata.obs[['parcellation_division',
                          'parcellation_division_color']].drop_duplicates()
parc_color_map = {
    row['parcellation_division']: row['parcellation_division_color']
    for _, row in parc_color_df.iterrows()
}

sample_sizes = {'Zhuang-ABCA-1-raw': 3, 'Zhuang-ABCA-3-raw': 0.25}

fig = plt.figure(figsize=(30, 12))
fig.set_facecolor('white')
gs_main = gridspec.GridSpec(
    2, 5, height_ratios=[1, 1], width_ratios=[1, 1, 1, 0.8, 0.8],
    wspace=0.2, hspace=0.3, figure=fig
)

for i, sample_name in enumerate(['Zhuang-ABCA-1-raw', 'Zhuang-ABCA-3-raw']):
    trait_name = 'Fibromyalgia'

    adata_sample = sc.read_h5ad(
        f'{input_dir}/gsMap_example_data/ST/{sample_name}.h5ad'
    )
    spatial_df = pd.DataFrame(
        adata_sample.obsm['spatial'],
        columns=['sx', 'sy'],
        index=adata_sample.obs.index
    )
    plot_data = adata_sample.obs.join(spatial_df)

    gsmap_plot_file = (
        f'{output_dir}/{sample_name}/report/{trait_name}/gsMap_plot/'
        f'{sample_name}_{trait_name}_gsMap_plot.csv'
    )
    logp_df = pl.read_csv(
        gsmap_plot_file, schema_overrides={'': pl.Utf8}
    ).to_pandas().set_index('')
    plot_data = plot_data.join(logp_df[['logp']])

    class_cauchy_file = (
        f'{output_dir}/{sample_name}/cauchy_combination/'
        f'{sample_name}_{trait_name}.Class.Cauchy.csv.gz'
    )
    parc_cauchy_file = (
        f'{output_dir}/{sample_name}/cauchy_combination/'
        f'{sample_name}_{trait_name}.Parcellation.Cauchy.csv.gz'
    )
    class_plot_df = pl.read_csv(class_cauchy_file).sort('p_cauchy')\
        .with_columns((-pl.col('p_cauchy').log10()).alias('-log10(p_cauchy)'))
    parc_plot_df = pl.read_csv(parc_cauchy_file).sort('p_cauchy')\
        .with_columns((-pl.col('p_cauchy').log10()).alias('-log10(p_cauchy)'))

    ax1 = fig.add_subplot(gs_main[i, 0])
    ax2 = fig.add_subplot(gs_main[i, 1])
    ax3 = fig.add_subplot(gs_main[i, 2])
    ax4 = fig.add_subplot(gs_main[i, 3])
    ax5 = fig.add_subplot(gs_main[i, 4])

    sns.scatterplot(
        data=plot_data, x='sx', y='sy', hue='class',
        palette=brain_color_map, s=sample_sizes[sample_name], linewidth=0,
        ax=ax1, legend=False, rasterized=True,
        hue_order=sorted(brain_color_map.keys())
    )
    ax1.set_facecolor('black')
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title(f'{sample_name} - Class', y=1.05)

    sns.scatterplot(
        data=plot_data, x='sx', y='sy', hue='parcellation_division',
        palette=parc_color_map, s=sample_sizes[sample_name], linewidth=0,
        ax=ax2, legend=False, rasterized=True,
        hue_order=sorted(parc_color_map.keys())
    )
    ax2.set_facecolor('black')
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title(f'{sample_name} - Parcellation', y=1.05)

    sns.scatterplot(
        data=plot_data, x='sx', y='sy', hue='logp', palette='magma',
        s=sample_sizes[sample_name], linewidth=0, ax=ax3, legend=False,
        rasterized=True
    )
    ax3.set_facecolor('black')
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title(trait_name, y=1.05)
    norm = plt.Normalize(plot_data['logp'].min(), plot_data['logp'].max())
    sm = plt.cm.ScalarMappable(cmap="magma", norm=norm)
    sm.set_array([])
    cbar_ax_bbox = ax3.get_position()
    cbar_ax = fig.add_axes(
        [cbar_ax_bbox.x0, cbar_ax_bbox.y0 - 0.05, cbar_ax_bbox.width, 0.02]
    )
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'$-\log_{10}(\mathrm{P\ value})$')

    class_plot_df_pd = class_plot_df.to_pandas()
    sns.barplot(
        data=class_plot_df_pd, x='-log10(p_cauchy)', y='annotation',
        hue='annotation', legend=False, palette=brain_color_map, ax=ax4,
        dodge=False, order=class_plot_df_pd['annotation'], rasterized=True
    )
    ax4.set_title('Class P-values')
    ax4.set_xlabel('')
    ax4.set_ylabel('')

    parc_plot_df_pd = parc_plot_df.to_pandas()
    sns.barplot(
        data=parc_plot_df_pd, x='-log10(p_cauchy)', y='annotation',
        hue='annotation', legend=False, palette=parc_color_map, ax=ax5,
        dodge=False, order=parc_plot_df_pd['annotation'], rasterized=True
    )
    ax5.set_title('Parcellation P-values')
    ax5.set_xlabel('')
    ax5.set_ylabel('')

path_svg = f'{figures_dir}/mouse_brain_{trait_name}_spatial_combined.svg'
path_png = f'{figures_dir}/mouse_brain_{trait_name}_spatial_combined.png'
plt.savefig(path_svg, bbox_inches='tight', pad_inches=0.1)
plt.savefig(path_png, bbox_inches='tight', pad_inches=0.1)
plt.close(fig)