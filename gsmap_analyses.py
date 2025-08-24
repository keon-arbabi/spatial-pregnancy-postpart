import os
import scanpy as sc
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as patches
from utils import run
from pathlib import Path
from ryp import r, to_r
from scipy.cluster.hierarchy import linkage, leaves_list
from typing import Optional, List

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 500

workdir = 'projects/rrg-wainberg/karbabi/spatial-pregnancy-postpart'
input_dir = f'{workdir}/gsmap/input'
output_dir = f'{workdir}/gsmap/output'
figures_dir = f'{workdir}/figures/gsmap'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

#region functions ##############################################################

def _insert_gap(arr, indices, axis=0):
    if isinstance(arr, pd.DataFrame):
        res = arr.copy()
        if axis == 0:
            for i in sorted(indices, reverse=True):
                res = pd.concat([
                    res.iloc[:i],
                    pd.DataFrame([[np.nan]*res.shape[1]], columns=res.columns),
                    res.iloc[i:]
                ]).reset_index(drop=True)
        elif axis == 1:
            for i in sorted(indices, reverse=True):
                res.insert(loc=int(i), column=f'gap_{i}', value=np.nan)
        return res

    is_series = isinstance(arr, pd.Series)
    if is_series:
        arr = arr.values

    val = np.nan
    if arr.dtype == object or arr.dtype.kind in ['U', 'S']:
        val = ''
    for i in sorted(indices, reverse=True):
        arr = np.insert(arr, i, val, axis=axis)
    
    if is_series:
        return pd.Series(arr)
    return arr

def _segments(mask):
    seg, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            seg.append((s, i))
            s = None
    if s is not None:
        seg.append((s, len(mask)))
    return seg

def plot_trait_ranking(output_dir, figures_dir, conditions):
    sample_map = {s: c for c, L in conditions.items() for s in L}
    files = Path(output_dir).glob('*/cauchy_combination/*.Cauchy.csv.gz')
    df = pl.concat([
        pl.scan_csv(f).with_columns(
            condition=pl.lit(sample_map.get(f.parts[-3])),
            trait=pl.lit(
                f.name.replace(f.parts[-3] + '_', '')
                .replace('.Cauchy.csv.gz', '')
            )
        ) for f in files if sample_map.get(f.parts[-3])
    ])

    stats_df = df.select(['trait', 'annotation']).collect()
    n_traits = stats_df['trait'].n_unique()
    n_annotations = stats_df['annotation'].n_unique()
    bonferroni_p = 0.05 / (n_traits * n_annotations)
    log_p_threshold = -np.log10(bonferroni_p)

    ranking_data = df\
        .with_columns(p_log=(-pl.col('p_cauchy').log10()))\
        .group_by('trait', 'annotation', 'condition')\
        .agg(pl.median('p_log').alias('median_log_p'))\
        .group_by('trait')\
        .agg(pl.max('median_log_p').alias('max_median_log_p'))\
        .sort('max_median_log_p', descending=True)\
        .collect()\
        .to_pandas()

    fig, ax = plt.subplots(figsize=(2.5, 3.5), facecolor='white')

    scores = ranking_data['max_median_log_p']
    norm = colors.Normalize(vmin=scores.min(), vmax=scores.max())
    cmap = plt.get_cmap('GnBu')

    sns.barplot(
        x='max_median_log_p', y='trait', data=ranking_data,
        hue='trait', palette=list(cmap(norm(scores.values))), 
        ax=ax, orient='h', legend=False
    )

    ax.axvline(
        x=log_p_threshold, color='black', linestyle='--',
        linewidth=2
    )

    ax.set_xlabel('Peak Score')
    ax.set_ylabel('')
    sns.despine(ax=ax)
    ax.tick_params(axis='y', length=0)

    fig.tight_layout()
    fig.savefig(f'{figures_dir}/trait_ranking.svg', bbox_inches='tight')
    fig.savefig(
        f'{figures_dir}/trait_ranking.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_gwas_heatmap(
    adata, output_dir, figures_dir, conditions, 
    traits_to_include: Optional[List[str]] = None
):
    sample_map = {s: c for c, L in conditions.items() for s in L}
    files = [f for f in Path(output_dir).glob(
        '*/cauchy_combination/*.Cauchy.csv.gz'
    ) if sample_map.get(f.parts[-3])]

    def get_trait(f):
        return f.name.replace(f.parts[-3] + '_', '')\
            .replace('.Cauchy.csv.gz', '')

    df = pl.concat([
        pl.scan_csv(f).with_columns(
            condition=pl.lit(sample_map.get(f.parts[-3])),
            trait=pl.lit(get_trait(f))
        ) for f in files
    ])

    agg_data = df\
        .with_columns(p_log=(-pl.col('p_cauchy').log10()).fill_null(0.0))\
        .group_by(['condition', 'trait', 'annotation'])\
        .agg(pl.col('p_log').median())\
        .group_by(['trait', 'annotation'])\
        .agg(pl.col('p_log').mean().alias('score'))\
        .collect()\
        .to_pandas()

    mat = agg_data.pivot(
        index='annotation', columns='trait', values='score'
    ).dropna().sort_index()

    if traits_to_include:
        mat = mat[[t for t in traits_to_include if t in mat.columns]]

    mat = mat.T
    if not mat.empty:
        mat = mat.reindex(mat.mean(axis=1).sort_values(ascending=False).index)

    type_info = adata.obs[['subclass', 'type']]\
        .drop_duplicates().set_index('subclass')
    col_df = pd.DataFrame(index=mat.columns).join(type_info)
    col_df['type'] = pd.Categorical(
        col_df['type'], categories=['Glut', 'Gaba', 'NN'], ordered=True
    )
    col_df = col_df.sort_values('type')

    ordered_cols = []
    for _, group in col_df.groupby('type', sort=False):
        subtypes = group.index
        if len(subtypes) > 1:
            avg_scores = mat[subtypes].mean(axis=0)
            subtypes = avg_scores.sort_values(ascending=False).index
        ordered_cols.extend(subtypes)
    mat = mat[ordered_cols]

    a_types, b_types = mat.index.tolist(), mat.columns.tolist()
    fig, ax = plt.subplots(figsize=(13.5, 3.5), facecolor='white')

    col_df_sorted = col_df.reindex(b_types)
    type_boundaries = col_df_sorted['type'].ne(
        col_df_sorted['type'].shift()
    ).cumsum()
    gaps = np.where(type_boundaries.diff() > 0)[0]

    plot_mat = _insert_gap(mat, gaps, axis=1)
    b_types_gapped = _insert_gap(pd.Series(b_types, dtype=object), gaps)

    im = ax.pcolormesh(plot_mat.values, cmap='GnBu', rasterized=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for r0, r1 in [(0, len(a_types))]:
        for c0, c1 in _segments(~plot_mat.isna().all(0)):
            ax.add_patch(patches.Rectangle(
                (c0, r0), c1 - c0, r1 - r0,
                fill=False, ec='black', lw=0.5
            ))

    ax.set_xlim(0, len(b_types_gapped))
    ax.set_ylim(len(a_types), 0)
    ax.set_xticks(np.arange(len(b_types_gapped)) + 0.5)
    ax.set_yticks(np.arange(len(a_types)) + 0.5)
    ax.set_xticklabels(b_types_gapped, rotation=45, ha='right')
    ax.set_yticklabels(a_types)
    ax.tick_params(length=0)
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('GWAS Trait')

    fig.tight_layout()
    fig.subplots_adjust(right=0.88, bottom=0.3, left=0.2)

    cbar_ax = fig.add_axes([0.9, 0.3, 0.015, 0.4])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'Mean of Median $-\log_{10}(P\text{-value})$')

    fig.savefig(
        f'{figures_dir}/gwas_association_heatmap.svg', bbox_inches='tight')
    fig.savefig(
        f'{figures_dir}/gwas_association_heatmap.png', 
        dpi=300, bbox_inches='tight'
    )
    plt.close(fig)

def plot_boxplot_profiles(
    output_dir, figures_dir, conditions, traits_of_interest: List[str],
    p_adj_threshold: float = 0.05
):
    condition_colors = {
        'control': '#7209b7',
        'pregnant': '#b5179e',
        'postpartum': '#f72585'
    }
    sample_map = {s: c for c, L in conditions.items() for s in L}
    all_samples = sorted([s for L in conditions.values() for s in L])

    files = [
        p for sample in all_samples for trait in traits_of_interest
        if (p := Path(output_dir) / sample / 'report' / trait /
                'gsMap_plot' / f'{sample}_{trait}_gsMap_plot.csv').exists()
    ]
    df = pl.concat([
        pl.scan_csv(f).with_columns(
            sample=pl.lit(f.parts[-5]),
            trait=pl.lit(f.parts[-3]),
            condition=pl.lit(sample_map.get(f.parts[-5]))
        ) for f in files
    ])\
    .with_columns(gsmap_score=-pl.col('p').log10())\
    .select(['gsmap_score', 'annotation', 'sample', 'trait', 'condition'])\
    .collect()\
    .to_pandas()

    median_scores = df.groupby(['trait', 'annotation', 'condition'])\
        ['gsmap_score'].median().unstack().dropna()
    median_scores['range'] = median_scores.max(axis=1) \
        - median_scores.min(axis=1)

    top_annotations_idx = median_scores.groupby('trait')['range']\
        .nlargest(5).index

    top_annotations_df = pd.DataFrame({
        'trait': top_annotations_idx.get_level_values(0),
        'annotation': top_annotations_idx.get_level_values(2)
    })

    plot_df = pd.merge(df, top_annotations_df, on=['trait', 'annotation'])
    plot_df = plot_df.merge(
        median_scores['range'].reset_index(), on=['trait', 'annotation']
    ).reset_index(drop=True)

    to_r(plot_df, 'plot_data_py')
    to_r(condition_colors, 'condition_colors')
    to_r(traits_of_interest, 'traits_of_interest')
    to_r(p_adj_threshold, 'p_adj_threshold')

    r_script = f'''
        suppressPackageStartupMessages({{
            library(ggplot2)
            library(ggpubr)
            library(dplyr)
            library(forcats)
            library(rstatix)
            library(svglite)
            library(ggbeeswarm)
            library(patchwork)
            library(lmerTest)
            library(emmeans)
            library(tidyr)
        }})

        plot_data <- as.data.frame(plot_data_py)
        condition_order <- c("control", "pregnant", "postpartum")
        plot_data$condition <- factor(
            plot_data$condition, levels = condition_order
        )

        create_trait_plot <- function(trait_name) {{
            trait_df <- plot_data %>% filter(trait == trait_name)
            
            annotation_order <- trait_df %>%
                group_by(annotation) %>%
                summarise(mean_score = mean(gsmap_score, na.rm = TRUE)) %>%
                arrange(desc(mean_score)) %>%
                pull(annotation)
            
            trait_df$annotation <- factor(
                trait_df$annotation, levels = annotation_order)

            plottable_annotations <- trait_df %>%
                group_by(annotation, condition) %>%
                summarise(n = n(), .groups = "drop_last") %>%
                filter(n >= 2) %>%
                summarise(n_conditions = n(), .groups = "drop") %>%
                filter(n_conditions == 3) %>%
                pull(annotation)

            trait_df_filtered <- trait_df %>%
                filter(annotation %in% plottable_annotations)

            if(nrow(trait_df_filtered) == 0) {{ return(NULL) }}

            omnibus_results <- trait_df_filtered %>%
                group_by(annotation) %>%
                do({{
                    df <- .
                    if (length(unique(df$sample)) > length(unique(df$condition))) {{
                        lmm_fit <- lmer(gsmap_score ~ condition + (1 | sample), data = df)
                        anova_res <- anova(lmm_fit)
                        data.frame(p.value = anova_res$`Pr(>F)`[1])
                    }} else {{
                        data.frame(p.value = NA_real_)
                    }}
                }}) %>%
                ungroup() %>%
                filter(!is.na(p.value))
            
            if(nrow(omnibus_results) > 0) {{
                omnibus_results <- omnibus_results %>%
                    mutate(p.adj = p.adjust(p.value, method = "fdr"))
                
                significant_annotations <- omnibus_results %>%
                    filter(p.adj < p_adj_threshold) %>%
                    pull(annotation)
            }} else {{
                significant_annotations <- c()
            }}

            if (length(significant_annotations) == 0) {{
                stat_test_sig <- data.frame()
            }} else {{
                trait_df_for_pairwise <- trait_df_filtered %>%
                    filter(annotation %in% significant_annotations)
                
                stat_test <- trait_df_for_pairwise %>%
                    group_by(annotation) %>%
                    do({{
                        df <- .
                        lmm_fit <- lmer(gsmap_score ~ condition + (1 | sample), data = df)
                        emm_res <- emmeans(lmm_fit, ~ condition)
                        pairs(emm_res) %>% as.data.frame()
                    }}) %>%
                    ungroup()

                if(nrow(stat_test) > 0) {{
                     stat_test <- stat_test %>%
                        mutate(p.adj = p.adjust(p.value, method = "fdr"))
                }}

                stat_test_sig <- stat_test %>%
                    filter(p.adj < p_adj_threshold) %>%
                    mutate(p.adj.signif = "*") %>%
                    separate(
                        contrast, 
                        into = c("group1", "group2"), 
                        sep = " - "
                    )
            }}

            max_scores <- trait_df_filtered %>%
                group_by(annotation) %>%
                summarise(max_score = max(gsmap_score, na.rm = TRUE))

            if(nrow(stat_test_sig) > 0) {{
                stat_test_sig <- stat_test_sig %>%
                    left_join(max_scores, by = "annotation") %>%
                    group_by(annotation) %>%
                    mutate(y.position = max_score + (row_number() * max_score * 0.12)) %>%
                    ungroup()
            }}

            p <- trait_df_filtered %>%
                ggplot(aes(x = condition, y = gsmap_score)) +
                facet_wrap(~annotation, nrow = 1, strip.position = "top") +
                geom_quasirandom(
                    aes(color = condition),
                    dodge.width = 0.8, alpha = 0.6, size = 0.8,
                    stroke = 0
                ) +
                geom_boxplot(
                    aes(fill = condition),
                    outlier.shape = NA, alpha = 0.4,
                    width = 0.6
                )
            
            if(nrow(stat_test_sig) > 0) {{
                p <- p + stat_pvalue_manual(
                    stat_test_sig, label = "p.adj.signif",
                    y.position = "y.position",
                    tip.length = 0.01,
                    hjust = 0.5, vjust = -0.2
                )
            }}

            p <- p +
                scale_color_manual(
                    values = condition_colors,
                    name = "Condition", guide = "none"
                ) +
                scale_fill_manual(
                    values = condition_colors,
                    name = "Condition"
                ) +
                labs(
                    y = bquote("gsMap Score ("~-log[10]~P~")"),
                    x = NULL, title = trait_name
                ) +
                theme_classic() +
                theme(
                    text = element_text(family = "DejaVu Sans"),
                    plot.title = element_text(
                        hjust = 0.5, size = 14
                    ),
                    legend.position = "none",
                    panel.border = element_rect(
                        colour = "black", fill=NA, linewidth=0.5),
                    strip.background = element_blank(),
                    strip.text = element_text(hjust = 0),
                    panel.spacing.y = unit(0.5, "lines"),
                    axis.text.x = element_text(angle = 45, hjust = 1)
                )
            return(p)
        }}

        plots <- lapply(traits_of_interest, create_trait_plot)
        plots <- plots[!sapply(plots, is.null)]

        if (length(plots) > 0) {{
            if (length(plots) > 1) {{
                combined_plot <- Reduce(`/`, plots) + 
                    plot_layout(guides = "collect") &
                    theme(legend.position = "bottom")
            }} else {{
                combined_plot <- plots[[1]]
            }}
            
            dir.create("{figures_dir}", showWarnings = FALSE, recursive = TRUE)
            ggsave(
                paste0("{figures_dir}/condition_comparison_boxplot.svg"),
                plot = combined_plot, width = 7, 
                height = 2.5 * length(plots)
            )
            ggsave(
                paste0("{figures_dir}/combined_divergent_profiles.png"),
                plot = combined_plot, width = 7, 
                height = 2.5 * length(plots), dpi = 300
            )
        }}
    '''
    r(r_script)



def plot_profile_heatmap(
    output_dir, figures_dir, conditions
):
    traits_of_interest = ['MDD', 'Neuroticism']
    sample_map = {s: c for c, L in conditions.items() for s in L}
    files = Path(output_dir).glob('*/cauchy_combination/*.Cauchy.csv.gz')
    df = pl.concat([
        pl.scan_csv(f).with_columns(
            condition=pl.lit(sample_map.get(f.parts[-3])),
            trait=pl.lit(
                f.name.replace(f.parts[-3] + '_', '')
                .replace('.Cauchy.csv.gz', '')
            )
        ) for f in files if sample_map.get(f.parts[-3])
    ]).filter(pl.col('trait').is_in(traits_of_interest)).collect()
    pivoted_df = df\
        .with_columns(p_log=-pl.col('p_cauchy').log10())\
        .group_by('trait', 'annotation', 'condition')\
        .agg(pl.median('p_log').alias('score'))\
        .pivot(
            index=['trait', 'annotation'],
            columns='condition',
            values='score'
        )\
        .drop_nulls()\
        .to_pandas()\
        .reset_index()
    
    pivoted_df['range'] = pivoted_df[list(conditions.keys())].max(axis=1) - \
                        pivoted_df[list(conditions.keys())].min(axis=1)
    
    def get_trend(row):
        c, p, pp = row['control'], row['pregnant'], row['postpartum']
        if c <= p and p <= pp: return 'Increasing'
        if c >= p and p >= pp: return 'Decreasing'
        if p >= c and p >= pp: return 'Peak Pregnancy'
        if p <= c and p <= pp: return 'Dip Pregnancy'
        return 'Other'

    pivoted_df['trend'] = pivoted_df.apply(get_trend, axis=1)
    
    for trait in traits_of_interest:
        trait_data = pivoted_df[pivoted_df['trait'] == trait] \
            .nlargest(25, 'range') \
            .reset_index(drop=True)
        
        to_r(trait_data, 'plot_data_py')
        
        r_script = f'''
            library(ComplexHeatmap)
            library(circlize)
            library(svglite)

            plot_data <- as.data.frame(plot_data_py)
            rownames(plot_data) <- plot_data$annotation

            trend_order <- c(
                "Increasing", "Peak Pregnancy", 
                "Dip Pregnancy", "Decreasing", "Other"
            )
            plot_data$trend <- factor(
                plot_data$trend, levels = trend_order
            )

            heatmap_matrix <- as.matrix(
                plot_data[, c("control", "pregnant", "postpartum")]
            )
            
            trend_colors <- c(
                "Increasing" = "#E66101", "Peak Pregnancy" = "#FDB863",
                "Dip Pregnancy" = "#5E3C99", "Decreasing" = "#B2ABD2",
                "Other" = "grey"
            )
            
            min_val <- min(heatmap_matrix, na.rm = TRUE)
            max_val <- max(heatmap_matrix, na.rm = TRUE)
            med_val <- median(heatmap_matrix, na.rm = TRUE)

            left_ha <- HeatmapAnnotation(
                "Trend" = plot_data$trend,
                col = list(Trend = trend_colors), which = "row",
                show_annotation_name = FALSE,
                annotation_width = unit(0.75, "cm")
            )
            
            right_ha <- HeatmapAnnotation(
                "Score Range" = anno_barplot(
                    plot_data$range, 
                    gp = gpar(col = NA, fill = "grey")
                ),
                which = "row", show_annotation_name = TRUE,
                annotation_width = unit(2, "cm"),
                annotation_name_side = "top",
                annotation_name_rot = 0,
                annotation_name_gp = gpar(fontsize = 10)
            )

            ht <- Heatmap(
                heatmap_matrix, name = "−log10(P-value)",
                col = colorRamp2(
                    c(min_val, med_val, max_val), 
                    c("white", "tomato", "darkred")
                ),
                border = TRUE,
                cluster_rows = FALSE, cluster_columns = FALSE,
                row_split = plot_data$trend, row_title = NULL,
                row_labels = rownames(heatmap_matrix),
                row_names_side = "left",
                row_names_gp = gpar(fontsize = 9),
                left_annotation = left_ha,
                right_annotation = right_ha,
                column_title = "{trait}",
                column_title_gp = gpar(fontsize = 16)
            )
            
            svglite("{figures_dir}/condition_comparison_heatmap_{trait}.svg", 
                    width = 6.5, height = 10,
                    system_fonts = list(sans = "DejaVu Sans"))
            draw(ht, heatmap_legend_side = "bottom", 
                 annotation_legend_side = "bottom")
            dev.off()

            png("{figures_dir}/condition_comparison_heatmap_{trait}.png", 
                width = 5, height = 10, units = "in", res = 300)
            draw(ht, heatmap_legend_side = "bottom", 
                 annotation_legend_side = "bottom")
            dev.off()
        '''
        r(r_script)





def plot_spatial_gwas(
    output_dir, figures_dir, conditions, trait, logp_threshold, cell_types
):
    fig, axes = plt.subplots(
        3, 1, figsize=(5, 9), facecolor='white', sharex=True
    )
    fig.subplots_adjust(hspace=0.1)
    
    all_dfs = []
    for condition in conditions.keys():
        sample_names = conditions.get(condition, [])
        condition_dfs = []
        for sample in sample_names:
            file_path = (
                Path(output_dir) / sample / 'report' / trait / 'gsMap_plot' /
                f'{sample}_{trait}_gsMap_plot.csv'
            )
            if file_path.exists():
                df = pd.read_csv(file_path)
                condition_dfs.append(df)
        if condition_dfs:
            condition_df = pd.concat(condition_dfs, ignore_index=True)
            condition_df['condition'] = condition
            all_dfs.append(condition_df)

    if not all_dfs:
        print(f"No data found for trait '{trait}'")
        plt.close(fig)
        return
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    is_selected_type = combined_df['annotation'].isin(cell_types)
    combined_df['alpha'] = np.where(is_selected_type, 1.0, 0.3)
    combined_df['edgecolor'] = np.where(is_selected_type, 'black', 'none')
    combined_df['linewidth'] = np.where(is_selected_type, 0.5, 0.0)
    combined_df['size'] = np.where(is_selected_type, 15, 5)

    vmin = combined_df['logp'].min()
    vmax = combined_df['logp'].max()

    for i, (condition, ax) in enumerate(zip(conditions.keys(), axes)):
        plot_df = combined_df[combined_df['condition'] == condition]
        plot_df = plot_df.sort_values(by='size').reset_index(drop=True)
        
        ax.scatter(
            x=plot_df['sx'], y=plot_df['sy'], c=plot_df['logp'],
            alpha=plot_df['alpha'], s=plot_df['size'],
            edgecolors=plot_df['edgecolor'], linewidths=plot_df['linewidth'],
            cmap='inferno', vmin=vmin, vmax=vmax, rasterized=True
        )

        ax.set_title(condition.capitalize())
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mappable = cm.ScalarMappable(norm=norm, cmap='inferno')

    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.015])
    cbar = fig.colorbar(
        mappable, cax=cbar_ax, orientation='horizontal'
    )
    cbar.set_label(r'$-\log_{10}(P)$')
    
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = f'{figures_dir}/spatial_plot_{trait}'
    fig.savefig(f'{fig_path}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{fig_path}.svg', bbox_inches='tight')
    plt.close(fig)

#endregion

#region prep data ##############################################################

if not os.path.exists(f'{input_dir}/gsMap_resource'):
    os.makedirs(input_dir, exist_ok=True)
    run(f'wget https://yanglab.westlake.edu.cn/data/gsMap/gsMap_resource.tar.gz '
        f'-P {input_dir}')
    run(f'tar -xvzf {input_dir}/gsMap_resource.tar.gz -C {input_dir}')
    run(f'rm {input_dir}/gsMap_resource.tar.gz')

adata_curio = sc.read_h5ad(
    f'{workdir}/output/data/adata_query_curio_final.h5ad')
adata_merfish = sc.read_h5ad(
    f'{workdir}/output/data/adata_query_merfish_final.h5ad')
for adata in [adata_curio, adata_merfish]:
    for col in ['class', 'subclass']:
        adata.obs[col] = adata.obs[col].astype(str)\
            .str.extract(r'^(\d+)\s+(.*)', expand=False)[1]
common_cell_types = (
    set(adata_curio.obs[
        adata_curio.obs['subclass_keep']]['subclass'])
    & set(adata_merfish.obs[
        adata_merfish.obs['subclass_keep']]['subclass']))
del adata_curio, adata_merfish

adata = sc.read_h5ad(f'{workdir}/output/data/adata_query_curio_final.h5ad')
adata.obsm['spatial'] = adata.obs[['x_ffd', 'y_ffd']].to_numpy()

for col in ['class', 'subclass']:
    adata.obs[col] = adata.obs[col].astype(str)\
        .str.extract(r'^(\d+)\s+(.*)', expand=False)[1]
    adata.obs['type'] = adata.obs['subclass']\
        .astype(str).str.extract(r'(\w+)$', expand=False)
    adata.obs['type'] = adata.obs['type'].replace({'IMN': 'Gaba'})
    adata.obs['type'] = adata.obs['type'].replace({'Chol': 'Gaba'})

all_sample_names = adata.obs['sample'].unique()
for sample_name in all_sample_names:
    if not os.path.exists(f'{input_dir}/ST/{sample_name}.h5ad'):
        os.makedirs(f'{input_dir}/ST', exist_ok=True)
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

conditions = {
    'control': [s for s in all_sample_names if 'CTRL' in s],
    'pregnant': [s for s in all_sample_names if 'PREG' in s],
    'postpartum': [s for s in all_sample_names if 'POSTPART' in s]
}

#endregion

#region run gsmap ##############################################################

for condition, sample_names in conditions.items():
    slice_mean_file = f'{output_dir}/{condition}_slice_mean.parquet'
    h5ad_paths = ' '.join([f'{input_dir}/ST/{name}.h5ad' for name in sample_names])
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

traits = []
for gwas_file in sorted(os.listdir(gwas_formatted_dir)):
    if gwas_file.endswith('.sumstats.gz'):
        trait = gwas_file.replace('.sumstats.gz', '')
        traits.append(trait)

for sample_name in all_sample_names:
    for trait_name in traits:
        cauchy_type_dir = f'{output_dir}/{sample_name}/cauchy_combination_type'
        os.makedirs(cauchy_type_dir, exist_ok=True)
        output_file = f'{cauchy_type_dir}/{sample_name}_{trait_name}.Cauchy.csv.gz'
        if os.path.exists(output_file):
            continue
        run(f'''
            gsmap run_cauchy_combination \
                --workdir '{output_dir}' \
                --sample_name '{sample_name}' \
                --trait_name '{trait_name}' \
                --annotation 'type' \
                --output_file '{output_file}'
        ''')

#endregion

#region analysis ###############################################################

adata = sc.read_h5ad(f'{workdir}/output/data/adata_query_curio_final.h5ad')
for col in ['class', 'subclass']:
    adata.obs[col] = adata.obs[col].astype(str)\
        .str.extract(r'^(\d+)\s+(.*)', expand=False)[1]
adata.obs['type'] = adata.obs['subclass']\
    .astype(str).str.extract(r'(\w+)$', expand=False)
adata.obs['type'] = adata.obs['type'].replace({'IMN': 'Gaba'})
adata.obs['type'] = adata.obs['type'].replace({'Chol': 'Gaba'})

all_sample_names = adata.obs['sample'].unique()
conditions = {
    'control': [s for s in all_sample_names if 'CTRL' in s],
    'pregnant': [s for s in all_sample_names if 'PREG' in s],
    'postpartum': [s for s in all_sample_names if 'POSTPART' in s]
}
traits = ['MDD', 'Neuroticism', 'ADHD', 'Autism', 'PTSD']

plot_trait_ranking(output_dir, figures_dir, conditions)

plot_gwas_heatmap(
    adata, output_dir, figures_dir, conditions, traits)

plot_boxplot_profiles(
    output_dir, figures_dir, conditions, traits_of_interest=['MDD'],
    p_adj_threshold=0.1
)

plot_profile_heatmap(output_dir, figures_dir, conditions)




plot_spatial_gwas(
    output_dir=output_dir,
    figures_dir=figures_dir,
    conditions=conditions,
    trait='MDD',
    logp_threshold=12.0,
    cell_types=['SI-MPO-LPO Lhx8 Gaba', 'MPO-LPO Lhx8 Gaba']
)

plot_spatial_gwas(
    output_dir=output_dir,
    figures_dir=figures_dir,
    conditions=conditions,
    trait='Neuroticism',
    logp_threshold=12.0,
    cell_types=['LSX Nkx2-1 Gaba', 'LSX Prdm12 Zeb2 Gaba']
)

#endregion