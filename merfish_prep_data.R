library(tidyverse)
library(Seurat)
library(anndata)
library(Matrix)

data_dir = "../../spatial/Kalish/pregnancy-postpart/MERFISH/raw-rds"
output_dir = "../../spatial/Kalish/pregnancy-postpart/MERFISH/raw-h5ad"
work_dir = "projects/def-wainberg/karbabi/spatial-pregnancy-postpart"
setwd(work_dir)
dir.create(output_dir, recursive = TRUE)

files = list.files(data_dir, pattern = "\\.rds$")
for (file in files) {
    sample = sub("^(.+)_.*\\.rds$", "\\1", file)
    sample = sub("CTL", "CTRL", sample)
    sobj = readRDS(paste(data_dir, file, sep='/'))
    
    counts = sobj@assays[["SCT"]]@counts
    norm = sobj@assays[["Vizgen"]]@counts[-1, ]
    metadata = sobj@meta.data %>% 
        rename(cell_id = "X", Custom_regions = "Custom.cell.groups") %>%
        select(-Unnamed..0)
    genes = rownames(counts)

    adata = AnnData(
        X = t(counts),
        obs = metadata,
        var = data.frame(gene = genes, row.names = genes),
        layers = list(orig_norm = t(norm))
    )
    write_h5ad(adata, filename = file.path(output_dir, paste0(sample, ".h5ad")))
    rm(sobj, adata); gc()
}

# files = list.files(data_dir)
# for (file in files) {
#     sample = sub("^(.+)_.*\\.rds$", "\\1", file)
#     sample = sub("CTL", "CTRL", sample)
#     sobj = readRDS(paste(data_dir, file, sep='/'))
#     counts = sobj@assays[["SCT"]]@counts
#     norm = sobj@assays[["Vizgen"]]@counts[-1, ]
#     metadata = sobj@meta.data %>% 
#         rename(cell_id = "X", Custom_regions = "Custom.cell.groups") %>%
#         select(-Unnamed..0)
#     genes = rownames(sobj@assays[["SCT"]]@counts)

#     set.seed(1)
#     n_cells = ncol(counts)
#     split_indices = split(sample(n_cells), rep(1:5, length.out = n_cells))
#     # check 
#     all_indices = sort(unlist(split_indices))
#     stopifnot(all(all_indices == 1:n_cells))
#     stopifnot(length(all_indices) == n_cells)

#     for (i in 1:5) {
#         subset_indices = split_indices[[i]]
#         subset_counts = counts[, subset_indices]
#         subset_norm = norm[, subset_indices]
#         subset_metadata = metadata[subset_indices, ]
        
#         adata = AnnData(
#             X = t(subset_counts),
#             obs = subset_metadata,
#             var = data.frame(gene = genes, row.names = genes),
#             layers = list(orig_norm = t(subset_norm))
#         )
#         write_h5ad(adata, 
#             filename = paste0(output_dir, "/", sample, "_", i, ".h5ad"))
#     }
#     rm(sobj); gc()
# }