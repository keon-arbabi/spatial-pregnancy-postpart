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
    metadata = sobj@meta.data %>% 
        rename(cell_id = "X", Custom_regions = "Custom.cell.groups") %>%
        select(-Unnamed..0)
    genes = rownames(counts)

    adata = AnnData(
        X = t(counts),
        obs = metadata,
        var = data.frame(gene = genes, row.names = genes)
    )
    write_h5ad(adata, filename = file.path(output_dir, paste0(sample, ".h5ad")))
    rm(sobj, adata); gc()
}
