library(Seurat)
library(anndata)
library(Matrix)

data_dir = "../../spatial/Kalish/pregnancy-postpart/curio/raw-rds"
output_dir = "../../spatial/Kalish/pregnancy-postpart/curio/raw-h5ad"
work_dir = "projects/def-wainberg/karbabi/spatial-pregnancy-postpart"
setwd(work_dir)
dir.create(output_dir, recursive = TRUE)

files = list.files(data_dir)
files = files[!grepl('broken', files)]

for (file in files) {
    sample = sub('^([^_]+_[0-9]+(?:_[0-9]+)?).*', '\\1', file)
    sobj = readRDS(paste(data_dir, file, sep='/'))
    counts = sobj@assays[["RNA"]]@counts
    metadata = sobj@meta.data
    metadata$cell_id = rownames(metadata)
    coords = sobj@images[["slice1"]]@coordinates
    spat_embed = as.data.frame(sobj@reductions[["SPATIAL"]]@cell.embeddings)
    stopifnot(!any(is.na(spat_embed$SPATIAL_1)))
    stopifnot(!any(is.na(spat_embed$SPATIAL_2)))
    metadata = merge(metadata, coords, by = "row.names", 
        all = TRUE, sort = FALSE)
    metadata = merge(metadata, spat_embed, 
        by.x = "cell_id", by.y = "row.names", all = TRUE, sort = FALSE)
    rownames(metadata) = metadata$cell_id

    adata  = AnnData(
        X = t(counts), obs = metadata,
        var = data.frame(gene = rownames(sobj@assays$RNA$counts), 
                        row.names = rownames(sobj@assays$RNA$counts)))
    write_h5ad(adata, 
        filename = paste0(output_dir, "/", sample, ".h5ad"))
    rm(sobj); gc()
}
