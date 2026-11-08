library(Matrix)
args <- commandArgs(trailingOnly = TRUE)
root <- args[1]
out <- file.path(root, "processed", "rds_export")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
for (modality in c("RNA", "ATAC")) {
  prefix <- if (modality == "RNA") "GSE196988" else "GSE196987"
  name <- paste0(prefix, "_", modality, "_ReadCounts_Final_Multiome.rds")
  x <- readRDS(file.path(root, "raw", name, name))
  stopifnot(inherits(x, "dgCMatrix"), all(x@x >= 0), all(x@x == floor(x@x)))
  stem <- file.path(out, tolower(modality))
  writeBin(as.integer(x@i), paste0(stem, ".indices"), size = 4, endian = "little")
  writeBin(as.integer(x@p), paste0(stem, ".indptr"), size = 4, endian = "little")
  writeBin(as.integer(x@x), paste0(stem, ".values"), size = 4, endian = "little")
  writeLines(rownames(x), paste0(stem, ".features"))
  writeLines(colnames(x), paste0(stem, ".cells"))
  cat(modality, dim(x), length(x@x), "exported\n")
  rm(x)
  gc()
}
