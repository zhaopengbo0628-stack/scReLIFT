args <- commandArgs(trailingOnly = TRUE)
directory <- args[[1]]
seed <- as.integer(args[[2]])
latent <- as.integer(args[[3]])
suppressPackageStartupMessages(library(PaletteSC))
suppressPackageStartupMessages(library(Matrix))
upstream <- new.env()
sys.source(args[[4]], envir = upstream)
stopifnot(identical(body(PaletteSC::Run.Palette), body(upstream$Run.Palette)),
          identical(body(PaletteSC:::Run_Multi_SVD), body(upstream$Run_Multi_SVD)))
set.seed(seed)
cells <- readLines(file.path(directory, "train_cells.txt"))
read_matrix <- function(modal) {
  x <- as(readMM(file.path(directory, paste0(modal, ".mtx"))), "dgCMatrix")
  rownames(x) <- readLines(file.path(directory, paste0(modal, "_features.txt")))
  colnames(x) <- cells
  x
}
inputs <- list(RNA = read_matrix("rna"), ATAC = read_matrix("atac"))
object <- Create.Palette.Object(inputs, samples = c("Source", "Source"),
                               modals = c("RNA", "ATAC"), filter = FALSE)
for (i in seq_along(inputs)) {
  object@Assays[[i]]@data <- inputs[[i]]
  embedding <- as.matrix(PaletteSC:::Run_Multi_SVD(list(inputs[[i]]), n.dims = 40)[[1]])
  rownames(embedding) <- paste0("svd", seq_len(nrow(embedding)))
  colnames(embedding) <- cells
  object@Assays[[i]]@embedding <- embedding
}
object <- Run.Palette(object, nDims = latent, supervised = FALSE, seed = seed)
embedding <- object@Int.result[["bind"]]
stopifnot(identical(rownames(embedding), cells), all(is.finite(embedding)))
saveRDS(object, file.path(directory, "source_palette.rds"))
write.table(embedding, file.path(directory, "source_embedding.tsv"), sep = "\t",
            row.names = FALSE, col.names = FALSE, quote = FALSE)
jsonlite::write_json(list(R = R.version.string,
                         PaletteSC = as.character(packageVersion("PaletteSC"))),
                    file.path(directory, "r_runtime.json"), auto_unbox = TRUE)
