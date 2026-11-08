args <- commandArgs(trailingOnly=TRUE)
dir <- args[[1]]; seed <- as.integer(args[[2]]); ns <- as.integer(args[[3]]); latent <- as.integer(args[[4]])
suppressPackageStartupMessages(library(PaletteSC)); suppressPackageStartupMessages(library(Matrix))
read_matrix <- function(path, features) {
  x <- as(readMM(path), 'dgCMatrix')
  rownames(x) <- readLines(file.path(dir, features))
  x
}
inputs <- list(); samples <- character(); modals <- character()
for (i in seq_len(ns)) {
  r <- read_matrix(file.path(dir,paste0('rna_',i-1,'.mtx')), 'rna_features.txt')
  a <- read_matrix(file.path(dir,paste0('atac_',i-1,'.mtx')), 'atac_features.txt')
  cells <- paste0('Source',i,'_',seq_len(ncol(r)))
  colnames(r) <- cells; colnames(a) <- cells
  inputs <- c(inputs,list(r,a)); samples <- c(samples,rep(paste0('Source',i),2)); modals <- c(modals,'RNA','ATAC')
}
names(inputs) <- paste0(modals,'_',samples)
object <- Create.Palette.Object(inputs,samples=samples,modals=modals,filter=FALSE)
for (i in seq_along(inputs)) {
  e <- as.matrix(PaletteSC:::Run_Multi_SVD(list(inputs[[i]]),n.dims=40)[[1]])
  rownames(e) <- paste0('svd',seq_len(nrow(e))); colnames(e) <- colnames(inputs[[i]])
  object@Assays[[i]]@data <- inputs[[i]]; object@Assays[[i]]@embedding <- e
}
set.seed(seed); object <- Run.Palette(object,nDims=latent,supervised=FALSE,seed=seed)
emb <- as.matrix(object@Int.result[['bind']]); stopifnot(all(is.finite(emb)))
saveRDS(object,file.path(dir,'source_palette.rds')); write.table(emb,file.path(dir,'source_embedding.tsv'),sep='\t',row.names=FALSE,col.names=FALSE,quote=FALSE)
target <- as.matrix(read.csv(file.path(dir,'target_rna.csv'),header=FALSE))
src_rna <- do.call(rbind,lapply(seq(0,ns-1),function(i) t(as.matrix(read_matrix(file.path(dir,paste0('rna_',i,'.mtx')), 'rna_features.txt')))))
w <- qr.solve(crossprod(src_rna) + diag(ncol(src_rna))*1e-4,crossprod(src_rna,emb))
z <- target %*% w; stopifnot(all(is.finite(z))); write.csv(z,file.path(dir,'rna_test_embedding.csv'),row.names=FALSE)
