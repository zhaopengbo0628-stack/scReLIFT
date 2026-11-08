args <- commandArgs(trailingOnly=TRUE)
out <- args[[1]]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
suppressPackageStartupMessages(library(Matrix))
path <- '/home/u202522802045/share/zpb/data/raw/PBMC/PBMC-MO-002/GSE196988_RNA_ReadCounts_Final_Multiome.rds/GSE196988_RNA_ReadCounts_Final_Multiome.rds'
cat('READ_RDS\n'); flush.console()
x <- readRDS(path)
cat('CLASS', class(x), 'DIM', dim(x), '\n'); flush.console()
stopifnot(inherits(x, 'sparseMatrix'))
cells <- readLines('/home/u202522802045/share/zpb/data/cache/s_mode/PBMC/PBMC-MO-002/pbmc_corrected_s10x20_v1_20261012/barcodes.txt')
stopifnot(length(cells)==6317, all(cells %in% colnames(x)))
sub <- x[,cells,drop=FALSE]
writeMM(sub, file.path(out,'rna_counts_genes_by_cells.mtx'))
writeLines(rownames(sub), file.path(out,'gene_names.txt'))
writeLines(colnames(sub), file.path(out,'barcodes.txt'))
cat('HDA_EXTRACT_COMPLETE',dim(sub),'NNZ',length(sub@x),'\n'); flush.console()
