args<-commandArgs(trailingOnly=TRUE)
run<-args[[1]]
.libPaths(c(file.path(run,'R_library'),.libPaths()))
suppressPackageStartupMessages(library(fgsea))
suppressPackageStartupMessages(library(data.table))
dir.create(file.path(run,'fgsea_expressed'),showWarnings=FALSE)
set.seed(20020130)
w<-fgseaSimple(pathways=list(positive=paste0('G',1:15),negative=paste0('G',106:120)),
    stats=setNames(seq(3,-3,length.out=120),paste0('G',1:120)),nperm=32,minSize=5,maxSize=100,nproc=2)
stopifnot(nrow(w)==2,all(is.finite(w$NES)))
writeLines(paste('FGSEA_CPU_WITNESS_PASS',as.character(packageVersion('fgsea'))),file.path(run,'audit/FGSEA_EXPRESSED_CPU_WITNESS.txt'))
manifest<-read.csv(file.path(run,'fgsea_expressed/jobs.csv'),stringsAsFactors=FALSE)
go<-gmtPathways(file.path(run,'input/GO_Biological_Process_2025.gmt'))
reactome<-gmtPathways(file.path(run,'input/Reactome_2022.gmt'))
names(go)<-paste0('GO_BP_2025::',names(go));names(reactome)<-paste0('Reactome_2022::',names(reactome))
pathways<-c(go,reactome)
for(i in seq_len(nrow(manifest))){
  rankfile<-manifest$rank_file[[i]]
  outfile<-manifest$result_file[[i]]
  if(file.exists(outfile))next
  ranks<-fread(rankfile)
  scores<-setNames(ranks$wilcoxon_score_signed,ranks$gene_symbol)
  stopifnot(length(scores)>1000,!anyDuplicated(names(scores)),all(is.finite(scores)))
  cat('FGSEA_START',i,nrow(manifest),manifest$candidate_id[[i]],manifest$contrast[[i]],'\n');flush.console()
  start<-proc.time()[['elapsed']]
  set.seed(20020130)
  res<-fgseaSimple(pathways=pathways,stats=scores,nperm=10000,minSize=10,maxSize=500,nproc=2,scoreType='std')
  res<-as.data.frame(res)
  res$leadingEdge<-vapply(res$leadingEdge,paste,collapse=';',FUN.VALUE=character(1))
  res$database<-sub('::.*','',res$pathway)
  res$Term<-sub('^[^:]+::','',res$pathway)
  res$candidate_id<-manifest$candidate_id[[i]];res$method<-manifest$method[[i]]
  res$reference_subclass<-manifest$reference_subclass[[i]];res$seed<-manifest$seed[[i]]
  res$contrast<-manifest$contrast[[i]]
  res$permutations<-10000;res$n_ranked_genes<-length(scores)
  res$implementation<-paste0('fgseaSimple ',as.character(packageVersion('fgsea')))
  res$anchor_rule<-'lowest available fixed seed; chosen before DEG'
  res$rank_tie_rule<-'score descending, Ensembl ID ascending; no jitter'
  res$seconds<-proc.time()[['elapsed']]-start
  temp<-paste0(outfile,'.tmp');write.csv(res,temp,row.names=FALSE);file.rename(temp,outfile)
  cat('FGSEA_DONE',i,nrow(res),res$seconds[[1]],'seconds\n');flush.console()
}
writeLines('COMPLETE',file.path(run,'fgsea_expressed/R_RUN_COMPLETE.txt'))
