import json, sys, numpy as np, pandas as pd
from pathlib import Path
from scipy import sparse
ROOT=Path('/home/u202522802045/share/zpb')
sys.path[:0]=[str(ROOT),str(ROOT/'main/core/regulatory/common/legacy_support')]
from main.core.regulatory.s_mode.single_source import normalize_counts, select_hvg
from v5d.regulatory_graph import read_tss
BENCH=ROOT/'data/benchmark/Mouse-brain/GSE229169_intra_v1_20261023/GSE229169'
RESULT=ROOT/'result_contain/83_gse229169_fullb2_preprocessing'
GTF=ROOT/'data/reference/genome_annotation/Mus_musculus/GRCm38/gencode_M25/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz'
rna=sparse.load_npz(BENCH/'rna_counts.npz').tocsr()
meta=pd.read_csv(BENCH/'metadata.csv',keep_default_na=False)
splits={s:np.load(BENCH/'splits.npz')[s+'_index'] for s in ('train','val','test')}
log=normalize_counts(rna)
mapping=pd.read_csv(BENCH/'identity_map/gene_mapping.tsv',sep='\t',keep_default_na=False)
symbols=np.load(BENCH/'gene_names.npy',allow_pickle=True).astype(str)
tss=read_tss(GTF)
eligible=np.flatnonzero(mapping.status.eq('unique').to_numpy() & mapping.ensembl_gene_id.isin(tss).to_numpy())
selected,scores=select_hvg(log[splits['train']][:,eligible],2000)
cols=eligible[selected]
hvg_det=np.asarray((log[splits['train']][:,cols]>0).mean(0)).ravel()
per_cell=np.asarray((log[:,cols]>0).sum(1)).ravel()
out={
 'eligible_genes':int(len(eligible)),
 'selected_hvg':int(len(cols)),
 'hvg_detection_train':{
   'mean':float(hvg_det.mean()),'median':float(np.median(hvg_det)),'min':float(hvg_det.min()),'max':float(hvg_det.max()),
   'lt_1pct':int((hvg_det<0.01).sum()),'lt_0.1pct':int((hvg_det<0.001).sum())},
 'all_gene_detection_train':{
   'median':float(np.median(np.asarray((log[splits['train']]>0).mean(0)).ravel()))},
 'hvg_per_cell':{'median':float(np.median(per_cell)),'p10':float(np.percentile(per_cell,10)),'p90':float(np.percentile(per_cell,90)),'zeros':int((per_cell==0).sum()),'min':int(per_cell.min()),'max':int(per_cell.max())},
 'split_hvg_per_cell':{s:{'median':float(np.median(per_cell[ix])),'zeros':int((per_cell[ix]==0).sum()),'p10':float(np.percentile(per_cell[ix],10))} for s,ix in splits.items()},
 'subclass_hvg_coverage':{k:{'cells':int(v),'mean_nonzero_hvg':float(per_cell[meta.subclass.to_numpy()==k].mean())} for k,v in meta.subclass.value_counts().items()},
 'score_detection_spearman':float(pd.Series(scores[selected]).corr(pd.Series(hvg_det),method='spearman')),
 'detection_filtered_alternative':{}
}
det=np.asarray((log[splits['train']]>0).mean(0)).ravel()
for th in [0.01,0.05]:
    elig=np.flatnonzero(mapping.status.eq('unique').to_numpy() & mapping.ensembl_gene_id.isin(tss).to_numpy() & (det>=th))
    if len(elig)>=2000:
        sel,sc=select_hvg(log[splits['train']][:,elig],2000); cc=elig[sel]
        hd=np.asarray((log[splits['train']][:,cc]>0).mean(0)).ravel(); pc=np.asarray((log[:,cc]>0).sum(1)).ravel()
        out['detection_filtered_alternative'][str(th)]={'eligible':int(len(elig)),'hvg_det_median':float(np.median(hd)),'lt1pct':int((hd<0.01).sum()),'per_cell_median':float(np.median(pc)),'zeros':int((pc==0).sum())}
(RESULT/'HVG_SPARSITY_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
