"""Export frozen PBMC D-mode B_D edges and basic per-fold statistics."""
import gzip, json, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse

ROOT=Path('/home/u202522802045/share/zpb')
BASE=ROOT/'result_contain/99_pbmc_lodo_dmode_20261024/full_b2/validated_source_artifacts'
OUT=ROOT/'result_contain/pbmc_dmode_regulatory_biology_20261026'
GTF=ROOT/'data/reference/genome_annotation/Homo_sapiens/GRCh38/gencode_32/gencode.v32.primary_assembly.annotation.gtf.gz'
FOLDS=['target_D1','target_D2','target_D3','target_D4']

def symbols():
    d={}
    with gzip.open(GTF,'rt') as f:
        for line in f:
            if line.startswith('#'): continue
            a=line.rstrip().split('\t')
            if len(a)<9 or a[2]!='gene': continue
            gid=re.search(r'gene_id "([^"]+)"',a[8]); name=re.search(r'gene_name "([^"]+)"',a[8])
            if gid: d[gid.group(1).split('.')[0]]=name.group(1) if name else ''
    return d

def parse_peak(p):
    m=re.match(r'^(chr[^:]+):(\d+)-(\d+)$',p)
    if not m: raise ValueError('bad peak '+p)
    return m.group(1),int(m.group(2)),int(m.group(3))

sym=symbols(); rows=[]
for fold in FOLDS:
    d=BASE/fold
    b=np.load(d/'B_D.npz'); mat=sparse.csr_matrix((b['data'],b['indices'],b['indptr']),shape=tuple(b['shape']))
    genes=np.load(d/'gene_ids.npy',allow_pickle=True).astype(str)
    peaks=np.load(d/'harmonized/canonical_peak_names.npy',allow_pickle=True).astype(str)
    assert mat.shape==(len(peaks),len(genes)),(mat.shape,len(peaks),len(genes))
    coo=mat.tocoo(); out=[]
    for p,g,w in zip(coo.row,coo.col,coo.data):
        if w==0: continue
        ch,st,en=parse_peak(peaks[p]); out.append((fold,ch,st,en,genes[g],sym.get(genes[g],''),float(w),abs(float(w)),int(np.sign(w)),int(p),int(g)))
    df=pd.DataFrame(out,columns=['fold','peak_chr','peak_start','peak_end','gene_id','gene_name','weight','abs_weight','sign','peak_index','gene_index'])
    with gzip.open(OUT/'edges'/f'{fold}_edges.tsv.gz','wt',newline='') as f: df.to_csv(f,sep='\t',index=False)
    rows.append(dict(fold=fold,sources=','.join(json.loads((d/'manifest.json').read_text())['sources']),genes=len(genes),peaks=len(peaks),
        candidate_edges=int(json.loads((d/'manifest.json').read_text())['selection']['candidate_edge_count']),
        nonzero_edges=len(df),positive=int((df.sign>0).sum()),negative=int((df.sign<0).sum()),zero_peaks=int(len(peaks)-len(set(df.peak_index))),
        zero_genes=int(len(genes)-len(set(df.gene_index)))))
    print(fold,rows[-1],flush=True)
pd.DataFrame(rows).to_csv(OUT/'edges/per_fold_stats.csv',index=False)
print('DONE')
