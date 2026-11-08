import json
from pathlib import Path
import numpy as np
ROOT=Path('/home/u202522802045/share/zpb')
caches={
 'D3':ROOT/'data/cache/s_mode/Mouse-brain/full_b2_intra_v1_20261022/D3',
 'D4':ROOT/'data/cache/s_mode/Mouse-brain/GSE229169_intra_v1_20261023/GSE229169',
 'Gao':ROOT/'data/cache/s_mode/Mouse-brain/adult_cortex_author_processed_intra_pre_final_20261023/GaoVIS',
 'FC':ROOT/'data/cache/s_mode/Mouse-brain/FC_author_labeled_v1_20261023/FC'}
genes={k:np.load(v/'gene_ids.npy',allow_pickle=True).astype(str) for k,v in caches.items()}
res={}
for target in genes:
    src=[k for k in genes if k!=target]
    inter=set(genes[src[0]])
    for k in src[1:]: inter &= set(genes[k])
    union=set().union(*[set(genes[k]) for k in src])
    res[target]=dict(sources=src,source_gene_counts={k:len(genes[k]) for k in src},
        common_genes=len(inter),union_genes=len(union),
        per_source_retained={k:len(set(genes[k]) & inter) for k in src})
print(json.dumps(res,indent=2))
print('pairwise')
for i,a in enumerate(genes):
    for b in list(genes)[i+1:]:
        print(a,b,len(set(genes[a])&set(genes[b])))
