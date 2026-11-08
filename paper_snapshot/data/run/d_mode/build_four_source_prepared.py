import json, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
ROOT=Path('/home/u202522802045/share/zpb')
OUT=ROOT/'data/cache/d_mode/Mouse-brain/four_dataset_v1_20261023/source_prepared'
OUT.mkdir(parents=True,exist_ok=True)
D3OLD=ROOT/'data/cache/d_mode/Mouse-brain/full_b2_v1_20261022/source_prepared/D3'
DS={
 'D3':dict(cache=ROOT/'data/cache/s_mode/Mouse-brain/full_b2_intra_v1_20261022/D3',bench=None),
 'D4':dict(cache=ROOT/'data/cache/s_mode/Mouse-brain/GSE229169_intra_v1_20261023/GSE229169',bench=ROOT/'data/benchmark/Mouse-brain/GSE229169_intra_v1_20261023/GSE229169'),
 'Gao':dict(cache=ROOT/'data/cache/s_mode/Mouse-brain/adult_cortex_author_processed_intra_pre_final_20261023/GaoVIS',bench=ROOT/'data/benchmark/Mouse-brain/adult_cortex_author_processed_v1_20261022/GaoVIS'),
 'FC':dict(cache=ROOT/'data/cache/s_mode/Mouse-brain/FC_author_labeled_v1_20261023/FC',bench=ROOT/'data/benchmark/Mouse-brain/FC_author_labeled_v1_20261023/FC')}
def sha(p):
    import hashlib
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
# D3 reuse old source_prepared via symlink
link=OUT/'D3'
if not link.exists():
    os.symlink(D3OLD, link, target_is_directory=True)
    print('linked D3')
for name,d in DS.items():
    if name=='D3': continue
    dest=OUT/name
    if dest.exists(): print('exists',name); continue
    dest.mkdir()
    c=d['cache']; b=d['bench']
    tr=np.load(c/'train_index.npy'); va=np.load(c/'val_index.npy'); te=np.load(c/'test_index.npy')
    n=len(tr)+len(va)+len(te)
    rna=np.zeros((n,2000),np.float32)
    rna[tr]=np.load(c/'rna_train.npy'); rna[va]=np.load(c/'rna_val.npy'); rna[te]=np.load(c/'rna_test.npy')
    np.save(dest/'rna.npy',rna)
    atac=sparse.load_npz(b/'atac_binary.npz' if (b/'atac_binary.npz').exists() else b/'atac_counts.npz').tocsr()
    atac=atac.astype(bool).astype(np.float32)
    sparse.save_npz(dest/'atac_binary.npz',atac)
    np.save(dest/'peak_names.npy',np.load(c/'peak_names.npy',allow_pickle=True))
    np.save(dest/'gene_ids.npy',np.load(c/'gene_ids.npy',allow_pickle=True))
    np.savez_compressed(dest/'metadata.npz',train_index=tr,val_index=va,test_index=te,labels=np.array([],np.int64))
    (dest/'manifest.json').write_text(json.dumps(dict(status='SOURCE_PREPARED',dataset=name,cells=n,genes=2000,peaks=int(atac.shape[1]),
        rna_sha256=sha(dest/'rna.npy'),atac_sha256=sha(dest/'atac_binary.npz'),peak_names_sha256=sha(dest/'peak_names.npy')),indent=2)+'\n')
    print('built',name,n,atac.shape,atac.nnz,flush=True)
print('DONE')
