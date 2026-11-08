import json, os
from pathlib import Path
import numpy as np
from scipy import sparse
ROOT=Path('/home/u202522802045/share/zpb')
V1=ROOT/'data/cache/d_mode/Mouse-brain/four_dataset_v1_20261023/source_prepared'
V2=ROOT/'data/cache/d_mode/Mouse-brain/four_dataset_v2_20261024/source_prepared'
V2.mkdir(parents=True,exist_ok=True)
for k in ['D3','D4','Gao']:
    link=V2/k
    if not link.exists(): os.symlink(V1/k,link,target_is_directory=True)
cache=ROOT/'data/cache/s_mode/Mouse-brain/ISSAAC_E-MTAB-11264_v1_20261023/ISSAAC'
bench=ROOT/'data/benchmark/Mouse-brain/ISSAAC_E-MTAB-11264_v1_20261023/ISSAAC'
out=V2/'ISSAAC'
if not out.exists():
    out.mkdir()
    tr=np.load(cache/'train_index.npy'); va=np.load(cache/'val_index.npy'); te=np.load(cache/'test_index.npy')
    n=len(tr)+len(va)+len(te); rna=np.zeros((n,2000),np.float32)
    rna[tr]=np.load(cache/'rna_train.npy'); rna[va]=np.load(cache/'rna_val.npy'); rna[te]=np.load(cache/'rna_test.npy')
    np.save(out/'rna.npy',rna)
    sparse.save_npz(out/'atac_binary.npz',sparse.load_npz(bench/'atac_binary.npz').tocsr().astype(np.float32))
    np.save(out/'peak_names.npy',np.load(cache/'peak_names.npy',allow_pickle=True))
    np.save(out/'gene_ids.npy',np.load(cache/'gene_ids.npy',allow_pickle=True))
    np.savez_compressed(out/'metadata.npz',train_index=tr,val_index=va,test_index=te,labels=np.array([],np.int64))
    (out/'manifest.json').write_text(json.dumps(dict(status='SOURCE_PREPARED',dataset='ISSAAC',cells=n,genes=2000,peaks=int(len(np.load(out/'peak_names.npy',allow_pickle=True)))),indent=2)+'\n')
print('source prepared ready')
