import json, sys
from pathlib import Path
import h5py, numpy as np
from scipy import sparse
ROOT=Path('/home/u202522802045/share/zpb')
RAW=ROOT/'data/raw/Mouse-brain/Zeppilli_SSp_GSE239477/downloads'
OUT=ROOT/'data/staging/Mouse-brain/GSE239477_SSp_v1_20261023/libraries'
OUT.mkdir(parents=True,exist_ok=True)
GSM={'N1':('GSM7665923','GSM7665924'),'N4':('GSM7665925','GSM7665926'),'N6':('GSM7665927','GSM7665928'),'N7':('GSM7665929','GSM7665930')}
def parse(s):
    p=RAW/f'GSE239477_{s}_filtered_feature_bc_matrix.h5'
    assert p.exists(), p
    with h5py.File(p,'r') as f:
        m=f['matrix']; feat=m['features']
        ft=np.array([x.decode() if isinstance(x,bytes) else str(x) for x in feat['feature_type'][:]])
        genome=np.array([x.decode() if isinstance(x,bytes) else str(x) for x in feat['genome'][:]])
        ids=np.array([x.decode() if isinstance(x,bytes) else str(x) for x in feat['id'][:]])
        names=np.array([x.decode() if isinstance(x,bytes) else str(x) for x in feat['name'][:]])
        intervals=np.array([x.decode() if isinstance(x,bytes) else str(x) for x in feat['interval'][:]])
        bcs=np.array([x.decode() if isinstance(x,bytes) else str(x) for x in m['barcodes'][:]])
        nfeat,ncell=int(m['shape'][0]),int(m['shape'][1])
        X=sparse.csc_matrix((m['data'][:],m['indices'][:],m['indptr'][:]),shape=(nfeat,ncell))
    print(s,'features',nfeat,'cells',ncell,'types',dict(zip(*np.unique(ft,return_counts=True))),'genome',dict(zip(*np.unique(genome,return_counts=True))),flush=True)
    gmask=ft=='Gene Expression'; pmask=ft=='Peaks'
    assert gmask.any() and pmask.any()
    rna=X[gmask,:].T.tocsr(); atac=X[pmask,:].T.tocsr()
    out=OUT/s; out.mkdir(parents=True,exist_ok=True)
    sparse.save_npz(out/'rna_counts.npz',rna)
    sparse.save_npz(out/'atac_counts.npz',atac)
    np.save(out/'barcodes.npy',bcs)
    np.save(out/'gene_ids.npy',ids[gmask]); np.save(out/'gene_names.npy',names[gmask])
    np.save(out/'peak_names.npy',intervals[pmask])
    np.save(out/'gene_genome.npy',genome[gmask]); np.save(out/'peak_genome.npy',genome[pmask])
    (out/'manifest.json').write_text(json.dumps(dict(library=s,GSM_RNA=GSM[s][0],GSM_ATAC=GSM[s][1],
        cells=ncell,genes=int(gmask.sum()),peaks=int(pmask.sum()),rna_nnz=int(rna.nnz),atac_nnz=int(atac.nnz),
        feature_types={str(k):int(v) for k,v in zip(*np.unique(ft,return_counts=True))},
        genome={str(k):int(v) for k,v in zip(*np.unique(genome,return_counts=True))}),indent=2)+'\n')
if __name__=='__main__':
    for s in ['N1','N4','N6','N7']:
        p=RAW/f'GSE239477_{s}_filtered_feature_bc_matrix.h5'
        if p.exists():
            try:
                parse(s)
            except Exception as e:
                print('SKIP',s,e,flush=True)
