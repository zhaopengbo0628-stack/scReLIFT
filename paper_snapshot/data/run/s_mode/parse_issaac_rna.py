import gzip, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
ROOT=Path('/home/u202522802045/share/zpb')
RAW=ROOT/'data/raw/Mouse-brain/ISSAAC_E-MTAB-11264/downloads'
ST=ROOT/'data/staging/Mouse-brain/ISSAAC_E-MTAB-11264_v1_20261023'
ST.mkdir(parents=True,exist_ok=True)
RNA=RAW/'mCortex_all_gene_expression_matrix.csv.gz'
INFO=RAW/'Sample_Info.csv.gz'
def main():
    t=time.time()
    with gzip.open(RNA,'rt') as f: header=f.readline().rstrip('\n').split(',')
    barcodes=np.array(header[1:],dtype=str)
    print('cells',len(barcodes),flush=True)
    chunks=[]; genes=[]
    for ch in pd.read_csv(RNA,index_col=0,chunksize=5000):
        genes.extend(ch.index.astype(str))
        chunks.append(sparse.csr_matrix(ch.values.astype(np.int32)))
    rna=sparse.vstack(chunks).tocsr().T.tocsr()
    print('rna',rna.shape,rna.nnz,'genes',len(genes),flush=True)
    assert rna.shape==(len(barcodes),len(genes))
    np.save(ST/'gene_names.npy',np.array(genes,dtype=str))
    np.save(ST/'barcodes.npy',barcodes)
    sparse.save_npz(ST/'rna_counts.npz',rna)
    info=pd.read_csv(INFO,index_col=0)
    assert info.index.tolist()==barcodes.tolist()
    info=info.reset_index().rename(columns={'Unnamed: 0':'barcode'})
    info['batch']=info['batch'].replace({'mCortex':'mCortex_rep1','mCortex_rep':'mCortex_rep2'})
    info.to_csv(ST/'metadata.csv',index=False)
    (ST/'rna_manifest.json').write_text(json.dumps(dict(cells=len(barcodes),genes=len(genes),rna_nnz=int(rna.nnz),
        rna_sha256='pending',source=str(RNA)),indent=2)+'\n')
    print('done',round(time.time()-t,1),flush=True)
if __name__=='__main__': main()
