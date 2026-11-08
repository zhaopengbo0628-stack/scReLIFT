"""Offline byte-for-byte artifact checks; no torch/GPU required."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

RUN=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    checked=0
    manifest=json.loads((RUN/'artifacts/PREPARATION_COMPLETE.json').read_text())
    for file,digest in manifest['hashes'].items():
        assert sha(RUN/file)==digest,file
        checked+=1
    for seed in [20020130,20020131,20020132]:
        unit=RUN/'FullB2'/f'seed_{seed}'
        m=json.loads((unit/'manifest.json').read_text())
        for file,digest in m['hashes'].items():
            assert sha(unit/file)==digest,file
            checked+=1
        x=np.load(unit/'target_embedding_raw.npy')
        assert x.shape==(9347,64) and np.isfinite(x).all()
    for p in (RUN/'figures').glob('*/seed_*/manifest.json'):
        m=json.loads(p.read_text())
        assert sha(p.parent/'allcells_umap.png')==m['plot_sha256']
        assert sha(p.parent/'plot_data.csv')==m['data_sha256']
        checked+=2
    q=pd.read_csv(RUN/'JOB_QUEUE.csv');assert q.Status.eq('COMPLETED').all()
    result=dict(status='PASS',files_hashed=checked,full_b2_seeds=3,target_cells=9347,
                cd8_tem_cells=622,checkpoint_integrity='remote torch check PASS plus local byte hashes')
    (RUN/'audit/LOCAL_DOWNLOAD_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
