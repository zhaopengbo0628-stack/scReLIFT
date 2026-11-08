import json, os, shutil
from pathlib import Path
import numpy as np
from scipy import sparse

base=Path('/home/u202522802045/share/zpb')
p=base/'data/cache/d_mode/PBMC/legacy_full_b2_corrected/_historical_dependencies/source_inputs/D2'
x=np.load(p/'rna.npy',mmap_mode='r')
with np.load(p/'metadata.npz',allow_pickle=True) as m:
    print('D2_META',[(k,m[k].shape,str(m[k].dtype),m[k].ravel()[:3].tolist()) for k in m.files])
print('D2_RNA',x.shape,x.dtype,'slice',x[:2,:20].tolist(),'max',float(np.max(x[:100])))
for f in ['gene_ids.npy','gene_names.npy','peak_names.npy']:
    a=np.load(p/f,allow_pickle=True);print(f,a.shape,a[:3].tolist())
for sub in ['GSE196987_ATAC_Metadata_Final_Multiome.txt','GSE196988_RNA_Metadata_Final_Multiome.txt']:
    f=base/'data/raw/PBMC/PBMC-MO-002'/sub/sub
    with f.open() as h:
        print('METADATA',sub,h.readline().strip(),h.readline().strip())
print('SCHEDULER', {s:shutil.which(s) for s in ['sbatch','squeue','srun','qsub','pbsnodes','bsub','condor_submit','flux','tmux']})
print('CGROUP',Path('/proc/self/cgroup').read_text())
print('CUDA_VISIBILITY',os.environ.get('CUDA_VISIBLE_DEVICES'))
for f in ['/sys/fs/cgroup/memory.max','/sys/fs/cgroup/cpu.max']:
    p=Path(f)
    if p.exists(): print(f,p.read_text().strip())
