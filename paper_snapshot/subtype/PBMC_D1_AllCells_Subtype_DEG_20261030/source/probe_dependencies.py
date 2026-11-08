import importlib.util,os,sys
from pathlib import Path
from analysis_common import *
for name in ['gseapy','scanpy','numpy','pandas','scipy','sklearn','matplotlib']:
    spec=importlib.util.find_spec(name)
    print(name,spec.origin if spec else None)
for p in [SHARED/'data/raw/PBMC/PBMC-MO-001',Path('/home/u202522802045/glue_156478')]:
    print('DIR',p,[f.name for f in p.iterdir()][:20])
print('CPU_THREADS', {k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']})
