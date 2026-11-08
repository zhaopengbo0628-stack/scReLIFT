"""Use the already-validated all-cell graph/UMAP renderer for each new fold."""
import os,sys,json,time,traceback
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'PBMC_D1_AllCells_Dmode_20261030'
fold=sys.argv[1];run=ROOT/'folds'/fold
while not (run/'fold_config.json').exists():time.sleep(15)
sys.path.insert(0,str(run/'source'))
s=(OLD/'source/postprocess.py').read_text()
s=s.replace("'artifacts/D1_barcodes.tsv'",f"'artifacts/{fold}_barcodes.tsv'").replace("'audit/D1_allcell_labels.csv'",f"'audit/{fold}_allcell_labels.csv'")
s=s.replace("'Reference labels | 9,347 cells'","f'Reference labels | {len(labs):,} cells'")
ns={'__name__':'frozen_fold_renderer'};exec(compile(s,str(OLD/'source/postprocess.py'),'exec'),ns)
ns['PALETTE'] += ['#6B8795','#C7A76C','#9A7895','#7A9D68','#D28E5B','#287C8E','#E5C494','#B3DE69','#FCCDE5','#BC80BD']
render=ns['render'];write=ns['write']
paths=[(m,seed,run/m/f'seed_{seed}') for seed in [20020130,20020131,20020132] for m in ['PCA_sourcefit','PCA_targetfit_fullnorm','FullB2']]
while paths:
    progress=False
    for m,seed,p in list(paths):
        flag=run/m/'manifest.json' if m.startswith('PCA') else p/'manifest.json'
        if not flag.exists():continue
        render(m,seed,np.load(p/'target_embedding_standardized.npy'))
        paths.remove((m,seed,p));progress=True
    if paths and not progress:time.sleep(30)
write(run/'audit/ALL_UMAPS_COMPLETE.json',dict(status='COMPLETE',fold=fold,views=9,all_three_seeds=True))
print('ALL_UMAPS_COMPLETE',fold,flush=True)
