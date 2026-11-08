import csv, json, math
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
    adjusted_mutual_info_score, accuracy_score, balanced_accuracy_score,
    f1_score, silhouette_score)
from sklearn.neighbors import NearestNeighbors

BASE=Path('/home/u202522802045/share/zpb/result_contain/99_pbmc_lodo_dmode_20261024')
ROOT=Path('/home/u202522802045/share/zpb/comparison')
UNITS=BASE/'units'
FOLDS=('target_D1','target_D2','target_D3','target_D4')
METHODS=('palette','multigrate','eclare','multivi')
TARGET={'target_D1':'PBMC-MO-001','target_D2':'PBMC-MO-002','target_D3':'PBMC-MO-003','target_D4':'PBMC-MO-004'}

def labels(fold):
    target=TARGET[fold]
    with (ROOT.parent/'data/cache/legacy/PBMC'/target/'processed_v1/label.csv').open() as f:
        by={r['barcode']:r['cell_type'] for r in csv.DictReader(f)}
    cache=ROOT.parent/'data/cache/s_mode/PBMC'/target/'pbmc_corrected_s10x20_v1_20261012'
    bars=(cache/'barcodes.txt').read_text().splitlines(); test=np.load(cache/'test_index.npy')
    return np.asarray([by[bars[i]] for i in test])

def unit_metrics(fold,path):
    y=labels(fold); z=np.load(path/'rna_test_embedding.npy'); z=(z-z.mean(0))/(z.std(0)+1e-8)
    classes,yi=np.unique(y,return_inverse=True); c=KMeans(n_clusters=len(classes),n_init=100,random_state=20020130,max_iter=1000).fit_predict(z)
    tab=np.zeros((len(classes),len(classes)),dtype=np.int64); np.add.at(tab,(yi,c),1); _,cc=linear_sum_assignment(-tab); mp=dict(zip(cc,np.arange(len(classes)))); m=np.asarray([mp[k] for k in c])
    nn=NearestNeighbors(n_neighbors=min(16,len(z))).fit(z).kneighbors(z,return_distance=False); nn=np.asarray([r[r!=i][:15] for i,r in enumerate(nn)])
    return dict(ARI=float(adjusted_rand_score(yi,c)),NMI=float(normalized_mutual_info_score(yi,c)),AMI=float(adjusted_mutual_info_score(yi,c)),ACC=float(accuracy_score(yi,m)),Macro_F1=float(f1_score(yi,m,average='macro',zero_division=0)),Balanced_ACC=float(balanced_accuracy_score(yi,m)),Weighted_F1=float(f1_score(yi,m,average='weighted',zero_division=0)),Purity15=float(np.mean(yi[nn]==yi[:,None])),Silhouette=float(silhouette_score(z,yi)) if len(classes)>1 else float('nan'))

def efficiency(method,path):
    p=path/'adapter_smoke.json'; a=json.loads(p.read_text())
    if method=='multigrate':
        e=a.get('efficiency_v2',{}); tr=e.get('training',{}); fit=e.get('source_fitting',{}); inf=e.get('inference',{})
        return dict(train_time=a.get('training_seconds'),trainable=a.get('trainable_parameters'),infer_params=a.get('inference_parameters'),peak_gpu=tr.get('peak_allocated_bytes'),infer_time=a.get('inference_seconds'),gpu=a.get('gpu_model','NVIDIA A30'))
    if method=='eclare':
        stages=a.get('stages',[]); train=sum(float(x.get('seconds',0)) for x in stages); params=sum(int(x.get('trainable_parameters',0)) for x in stages); peaks=[int(x.get('peak_allocated_bytes',0)) for x in stages]
        infp=None; cp=path/'best.pt'
        if cp.exists():
            import torch
            x=torch.load(cp,map_location='cpu',weights_only=False); sd=x.get('state_dict',{}); infp=sum(v.numel() for v in sd.values() if hasattr(v,'numel'))
        return dict(train_time=train,trainable=params,infer_params=infp,peak_gpu=max(peaks) if peaks else None,infer_time=None,gpu='NVIDIA A30')
    if method=='multivi':
        cp=path/'checkpoint/model.pt'; params=None
        if cp.exists():
            import torch
            x=torch.load(cp,map_location='cpu',weights_only=False); sd=x.get('model_state_dict',{}); params=sum(v.numel() for v in sd.values() if hasattr(v,'numel'))
        return dict(train_time=None,trainable=params,infer_params=params,peak_gpu=None,infer_time=None,gpu='NVIDIA A30')
    return dict(train_time=None,trainable=None,infer_params=None,peak_gpu=None,infer_time=None,gpu='CPU')

def fmt(vals, digits=3):
    vals=[float(x) for x in vals if x is not None and np.isfinite(x)]
    if not vals: return 'NA'
    return f'{np.mean(vals):.{digits}f} ± {np.std(vals,ddof=1):.{digits}f}' if len(vals)>1 else f'{vals[0]:.{digits}f}'

rows=[]
for method in METHODS:
    print("START", method, flush=True)
    name={'palette':'Palette-U','multigrate':'Multigrate','eclare':'ECLARE','multivi':'MultiVI'}[method]; mm=[]; ee=[]
    for fold in FOLDS:
        for seed in (20020130,20020131,20020132):
            p=UNITS/method/fold/f'seed_{seed}'
            mm.append(unit_metrics(fold,p)); ee.append(efficiency(method,p))
    row={'Method':name}
    for k in mm[0]: row[k]=fmt([x[k] for x in mm])
    row.update({'Training/fit time':fmt([x['train_time'] for x in ee],2),'Trainable params':fmt([x['trainable'] for x in ee],0),'Inference params':fmt([x['infer_params'] for x in ee],0),'Peak GPU':fmt([x['peak_gpu']/1024**2 for x in ee],1)+' MiB' if any(x['peak_gpu'] for x in ee) else 'NA','Inference time':fmt([x['infer_time'] for x in ee],3),'GPU':sorted(set(x['gpu'] for x in ee))})
    rows.append(row)
out=BASE/'metrics/summary_all_methods.csv'; out.parent.mkdir(parents=True,exist_ok=True)
fields=['Method','ARI','NMI','AMI','ACC','Macro_F1','Balanced_ACC','Weighted_F1','Purity15','Silhouette','Training/fit time','Trainable params','Inference params','Peak GPU','Inference time','GPU']
with out.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
(BASE/'metrics/summary_all_methods.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
for r in rows: print(json.dumps(r,ensure_ascii=False))
