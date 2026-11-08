"""Strict PBMC D-mode LODO adapter smoke tests.

Assembles source-only folds, applies a source-only peak cap, and calls the
existing official method runners. Target cells are loaded only for frozen RNA
inference.
"""
from pathlib import Path
import argparse, json, subprocess, sys, time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, accuracy_score, f1_score
from scipy.optimize import linear_sum_assignment

BASE = Path('/home/u202522802045/share/zpb/result_contain/99_pbmc_lodo_dmode_20261024')
ART = BASE/'full_b2/validated_source_artifacts'
TGT = BASE/'full_b2/validated_cache'
ROOT = Path('/home/u202522802045/share/zpb/comparison')
SEEDS = (20020130, 20020131, 20020132)
FOLDS = {
    'target_D1': ('PBMC-MO-002','PBMC-MO-003','PBMC-MO-004'),
    'target_D2': ('PBMC-MO-001','PBMC-MO-003','PBMC-MO-004'),
    'target_D3': ('PBMC-MO-001','PBMC-MO-002','PBMC-MO-004'),
    'target_D4': ('PBMC-MO-001','PBMC-MO-002','PBMC-MO-003'),
}
TARGET_NAME = {'target_D1':'PBMC-MO-001','target_D2':'PBMC-MO-002','target_D3':'PBMC-MO-003','target_D4':'PBMC-MO-004'}

def source_peak_columns(fold, sources, cap=2048):
    mats=[]
    for source in sources:
        d=ART/fold/'harmonized'/source
        a=sparse.load_npz(d/'atac_binary.npz').tocsr()
        m=np.load(d/'metadata.npz')
        mats.append(a[m['train_index']])
    x=sparse.vstack(mats,format='csc')
    sums=np.asarray(x.sum(axis=0)).ravel(); sq=np.asarray(x.multiply(x).sum(axis=0)).ravel()
    var=sq-sums*sums/max(1,x.shape[0])
    return np.sort(np.argsort(-var,kind='stable')[:min(cap,x.shape[1])])

def load_fold(fold, cap=2048):
    sources=FOLDS[fold]; cols=source_peak_columns(fold,sources,cap)
    train_r=[]; val_r=[]; train_a=[]; val_a=[]; source_ids=[]
    for source in sources:
        d=ART/fold/'harmonized'/source; r=np.load(d/'rna.npy',mmap_mode='r')
        a=sparse.load_npz(d/'atac_binary.npz').tocsr()[:,cols]; m=np.load(d/'metadata.npz')
        ti,vi=m['train_index'],m['val_index']
        train_r.append(sparse.csr_matrix(r[ti])); val_r.append(sparse.csr_matrix(r[vi]))
        train_a.append(a[ti]); val_a.append(a[vi]); source_ids.extend([source]*len(ti))
    target=np.load(TGT/fold/'target_rna_test.npy',mmap_mode='r')
    train_r=sparse.vstack(train_r,format='csr'); val_r=sparse.vstack(val_r,format='csr')
    train_a=sparse.vstack(train_a,format='csr'); val_a=sparse.vstack(val_a,format='csr')
    tf=TfidfTransformer(norm='l2',use_idf=True,smooth_idf=True,sublinear_tf=True); tf.fit(train_a)
    return dict(fold=fold,sources=sources,peak_columns=cols,source_ids=source_ids,
        rna_train=train_r,rna_val=val_r,atac_train=train_a,atac_val=val_a,
        atac_tfidf_train=tf.transform(train_a).tocsr(),atac_tfidf_val=tf.transform(val_a).tocsr(),
        rna_log_train=train_r,rna_log_val=val_r,rna_test=sparse.csr_matrix(target),
        rna_log_test=sparse.csr_matrix(target),target_rna_shape=target.shape)

def save_protocol(out,data,method,seed,status,extra=None):
    out.mkdir(parents=True,exist_ok=True)
    payload=dict(status=status,method=method,fold=data['fold'],seed=seed,sources=list(data['sources']),
        target_protocol='RNA-only frozen inference',target_atac_used=False,target_labels_used=False,
        target_fitting=False,source_only_peak_cap=int(len(data['peak_columns'])),source_ids_retained=True,configured_seeds=list(SEEDS),run_tier='pre_final_explicit_3seeds',
        source_counts={s:data['source_ids'].count(s) for s in data['sources']},target_rna_shape=list(data['target_rna_shape']))
    if extra: payload.update(extra)
    (out/'adapter_smoke.json').write_text(json.dumps(payload,indent=2,default=str)+'\n')

def evaluate_target(data,out):
    """Posthoc clustering only; labels are read after frozen target embedding."""
    import csv
    target=TARGET_NAME[data['fold']];
    label_path=ROOT.parent/'data/cache/legacy/PBMC'/target/'processed_v1/label.csv'
    labels_by_barcode={r['barcode']:r['cell_type'] for r in csv.DictReader(label_path.open())}
    cache=ROOT.parent/'data/cache/s_mode/PBMC'/target/'pbmc_corrected_s10x20_v1_20261012'
    barcodes=(cache/'barcodes.txt').read_text().splitlines(); test=np.load(cache/'test_index.npy')
    labels=np.asarray([labels_by_barcode[barcodes[i]] for i in test])
    z=np.load(out/'rna_test_embedding.npy'); z=(z-z.mean(0))/(z.std(0)+1e-8); classes,yi=np.unique(labels,return_inverse=True)
    clusters=KMeans(n_clusters=len(classes),n_init=100,random_state=20020130,max_iter=1000).fit_predict(z)
    table=np.zeros((len(classes),len(classes)),dtype=np.int64); np.add.at(table,(yi,clusters),1); rr,cc=linear_sum_assignment(-table); mapping=dict(zip(cc,rr)); matched=np.asarray([mapping[c] for c in clusters])
    acc=float(accuracy_score(yi,matched)); metrics={'ari':float(adjusted_rand_score(yi,clusters)),'nmi':float(normalized_mutual_info_score(yi,clusters)),'ami':float(adjusted_mutual_info_score(yi,clusters)),'acc':acc,'macro_f1':float(f1_score(yi,matched,average='macro',zero_division=0)),'target_labels_loaded_after_embedding':True,'target_atac_used':False}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n'); return metrics

def smoke_multigrate(data,out,seed,device,smoke=True):
    out.mkdir(parents=True,exist_ok=True)
    # The checked-out official Multigrate source targets scvi-tools >=1.4,
    # while its pinned runtime is scvi-tools 0.19. The only missing symbol in
    # the model import is this compatibility helper; keep the source untouched.
    import scvi.model._utils as scvi_utils
    if not hasattr(scvi_utils,'parse_device_args'):
        scvi_utils.parse_device_args=lambda **kwargs:(None,None,kwargs.get('accelerator','cpu'))
    sys.path.insert(0,str(ROOT/'run')); import s_mode_intra
    import torch
    from multigrate.module import MultiVAETorch
    def legacy_make_model(_method,dims,latent,dev):
        return MultiVAETorch(modality_lengths=dims,z_dim=latent,losses=['nb','mse'],condition_encoders=False,condition_decoders=False,normalization='layer',cat_covariate_dims=[],cont_covariate_dims=[]).to(dev)
    s_mode_intra.make_model=legacy_make_model
    torch_run=s_mode_intra.torch_run
    config=json.loads((ROOT/'configs/multigrate/s_mode_pbmc_intra.json').read_text())
    if smoke: config.update(max_epochs=2,batch_size=64)
    import fcntl
    with (ROOT/'.benchmark_a30.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX); result=torch_run('multigrate',data,config,out,seed,smoke,device,time.perf_counter())
    save_protocol(out,data,'Multigrate',seed,'SMOKE_PASS' if smoke else 'TRAINED',result)

def smoke_multivi(data,out,seed,device,smoke=True):
    import torch, anndata as ad
    from scvi.model import MULTIVI
    torch.manual_seed(seed); np.random.seed(seed); r=np.asarray(data['rna_train'].toarray(),dtype=np.float32); a=np.asarray(data['atac_train'].toarray(),dtype=np.float32)
    x=np.concatenate([np.maximum(np.expm1(r),0),a],axis=1); src=ad.AnnData(X=x,obs={'source_id':np.asarray(data['source_ids'],dtype=str)})
    src.var_names=[f'g{i}' for i in range(r.shape[1])]+[f'p{i}' for i in range(a.shape[1])]; MULTIVI.setup_anndata(src)
    model=MULTIVI(src,n_genes=r.shape[1],n_regions=a.shape[1],n_latent=32,n_hidden=64)
    model.train(max_epochs=2 if smoke else 200,check_val_every_n_epoch=None,accelerator='gpu' if device=='cuda' else 'cpu',devices=1)
    tr=np.asarray(data['rna_test'].toarray(),dtype=np.float32); tx=np.concatenate([np.maximum(np.expm1(tr),0),np.zeros((len(tr),a.shape[1]),dtype=np.float32)],axis=1)
    tgt=ad.AnnData(X=tx,obs={'_indices':np.arange(len(tx))}); tgt.var_names=src.var_names; z=model.get_latent_representation(tgt,modality='expression',batch_size=64); assert np.isfinite(z).all()
    model.save(out/'checkpoint',overwrite=True); np.save(out/'rna_test_embedding.npy',z)
    save_protocol(out,data,'MultiVI',seed,'SMOKE_PASS' if smoke else 'TRAINED',dict(latent_shape=list(z.shape),rna_transform='expm1(log-normalized RNA), clipped at zero',native_support='MULTIVI setup_anndata + expression-only latent inference'))

def smoke_palette(data,out,seed,rscript,smoke=True):
    from scipy.io import mmwrite
    out.mkdir(parents=True,exist_ok=True); start=0; train_cells=[]
    for i,s in enumerate(data['sources']):
        n=data['source_ids'].count(s); end=start+n; mmwrite(out/f'rna_{i}.mtx',data['rna_train'][start:end].T); mmwrite(out/f'atac_{i}.mtx',data['atac_tfidf_train'][start:end].T)
        train_cells.extend([f'{s}_{j}' for j in range(n)]); start=end
    (out/'train_cells.txt').write_text('\n'.join(train_cells)+'\n'); np.savetxt(out/'target_rna.csv',data['rna_test'].toarray(),delimiter=',')
    (out/'rna_features.txt').write_text('\n'.join(f'g{i}' for i in range(data['rna_train'].shape[1]))+'\n')
    (out/'atac_features.txt').write_text('\n'.join(f'p{i}' for i in range(data['atac_train'].shape[1]))+'\n')
    cmd=[rscript,str(BASE/'source/fit_palette_lodo.R'),str(out),str(seed),str(len(data['sources'])),str(64)]; subprocess.run(cmd,check=True)
    z=np.loadtxt(out/'rna_test_embedding.csv',delimiter=',',skiprows=1,ndmin=2); assert np.isfinite(z).all(); np.save(out/'rna_test_embedding.npy',z)
    save_protocol(out,data,'Palette-U',seed,'SMOKE_PASS' if smoke else 'TRAINED',dict(latent_shape=list(z.shape),native_support='Palette Create.Palette.Object with three paired source samples'))

def smoke_eclare(data,out,seed,device,smoke=True):
    out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/'run')); from neural_s_mode_intra import training
    import importlib.util
    spec=importlib.util.spec_from_file_location('eclare_lodo_adapter',ROOT/'run/eclare/s_mode_adapter.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    fit,predict=mod.fit,mod.predict
    config=json.loads((ROOT/'configs/eclare/s_mode_pbmc_intra.json').read_text())
    if smoke: config.update(max_epochs=1,batch_size=32,teachers=3)
    # Official ECLARE's CLIP constructor is fixed at 2000 RNA inputs. Pad only
    # the source-derived harmonized RNA space; the added columns are all zero.
    if data['rna_log_train'].shape[1] < 2000:
        pad=2000-data['rna_log_train'].shape[1]
        data=dict(data)
        for key in ('rna_log_train','rna_log_val','rna_log_test'):
            data[key]=sparse.hstack([data[key],sparse.csr_matrix((data[key].shape[0],pad),dtype=data[key].dtype)],format='csr')
    hist=[]; stages=[]; import fcntl
    with (ROOT/'.benchmark_a30.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX); model,result=fit(data,config,out,seed,smoke,device,None,training(data,config,seed,smoke,device,hist,stages))
    z=predict(model,data['rna_log_test'],{},device,32); np.save(out/'rna_test_embedding.npy',z)
    save_protocol(out,data,'ECLARE',seed,'SMOKE_PASS' if smoke else 'TRAINED',dict(latent_shape=list(z.shape),native_support='official ECLARE CLIP teacher/student path',stages=stages))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--method',choices=['multigrate','multivi','palette','eclare'],required=True); ap.add_argument('--fold',choices=sorted(FOLDS),required=True); ap.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0]); ap.add_argument('--out',required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--peak-cap',type=int,default=2048); ap.add_argument('--formal',action='store_true'); args=ap.parse_args()
    data=load_fold(args.fold,args.peak_cap); out=Path(args.out)
    if args.method=='multigrate': smoke_multigrate(data,out,args.seed,args.device,not args.formal)
    elif args.method=='multivi': smoke_multivi(data,out,args.seed,args.device,not args.formal)
    elif args.method=='palette': smoke_palette(data,out,args.seed,'/home/u202522802045/envs/supermap_r/bin/Rscript',not args.formal)
    else: smoke_eclare(data,out,args.seed,args.device,not args.formal)
    if args.formal: print(json.dumps(evaluate_target(data,out),sort_keys=True))
    print('SMOKE_PASS' if not args.formal else 'TRAINED',args.method,args.fold,args.seed)
if __name__=='__main__': main()
