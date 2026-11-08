"""All-source refit; frozen architecture/loss/batching; no Target selection."""
import argparse
import copy
import importlib.util
import time
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from common import *

ART=RUN/'artifacts'
spec=importlib.util.spec_from_file_location('frozen_b2',FROZEN/'run_b2_strict_lodo.py')
b2=importlib.util.module_from_spec(spec);spec.loader.exec_module(b2)

def sources():
    prep=json.loads((ART/'PREPARATION_COMPLETE.json').read_text())
    assert prep['sources']==['D2','D3','D4'] and prep['target_training'] is False
    for filename,digest in prep['hashes'].items():
        if filename.startswith('artifacts/D1_'): continue
        assert sha(RUN/filename)==digest, f'Frozen Source artifact changed: {filename}'
    stats=np.load(ART/'source_scaler.npz');mu,std=stats['mean'],stats['std']
    data=[];inputs={}
    for key in ['D2','D3','D4']:
        p=ART/'harmonized'/key
        raw=np.load(p/'rna.npy')
        field=np.load(ART/f'{key}_F.npy')
        assert raw.shape==field.shape and np.isfinite(raw).all() and np.isfinite(field).all()
        values=dict(rna=((raw-mu)/(std+1e-6)).astype(np.float32),field=field,raw_nonzero=raw!=0)
        data.append(dict(name=key,key=key,train=values,val=values))
        for f in [p/'rna.npy',ART/f'{key}_F.npy',ART/'native'/key/'barcodes.tsv']:
            inputs[str(f)]=sha(f)
    for f in [ART/'source_scaler.npz',ART/'gene_ids.npy',ART/'B_D.npz',SRC/'run_model.py',FROZEN/'run_b2_strict_lodo.py']:
        inputs[str(f)]=sha(f)
    return data,mu,std,inputs

def evaluate(unit,embedding,method,seed):
    # Embedding must be saved before reading any Target annotation.
    path=unit/'target_embedding_raw.npy';np.save(path,embedding)
    frozen=sha(path)
    labels=pd.read_csv(RUN/'audit/D1_allcell_labels.csv',index_col=0)
    cells=(ART/'D1_barcodes.tsv').read_text().splitlines()
    assert labels.index.tolist()==cells and len(cells)==9347
    labels=labels['cell_type'].astype(str).to_numpy()
    levels,y=np.unique(labels,return_inverse=True)
    metrics=b2.evaluate(embedding,y)
    z=StandardScaler().fit_transform(embedding).astype(np.float32)
    np.save(unit/'target_embedding_standardized.npy',z)
    clusters=KMeans(n_clusters=len(levels),**b2.EVALUATOR).fit_predict(z)
    neighbors=NearestNeighbors(n_neighbors=15).fit(z).kneighbors(return_distance=False)
    metrics.update(purity_at_15=float((y[neighbors]==y[:,None]).mean()),method=method,seed=seed,
                   cells=9347,subclasses=len(levels),cd8_tem_cells=int((labels=='CD8 TEM').sum()),
                   target_embedding_sha256=frozen,clustering='known-K posthoc; not blind subtype discovery')
    assert sha(path)==frozen
    pd.DataFrame(dict(barcode=cells,reference_label=labels,cluster=clusters)).to_csv(unit/'cluster_assignments.csv',index=False)
    pd.crosstab(pd.Series(labels,name='reference'),pd.Series(clusters,name='cluster')).to_csv(unit/'cluster_composition.csv')
    write(unit/'metrics.json',metrics)
    return metrics

def pca():
    data,mu,std,inputs=sources()
    fit=np.concatenate([s['train']['rna'] for s in data])
    x=np.load(ART/'D1_rna_all.npy')
    scaled=((x-mu)/(std+1e-6)).astype(np.float32)
    n=min(50,fit.shape[1],fit.shape[0]-1)
    for method,fit_x,transform_x in [('PCA_sourcefit',fit,scaled),('PCA_targetfit_fullnorm',x,x)]:
        unit=RUN/method;unit.mkdir(exist_ok=True)
        model=PCA(n_components=n,svd_solver='full')
        embedding=model.fit_transform(fit_x) if method.startswith('PCA_target') else model.fit(fit_x).transform(transform_x)
        embedding=embedding.astype(np.float32)
        np.savez_compressed(unit/'pca_state.npz',components=model.components_,mean=model.mean_,
              explained_variance=model.explained_variance_,explained_variance_ratio=model.explained_variance_ratio_,
              source_mean=mu,source_std=std,genes=np.load(ART/'gene_ids.npy'))
        rows=[]
        for seed in SEEDS:
            out=unit/f'seed_{seed}';out.mkdir(exist_ok=True)
            rows.append(evaluate(out,embedding,method,seed))
        pd.DataFrame(rows).to_csv(unit/'metrics.csv',index=False)
        write(unit/'manifest.json',dict(status='COMPLETE',method=method,fit_cells=len(fit_x),target_cells=9347,
              source_hvg_panel_shared_with_b2=True,full_library_normalization=True,
              target_used_for_pca_fit=method.startswith('PCA_target'),targetfit_role='descriptive sensitivity only',
              components=n,pca_deterministic=True,seed_note='PCA SVD and known-K KMeans fixed; seeds label downstream visualization runs, not independent PCA fits',
              input_hashes=inputs))
    print('PCA_COMPLETE',flush=True)

def full_b2(seed):
    import torch
    assert seed in SEEDS and torch.cuda.is_available()
    torch.set_num_threads(2)
    unit=RUN/'FullB2'/f'seed_{seed}';unit.mkdir(parents=True,exist_ok=True)
    data,mu,std,inputs=sources()
    b2.CONFIG.update(patience=101,epochs=100,validation='training-only monitor; no independent validation',
                     checkpoint_selection='fixed epoch 100 final; no Target selection',
                     source_gate_scope='training diagnostic only',run_tier='all_source_all_target_refit')
    write(unit/'protocol.json',dict(sources=['D2','D3','D4'],source_cells=[len(s['train']['rna']) for s in data],
          source_all_cells=True,target='D1',target_training=False,target_cells=9347,
          source_monitor_aliases_train=True,independent_validation=False,
          fixed_epochs=100,early_stopping=False,selected_checkpoint='latest.pt at epoch 100',
          frozen_model_sha256=sha(FROZEN/'run_b2_strict_lodo.py'),config=b2.CONFIG))
    device=torch.device('cuda');b2.seed_all(seed)
    teacher=b2.Teacher(len(mu),len(mu)).to(device)
    td=unit/'checkpoints/teacher'
    teacher=b2.train_stage(teacher,data,td,seed,device,True,inputs)
    checkpoint=torch.load(td/'latest.pt',map_location=device,weights_only=False)
    assert checkpoint['epoch']==100
    teacher.load_state_dict(checkpoint['model']);teacher.eval().requires_grad_(False)
    for s in data:
        outputs=b2.forward_numpy(teacher,s['train']['rna'],device,s['train']['field'])
        z=outputs[2]
        assert np.isfinite(z).all() and np.any(np.var(z,axis=0)>0)
        s['train']['teacher_latent']=z
        np.save(unit/f"teacher_{s['key']}_embedding.npy",z)
    latent_mu=np.mean([s['train']['teacher_latent'].mean(0) for s in data],axis=0)
    latent_std=np.sqrt(np.mean([s['train']['teacher_latent'].var(0)+
             (s['train']['teacher_latent'].mean(0)-latent_mu)**2 for s in data],axis=0)).astype(np.float32)
    for s in data:
        s.update(latent_mean=latent_mu,latent_std=latent_std,raw_zero_standardized=-mu/(std+1e-6))
    np.savez_compressed(unit/'kd_stats.npz',mean=latent_mu,std=latent_std)
    student_inputs=dict(inputs);student_inputs[str(td/'latest.pt')]=sha(td/'latest.pt')
    b2.seed_all(seed)
    student=copy.deepcopy(teacher.rna_encoder).requires_grad_(True).to(device)
    sd=unit/'checkpoints/student'
    student=b2.train_stage(student,data,sd,seed,device,False,student_inputs)
    checkpoint=torch.load(sd/'latest.pt',map_location=device,weights_only=False)
    assert checkpoint['epoch']==100
    student.load_state_dict(checkpoint['model']);student.eval().requires_grad_(False)
    write(unit/'selected_checkpoints.json',dict(teacher=str(td/'latest.pt'),student=str(sd/'latest.pt'),
          teacher_sha256=sha(td/'latest.pt'),student_sha256=sha(sd/'latest.pt'),epoch=100,
          original_best_preserved_but_not_selected=True))
    # First Target expression read in the training process occurs after both final checkpoints freeze.
    target=np.load(ART/'D1_rna_all.npy')
    z=b2.forward_numpy(student,((target-mu)/(std+1e-6)).astype(np.float32),device)
    metrics=evaluate(unit,z,'FullB2',seed)
    for s in data:
        np.save(unit/f"student_{s['key']}_embedding.npy",b2.forward_numpy(student,s['train']['rna'],device))
    write(unit/'manifest.json',dict(status='COMPLETE',seed=seed,metrics=metrics,
          target_rna_only=True,target_used_for_training=False,target_used_for_checkpoint_selection=False,
          checkpoint_reused=False,source_all_cells=True,final_epoch=100,
          hashes={str(p.relative_to(unit)):sha(p) for p in unit.rglob('*') if p.is_file() and p.name!='manifest.json'}))
    print('FULL_B2_COMPLETE',seed,metrics['ari'],metrics['nmi'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['pca','full_b2'],required=True);p.add_argument('--seed',type=int)
    args=p.parse_args()
    if args.method=='pca': pca()
    else: full_b2(args.seed)
