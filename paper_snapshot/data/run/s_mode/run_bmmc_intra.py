"""Run BMMC Site-level Full B2 and RNA-only AE intra baselines, 3 seeds."""
import argparse, importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
    adjusted_mutual_info_score, accuracy_score, f1_score, balanced_accuracy_score, silhouette_score)
from sklearn.neighbors import NearestNeighbors

ROOT=Path('/home/u202522802045/share/zpb')
sys.path[:0]=[str(ROOT),str(ROOT/'data/run/s_mode'),str(ROOT/'main/run/d_mode'),str(ROOT/'main/core/regulatory/common/legacy_support')]
from prepare_bmmc_site_intra import CACHE,SITES
spec=importlib.util.spec_from_file_location('frozen_smode_trainer',ROOT/'main/run/s_mode/train.py')
trainer=importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
b=trainer.b
rspec=importlib.util.spec_from_file_location('rna_only_ae',ROOT/'comparison/methods/rna_baseline/rna_only_ae.py')
rna=importlib.util.module_from_spec(rspec); rspec.loader.exec_module(rna)
OUT=ROOT/'result_contain/97_bmmc_b2full_rnaonly_3seed_intra_20261024'
SEEDS=[20020130,20020131,20020132]


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str)+'\n')


def score(z,labels):
    names,y=np.unique(labels,return_inverse=True)
    scaled=StandardScaler().fit_transform(z)
    pred=KMeans(n_clusters=len(names),init='k-means++',n_init=100,random_state=20020130,
        max_iter=1000,tol=1e-4,algorithm='lloyd').fit_predict(scaled)
    table=np.zeros((len(names),len(names)),np.int64); np.add.at(table,(y,pred),1)
    a,c=b.linear_sum_assignment(-table); mapping=dict(zip(c,a))
    matched=np.array([mapping[int(k)] for k in pred])
    nn=NearestNeighbors(n_neighbors=min(16,len(y))).fit(scaled).kneighbors(scaled,return_distance=False)
    purity=np.mean([np.mean(y[row[row!=i][:15]]==y[i]) for i,row in enumerate(nn)])
    return dict(ARI=float(adjusted_rand_score(y,pred)),NMI=float(normalized_mutual_info_score(y,pred)),
        AMI=float(adjusted_mutual_info_score(y,pred)),ACC=float(accuracy_score(y,matched)),
        Macro_F1=float(f1_score(y,matched,average='macro',zero_division=0)),
        Balanced_ACC=float(balanced_accuracy_score(y,matched)),
        Weighted_F1=float(f1_score(y,matched,average='weighted',zero_division=0)),
        Purity_at_15=float(purity),Silhouette=float(silhouette_score(scaled,y)),
        cells=int(len(y)),classes=int(len(names)),known_k=True),y,pred,names[matched]


def load_x(cache):
    state=np.load(cache/'rna_state.npz')
    x=((np.load(cache/'rna_test.npy')-state['mean'])/(state['std']+1e-6)).astype(np.float32)
    labels=np.load(cache/'labels.npy',allow_pickle=True).astype(str)[np.load(cache/'test_index.npy')]
    return x,labels


def b2_eval(site,seed):
    unit=OUT/'models'/site/f'seed_{seed}'; cache=CACHE/site
    evaluation=unit/'evaluation'/f'{site}_to_{site}'
    if (evaluation/'manifest.json').exists(): return
    x,labels=load_x(cache)
    student=b.Encoder(2000).cuda()
    ckpt=unit/'checkpoints/student/best.pt'
    student.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=False)['model'])
    student.eval().requires_grad_(False)
    with torch.inference_mode():
        z=b.forward_numpy(student,x,torch.device('cuda:0'))
    assert z.shape==(len(x),64) and np.isfinite(z).all()
    evaluation.mkdir(parents=True,exist_ok=True); np.save(evaluation/'embedding.npy',z)
    pd.DataFrame(dict(barcode=np.load(cache/'barcodes.npy',allow_pickle=True).astype(str)[np.load(cache/'test_index.npy')],
        reference_label=labels)).to_csv(evaluation/'cell_order.csv',index=False)
    m,y,pred,matched=score(z,labels)
    pd.DataFrame(dict(reference_label=labels,cluster=pred,matched_reference=matched)).to_csv(evaluation/'clusters.csv',index=False)
    write(evaluation/'metrics.json',m)
    write(evaluation/'manifest.json',dict(status='COMPLETE',method='Full B2-S',site=site,seed=seed,metrics=m,
        checkpoint_sha256=trainer.legacy.sha(ckpt),embedding_sha256=trainer.legacy.sha(evaluation/'embedding.npy'),
        cells=len(x),kmeans=dict(n_init=100,random_state=20020130,max_iter=1000,tol=1e-4,algorithm='lloyd')))
    write(unit/'manifest.json',dict(status='COMPLETE',method='Full B2-S',site=site,seed=seed,metrics=m,
        checkpoint_sha256=trainer.legacy.sha(ckpt),evaluation=str(evaluation)))
    print('B2_COMPLETE',site,seed,m['ARI'],m['NMI'],flush=True)


def rna_eval(site,seed):
    unit=OUT/'rna_only/models'/site/f'seed_{seed}'; cache=CACHE/site
    evaluation=unit/'evaluation'/f'{site}_to_{site}'
    if (evaluation/'manifest.json').exists(): return
    x,labels=load_x(cache)
    encoder,_=rna.load_encoder(unit/'best.pt')
    with torch.inference_mode(): z=rna.encode(encoder,x)
    assert z.shape==(len(x),64) and np.isfinite(z).all()
    evaluation.mkdir(parents=True,exist_ok=True); np.save(evaluation/'embedding.npy',z)
    pd.DataFrame(dict(barcode=np.load(cache/'barcodes.npy',allow_pickle=True).astype(str)[np.load(cache/'test_index.npy')],
        reference_label=labels)).to_csv(evaluation/'cell_order.csv',index=False)
    m,y,pred,matched=score(z,labels)
    pd.DataFrame(dict(reference_label=labels,cluster=pred,matched_reference=matched)).to_csv(evaluation/'clusters.csv',index=False)
    write(evaluation/'metrics.json',m)
    write(evaluation/'manifest.json',dict(status='COMPLETE',method='RNA-only AE',site=site,seed=seed,metrics=m,
        checkpoint_sha256=rna.sha(unit/'best.pt'),embedding_sha256=rna.sha(evaluation/'embedding.npy'),
        cells=len(x),kmeans=dict(n_init=100,random_state=20020130,max_iter=1000,tol=1e-4,algorithm='lloyd')))
    write(unit/'manifest.json',dict(status='COMPLETE',method='RNA-only AE',site=site,seed=seed,metrics=m,
        checkpoint_sha256=rna.sha(unit/'best.pt'),evaluation=str(evaluation)))
    print('RNA_COMPLETE',site,seed,m['ARI'],m['NMI'],flush=True)


def run_b2(site,seed):
    unit=OUT/'models'/site/f'seed_{seed}'
    if (unit/'manifest.json').exists() and json.loads((unit/'manifest.json').read_text()).get('status')=='COMPLETE':
        b2_eval(site,seed); return
    cache=CACHE/site; manifest=json.loads((cache/'manifest.json').read_text())
    unit.mkdir(parents=True,exist_ok=True)
    trainer.train(site,seed,OUT,cache,manifest,evaluation_targets=())
    b2_eval(site,seed)


def run_rna(site,seed):
    unit=OUT/'rna_only/models'/site/f'seed_{seed}'
    if (unit/'best.pt').exists():
        rna_eval(site,seed); return
    cache=CACHE/site
    state=dict(np.load(cache/'rna_state.npz')); genes=np.load(cache/'gene_ids.npy',allow_pickle=True).astype(str)
    unit.mkdir(parents=True,exist_ok=True)
    rna.fit(np.load(cache/'rna_train.npy'),np.load(cache/'rna_val.npy'),state,genes,unit,seed,
        dict(run_tier='pre_final',dataset=site,seed=seed),device='cuda')
    rna_eval(site,seed)


def summarize():
    (OUT/'metrics').mkdir(parents=True,exist_ok=True)
    rows=[]
    for method,root in [('Full B2-S',OUT/'models'),('RNA-only AE',OUT/'rna_only/models')]:
        for site in SITES:
            for seed in SEEDS:
                p=root/site/f'seed_{seed}'/'evaluation'/f'{site}_to_{site}'/'metrics.json'
                if p.exists(): rows.append(dict(Method=method,Site=site,Seed=seed,**json.loads(p.read_text())))
    df=pd.DataFrame(rows); df.to_csv(OUT/'metrics/per_unit.csv',index=False)
    if len(df):
        metrics=['ARI','NMI','AMI','ACC','Macro_F1','Balanced_ACC','Weighted_F1','Purity_at_15','Silhouette']
        summary=df.groupby(['Method','Site'])[metrics].agg(['mean','std']).reset_index()
        summary.columns=['Method','Site']+[f'{m}_{s}' for m in metrics for s in ('mean','std')]
        summary.to_csv(OUT/'metrics/summary.csv',index=False)
        eq=df.groupby(['Method','Seed'])[metrics].mean().reset_index()
        eqsum=eq.groupby('Method')[metrics].agg(['mean','std']).reset_index()
        eqsum.columns=['Method']+[f'{m}_{s}' for m in metrics for s in ('mean','std')]
        eqsum.to_csv(OUT/'metrics/equal_weight_summary.csv',index=False)
        cols=['Method','Site']+[f'{m}_mean' for m in metrics]
        cols=['Method','Site']+metrics
        table=df.groupby(['Method','Site'])[metrics].mean().reset_index()
        lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
        lines += ['| '+' | '.join(str(v) for v in row)+' |' for row in table[cols].itertuples(index=False,name=None)]
        (OUT/'metrics/summary.md').write_text('# BMMC 3-seed intra\n\n'+'\n'.join(lines)+'\n')
    write(OUT/'metrics/summary.json',dict(status='COMPLETE' if len(rows)==24 else 'PARTIAL',rows=len(rows),expected=24,
        seeds=SEEDS,methods=['Full B2-S','RNA-only AE'],sites=SITES))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['b2','rna','all','summarize'],required=True)
    parser.add_argument('--site',choices=SITES); parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.stage=='summarize': summarize(); raise SystemExit
    if args.site and args.seed:
        (run_b2 if args.stage=='b2' else run_rna)(args.site,args.seed)
    else:
        for site in SITES:
            for seed in SEEDS:
                if args.stage in ('b2','all'): run_b2(site,seed)
                if args.stage in ('rna','all'): run_rna(site,seed)
        summarize()
