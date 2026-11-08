"""Identical frozen P2 evaluation for Full B2, RNA AE and SOURCE-fit PCA64."""
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import ad_runtime as a

def transform_chunks(x,func):
    chunks=[]
    for i in range(0,len(x),1024):
        z=func(np.asarray(x[i:i+1024],np.float32));assert z.shape==(min(1024,len(x)-i),64)
        chunks.append(z)
    out=np.concatenate(chunks);assert np.isfinite(out).all();return out

def projection(fold,seed,method):
    d=a.OUT/'evaluation'/fold/f'seed_{seed}'/method
    arr=np.load(d/'donor_p2_features.npz');model=joblib.load(d/'probe.joblib')
    sx=model['scaler'].transform(arr['source_imputed']);tx=model['scaler'].transform(arr['target_imputed'])
    pca=PCA(n_components=2,svd_solver='full').fit(sx)
    sz,tz=pca.transform(sx),pca.transform(tx)
    su=pd.read_csv(d/'probe_training_units.csv',dtype={'unit_id':str})
    tu=pd.read_csv(d/'predictions.csv',dtype={'unit_id':str})
    rows=[]
    for frame,z,scope,label in [(su,sz,'SOURCE','binary_label'),(tu,tz,'TARGET','y_true')]:
        for row,xy in zip(frame.to_dict('records'),z):
            rows.append(dict(cohort=row['cohort'],unit_id=row['unit_id'],scope=scope,
                             diagnosis=row[label],PC1=xy[0],PC2=xy[1]))
    coords=pd.DataFrame(rows);coords.to_csv(d/'PCA2_coordinates.csv',index=False)
    joblib.dump(pca,d/'PCA2_source_fit.joblib')
    fig,ax=plt.subplots(figsize=(6.4,4.8))
    for scope,marker in [('SOURCE','o'),('TARGET','^')]:
        for label,color in [(0,'#2878b5'),(1,'#d64a3b')]:
            sub=coords[(coords.scope==scope)&(coords.diagnosis==label)]
            ax.scatter(sub.PC1,sub.PC2,c=color,marker=marker,s=35 if scope=='SOURCE' else 80,
                       alpha=.65 if scope=='SOURCE' else .95,label=f'{scope} '+('AD' if label else 'Control'),
                       edgecolors='white',linewidths=.4)
    variance=pca.explained_variance_ratio_*100
    ax.set(xlabel=f'PC1 ({variance[0]:.1f}%)',ylabel=f'PC2 ({variance[1]:.1f}%)',
           title=f'{method} | {fold} | seed {seed}\nPCA fitted on SOURCE donor features')
    ax.legend(fontsize=8,frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(d/'PCA2_donors.png',dpi=200);plt.close(fig)

def run(fold,seed,method):
    prot=a.protocol();cfg=prot['folds'][fold];fd=a.CACHE[fold]
    dest=a.OUT/'evaluation'/fold/f'seed_{seed}'/method
    if (dest/'COMPLETE.json').exists():
        for path,digest in a.read(dest/'COMPLETE.json')['hashes'].items():assert a.sha(path)==digest,path
        if not (dest/'PCA2_donors.png').exists():projection(fold,seed,method)
        return a.read(dest/'metrics.json')
    cm=a.read(fd/'COMPLETE.json');names=prot['probe_sources'][fold]+[cfg['target']]
    arrays={};inputs={str(a.OUT/'frozen_protocol.json'):a.EXPECTED_PROTOCOL}
    for name in set(cfg['sources']+names):
        p=fd/name/'rna.npy';assert a.sha(p)==cm['hashes'][str(p)]
        inputs[str(p)]=a.sha(p);arrays[name]=np.load(p,mmap_mode='r')
    if method=='PCA_RNA64':
        rng=np.random.default_rng(seed);parts=[]
        splits={n:np.load(fd/n/'metadata.npz')['train_index'] for n in cfg['sources']}
        count=min(20000,min(len(v) for v in splits.values()))
        for n in cfg['sources']:
            idx=np.sort(rng.choice(splits[n],count,replace=False));parts.append(np.asarray(arrays[n][idx]))
            dest.mkdir(parents=True,exist_ok=True);np.save(dest/(n+'_PCA_train_rows.npy'),idx)
        train=np.concatenate(parts);scaler=StandardScaler().fit(train)
        pca=PCA(n_components=64,svd_solver='randomized',random_state=seed).fit(scaler.transform(train))
        joblib.dump(dict(scaler=scaler,pca=pca),dest/'pca64.joblib')
        inputs[str(dest/'pca64.joblib')]=a.sha(dest/'pca64.joblib')
        encode=lambda x:pca.transform(scaler.transform(x)).astype(np.float32)
        embeddings={n:transform_chunks(arrays[n],encode) for n in names}
        del train,parts
    else:
        torch.set_num_threads(2);assert torch.cuda.is_available()
        b=a.frozen_b2();unit=a.OUT/'models'/fold/f'seed_{seed}'
        done=a.read(unit/'TRAINING_COMPLETE.json')
        for path,digest in done['checkpoints'].items():assert a.sha(path)==digest,path
        norm=np.load(unit/'rna_normalization.npz');mu,sigma=norm['mean'],norm['std']
        enc=b.Encoder(len(mu));which='student' if method=='Full_B2' else 'rna_ae'
        ck=torch.load(unit/which/'best.pt',map_location='cpu',weights_only=False)
        state=ck['model']
        if which=='rna_ae':state={k[len('encoder.'):]:v for k,v in state.items() if k.startswith('encoder.')}
        enc.load_state_dict(state);enc=enc.to('cuda').eval().requires_grad_(False)
        inputs[str(unit/which/'best.pt')]=a.sha(unit/which/'best.pt')
        inputs[str(unit/'rna_normalization.npz')]=a.sha(unit/'rna_normalization.npz')
        def encode(x):
            with torch.inference_mode():return enc(torch.from_numpy((x-mu)/(sigma+1e-6)).to('cuda')).cpu().numpy()
        embeddings={n:transform_chunks(arrays[n],encode) for n in names}
    m=a.evaluate_p2(fold,seed,method,embeddings,inputs);projection(fold,seed,method)
    print('EVALUATION_COMPLETE',fold,seed,method,flush=True)
    return m

def summarize():
    rows=[a.read(p) for p in (a.OUT/'evaluation').glob('fold*/seed_*/*/metrics.json')]
    if not rows:return
    f=pd.DataFrame(rows);f['probe_sources']=f.probe_sources.map(lambda v:'+'.join(v))
    f.to_csv(a.OUT/'metrics_per_fold_seed.csv',index=False)
    cols=a.protocol()['metrics'];summary=f.groupby(['fold','target','method'])[cols].agg(['mean','std','count'])
    summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(a.OUT/'metrics_summary.csv',index=False)
    macro=[]
    for (method,seed),g in f.groupby(['method','seed']):
        if g.fold.nunique()==4:macro.append(dict(method=method,seed=seed,**g[cols].mean().to_dict()))
    if macro:
        macro=pd.DataFrame(macro);macro.to_csv(a.OUT/'macro_per_seed.csv',index=False)
        s=macro.groupby('method')[cols].agg(['mean','std','count']);s.columns=['_'.join(c) for c in s.columns]
        s.reset_index().to_csv(a.OUT/'macro_summary.csv',index=False)
    preds=[pd.read_csv(p,dtype={'unit_id':str}) for p in (a.OUT/'evaluation').glob('fold*/seed_*/*/predictions.csv')]
    pd.concat(preds,ignore_index=True).to_csv(a.OUT/'predictions_all.csv',index=False)
    if len(f)==36 and all(len(g)==3 for _,g in f.groupby(['fold','method'])):
        a.write(a.OUT/'EVALUATION_COMPLETE.json',dict(status='COMPLETE',folds=4,seeds=a.SEEDS,methods=3,
                runs=36,std_ddof=1,target_tuned=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',choices=list(a.CACHE));p.add_argument('--seed',type=int)
    p.add_argument('--method',choices=['Full_B2','RNA_only_Encoder','PCA_RNA64']);p.add_argument('--summarize',action='store_true')
    v=p.parse_args()
    if v.summarize:summarize()
    else:run(v.fold,v.seed,v.method);summarize()
