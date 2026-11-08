"""Frozen cohorts and donor-level P2/Linear Probe shared by all methods."""
from pathlib import Path
import hashlib, json, importlib.util
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
ROOT=Path('/home/u202522802045/share/zpb')
RESULTS=ROOT/'result_contain'
OUT=RESULTS/'AD_FOURFOLD_P2_LINEAR_GSE288063_20261028_rev04'
PREP=OUT/'preprocessing'
EXTRA=PREP
CACHE={f'fold{i}':PREP/'folds'/f'fold{i}' for i in range(1,5)}
TYPES=['Excitatory','Inhibitory','Astrocytes','Oligodendrocytes','OPC']
EIGHT=TYPES+['Microglia/PVM','Endothelial','Pericytes/VLMC']
SEEDS=[20020130,20020131,20020132]
EXPECTED_PROTOCOL='698ac4cdcdf268e16a7f0ed20d170442f7791f478d7b1f2a20ec1cf87caca451'
REFERENCE=ROOT/'main/result/d_mode/35_b2_shared_regulatory_teacher_dense_kd_20261003/source/run_b2_strict_lodo.py'
REFERENCE_SHA='3de1ad3642e7af276658d9b239ccaf9457adf7f8e6078b95a237cf4605f2c7e1'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False,ensure_ascii=False)+'\n');temp.replace(path)
def protocol():
    assert sha(OUT/'frozen_protocol.json')==EXPECTED_PROTOCOL
    return read(OUT/'frozen_protocol.json')
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def frozen_b2():
    assert sha(REFERENCE)==REFERENCE_SHA
    return load_module(REFERENCE,'frozen_b2')
def cells(name):
    return pd.read_csv(PREP/'cohorts'/name/'cells.csv.gz',low_memory=False,
                      dtype={'donor_id':str,'sample_group_id':str,'Donor ID':str})
def identities(name,frame=None):
    frame=cells(name) if frame is None else frame
    barcode=next(k for k in ['barcode','cell_id','index'] if k in frame)
    donor={'GSE214979':'donor_id','GSE268609':'sample_group_id','SEAAD_MTG':'Donor ID','GSE288063_GSE278576':'donor_id'}[name]
    out=pd.DataFrame({'barcode':frame[barcode].astype(str),'unit_id':frame[donor].astype(str)})
    assert out.barcode.is_unique and not out.isna().any().any()
    return out
def roster(name):
    x=pd.read_csv(PREP/'cohorts'/name/'evaluation_units.csv',dtype={'unit_id':str})
    assert not x.unit_id.duplicated().any() and x.binary_label.isin([0,1]).all()
    return x
def author_types(name,frame=None):
    frame=cells(name) if frame is None else frame
    if name=='GSE214979':
        mp={t:t for t in TYPES};mp.update(OPCs='OPC',Microglia='Microglia/PVM',Endothelial='Endothelial',Pericytes='Pericytes/VLMC')
        return frame.author_predicted_id.map(mp).fillna('Unknown').to_numpy(str)
    if name=='GSE268609':
        mp=dict(mGC='Excitatory',CA_neurons='Excitatory',Astrocytes='Astrocytes',GABA_neurons='Inhibitory',
                mOli='Oligodendrocytes',OPCs='OPC',Microglia='Microglia/PVM',Endothelial='Endothelial')
        mp['CA2-4_neurons']='Excitatory'
        return frame.Cluster.map(mp).fillna('Unknown').to_numpy(str)
    if name=='SEAAD_MTG':
        out=np.full(len(frame),'Unknown',object)
        out[frame.Class.str.contains('Glutamatergic',na=False)]='Excitatory'
        out[frame.Class.str.contains('GABAergic',na=False)]='Inhibitory'
        for k,v in [('Astrocyte','Astrocytes'),('Oligodendrocyte','Oligodendrocytes'),('OPC','OPC'),('Endothelial','Endothelial')]:out[frame.Subclass.eq(k)]=v
        out[frame.Supertype.str.startswith('Micro-PVM',na=False)]='Microglia/PVM'
        out[frame.Supertype.str.startswith(('Pericyte','VLMC'),na=False)]='Pericytes/VLMC'
        return out.astype(str)
    if name=='GSE288063_GSE278576':
        from ad_new_cohort import broad_types
        return broad_types(frame)
    raise ValueError(name)
def annotated(name):
    ids=identities(name);ids['broad_type']=author_types(name);return ids

def pooled(z,annotation,units):
    z=np.asarray(z);assert z.shape==(len(annotation),64) and np.isfinite(z).all()
    x=np.full((len(units),320),np.nan);counts=np.zeros((len(units),5),np.int64)
    for i,u in enumerate(units.unit_id.astype(str)):
        for j,t in enumerate(TYPES):
            idx=np.flatnonzero(annotation.unit_id.eq(u)&annotation.broad_type.eq(t));counts[i,j]=len(idx)
            if len(idx):x[i,j*64:(j+1)*64]=z[idx].mean(axis=0,dtype=np.float64)
    assert set(annotation.unit_id)==set(units.unit_id.astype(str))
    return x,counts
def fit_probe(x,y):
    """TARGET data and labels are deliberately absent from this interface."""
    assert x.ndim==2 and x.shape[1]==320 and len(x)==len(y)
    assert np.unique(y).tolist()==[0,1] and not np.isnan(x).all(axis=0).any()
    model=Pipeline([('imputer',SimpleImputer(strategy='mean')),('scaler',StandardScaler()),
                    ('probe',LogisticRegression(**protocol()['probe']))])
    return model.fit(x,y)
def metrics(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float);pred=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(AUROC=float(roc_auc_score(y,p)),AUPRC=float(average_precision_score(y,p)),
        Balanced_Accuracy=float(balanced_accuracy_score(y,pred)),
        Macro_F1=float(f1_score(y,pred,average='macro',labels=[0,1],zero_division=0)),
        Sensitivity=float(tp/(tp+fn)),Specificity=float(tn/(tn+fp)),Accuracy=float(accuracy_score(y,pred)),
        TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp))
def evaluate_p2(fold,seed,method,embedding_by_cohort,model_inputs):
    prot=protocol();cfg=prot['folds'][fold];dest=OUT/'evaluation'/fold/f'seed_{seed}'/method
    dest.mkdir(parents=True,exist_ok=True);names=prot['probe_sources'][fold];target=cfg['target']
    assert set(embedding_by_cohort)==set(names+[target]) and len(names)==3 and 'retired_legacy_cohort' not in names
    vectors=[];units=[];all_counts=[]
    for name in names+[target]:
        u=roster(name);a=annotated(name);x,c=pooled(embedding_by_cohort[name],a,u)
        vectors.append(x);units.append(u);all_counts.append(c)
    su=pd.concat(units[:-1],ignore_index=True);sx=np.concatenate(vectors[:-1]);tx=vectors[-1]
    assert len(su)==prot['probe_unit_counts'][fold] and not su.duplicated(['cohort','unit_id']).any()
    model=fit_probe(sx,su.binary_label.to_numpy(int));joblib.dump(model,dest/'probe.joblib')
    np.savez_compressed(dest/'donor_p2_features.npz',source=sx,target=tx,
        source_imputed=model['imputer'].transform(sx),target_imputed=model['imputer'].transform(tx))
    su.to_csv(dest/'probe_training_units.csv',index=False);coverage=[]
    for u,c in zip(units,all_counts):
        for row,counts in zip(u.to_dict('records'),c):
            coverage.append({**{k:row[k] for k in ['cohort','unit_id','unit_type']},**dict(zip(TYPES,counts.tolist())),
                             'missing_blocks':int((counts==0).sum())})
    pd.DataFrame(coverage).to_csv(dest/'p2_coverage.csv',index=False)
    p=model.predict_proba(tx)[:,1]
    pred=units[-1][['cohort','unit_id','unit_type']].copy();pred['probability_AD']=p;pred['prediction']=(p>=.5).astype(int)
    pred.to_csv(dest/'predictions_blind.csv',index=False);blind_sha=sha(dest/'predictions_blind.csv')
    # The prediction file is persisted before TARGET labels are consulted for metrics.
    pred['y_true']=units[-1].binary_label.to_numpy(int);pred['fold']=fold;pred['seed']=seed;pred['method']=method
    pred.to_csv(dest/'predictions.csv',index=False)
    m=dict(fold=fold,seed=seed,method=method,target=target,probe_sources=names,n_probe_units=len(su),
           n_target_units=len(pred),**metrics(pred.y_true,p));write(dest/'metrics.json',m)
    write(dest/'COMPLETE.json',dict(status='COMPLETE',inputs=model_inputs,protocol_sha256=EXPECTED_PROTOCOL,
        blind_predictions_sha256=blind_sha,probabilities_before_scoring=True,target_fit=False,
        hashes={str(dest/f):sha(dest/f) for f in ['probe.joblib','donor_p2_features.npz','probe_training_units.csv',
                                               'p2_coverage.csv','predictions.csv','metrics.json']}))
    return m
