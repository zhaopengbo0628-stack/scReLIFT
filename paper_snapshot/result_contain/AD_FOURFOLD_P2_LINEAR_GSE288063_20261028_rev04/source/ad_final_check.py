"""Deterministic final evidence checks; reads results, never retrains or tunes."""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
import ad_runtime as a

def main(folds=None):
    prot=a.protocol();runs=[];notes=[]
    chosen=list(prot['folds']) if folds is None else folds
    assert chosen and len(chosen)==len(set(chosen)) and set(chosen)<=set(prot['folds'])
    suffix='' if len(chosen)==4 else '_'+('_'.join(chosen))
    freeze=a.read(a.OUT/'IMPLEMENTATION_FREEZE.json');dependency_hashes={}
    for name in ['ad_runtime.py','ad_train.py','ad_rna.py','ad_new_cohort.py','ad_evaluate.py','ad_biology.py']:
        path=a.OUT/'source'/name;expected=freeze['source_hashes'][str(path)]
        assert a.sha(path)==expected,('scientific source changed without recorded repair',name)
        dependency_hashes[str(path)]=expected
    for name in ['GSE214979','GSE268609','SEAAD_MTG','GSE288063_GSE278576']:
        for file in ['cells.csv.gz','evaluation_units.csv','metadata.npz']:
            p=a.PREP/'cohorts'/name/file;dependency_hashes[str(p)]=a.sha(p)
        p=a.PREP/'hvg'/f'{name}.csv';dependency_hashes[str(p)]=a.sha(p)
    for p in [a.REFERENCE,a.OUT/'frozen_protocol.json',a.OUT/'software_versions.json',a.OUT/'IMPLEMENTATION_FREEZE.json']:
        dependency_hashes[str(p)]=a.sha(p)
    for fd in a.CACHE.values():
        p=fd/'COMPLETE.json';dependency_hashes[str(p)]=a.sha(p)
        cm=a.read(p)
        # Verify consumed identity metadata against the relevant immutable CPU manifest.
        for path,digest in cm['hashes'].items():
            pp=Path(path)
            if pp.name in ['cells.csv.gz','evaluation_units.csv','metadata.npz','gene_ids.npy']:
                assert a.sha(pp)==digest,pp
                dependency_hashes[str(pp)]=digest
    for fold,cfg in prot['folds'].items():
        if fold not in chosen:continue
        for seed in a.SEEDS:
            unit=a.OUT/'models'/fold/f'seed_{seed}'
            done=a.read(unit/'TRAINING_COMPLETE.json')
            assert done['status']=='COMPLETE' and done['target_used'] is False
            for path,digest in done['checkpoints'].items():assert a.sha(path)==digest,path
            for stage in ['teacher','student','rna_ae']:
                ck=torch.load(unit/stage/'best.pt',map_location='cpu',weights_only=False)
                h=a.read(unit/stage/'history.json')
                vals=np.asarray([r['val'][-1] if isinstance(r['val'],list) else r['val'] for r in h])
                assert np.isfinite(vals).all()
                assert np.isclose(ck['best_score'],vals.min(),rtol=1e-10,atol=1e-10)
                assert int(ck['epoch'])==int(np.argmin(vals)+1)
                assert all(cfg['target'] not in r.get('val_by_source',{}) for r in h
                           if isinstance(r.get('val_by_source'),dict))
            annotations=[];training_units=[];target_units=[]
            for method in ['Full_B2','RNA_only_Encoder','PCA_RNA64']:
                d=a.OUT/'evaluation'/fold/f'seed_{seed}'/method;cm=a.read(d/'COMPLETE.json')
                assert cm['protocol_sha256']==a.EXPECTED_PROTOCOL and cm['target_fit'] is False
                for path,digest in cm['hashes'].items():assert a.sha(path)==digest,path
                dependency_hashes.update(cm['hashes'])
                for path,digest in cm['inputs'].items():assert a.sha(path)==digest,path
                dependency_hashes.update(cm['inputs'])
                assert a.sha(d/'predictions_blind.csv')==cm['blind_predictions_sha256']
                su=pd.read_csv(d/'probe_training_units.csv',dtype={'unit_id':str})
                pred=pd.read_csv(d/'predictions.csv',dtype={'unit_id':str})
                expected=pd.concat([a.roster(n) for n in prot['probe_sources'][fold]],ignore_index=True)
                assert su[['cohort','unit_id','binary_label']].equals(expected[['cohort','unit_id','binary_label']])
                assert len(su)==prot['probe_unit_counts'][fold]
                assert 'retired_legacy_cohort' not in set(su.cohort) and cfg['target'] not in set(su.cohort)
                assert not su.duplicated(['cohort','unit_id']).any() and not pred.duplicated(['cohort','unit_id']).any()
                target=a.roster(cfg['target']);assert np.array_equal(target.unit_id,pred.unit_id)
                assert np.array_equal(target.binary_label,pred.y_true)
                saved=joblib.load(d/'probe.joblib');params=saved['probe'].get_params()
                for k,v in prot['probe'].items():assert params[k]==v,(method,k,params[k],v)
                x=np.load(d/'donor_p2_features.npz');src=x['source'];tgt=x['target']
                assert src.shape==(len(su),320) and tgt.shape==(len(pred),320)
                assert np.allclose(saved['imputer'].statistics_,np.nanmean(src,axis=0),rtol=1e-12,atol=1e-12)
                filled=saved['imputer'].transform(src)
                assert np.allclose(saved['scaler'].mean_,filled.mean(0),rtol=1e-12,atol=1e-12)
                assert int(saved['scaler'].n_samples_seen_)==len(su)
                pp=saved.predict_proba(tgt)[:,1]
                assert np.allclose(pp,pred.probability_AD,rtol=1e-12,atol=1e-12)
                m=a.read(d/'metrics.json');recomputed=a.metrics(pred.y_true,pp)
                for k,v in recomputed.items():assert np.isclose(v,m[k],rtol=1e-12,atol=1e-12),(fold,seed,method,k)
                if np.max(saved['probe'].n_iter_)>=prot['probe']['max_iter']:
                    notes.append(dict(fold=fold,seed=seed,method=method,note='Probe hit frozen iteration limit; no tuning performed'))
                coverage=pd.read_csv(d/'p2_coverage.csv',dtype={'unit_id':str})
                annotations.append(coverage);training_units.append(su[['cohort','unit_id']]);target_units.append(pred[['cohort','unit_id']])
                bio=a.read(a.OUT/'biology'/fold/f'seed_{seed}'/method/'COMPLETE.json')
                positive=int(pred.prediction.sum());negative=len(pred)-positive
                assert bio['n_predicted_AD']==positive and bio['n_predicted_Control']==negative
                assert bio['prediction_sha256']==cm['blind_predictions_sha256'] and bio['true_diagnosis_used'] is False
                assert bio['status']==('COMPLETE' if min(positive,negative)>=2 else 'NOT_EVALUABLE')
                runs.append(dict(fold=fold,seed=seed,method=method,n_probe=len(su),n_target=len(pred),
                        replay_probability_max_error=float(np.max(np.abs(pp-pred.probability_AD))),
                        pipeline_fit_target=False,metrics_recomputed=True,best_source_checkpoint_verified=True,
                        biology_status=bio['status'],status='PASS'))
            for group in [annotations,training_units,target_units]:
                assert all(group[0].equals(other) for other in group[1:]),(fold,seed,'method roster/coverage mismatch')
    assert len(runs)==9*len(chosen)
    pd.DataFrame(runs).to_csv(a.OUT/('reports/final_integrity_checks'+suffix+'.csv'),index=False)
    a.write(a.OUT/('reports/FINAL_DEPENDENCY_MANIFEST'+suffix+'.json'),dict(
        status='FINAL_SNAPSHOT_RECORDED',hashes=dependency_hashes,
        scientific_sources_match_original_freeze_and_recorded_repair=True,
        note='Final current-file binding; this does not retroactively prove all execution-time dependencies were automatically enforced.'))
    a.write(a.OUT/('reports/FINAL_INTEGRITY_CHECK'+suffix+'.json'),dict(status='PASS',runs=len(runs),training_units=3*len(chosen),
        folds=chosen,scope='ALL_FOUR_FOLDS' if len(chosen)==4 else 'PARTIAL_COMPLETED_FOLDS',
        checks=['checkpoint_hashes','best_SOURCE_val_checkpoint','prediction_and_model_hashes','exact_frozen_Probe',
          'SOURCE_eligible_rosters','one_row_per_donor','SOURCE_only_imputation','SOURCE_only_scaling',
          'replayed_probabilities','recomputed_metrics','same_method_rosters_and_P2_coverage',
          'biology_prediction_group_counts','NOT_EVALUABLE_policy'],notes=notes,
        semantic_audit='separate; this is a deterministic verification, not a semantic reviewer verdict'))
    print('FINAL_INTEGRITY_CHECK_PASS',len(runs),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--fold',action='append',choices=list(a.CACHE))
    main(p.parse_args().fold)
