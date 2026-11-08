"""Replacement-specific leakage, identity, pooling and fail-closed regression checks."""
import ast,tempfile
from pathlib import Path
import h5py,numpy as np,pandas as pd
import ad_runtime as a
import ad_new_cohort as n
import ad_verify_bundle as v

def main():
    p=a.protocol();assert len(p['folds'])==4 and p['seeds']==a.SEEDS
    assert p['probe_unit_counts']==dict(fold1=47,fold2=49,fold3=38,fold4=52)
    for cfg in p['folds'].values():
        assert len(cfg['sources'])==3 and cfg['target'] not in cfg['sources']
        assert 'retired_legacy_cohort' not in cfg['sources']+[cfg['target']]
    frame=pd.DataFrame({'class':['Excitatory_Neuron','Inhibitory_Neuron']+['Non_Neuronal']*5,
        'subclass':['CA1','SST','Astro','Oligo','OPC','Microglia','T-Cell']})
    assert n.broad_types(frame).tolist()==a.TYPES+['Microglia/PVM','Unknown']
    ids=pd.DataFrame({'unit_id':['ad22']*3,'broad_type':['Excitatory','Excitatory','OPC']})
    z=np.stack([np.ones(64),np.ones(64)*3,np.ones(64)*5])
    units=pd.DataFrame({'unit_id':['ad22']});x,c=a.pooled(z,ids,units)
    assert x.shape==(1,320) and np.all(x[0,:64]==2) and np.all(x[0,-64:]==5) and c.tolist()==[[2,0,0,0,1]]
    rng=np.random.default_rng(20020130);sx=rng.normal(size=(12,320));sx[0,:64]=np.nan
    y=np.array([0,1]*6);model=a.fit_probe(sx,y)
    assert np.allclose(model['imputer'].statistics_,np.nanmean(sx,axis=0))
    assert model['scaler'].n_samples_seen_==12
    assert all(model['probe'].get_params()[k]==val for k,val in p['probe'].items())
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'valid.h5'
        with h5py.File(path,'w') as h:
            m=h.create_group('matrix');m['shape']=np.array([2,1]);m['barcodes']=np.array([b'AAA-1'])
            m['features/feature_type']=np.array([b'Gene Expression',b'Peaks']);m['indptr']=[0,2];m['indices']=[0,1];m['data']=[2,3]
        item=dict(path=str(path),donor='ad22',bytes=path.stat().st_size,sha256=v.sha(path))
        ann=pd.DataFrame({'cell_barcode':['AAA-1'],'library':['ad22_rep1']})
        assert v.inspect(item,ann)['metadata_subset_of_h5']
        bad=ann.copy();bad['cell_barcode']='MISSING-1'
        try:v.inspect(item,bad)
        except AssertionError:pass
        else:raise AssertionError('Barcode gate did not fail closed')
        with h5py.File(path,'r+') as h:h['matrix/data'][1]=0
        item.update(sha256=v.sha(path),bytes=path.stat().st_size)
        try:v.inspect(item,ann)
        except AssertionError:pass
        else:raise AssertionError('Missing ATAC gate did not fail closed')
    a.write(a.OUT/'REPLACEMENT_TESTS.json',dict(status='PASS',
        checks=['four_folds_three_probe_sources','fixed_seeds_probe','author_mapping','technical_libraries_one_donor',
        'SOURCE_only_imputation_scaling','missing_barcode_rejected','zero_ATAC_rejected']))
    print('REPLACEMENT_TESTS_PASS',flush=True)

if __name__=='__main__':main()
