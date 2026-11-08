"""Source-only ATAC TF-IDF for the B2-S joint B/F removal ablation."""
import json
import time
from pathlib import Path

import numpy as np

from data.code.s_mode.prepare import legacy, paired_atac, splits

ROOT = Path('/home/u202522802045/share/zpb')
RNA_CACHE = ROOT/'data/cache/s_mode/PBMC'


def prepare(key, cache_root, expected_identity):
    cache = Path(cache_root)/key
    if (cache/'manifest.json').exists():
        manifest = legacy.checked(cache/'manifest.json')
        assert manifest['identity']==expected_identity
        return cache, manifest
    if cache.exists() and any(cache.iterdir()):
        raise RuntimeError(f'Preserve and inspect incomplete cache: {cache}')
    started=time.perf_counter()
    base=legacy.cache_dir(RNA_CACHE,key)
    original=json.loads((base/'manifest.json').read_text())
    identity_path=legacy.DATA[key]/'DATASET_IDENTITY.json'
    assert original['identity']==expected_identity==json.loads(identity_path.read_text())['canonical_identity']
    index,cells=splits(key)
    assert cells==(base/'barcodes.txt').read_text().splitlines()
    names=['rna_train.npy','rna_val.npy','rna_state.npz','gene_ids.npy','barcodes.txt',
           'train_index.npy','val_index.npy','test_index.npy']
    for name in names:
        path=base/name
        assert legacy.sha(path)==legacy.expected_hash(original['hashes'],path),path
    for split,rows in index.items():
        np.testing.assert_array_equal(rows,np.load(base/f'{split}_index.npy'))
    state=np.load(base/'rna_state.npz')
    np.testing.assert_array_equal(state['fit_cell_index'],index['train'])
    rows=np.concatenate([index['train'],index['val']])
    atac,peaks,dependencies=paired_atac(key,rows,cells)
    ntrain=len(index['train'])
    keep=np.asarray(atac[:ntrain].sum(0)).ravel()>=np.ceil(.005*ntrain)
    peaks=peaks[keep];atac=atac[:,keep].tocsr()
    assert len(peaks)>0
    idf=np.log1p(ntrain/np.maximum(np.asarray(atac[:ntrain].sum(0)).ravel(),1)).astype(np.float32)
    values=atac.multiply(1/np.maximum(np.asarray(atac.sum(1)).ravel(),1)[:,None]).multiply(idf).tocsr().astype(np.float32)
    mean=np.asarray(values[:ntrain].mean(0)).ravel()
    std=np.sqrt(np.maximum(np.asarray(values[:ntrain].power(2).mean(0)).ravel()-mean**2,0))
    cache.mkdir(parents=True,exist_ok=False)
    for name in names:
        (cache/name).symlink_to(base/name)
    np.save(cache/'peak_names.npy',peaks)
    np.savez(cache/'atac_state.npz',idf=idf,mean=mean,std=std,fit_cell_index=index['train'])
    for split,lo,hi in [('train',0,ntrain),('val',ntrain,len(rows))]:
        output=np.lib.format.open_memmap(cache/f'atac_{split}.npy',mode='w+',dtype=np.float32,shape=(hi-lo,len(peaks)))
        for start in range(lo,hi,256):
            block=(values[start:min(start+256,hi)].toarray()-mean)/(std+1e-6)
            assert np.isfinite(block).all()
            output[start-lo:start-lo+len(block)]=block
        output.flush();del output
    dependencies += [identity_path,base/'manifest.json',Path(__file__),ROOT/'data/code/s_mode/prepare.py']
    dependencies += list(cache.iterdir())
    manifest=dict(status='PREPARED',dataset=key,identity=expected_identity,cohort_cells=len(cells),
        split_counts={s:len(i) for s,i in index.items()},hvg_count=2000,atac_peaks=len(peaks),
        preprocessing_seconds=time.perf_counter()-started,fitting_cells=index['train'].tolist(),
        B_used=False,F_used=False,regulatory_selection_used=False,labels_used=False,test_atac_loaded=False,
        RNA_rule='reuse task45 Source-train HVGs and frozen RNA statistics',
        ATAC_rule='Source-train prevalence >=ceil(.005*N); binary TF over selected peaks; train IDF; Source-train per-peak zscore/std+1e-6; no regulatory graph/projection or BCE clipping',
        hashes={str(p):legacy.sha(p) for p in dependencies if p.is_file()})
    legacy.write(cache/'manifest.json',manifest)
    print('NO_BF_CACHE_READY',key,len(peaks),flush=True)
    return cache,manifest
