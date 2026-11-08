"""After the existing verified export completes, remove all Low ADNC cells on disk."""
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent
OLD = OUT.parent/'AD_Dmode_Goal_20261024/rev02_three_source_seaad_holdout'
SRC = OLD/'preprocessing/seaad_paired_counts'
DEST = OUT/'data_audit/seaad_source3_eligible_counts'
STATE = OUT/'logs/seaad_eligibility_preparation_status.json'


def state(status, **fields):
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(status=status,training_started=False,**fields),indent=2)+'\n')
    os.replace(temporary,STATE)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def subset(source, rows, target):
    with h5py.File(source) as src, h5py.File(target,'w') as dst:
        x = src['X']; ptr = x['indptr'][:]
        assert x.attrs['encoding-type'] == 'csr_matrix' and np.all(np.diff(rows)>0)
        lengths = np.diff(ptr)[rows]; pointers = np.r_[0,np.cumsum(lengths)].astype(np.int64)
        g = dst.create_group('X'); g.attrs.update(dict(x.attrs)); g.attrs['shape'] = (len(rows),int(x.attrs['shape'][1]))
        g.create_dataset('indptr',data=pointers)
        arrays = {name:g.create_dataset(name,(int(pointers[-1]),),dtype=x[name].dtype,
                   chunks=True,compression='gzip',compression_opts=1) for name in ['data','indices']}
        for first in range(0,len(rows),256):
            part = rows[first:first+256]; lo,hi = int(ptr[part[0]]),int(ptr[part[-1]+1])
            # Input is already fully verified; still validate every retained value.
            for name, output in arrays.items():
                values = x[name][lo:hi]
                keep = np.concatenate([values[ptr[r]-lo:ptr[r+1]-lo] for r in part])
                if name == 'data' and (not np.isfinite(keep).all() or np.any(keep<0) or np.any(keep!=np.floor(keep))):
                    raise ValueError('Invalid retained counts')
                output[pointers[first]:pointers[first+len(part)]] = keep


if __name__ == '__main__':
    if os.name != 'posix':
        raise RuntimeError('Run on storage server only')
    if (DEST/'manifest.json').exists():
        raise FileExistsError('Preserve existing eligible counts')
    state('WAITING_FOR_EXISTING_VERIFIED_SEAAD_EXPORT',source=str(SRC))
    while not (SRC/'manifest.json').exists():
        time.sleep(30)
    manifest = json.loads((SRC/'manifest.json').read_text())
    if manifest['status'] != 'COMPLETE':
        raise RuntimeError('Input export is not COMPLETE')
    cells = pd.read_csv(SRC/'cells_rna_author.csv',dtype={'sample_id':str,'Donor ID':str})
    eligibility = pd.read_csv(OUT/'metadata/seaad_cell_eligibility_by_sample_id.csv.gz',dtype={'sample_id':str,'Donor ID':str})
    cells['input_row'] = np.arange(len(cells))
    merged = cells.merge(eligibility[['sample_id','Donor ID','include_source3','binary_label']],
                         on=['sample_id','Donor ID'],how='left',validate='one_to_one')
    assert len(merged) == 133084 and not merged.include_source3.isna().any()
    kept = merged[merged.include_source3].sort_values('input_row')
    assert len(kept) == 113019 and kept['Donor ID'].nunique() == 24
    assert not kept['Overall AD neuropathological Change'].eq('Low').any()
    assert kept.groupby('Donor ID').binary_label.first().value_counts().to_dict() == {1.0:21,0.0:3}
    DEST.mkdir(exist_ok=False)
    state('FILTERING_LOW_ADNC_FROM_BOTH_REAL_COUNT_MATRICES')
    rows = kept.input_row.to_numpy(np.int64)
    for modality in ['rna','atac']:
        subset(SRC/(modality+'_counts.h5'),rows,DEST/(modality+'_counts.h5'))
    kept.to_csv(DEST/'cells.csv.gz',index=False)
    result = dict(status='COMPLETE',cells=len(kept),donors=24,positive_donors=21,negative_donors=3,
        excluded_donors=4,excluded_cells=20065,endpoint='ADNC_IntermediateHigh_vs_NotAD',
        RNA_ATAC_row_order='identical cells.csv.gz order',both_modalities_filtered=True,
        normalized=False,hvg_selected=False,model_trained=False,
        input_manifest=str(SRC/'manifest.json'),input_manifest_sha256=sha(SRC/'manifest.json'),
        raw_input_hashes=manifest['input_sha256'],
        output_sha256={p.name:sha(p) for p in DEST.iterdir() if p.is_file()})
    (DEST/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    state('ELIGIBLE_SEAAD_PAIRED_COUNTS_COMPLETE',output=str(DEST))
    print(json.dumps(result),flush=True)
