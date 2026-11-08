"""Fail-closed library-qualified Multiome and author-barcode gate."""
from pathlib import Path
import collections,hashlib,json,os,time,traceback
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from ad_new_cohort import BUNDLE,EXPECTED,broad_types

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def write(p,obj):
    if p.exists() and json.loads(p.read_text())==obj:return
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def inspect(item,ann):
    p=Path(item['path']);assert len(ann)>0 and ann.cell_barcode.is_unique
    assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    assert not Path(str(p)+'.st').exists()
    with h5py.File(p,'r') as h:
        m=h['matrix'];shape=m['shape'][:].astype(int);bc=m['barcodes'][:].astype(str)
        ft=m['features/feature_type'][:].astype(str);ptr=m['indptr'][:]
        assert shape.tolist()==[len(ft),len(bc)] and len(ptr)==len(bc)+1
        assert ptr[0]==0 and ptr[-1]==len(m['data'])==len(m['indices']) and (np.diff(ptr)>=0).all()
        assert len(set(bc))==len(bc)
        types={k:int(v) for k,v in zip(*np.unique(ft,return_counts=True))}
        assert set(types)=={'Gene Expression','Peaks'} and min(types.values())>0
        rows=np.flatnonzero(np.isin(bc,ann.cell_barcode.to_numpy(dtype=bc.dtype)))
        observed=set(bc);expected=set(ann.cell_barcode.astype(str))
        assert expected<=observed and len(rows)==len(ann),(p.name,len(rows),len(ann))
        # Check both modalities in exactly the author-retained columns. No raw empty droplets enter modeling.
        rna_nonzero=atac_nonzero=0
        for first in range(0,len(rows),256):
            rr=rows[first:first+256];lo,hi=int(ptr[rr[0]]),int(ptr[rr[-1]+1])
            data=m['data'][lo:hi];idx=m['indices'][lo:hi]
            assert np.isfinite(data).all() and (data>=0).all() and (data==np.floor(data)).all()
            assert (idx>=0).all() and (idx<len(ft)).all()
            x=sparse.csr_matrix((data,idx,ptr[rr[0]:rr[-1]+2]-lo),shape=(rr[-1]-rr[0]+1,len(ft)))[rr-rr[0]]
            rna_nonzero+=int((x[:,ft=='Gene Expression'].sum(1).A1>0).sum())
            atac_nonzero+=int((x[:,ft=='Peaks'].sum(1).A1>0).sum())
        assert rna_nonzero==atac_nonzero==len(ann),(p.name,rna_nonzero,atac_nonzero,len(ann))
    return dict(file=p.name,donor=item['donor'],library=str(ann.library.iloc[0]),bytes=item['bytes'],
        sha256=item['sha256'],matrix_shape=shape.tolist(),feature_types=types,raw_barcodes=len(bc),
        metadata_barcodes=len(expected),barcode_intersection=len(rows),metadata_subset_of_h5=True,
        barcode_exact=expected==observed,paired_feature_types=True,
        author_cells_with_nonzero_RNA=rna_nonzero,author_cells_with_nonzero_ATAC=atac_nonzero)

def main():
    manifest=json.loads((BUNDLE/'download_manifest.json').read_text())
    if manifest.get('status')!='COMPLETE':return False
    df=pd.read_csv(manifest['metadata_path'],sep='\t');rows=[]
    chosen=df[df.donor.isin(EXPECTED)].copy()
    assert chosen.groupby('donor').size().to_dict()==EXPECTED
    assert not chosen.duplicated(['library','cell_barcode']).any()
    assert len(manifest['files'])==11 and len(manifest['selected_donors'])==10
    for item in manifest['files']:
        lib=item['donor']
        if 'ad22rep1' in item['filename']:lib='ad22_rep1'
        if 'ad22rep2' in item['filename']:lib='ad22_rep2'
        ann=chosen[chosen.library.eq(lib)]
        assert set(ann.donor)=={item['donor']}
        rows.append(inspect(item,ann));print('VERIFIED',lib,len(ann),flush=True)
    assert {x['library'] for x in rows}==set(chosen.library)
    assert sum(x['metadata_barcodes'] for x in rows)==54999
    chosen['broad_type']=broad_types(chosen)
    coverage=chosen.groupby(['donor','group','broad_type']).size().unstack(fill_value=0).reset_index()
    coverage.to_csv(BUNDLE/'author_cell_type_coverage_rev04.csv',index=False)
    pd.DataFrame(manifest['selected_donors']).to_csv(BUNDLE/'donor_summary_rev04.csv',index=False)
    pd.DataFrame([{k:v for k,v in x.items() if k not in ['feature_types','matrix_shape']}|
        {'RNA_features':x['feature_types']['Gene Expression'],'ATAC_features':x['feature_types']['Peaks']} for x in rows]).to_csv(BUNDLE/'file_verification_rev04.csv',index=False)
    report=dict(status='PASS',files=rows,donors=manifest['selected_donors'],all_paired=True,
        all_metadata_barcodes_observed=True,total_author_cells=54999,independent_donors=10,
        AD_donors=5,Control_donors=5,metadata_sha256=sha(manifest['metadata_path']),
        manifest_sha256=sha(BUNDLE/'download_manifest.json'),validator_sha256=sha(__file__),
        control_policy='restrict raw control H5 to author Library+Barcode QC subset',
        identity='library-qualified barcode; ad22 two libraries are one donor',
        overlap_limitation='No metadata donor-ID overlap found in prior audit; hidden biological overlap cannot be ruled out by IDs alone')
    write(BUNDLE/'verification_report_rev04.json',report);return True

if __name__=='__main__':
    try:
        if not main():print('WAIT_DOWNLOAD',flush=True);raise SystemExit(75)
    except Exception:
        write(BUNDLE/'verification_report_rev04.json',dict(status='FAIL',traceback=traceback.format_exc()))
        raise
