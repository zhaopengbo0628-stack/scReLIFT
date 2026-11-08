"""Unfiltered TARGET QC measurements and frozen-dictionary coverage; no model fitting."""
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path('E:/my_research')
OUT = Path(__file__).resolve().parent.parent
RAW = ROOT/'data/downloads/GSE272082_original_20261024'


if __name__ == '__main__':
    report_path = OUT/'qc/target_raw_counts_qc_audit.json'
    if report_path.exists():
        raise FileExistsError('Preserve completed audit')
    common = (ROOT/'data/benchmark/AD_Dmode_gene_overlap_audit_20261024/common_stable_gene_ids.txt').read_text().splitlines()
    assert len(common) == len(set(common)) == 36601
    samples = pd.read_csv(OUT/'metadata/target_libraries.csv')
    per_cell, feature_rows = [], []
    for _, row in samples.iterrows():
        path = RAW/'matrices'/Path(row.absolute_path).name
        with h5py.File(path) as f:
            x = f['matrix']; shape = tuple(map(int,x['shape'][:]))
            ids = x['features/id'].asstr()[:]
            symbols = x['features/name'].asstr()[:]
            types = x['features/feature_type'].asstr()[:]
            barcodes = x['barcodes'].asstr()[:]
            rna = types == 'Gene Expression'; atac = types == 'Peaks'
            mito = rna & np.char.startswith(symbols.astype(str),'MT-')
            assert rna.any() and atac.any() and len(barcodes) == shape[1]
            rna_ids = ids[rna]
            stable = np.array([re.sub(r'^(ENSG\d+)\.\d+$', r'\1', g) for g in rna_ids])
            if len(set(stable)) != len(stable):
                raise ValueError('Duplicate stable ID requires explicit biological audit')
            lookup = {g:i for i,g in enumerate(stable)}
            mapping = np.array([lookup.get(g,-1) for g in common], dtype=np.int64)
            np.save(OUT/'metadata'/(row.library_id+'_common_gene_to_RNA_feature.npy'), mapping)
            feature_rows.append(dict(library_id=row.library_id, donor_id=row.donor_id,
                RNA_features=int(rna.sum()), ATAC_features=int(atac.sum()), barcodes=shape[1],
                stable_Ensembl_features=int(sum(bool(re.fullmatch(r'ENSG\d+',g)) for g in stable)),
                versioned_Ensembl_features=int((stable != rna_ids).sum()),
                duplicate_stable_ids=len(stable)-len(set(stable)),
                common_36601_covered=int((mapping>=0).sum()),
                common_36601_missing=int((mapping<0).sum()),
                target_feature_data_used_for_fitting=False))
            ptr = x['indptr'][:]
            assert len(ptr) == shape[1]+1 and ptr[-1] == len(x['data'])
            for start in range(0,shape[1],512):
                stop = min(start+512,shape[1]); lo,hi = map(int,(ptr[start],ptr[stop]))
                values = x['data'][lo:hi]; ix = x['indices'][lo:hi]
                if not np.isfinite(values).all() or (values<0).any() or (values!=np.floor(values)).any():
                    raise ValueError('Invalid raw counts')
                m = sparse.csc_matrix((values,ix,ptr[start:stop+1]-lo),shape=(shape[0],stop-start))
                r = m[rna]; a = m[atac]
                total = np.asarray(r.sum(0),dtype=np.float64).ravel()
                mt = np.asarray(m[mito].sum(0),dtype=np.float64).ravel()
                per_cell.append(pd.DataFrame(dict(library_id=row.library_id,donor_id=row.donor_id,
                    brain_region=row.tissue,barcode=barcodes[start:stop],
                    nCount_RNA=total,nFeature_RNA=np.asarray((r>0).sum(0)).ravel(),
                    percent_MT=np.divide(100*mt,total,out=np.full_like(total,np.nan),where=total>0),
                    nCount_ATAC=np.asarray(a.sum(0)).ravel(),nFeature_ATAC=np.asarray((a>0).sum(0)).ravel(),
                    author_final_QC_recovered=False,retained_for_model=False,
                    QC_status='AUDIT_ONLY_FINAL_QC_PENDING')))
        print('TARGET_QC_MEASURED',row.library_id,shape[1],flush=True)
    cells = pd.concat(per_cell,ignore_index=True)
    assert len(cells) == 96498 and not cells.duplicated(['library_id','barcode']).any()
    cells.to_csv(OUT/'qc/target_unfiltered_per_cell_metrics.csv.gz',index=False)
    pd.DataFrame(feature_rows).to_csv(OUT/'data_audit/target_gene_dictionary_coverage.csv',index=False)
    cells.groupby(['donor_id','brain_region']).agg(cells=('barcode','size'),
        RNA_median_counts=('nCount_RNA','median'),RNA_median_genes=('nFeature_RNA','median'),
        median_percent_MT=('percent_MT','median'),ATAC_median_counts=('nCount_ATAC','median'),
        ATAC_median_features=('nFeature_ATAC','median')).to_csv(OUT/'qc/target_unfiltered_donor_region_metrics.csv')
    report = dict(status='RAW_COUNT_QC_MEASUREMENTS_COMPLETE_FINAL_QC_PENDING',
        libraries=len(samples),donors=cells.donor_id.nunique(),unfiltered_barcodes=len(cells),
        final_model_cells=None,cell_filtering_performed=False,normalization_performed=False,
        target_feature_selection=False,target_training=False,target_label_based_QC=False,
        all_libraries_cover_common_36601=all(r['common_36601_missing']==0 for r in feature_rows),
        unavailable=['CellBender_corrected_counts','author_retained_barcodes','doublet_calls',
                     'per_cell_TSS','per_cell_nucleosome','author_celltype_annotation'],
        warning='nCount_ATAC is peak-matrix sum, not unique fragments or FRiP; mitochondrial symbols are audited without filtering')
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
