"""Build Zeppilli GSE239477 SSp paired counts/peak harmonization. Labels unavailable -> no cache."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
ROOT=Path('/home/u202522802045/share/zpb')
sys.path[:0]=[str(ROOT/'main/core/regulatory/common/legacy_support')]
from v5d.harmonize import _canonical_peaks_from_sources, _overlap_mapping, _aggregate_binary, parse_peak
ST=ROOT/'data/staging/Mouse-brain/GSE239477_SSp_v1_20261023/libraries'
BENCH=ROOT/'data/benchmark/Mouse-brain/GSE239477_SSp_v1_20261023/ZeppilliSSp'
RESULT=ROOT/'result_contain/89_brain_two_new_sources'
LIBS=['N1','N4','N6','N7']
GSM={'N1':('GSM7665923','GSM7665924'),'N4':('GSM7665925','GSM7665926'),'N6':('GSM7665927','GSM7665928'),'N7':('GSM7665929','GSM7665930')}
def main():
    RESULT.mkdir(parents=True,exist_ok=True)
    assert all((ST/s/'manifest.json').exists() for s in LIBS), [s for s in LIBS if not (ST/s/'manifest.json').exists()]
    assert not BENCH.exists(); BENCH.mkdir(parents=True)
    roots=[ST/s for s in LIBS]
    canonical=_canonical_peaks_from_sources(roots)
    all_peaks=[np.load(root/'peak_names.npy',allow_pickle=True).astype(str) for root in roots]
    exact_union=set().union(*[set(x.tolist()) for x in all_peaks])
    rna=[]; atac=[]; bcs=[]; avail=[]; pair_rows=[]; peak_rows=[]
    gene_ids=None; gene_names=None
    for s,root in zip(LIBS,roots):
        m=json.load(open(root/'manifest.json'))
        r=sparse.load_npz(root/'rna_counts.npz').tocsr(); a=sparse.load_npz(root/'atac_counts.npz').tocsr()
        gid=np.load(root/'gene_ids.npy',allow_pickle=True).astype(str); gnm=np.load(root/'gene_names.npy',allow_pickle=True).astype(str)
        if gene_ids is None: gene_ids=gid; gene_names=gnm
        assert gid.tolist()==gene_ids.tolist()
        peaks=np.load(root/'peak_names.npy',allow_pickle=True).astype(str)
        mapping,mm=_overlap_mapping(peaks,canonical)
        ab=_aggregate_binary(a.astype(bool).astype(np.float32),mapping,len(canonical)); ab.data[:]=1.0
        rr=np.load(root/'barcodes.npy',allow_pickle=True).astype(str)
        rna.append(r); atac.append(ab); bcs.append(np.array([s+'::'+x for x in rr]))
        avail.append(np.bincount(mapping[mapping>=0],minlength=len(canonical))>0)
        pair_rows.append(dict(library=s,GSM_RNA=GSM[s][0],GSM_ATAC=GSM[s][1],cells=len(rr),genes=r.shape[1],peaks=int(a.shape[1]),
            rna_total_median=float(np.median(np.asarray(r.sum(1)).ravel())),rna_detected_median=float(np.median(np.asarray((r>0).sum(1)).ravel())),
            atac_total_median=float(np.median(np.asarray(a.sum(1)).ravel())),atac_detected_median=float(np.median(np.asarray((a>0).sum(1)).ravel())),
            canonical_mapped=int((mapping>=0).sum()),canonical_coverage=float((mapping>=0).mean()),
            native_to_canonical_one_to_many=0))
        widths=np.array([parse_peak(p)[2]-parse_peak(p)[1] for p in peaks])
        peak_rows.append(dict(library=s,native_peaks=len(peaks),mapped_peaks=int((mapping>=0).sum()),unmapped=int((mapping<0).sum()),
            exact_union=len(exact_union),canonical=len(canonical),native_p50=float(np.percentile(widths,50)),native_p90=float(np.percentile(widths,90)),
            native_p99=float(np.percentile(widths,99)),native_max=int(widths.max())))
    R=sparse.vstack(rna).tocsr(); A=sparse.vstack(atac).tocsr(); B=np.concatenate(bcs)
    sparse.save_npz(BENCH/'rna_counts.npz',R); sparse.save_npz(BENCH/'atac_binary.npz',A)
    np.save(BENCH/'peak_names.npy',canonical); np.save(BENCH/'barcodes.npy',B)
    np.save(BENCH/'gene_ids.npy',gene_ids); np.save(BENCH/'gene_names.npy',gene_names)
    sparse.save_npz(BENCH/'peak_availability.npz',sparse.csr_matrix(np.vstack(avail).astype(np.float32)))
    meta=pd.DataFrame(dict(barcode=B,library=np.repeat(LIBS,[x['cells'] for x in pair_rows]),
        RNA_GSM=np.repeat([GSM[s][0] for s in LIBS],[x['cells'] for x in pair_rows]),
        ATAC_GSM=np.repeat([GSM[s][1] for s in LIBS],[x['cells'] for x in pair_rows]),
        reference_label='',label_status='AUTHOR_LABELS_UNAVAILABLE'))
    idx=np.arange(len(meta)); from sklearn.model_selection import train_test_split
    tr,held=train_test_split(idx,train_size=.7,random_state=20020130); va,te=train_test_split(held,test_size=.5,random_state=20020130)
    splits={k:np.sort(v) for k,v in [('train',tr),('val',va),('test',te)]}
    meta['split']=''
    for k,v in splits.items(): meta.loc[v,'split']=k
    meta.to_csv(BENCH/'metadata.csv',index=False); np.savez_compressed(BENCH/'splits.npz',**{k+'_index':v for k,v in splits.items()})
    pd.DataFrame(pair_rows).to_csv(RESULT/'ZEPPILLI_PAIRING_LIBRARIES.csv',index=False)
    pd.DataFrame(peak_rows).to_csv(RESULT/'ZEPPILLI_PEAK_LIBRARIES.csv',index=False)
    widths=np.array([parse_peak(p)[2]-parse_peak(p)[1] for p in canonical])
    (BENCH/'manifest.json').write_text(json.dumps(dict(status='ZEPPILLI_COUNTS_COMPLETE_AUTHOR_LABELS_BLOCKED',cells=len(meta),genes=len(gene_ids),
        peaks=len(canonical),exact_union=len(exact_union),libraries=4,cells_by_library={x['library']:x['cells'] for x in pair_rows},
        labels=0,author_labels=False,author_qc='Cell Ranger filtered only; no public author barcode list',canonical_width_p50=float(np.percentile(widths,50)),
        canonical_width_p90=float(np.percentile(widths,90)),canonical_width_p99=float(np.percentile(widths,99)),canonical_width_max=int(widths.max()),
        splits={k:len(v) for k,v in splits.items()}),indent=2)+'\n')
    print(json.dumps(dict(cells=len(meta),genes=len(gene_ids),peaks=len(canonical),exact_union=len(exact_union),split={k:len(v) for k,v in splits.items()}),indent=2))
if __name__=='__main__': main()
