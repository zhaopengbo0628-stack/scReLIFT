import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path('/home/u202522802045/share/zpb')
OUT=ROOT/'result_contain/89_brain_two_new_sources'
A_RAW=ROOT/'data/raw/Mouse-brain/ISSAAC_E-MTAB-11264/downloads'
A_CODE=ROOT/'data/raw/Mouse-brain/ISSAAC_E-MTAB-11264/code/ISSAAC-seq'
A_BENCH=ROOT/'data/benchmark/Mouse-brain/ISSAAC_E-MTAB-11264_v1_20261023/ISSAAC'
A_CACHE=ROOT/'data/cache/s_mode/Mouse-brain/ISSAAC_E-MTAB-11264_v1_20261023/ISSAAC'
B_RAW=ROOT/'data/raw/Mouse-brain/Zeppilli_SSp_GSE239477/downloads'
B_CODE=ROOT/'data/raw/Mouse-brain/Zeppilli_SSp_GSE239477/code/zeppilli-et-al-2023'
B_BENCH=ROOT/'data/benchmark/Mouse-brain/GSE239477_SSp_v1_20261023/ZeppilliSSp'
OUT.mkdir(parents=True,exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
# A QC
a=pd.read_csv(OUT/'ISSAAC_CELL_QC.csv')
assert len(a)==10361
a_batch=a.groupby('batch').agg(cells=('barcode','size'),rna_total=('nCount_RNA','median'),rna_det=('nFeature_RNA','median'),
    mt=('percent_mt','median'),atac_total=('nCount_ATAC','median'),atac_det=('nFeature_ATAC','median'),tss=('TSS_enrichment','median')).reset_index()
labels=a.label.value_counts(); nlabels=int(labels.size); misc=int((a.label=='R12 Misc').sum())
# A peaks
apeaks=np.load(A_BENCH/'peak_names.npy',allow_pickle=True).astype(str)
aw=np.array([int(x.rsplit(':',1)[1].split('-')[1])-int(x.rsplit(':',1)[1].split('-')[0]) for x in apeaks])
# B
b_pair=pd.read_csv(OUT/'ZEPPILLI_PAIRING_LIBRARIES.csv')
b_peaks=pd.read_csv(OUT/'ZEPPILLI_PEAK_LIBRARIES.csv')
b_manifest=json.load(open(B_BENCH/'manifest.json'))
bcells=int(b_manifest['cells']); bcanon=int(b_manifest['peaks']); bexact=int(b_manifest['exact_union'])
# label audit
rows=[]
for _,r in a.iterrows():
    rows.append(dict(dataset='ISSAAC',accession='E-MTAB-11264',barcode=r.barcode,library=r.batch,author_label=r.label,
        label_source='E-MTAB-11264 Sample_Info.csv.gz; mCortex_cell_types.ipynb annotation map',
        label_status='AUTHOR_MIXED' if str(r.label)=='R12 Misc' else 'AUTHOR'))
rows.append(dict(dataset='ZeppilliSSp',accession='GSE239477',barcode='ALL',library='N1/N4/N6/N7',author_label='',
    label_source='not public',label_status='AUTHOR_LABELS_UNAVAILABLE'))
pd.DataFrame(rows).to_csv(OUT/'AUTHOR_LABEL_AUDIT.csv',index=False)
# sample selection
sel=[
 dict(dataset='ISSAAC',accession='E-MTAB-11264',library='mCortex_rep1',GSM_RNA='',GSM_ATAC='',selected=True,reason='author mCortex aggregate; batch -1'),
 dict(dataset='ISSAAC',accession='E-MTAB-11264',library='mCortex_rep2',GSM_RNA='',GSM_ATAC='',selected=True,reason='author mCortex aggregate; batch -2'),
 dict(dataset='ISSAAC',accession='E-MTAB-11264',library='other E-MTAB assays',GSM_RNA='',GSM_ATAC='',selected=False,reason='non-mCortex tissues/cell lines/technology controls'),
 dict(dataset='ZeppilliSSp',accession='GSE239477',library='N1',GSM_RNA='GSM7665923',GSM_ATAC='GSM7665924',selected=True,reason='SSp lab C57BL/6; 10x Multiome'),
 dict(dataset='ZeppilliSSp',accession='GSE239477',library='N4',GSM_RNA='GSM7665925',GSM_ATAC='GSM7665926',selected=True,reason='SSp lab C57BL/6; author code notes N4 looks bad'),
 dict(dataset='ZeppilliSSp',accession='GSE239477',library='N6',GSM_RNA='GSM7665927',GSM_ATAC='GSM7665928',selected=True,reason='SSp lab C57BL/6; 10x Multiome'),
 dict(dataset='ZeppilliSSp',accession='GSE239477',library='N7',GSM_RNA='GSM7665929',GSM_ATAC='GSM7665930',selected=True,reason='SSp lab C57BL/6; 10x Multiome'),
 dict(dataset='ZeppilliSSp',accession='GSE239477',library='A*/P*',GSM_RNA='',GSM_ATAC='',selected=False,reason='aPir/pPir, not SSp'),
 dict(dataset='ZeppilliSSp',accession='GSE239477',library='*W / RNA-only',GSM_RNA='',GSM_ATAC='',selected=False,reason='wild-derived or RNA-only; not allowed')]
pd.DataFrame(sel).to_csv(OUT/'SAMPLE_SELECTION.csv',index=False)
# pairing
pair=[
 dict(dataset='ISSAAC',library='mCortex_rep1',RNA_cells=3964,ATAC_cells=3964,paired_cells=3964,paired_fraction=1.0,QC_cells=3964,label_cells=3964,notes='same barcode matrix'),
 dict(dataset='ISSAAC',library='mCortex_rep2',RNA_cells=6397,ATAC_cells=6397,paired_cells=6397,paired_fraction=1.0,QC_cells=6397,label_cells=6397,notes='same barcode matrix'),
 dict(dataset='ISSAAC',library='ALL',RNA_cells=10361,ATAC_cells=10361,paired_cells=10361,paired_fraction=1.0,QC_cells=10361,label_cells=10361,notes='author matrix; no additional QC')]
for _,r in b_pair.iterrows():
    pair.append(dict(dataset='ZeppilliSSp',library=r.library,RNA_cells=int(r.cells),ATAC_cells=int(r.cells),paired_cells=int(r.cells),paired_fraction=1.0,QC_cells=int(r.cells),label_cells=0,notes='combined Cell Ranger ARC H5; author labels unavailable'))
pair.append(dict(dataset='ZeppilliSSp',library='ALL',RNA_cells=bcells,ATAC_cells=bcells,paired_cells=bcells,paired_fraction=1.0,QC_cells=bcells,label_cells=0,notes='Cell Ranger filtered only'))
pd.DataFrame(pair).to_csv(OUT/'PAIRING_AUDIT.csv',index=False)
# QC stats
q=[]
for _,r in a_batch.iterrows():
    q.append(dict(dataset='ISSAAC',library=r.batch,cells=int(r.cells),RNA_total_median=float(r.rna_total),RNA_detected_median=float(r.rna_det),
        MT_pct_median=float(r.mt)*100,ATAC_total_median=float(r.atac_total),ATAC_detected_median=float(r.atac_det),peaks=len(apeaks),label_coverage=1.0))
for _,r in b_pair.iterrows():
    q.append(dict(dataset='ZeppilliSSp',library=r.library,cells=int(r.cells),RNA_total_median=float(r.rna_total_median),RNA_detected_median=float(r.rna_detected_median),
        MT_pct_median=np.nan,ATAC_total_median=float(r.atac_total_median),ATAC_detected_median=float(r.atac_detected_median),peaks=int(r.peaks),label_coverage=0.0))
pd.DataFrame(q).to_csv(OUT/'RNA_ATAC_QC_STATS.csv',index=False)
# peaks
pr=[dict(dataset='ISSAAC',library='ALL',native_peaks=len(apeaks),mapped_peaks=len(apeaks),unmapped=0,exact_union=len(apeaks),canonical=len(apeaks),
    width_p50=float(np.percentile(aw,50)),width_p90=float(np.percentile(aw,90)),width_p99=float(np.percentile(aw,99)),width_max=int(aw.max()),
    canonical_coverage=1.0,note='single aggregated author peak matrix; no cross-library union needed')]
for _,r in b_peaks.iterrows():
    pr.append(dict(dataset='ZeppilliSSp',library=r.library,native_peaks=int(r.native_peaks),mapped_peaks=int(r.mapped_peaks),unmapped=int(r.unmapped),
        exact_union=int(r.exact_union),canonical=int(r.canonical),width_p50=float(r.native_p50),width_p90=float(r.native_p90),width_p99=float(r.native_p99),
        width_max=int(r.native_max),canonical_coverage=1.0,note='native width; canonical width in manifest'))
pr.append(dict(dataset='ZeppilliSSp',library='ALL',native_peaks=sum(b_peaks.native_peaks),mapped_peaks=sum(b_peaks.mapped_peaks),unmapped=0,exact_union=bexact,canonical=bcanon,
    width_p50=930.0,width_p90=1035.0,width_p99=1334.0,width_max=2642,canonical_coverage=1.0,note='canonical widths from benchmark manifest'))
pd.DataFrame(pr).to_csv(OUT/'PEAK_HARMONIZATION_AUDIT.csv',index=False)
# HVG
pd.DataFrame([dict(dataset='ISSAAC',HVG=2000,detection_lt_1pct=1558,median_nonzero_per_cell=20.0,test_all_zero_cells=2,status='HIGH_SPARSITY'),
              dict(dataset='ZeppilliSSp',HVG=0,detection_lt_1pct=None,median_nonzero_per_cell=None,test_all_zero_cells=None,status='NOT_BUILT_AUTHOR_LABELS_BLOCKED')]).to_csv(OUT/'HVG_QUALITY_AUDIT.csv',index=False)
# download manifest
files=[
 ('ISSAAC','E-MTAB-11264','mCortex_all_gene_expression_matrix.csv.gz',A_RAW/'mCortex_all_gene_expression_matrix.csv.gz','https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/264/E-MTAB-11264/Files/mCortex_all_gene_expression_matrix.csv.gz','complete','author processed RNA matrix'),
 ('ISSAAC','E-MTAB-11264','mCortex_all_ATAC_peak_matrix.csv.gz',A_RAW/'mCortex_all_ATAC_peak_matrix.csv.gz','https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/264/E-MTAB-11264/Files/mCortex_all_ATAC_peak_matrix.csv.gz','complete','author processed ATAC peak matrix'),
 ('ISSAAC','E-MTAB-11264','Sample_Info.csv.gz',A_RAW/'Sample_Info.csv.gz','https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/264/E-MTAB-11264/Files/Sample_Info.csv.gz','complete','author per-cell metadata/labels/QC'),
 ('ISSAAC','E-MTAB-11264','E-MTAB-11264.sdrf.txt',A_RAW/'E-MTAB-11264.sdrf.txt','https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/264/E-MTAB-11264/Files/E-MTAB-11264.sdrf.txt','complete','sample/data relation'),
 ('ISSAAC','E-MTAB-11264','fragments_all.tsv.gz',A_RAW/'fragments_all.tsv.gz','https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/264/E-MTAB-11264/Files/fragments_all.tsv.gz','available_not_downloaded','3.0 GB; peak matrix used instead'),
 ('ZeppilliSSp','GSE239477','GSE239477_N1_filtered_feature_bc_matrix.h5',B_RAW/'GSE239477_N1_filtered_feature_bc_matrix.h5','https://ftp.ncbi.nlm.nih.gov/geo/series/GSE239nnn/GSE239477/suppl/GSE239477_N1_filtered_feature_bc_matrix.h5','complete','N1 SSp GEX+ATAC'),
 ('ZeppilliSSp','GSE239477','GSE239477_N4_filtered_feature_bc_matrix.h5',B_RAW/'GSE239477_N4_filtered_feature_bc_matrix.h5','https://ftp.ncbi.nlm.nih.gov/geo/series/GSE239nnn/GSE239477/suppl/GSE239477_N4_filtered_feature_bc_matrix.h5','complete','N4 SSp GEX+ATAC'),
 ('ZeppilliSSp','GSE239477','GSE239477_N6_filtered_feature_bc_matrix.h5',B_RAW/'GSE239477_N6_filtered_feature_bc_matrix.h5','https://ftp.ncbi.nlm.nih.gov/geo/series/GSE239nnn/GSE239477/suppl/GSE239477_N6_filtered_feature_bc_matrix.h5','complete','N6 SSp GEX+ATAC'),
 ('ZeppilliSSp','GSE239477','GSE239477_N7_filtered_feature_bc_matrix.h5',B_RAW/'GSE239477_N7_filtered_feature_bc_matrix.h5','https://ftp.ncbi.nlm.nih.gov/geo/series/GSE239nnn/GSE239477/suppl/GSE239477_N7_filtered_feature_bc_matrix.h5','complete','N7 SSp GEX+ATAC')]
dm=[]
for ds,acc,name,path,url,status,note in files:
    dm.append(dict(dataset=ds,accession=acc,file=name,bytes=(path.stat().st_size if path.exists() else None),sha256=(sha(path) if path.exists() else ''),url=url,status=status,note=note))
pd.DataFrame(dm).to_csv(OUT/'DOWNLOAD_MANIFEST.csv',index=False)
# provenance
prov=f'''# Source provenance

## Candidate A: ISSAAC-seq E-MTAB-11264

- Study: https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-11264
- Author code: https://github.com/dbrg77/ISSAAC-seq (HEAD `713eaa492baf11c6cbb8bc68768e59304c20e79f`)
- Selected data: `mCortex` aggregate; 2 batches (`mCortex_rep1` 3,964; `mCortex_rep2` 6,397), 10,361 cells.
- Processed RNA: `mCortex_all_gene_expression_matrix.csv.gz` (10,361 × 32,285 symbols).
- Processed ATAC: `mCortex_all_ATAC_peak_matrix.csv.gz` (169,180 peaks × 10,361 cells).
- Labels/QC: `Sample_Info.csv.gz`; annotation map in `mCortex_cell_types.ipynb`.
- Fragments are available but not downloaded (3.0 GB); peak counts were mapped to a binary any-hit ATAC matrix.
- No non-cortex cell-line or technical-control assays were used.

## Candidate B: Zeppilli SSp GSE239477

- GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE239477
- Author code: https://gitlab.com/fleischmann-lab/papers/zeppilli-et-al-2023 (HEAD `fa38a8e585768e6f0276067c4c538cda4248c2b6`)
- GEO title is the mouse olfactory cortex study; the series contains aPir, pPir and SSp samples. Only the four SSp lab-mouse 10x Multiome libraries were selected: N1, N4, N6, N7.
- N1/N4/N6/N7 H5 are Cell Ranger ARC `filtered_feature_bc_matrix.h5` with both `Gene Expression` (32,285) and `Peaks` (123k–144k per library) in one file.
- GEO SOFT confirms `tissue: primary somatosensory cortex` and `strain: C57Bl/6 mice (lab mice)` for all four.
- Author code notes `N4 looks bad, use with awareness`; N4 is retained and flagged, not silently removed.
- No public per-cell author labels or author final-QC barcode list were found in GEO, the GitLab repository or supplementary files. Labels are therefore not fabricated.
'''
(OUT/'SOURCE_PROVENANCE.md').write_text(prov,encoding='utf-8')
# author QC
aqc=f'''# Author QC audit

## ISSAAC E-MTAB-11264

- Author R script creates a 10,378-cell object from 3,974 (`rep1`) + 6,404 (`rep2`) cells and computes QC metrics; it does **not** apply a doublet/QC subset in the recovered script.
- The processed matrix actually contains **10,361** cells: 3,964 `rep1` + 6,397 `rep2`. The 17-cell difference is not documented in the recovered files; no extra filter is applied here.
- Labels: 23 `RNA_Cluster_Annotation` classes; 100% coverage; `R12 Misc` = **555** cells (author mixed/unresolved class, retained and flagged).
- No `Mis`, `unknown` or doublet label column exists in `Sample_Info.csv.gz`; no cells were removed as doublets.
- RNA medians: 3,748.5 / 5,874.0 counts; 1,663 / 1,968 genes. ATAC medians: 13,355 / 11,670 counts; 6,844 / 5,804 peaks. TSS enrichment medians 3.92 / 3.54.
- Decision: reuse the 10,361 author matrix cohort; no new QC filter.

## Zeppilli GSE239477 SSp

- Only Cell Ranger ARC `filtered_feature_bc_matrix.h5` files are public; no author per-cell metadata, doublet list or final-QC barcode list is in GEO or the GitLab repo.
- All 1,872 recovered cells are Cell Ranger filtered (N1 459, N4 512, N6 494, N7 407); no additional QC filter is applied.
- N4 is flagged by the author code as `looks bad`; retained with awareness.
- Author cell-type/subclass labels are **not publicly available**; no custom clustering is substituted. `AUTHOR_FINAL_QC_NOT_REPRODUCED` and `AUTHOR_LABELS_UNAVAILABLE` apply.
'''
(OUT/'AUTHOR_QC_AUDIT.md').write_text(aqc,encoding='utf-8')
# preflight
pre=f'''# B/F preflight

## ISSAAC

- Cache loads: 10,361 cells; 169,180 peaks; 2,000 HVGs.
- B_S: 169,180 × 2,000 (29,312 nonzero edges); F_train/F_val: 8,806 × 2,000.
- 8-cell minibatch: RNA 8×2,000; F 8×2,000; B 169,180×8; all finite.
- Labels 23/23; `FULL_B2_INTRA_READY=YES`.
- HVG caveat: 1,558/2,000 detected in <1% of train cells; median 20 nonzero HVGs/cell; 2 test all-zero cells.

## Zeppilli SSp

- Counts benchmark loads: 1,872 cells; 32,285 genes; 180,604 canonical peaks; 4 libraries.
- B/F cache **not built**: no public author labels, so this cannot be a formal labeled intra Source. `FULL_B2_INTRA_READY=NO`.
'''
(OUT/'BF_PREFLIGHT.md').write_text(pre,encoding='utf-8')
# final report
final=f'''# Final report — two new mouse sources

## Final table

| Dataset | Raw paired | QC paired | Author label coverage | Classes | Libraries | RNA QC (median) | ATAC QC (median) | Intra Ready |
|---|---:|---:|---:|---:|---:|---|---|---|
| ISSAAC E-MTAB-11264 | 10,361 | 10,361 | 100% | 23 | 2 | 3,748/5,874 counts; 1,663/1,968 genes | 13,355/11,670 counts; 6,844/5,804 peaks | **YES** |
| Zeppilli SSp GSE239477 | 1,872 | 1,872 | **0%** | **0** | 4 | 2,438–18,801 counts | 4,948–70,865 counts | **NO — BLOCKED** |

## Status

- **ISSAAC: READY.** Paired RNA+ATAC, author labels (23), Full B2 Intra cache, B/F and 8-cell preflight all present. HVG sparsity is HIGH (1,558/2,000 <1%, median 20), recorded as a risk.
- **Zeppilli SSp: BLOCKED.** Counts, pairing, peak harmonization and QC are recovered, but no public author cell labels/final-QC barcode list exists. We do not substitute custom clusters; no cache is built.

## Key numbers

- ISSAAC: 10,361 cells; 169,180 peaks; exact union = canonical = 169,180; widths P50/P90/P99/max = 874/918/942/1,610 bp; B 29,312 edges; F 8,806 × 2,000.
- Zeppilli: 1,872 cells; 4 libraries; native peaks 123,138–144,252; exact union 518,227; canonical 180,604; canonical widths P50/P90/P99/max = 930/1,035/1,334/2,642 bp; all native peaks mapped.
- Zeppilli N1/N4/N6/N7 = 459/512/494/407 cells; all H5 contain Gene Expression + Peaks and paired barcodes. N4 author-code caveat retained.

## Blockers

1. Zeppilli author labels absent — cannot build formal labeled intra cache.
2. Zeppilli author final QC not reproduced; Cell Ranger filtered only.
3. ISSAAC HVG sparsity high; cache is usable but model signal is at risk.
'''
(OUT/'FINAL_REPORT.md').write_text(final,encoding='utf-8')
print('wrote reports')
