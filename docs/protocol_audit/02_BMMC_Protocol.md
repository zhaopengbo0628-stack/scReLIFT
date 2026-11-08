# BMMC Experimental Protocol

Audit snapshot: 2026-11-04. BMMC is documented independently from PBMC and AD. Local mirror evidence is available; live SSH state was not rechecked.

## 1. Experimental objective

Audit the paired multiome dataset GSE194122 and document (a) scReLIFT per-Site Intra evaluation and (b) the current comparison-method Task98 results. Keep historical Site-to-Site zero-shot experiments separate. **VERIFIED** from the data audit, preparation source and result reports.

## 2. Dataset inventory

| Item | Verified value |
|---|---:|
| Dataset | GEO GSE194122, OpenProblems NeurIPS 2021 BMMC processed Multiome |
| Paired cells | 69,249 |
| AnnData shape | 69,249 × 129,921 |
| GEX genes | 13,431 |
| ATAC peaks | 116,490 |
| `cell_type` classes | 22 overall; no missing/unknown/doublet-like labels |
| Sites | 4: Site1–Site4 |
| Unique `DonorID`s | 10 |
| Site–donor observations | 13 |
| Raw counts | `layers/counts`, CSR, non-negative integer; no RNA/ATAC all-zero cells |

Per Site: Site1 17,243 cells / 3 donors / 22 classes; Site2 15,226 / 3 / 20; Site3 14,556 / 4 / 21; Site4 22,224 / 3 / 20. Donor1 occurs in Site1, Site2 and Site4; donor3 occurs in Site1 and Site3. Thus Site results are not independent donor replications.

## 3. Original data and processed file paths

- H5AD: `data/raw/BMMC/GSE194122/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad`.
- Data integrity audit: `result_contain/96_bmmc_multiome_recovery/MODALITY_AUDIT.json`, `SITE_CELL_COUNTS.csv`, `SITE_CELLTYPE_DISTRIBUTION.csv`, `SITE_DONOR_DISTRIBUTION.csv`, `DONOR_OVERLAP_MATRIX.csv`, `FINAL_REPORT.md`.
- scReLIFT Site caches: `data/cache/s_mode/BMMC/bmmc_site_intra_v1_20261024/{BMMC_Site1,...,BMMC_Site4}/` (Linux path recorded in cache manifests).
- Comparison results: `comparison/result/BMMC_intra/`.

Official download manifest reports the compressed GEO source, 2,917,117,242 bytes, SHA256 `53b516e7a35518e7678f90a2f27867ae2a0054dba6ac0c52fafb67b5a80f415a`; decompressed H5AD 3,116,868,580 bytes. The local transfer manifest marks copied files checksum-verified at transfer time.

## 4. Complete data-processing pipeline

### Data audit

`data/run/audit_bmmc.py` loads the processed object and uses `obs`, `var.feature_types`, `var.gene_id`, and `layers/counts` to calculate modality and cell metadata. It reuses the author processed cohort and its QC metadata; it does not re-filter cells, recluster, or change labels. **VERIFIED** from implementation and its generated reports.

### scReLIFT-S Site Intra

`data/run/s_mode/prepare_bmmc_site_intra.py` defines `load()` and `prepare(site, ...)`. It extracts the Site rows, stratifies author cell-type labels into train/validation/test, uses `random_state=20020130`, estimates the top 2,000 HVGs on Site-train normalized RNA, scales using training RNA statistics, binarizes ATAC, builds a site-specific graph and `B_S` from training cells, then creates fields for train and validation cells. The test ATAC is excluded from field construction. RNA/ATAC matrices and barcode/label arrays are cached with manifests.

The script uses local Linux absolute root `/home/u202522802045/share/zpb`; it is not Windows-ready without path/environment adaptation. Exact downstream scReLIFT training settings and final metric bundle for the four Sites are **UNRESOLVED** in the inspected result summary; Task98 below is a separate baseline comparison.

### Task98 baseline comparison

`comparison/result/BMMC_intra/TASK98_FINAL_REPORT.md` reports 4 methods × 4 Sites × 3 seeds = 48 complete units, with no failed units. scMomer-S was excluded by user and must not be counted. `finalize_bmmc_intra.py` and `bmmc_s_mode_intra.py` are result orchestration/evaluation entry points.

## 5. Feature construction and harmonization

For scReLIFT Site caches, source-specific Site train data selects 2,000 RNA HVGs from full-library 10k-normalized log1p expression; train mean/std are saved. ATAC is author peak-space binary data; regulatory graph uses GRCh38 GENCODE v32 primary assembly GTF; `build_bs` and `build_field` come from `main/core/regulatory/s_mode/single_source.py`. The resulting Stable Regulatory Template is site-specific. These implementation details are direct source evidence.

For comparison methods, Task98 reports 2,000 train-only RNA HVGs and full-library 10k normalization; ATAC uses author peaks, binary counts for VAE methods, and train-fitted TF-IDF for Palette-U/ECLARE-S. Test ATAC is not used. **VERIFIED** from Task98 report; method-specific fitting internals remain in each runner.

## 6. Experimental split and validation protocol

The scReLIFT preparation script uses stratified 70/15/15 train/validation/test split per Site, fixed split seed 20020130. Task98 reuses Task96/97 raw counts and fixed Site splits; report counts are:

| Site | Train | Validation | Test |
|---|---:|---:|---:|
| Site1 | 12,070 | 2,586 | 2,587 |
| Site2 | 10,658 | 2,284 | 2,284 |
| Site3 | 10,189 | 2,183 | 2,184 |
| Site4 | 15,556 | 3,334 | 3,334 |

Labels are used for stratified splitting and post-hoc scoring; not for representation model fitting. Four Site analyses are within-study evaluations. Donor overlap prevents calling them four independent cohorts.

Historical Inter setting: one Site as source and other Sites as targets was explored in earlier runs. The current Task98 final report is explicitly “intra-dataset benchmark”; Inter is not part of its 48 units. Whether any older Inter result is retained in the current manuscript is **UNRESOLVED**; do not mix it with Task98.

## 7. Model input/output and training procedure

scReLIFT input: paired RNA plus train-derived regulatory field for train/validation; target/test embedding inference uses RNA only. The Student output is a cell-level latent embedding. Exact BMMC Student dimension, optimizer, epoch, batch, distillation weight, and checkpoint selection are **UNRESOLVED** from the inspected BMMC run summary and should be taken from the exact per-run manifests before writing manuscript methods.

Task98 methods: Multigrate latent 64, 200 epochs, batch 256, lr 5e-4; Cobolt latent 64, 100 epochs, batch 128, lr 5e-3 with one Site1 seed0 retry at 1e-3 after nonfinite gradient; Palette-U latent 64; ECLARE-S uses 3 independent source CLIP teachers and ensemble KL/OT Student, 100 epochs, batch 128, lr 1e-3, latent 64. These values describe comparator runs only.

## 8. Evaluation metrics and statistical tests

Task98 evaluates frozen test embeddings after `StandardScaler`; KMeans uses k-means++, `n_init=100`, `random_state=20020130`, `max_iter=1000`, `tol=1e-4`, algorithm `lloyd`. Hungarian matching is post-hoc for ACC/Macro-F1. Metrics: ARI, NMI, AMI, ACC, Macro-F1, Balanced Accuracy, Weighted-F1, Purity@15, silhouette. Results are reported by Site and equal-weight over 12 Site-seed units per method, mean ± sample SD over seeds as documented in report. Do not pool cells across sites as if independent.

## 9. Exact parameters and random seeds

| Item | Value | Status |
|---|---|---|
| Dataset input | 69,249 × 129,921; 13,431 GEX / 116,490 ATAC | VERIFIED |
| Intra split | 70/15/15, stratified by cell_type, seed 20020130 | VERIFIED in code/report |
| scReLIFT HVGs | 2,000, train-only, full-library log1p; variance/mean ranking formula in manifest | VERIFIED |
| BMMC comparator seeds | 0, 1, 2 | VERIFIED |
| BMMC comparator units | 48/48 complete | VERIFIED |
| scReLIFT training values | Need exact cache-linked run manifests | UNRESOLVED |

## 10. Output files and reproducibility information

Data: `result_contain/96_bmmc_multiome_recovery/`. Comparison: `comparison/result/BMMC_intra/results_raw.csv`, `results_summary.csv`, `results_equal_weight.csv`, `TASK98_FINAL_REPORT.md`, plus method/Site/seed manifests and frozen embeddings. scReLIFT preparation source hashes dependencies into each cache manifest.

## 11. Verified facts

- GSE194122 processed paired data has 69,249 cells, 22 classes, and four Sites.
- There are 10 unique donors and 13 Site–donor observations; donor overlap is explicit.
- Author QC was reused; no local cell removal or reannotation is described by the audit.
- Task98 is a 48-unit Intra comparison across four comparator methods and three seeds; scMomer-S is excluded.

## 12. Unresolved issues and discrepancies

- Current paper inclusion of legacy Site-to-Site Inter experiments.
- Exact scReLIFT BMMC checkpoint/embedding/training config and final method-specific score package.
- Some sites have 20–22 represented cell classes rather than all 22; evaluation class policy by setting should be read from per-unit metrics if needed.
- Site-level split is cell-stratified, not donor-held-out; donor overlap can leak donor-specific structure across Site comparisons.

## 13. Obsolete protocol history

- Historical one-source-to-three-target Site Inter results are separate from current four-Site Intra Task98.
- scMomer-S was deleted/excluded by user; no result should be included.
- Do not label four Sites as four independent studies or donors.
