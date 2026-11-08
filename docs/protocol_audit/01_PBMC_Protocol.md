# PBMC Experimental Protocol

Audit status: 2026-11-04. Evidence is from the local mirror `E:\my_research\zpb`; server-only live state was not rechecked. Historical names in paths are retained for provenance; this document uses scReLIFT naming.

## 1. Experimental objective

Evaluate paired RNA–ATAC representation learning on four PBMC Multiome datasets under cross-dataset generalization (scReLIFT-D, LODO) and single-source learning (scReLIFT-S, Intra and Inter). Cell-type labels are used for post-hoc clustering evaluation, not as supervised model targets. **VERIFIED** for audited protocols and reports.

## 2. Dataset inventory

| ID | Source identity / accession | Cells | Aligned genes / labels | Modalities and annotation | Donor metadata |
|---|---|---:|---|---|---|
| D1 / PBMC-MO-001 | 10x `pbmc_granulocyte_sorted_10k`, ARC1.0.0 | 9,347 | 36,601 genes; 25 labels | paired GEX + ATAC; label.csv | Donor/physiological status not specified |
| D2 / PBMC-MO-002 | GEO GSE196990, HDa; source files GSE196988 + GSE196987 | 6,317 | 36,601 aligned genes; 26 cell types | paired RNA + ATAC; `predicted.celltype.l2`, paired annotations checked identical | one Healthy sample HDa, female, 36 years |
| D3 / PBMC-MO-003 | 10x `pbmc_unsorted_10k`, ARC2.0.0 | 8,105 | 36,601 genes; 24 labels | paired GEX + ATAC; original and harmonized label files | Not specified |
| D4 / PBMC-MO-004 | 10x `pbmc_unsorted_3k`, ARC2.0.0 | 2,413 | 36,601 genes; 23 labels | paired GEX + ATAC; original and harmonized label files | Not specified |

The audited independence table reports zero exact count duplicates between every source pair and no within-cohort duplicate cells. It reports barcode-string overlap for D1–D3 (120), D1–D4 (25), and D3–D4 (30), while the pairs involving D2 have 0. This is not evidence of cell identity overlap without library-qualified barcodes. **VERIFIED** from `data/docs/UPDATED_FOUR_DATASET_INDEPENDENCE_AUDIT.md`; biological donor independence is **PARTIALLY VERIFIED** because donor metadata is absent for most sources.

Publication citation for the 10x demo datasets: **UNRESOLVED** in local dataset identity records. D2 title and GEO accession are verified as “Durable Alterations of Hematopoiesis and Chromatin post-COVID-19” (GSE196990). Do not invent a citation.

## 3. Original data and processed file paths

Local mirror root: `E:\my_research\zpb\`.

| Source | Raw/author source | Processed or identity records |
|---|---|---|
| D1 | `data/raw/PBMC/PBMC-MO-001/pbmc_granulocyte_sorted_10k_raw_feature_bc_matrix.h5`; peaks BED adjacent | `data/benchmark/PBMC/PBMC-MO-001/legacy_corrected_refs_v1/`; `data/staging/PBMC/legacy_dmode_data_20261004/derived/common/D1/` |
| D2 | `data/raw/PBMC/PBMC-MO-002/`; GEO paired RNA/ATAC RDS and metadata | `data/benchmark/PBMC/PBMC-MO-002/legacy_corrected_refs_v1/`; `data/cache/d_mode/PBMC/legacy_full_b2_corrected/_historical_dependencies/source_inputs/D2/` |
| D3 | `data/raw/PBMC/PBMC-MO-003/pbmc_unsorted_10k_filtered_feature_bc_matrix.h5`, fragments and peak BED adjacent | `data/benchmark/PBMC/PBMC-MO-003/legacy_corrected_refs_v1/`; `data/staging/PBMC/legacy_dmode_data_20261004/derived/common/D3/` |
| D4 | `data/raw/PBMC/PBMC-MO-004/pbmc_unsorted_3k_filtered_feature_bc_matrix.h5`, fragments and peak BED adjacent | `data/benchmark/PBMC/PBMC-MO-004/legacy_corrected_refs_v1/`; `data/staging/PBMC/legacy_dmode_data_20261004/derived/common/D4/` |

Important: the `legacy_corrected_refs_v1/manifest.json` files explicitly state `post_qc_h5ad_present=false` and describe identity/label/split references, not a complete post-QC H5AD package. Use the verified raw and derived paths above; do not cite the directory name as proof of a new QC operation.

## 4. Complete data-processing pipeline

### scReLIFT-D

The audited scReLIFT-D preparation uses the four-source PBMC cohort and fold-specific SOURCE data. See `main/run/d_mode/prepare_updated_b2_sources.py`, `main/run/d_mode/run_updated_b2_validated.py`, `main/core/regulatory/common/` and the fixed historical implementation under `main/core/regulatory/d_mode/` (historical paths retained verbatim; exact lower-level function mapping is **PARTIALLY VERIFIED** because the complete training invocation and all source hashes were not independently replayed in this audit).

| Stage | Protocol evidence | Status |
|---|---|---|
| Source/target folds | Four folds, each PBMC dataset held out once; three remaining datasets train. The archived three-group audit records targets D1–D4 and source-only selection. | VERIFIED for the inspected three-seed pre-final package |
| RNA identity | Common stable gene identity, audited dictionary size 36,601; full scReLIFT-D source files use the matching raw gene universe. | VERIFIED |
| ATAC harmonization | Peak dictionaries and mapping are constructed from SOURCEs; target ATAC is excluded from target inference/evaluation input. Exact common-to-native mapping statistics per fold are not collected into this document. | PARTIALLY VERIFIED |
| Stable Regulatory Template | Source-only reproducibility selection builds `B_D`; Cell-specific Regulatory Field `F` is formed with shared core regulatory operations. The final audit states `B_D/F valid in all four folds`. | VERIFIED for archived audit scope |
| Teacher–Student | Multi-source Teacher uses RNA and regulatory-field inputs; RNA-only Student is distilled and used for RNA-only target embedding. MSE is the approved distillation term; decoder/local-KL claims are not supported for the current scReLIFT specification. | PARTIALLY VERIFIED; exact epochs, batch sizes, dimensions and checkpoint rule for each PBMC package need the package-specific config |
| Seeds | scReLIFT-D audited comparison used 20020130, 20020131, 20020132. | VERIFIED for the explicit `pre_final` audit; do not extend to other runs |

The file `main/docs/B2_THREE_GROUP_FINAL_AUDIT.md` reports 4 folds, 3 seeds, 3 model variants and 36 complete unit manifests, 2,072 artifacts checked, 36/36 leakage audits passed, 108 efficiency rows reconciled, and independent re-scoring with zero metric mismatches. It explicitly labels the run `pre_final_explicit_3seeds`. The comparator summary at `result_contain/99_pbmc_lodo_dmode_20261024/metrics/summary_all_methods.csv` has Palette-U, Multigrate, ECLARE and MultiVI aggregates. It is a separate comparison artifact; aggregate rows must not be attributed to a different checkpoint tier without checking per-unit manifests.

### scReLIFT-S

Source-specific templates and fields are constructed by `main/core/regulatory/s_mode/single_source.py` (`build_bs`, `build_field`, `normalize_counts`, `select_hvg`); orchestration is in `main/run/s_mode/train.py`, `execute.py`, and `summarize.py`. D1–D4 each train one source-specific model. Training-only fitting is reflected in the scored package identity and source hashes.

## 5. Feature construction and harmonization

RNA is represented in a common gene-ID coordinate system. ATAC peaks remain native to their source in scReLIFT-S; scReLIFT-D uses source-derived peak reconciliation and a regulatory graph to form gene-aligned Cell-specific Regulatory Fields. Full operational formulas and exact numerical feature dimensions should be cited from the matching frozen config/manifest for a specific run. The four PBMC per-source identity manifests say gene counts are 36,601 for the scReLIFT-D aligned dictionaries, not necessarily the number of selected HVGs in every run.

For scReLIFT-D stable edge selector thresholds and scReLIFT-S HVG criteria, use the exact run manifest rather than infer from another dataset. **UNRESOLVED** here: one common final PBMC config with all selected-feature counts, graph radius, edge thresholds and run-specific normalization values was not located among the selected local summaries.

## 6. Experimental split and validation protocol

| Setting | Split / evaluation |
|---|---|
| scReLIFT-D LODO | Four target folds; target dataset wholly held out from encoder/template fitting; three SOURCE datasets train. |
| scReLIFT-S Intra | Train one source model, then evaluate on the same source using frozen embeddings and the held-out test cells recorded in each setting. |
| scReLIFT-S Inter | Train on one source and apply its frozen RNA-only Student to another dataset; no target fitting. The inspected evaluation tree includes source-to-target combinations, but the exact count of retained comparisons varies with available mapped labels. |

The five-seed scReLIFT-S scoring manifest expects 80 evaluations, completed 80; timing is present for 65 and absent for 15, and the end-to-end historical workflow was not timed because checkpoints were reused. Run tier is `pre_final`. Exact split counts for each of the 80 combinations are **UNRESOLVED** in the summary manifest; consult their per-evaluation manifests.

## 7. Model input/output and training procedure

Input: paired RNA and ATAC on SOURCE training cells; target representation inference uses RNA only. Output: cell embedding for post-hoc clustering. scReLIFT-D has a shared multi-source Teacher and RNA-only Student; scReLIFT-S has one-source Teacher/Student. Stable Regulatory Template and Cell-specific Regulatory Field terminology is used. The exact embedding size, epochs, batch size, optimizer, loss weights and checkpoint selection are **UNRESOLVED** for a unified PBMC protocol because the audited reports point to distinct run packages and not one common config.

## 8. Evaluation metrics and statistical tests

The scReLIFT-D comparison summary reports ARI, NMI, AMI, ACC, Macro-F1, Balanced Accuracy, Weighted-F1, Purity@15, silhouette, time, parameter counts, peak GPU memory, inference time and GPU. The scReLIFT-S report is a frozen-embedding scoring/timing audit; metrics and timing are stored per unit. Clustering labels are post-hoc. The exact KMeans parameters for each PBMC run are **UNRESOLVED** from the selected summary evidence; do not import BMMC's KMeans configuration into PBMC.

## 9. Exact parameters and random seeds

| Parameter | Value | Evidence / status |
|---|---|---|
| scReLIFT-D seeds | 20020130, 20020131, 20020132 | `B2_THREE_GROUP_FINAL_AUDIT.md`; VERIFIED for that run only |
| scReLIFT-S scoring seeds | 20020130–20020134 | `62_.../manifest.json`; VERIFIED |
| scReLIFT-S scoring evaluations | 80 complete, 65 timed, 15 timing gaps | Same manifest; VERIFIED |
| RNA aligned input | 36,601 genes | Dataset independence audit; VERIFIED for raw dictionary, not selected feature count |
| Other model/training settings | Per-run config required | UNRESOLVED in consolidated protocol |

## 10. Output files and reproducibility information

- Data identity and hash manifests: `data/benchmark/PBMC/PBMC-MO-00{1,2,3,4}/legacy_corrected_refs_v1/`.
- scReLIFT-D final-audit entry: `main/docs/B2_THREE_GROUP_FINAL_AUDIT.md`; scores: `result_contain/99_pbmc_lodo_dmode_20261024/metrics/summary_all_methods.csv`.
- scReLIFT-S scoring: `main/result/s_mode/intra/62_smode_full_b2_pbmc_scoring_timing_5seed_20261014/manifest.json` and `source/config.json`.
- scReLIFT-S corrected run: `main/result/s_mode/intra/45_smode_full_b2_corrected_pbmc_20261012/`.

## 11. Verified facts

- Four cohort identities and cell totals in the dataset table; exact count duplicate audit passes.
- D2 is GEO-derived Healthy HDa, not the old 10x sorted dataset; the legacy directory name is invalid and retained for compatibility.
- scReLIFT-D audit is explicitly pre-final and three-seed.
- scReLIFT-S scoring reused frozen embeddings, completed its 80 expected evaluations, and has incomplete timing coverage.

## 12. Unresolved issues and discrepancies

- Exact final publication citation for 10x PBMC-MO-001/003/004 and per-cell donor identity for these sources.
- Unified final PBMC configuration tying exact epoch/optimizer/dimension/feature selection and checkpoint rule to each results table.
- Exact scReLIFT-D run tier relationship between the three-seed main-method audit and the `99_pbmc_lodo_dmode_20261024` comparator summary.
- Per-setting scReLIFT-S split counts and whether each reported result is Intra or Inter in the 80-row evaluation inventory.
- Nature of barcode-only cross-dataset overlaps; count matrices are not duplicates, but barcode strings alone are ambiguous.
- Full D/S baseline configuration and model-specific runtime/GPU coverage for every row.

## 13. Obsolete protocol history

- Old D1 TEA-seq assignment is incorrect; current D1 is the 10x granulocyte-sorted Multiome. D2 is GSE196990 HDa.
- `legacy_corrected_refs_v1` naming is historical compatibility metadata; its own manifests say it is not a complete post-QC H5AD package.
- Old/early PBMC evaluations and the three-seed pre-final audit must not be represented as a final 15-seed benchmark.
- Use `Palette-U` for the unsupervised Palette setting. The scReLIFT-D comparison file's historical ECLARE identifier maps to display name ECLARE-S only where the method actually uses that configuration; preserve the source identifier in provenance.
