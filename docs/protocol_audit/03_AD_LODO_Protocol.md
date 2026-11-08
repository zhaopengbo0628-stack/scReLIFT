# AD Four-Group LODO Protocol

Audit snapshot: 2026-11-04. This is the highest-priority protocol. Current formal result package is rev04. The local mirror is checksum-verified as of transfer, but the live SSH endpoint could not be rechecked.

## 1. Experimental objective

Evaluate frozen scReLIFT representations for donor/sample-group-level AD versus control classification under four-group leave-one-dataset-out (LODO). Compare Full B2 historical representation identifier (display as scReLIFT), RNA-only Encoder, and PCA-RNA64 using the same P2 aggregation and Linear Probe. Target disease labels are opened only after predictions are frozen for scoring. **VERIFIED** from `frozen_protocol.json`, `FINAL_RESULTS.md` and integrity outputs.

## 2. Dataset inventory

| Group | Cohort | Labels / units | Paired data / QC | Status and caveats |
|---|---|---|---|---|
| G1 | GSE214979 | 15 donors; exact AD/control distribution **UNRESOLVED here** | Paired RNA/ATAC cohort; source manifest and fold-specific metadata | Label split should be confirmed from `AD_FORMAL_DONOR_MANIFEST.csv` before methods prose |
| G2 | GSE268609 | 13 technical groups: 8 AD, 5 HA; 51,246 cells in frozen protocol | Multimodal cohort | Do not call 13 independently verified biological donors |
| G3 | SEA-AD MTG | 24 retained donors: 21 positive (Intermediate/High ADNC) and 3 Not AD; ADNC-Low excluded | Paired subset | Endpoint is AD neuropathological change, not equivalent to clinical AD diagnosis |
| G4 | GSE288063 + GSE278576 | 10 donors: 5 AD + 5 Control; 11 libraries; 54,999 author-QC cells | Paired RNA/ATAC; author Library+Barcode subset | Combined as one LODO group; GSE288063 AD and GSE278576 controls; ad22's two libraries pooled as one donor |

Current group order/folds is confirmed in `frozen_protocol.json`. Exact G1 donor label counts, G2 group-to-biological-donor map, and full G4 donor IDs are retained in benchmark manifests but not reproduced here without a second independent cross-check. **PARTIALLY VERIFIED**.

## 3. Original data and processed file paths

- Frozen output root: `result_contain/AD_FOURFOLD_P2_LINEAR_GSE288063_20261028_rev04/`.
- Formal registry: `data/benchmark/AD_FORMAL_DATASET_REGISTRY.csv`.
- Donor mapping: `data/benchmark/AD_FORMAL_DONOR_MANIFEST.csv`.
- Formal fold config: `data/benchmark/AD_FOURFOLD_LODO_FORMAL.yaml`.
- G4 audit/author files: `result_contain/AD_LODO_4FOLD_RECOVERY_20261027/GSE288063_AUDIT/recheck_20261029/` and `data/raw/AD/GSE288063_GSE278576/`.
- Fold-specific prepared RNA, ATAC and metadata: `result_contain/AD_FOURFOLD_P2_LINEAR_GSE288063_20261028_rev04/preprocessing/folds/fold{1..4}/`.
- Raw source data are preserved under `data/raw/AD/`; the local mirror omits some older benchmark files by design. Consult transfer inventory for inclusion reasons.

## 4. Complete data-processing pipeline

### Cohort verification and held-out protocol

`disease/ACTIVE_AD_PROTOCOL_rev04.md` and `frozen_protocol.json` fix groups and fold direction. Fold targets: F1 GSE214979; F2 GSE268609; F3 SEAAD_MTG; F4 GSE288063_GSE278576. Each remaining three groups are SOURCEs for encoder/template and Probe. GSE272082 is retired and forbidden.

For new G4 data, the protocol requires all 11 H5 libraries to pass `verification_report_rev04.json` before preprocessing; retain only author-QC Library+Barcode cells; harmonize by Library + Barcode; exclude raw control droplets. The transfer contains a pretraining-reader check and replacement tests. This is a fixed cohort selection, not target-performance selection.

### SOURCE-only feature processing

Implementation: `result_contain/AD_FOURFOLD_P2_LINEAR_GSE288063_20261028_rev04/preprocessing/provenance/ad_fourfold_prepare.py`, plus linked historical core `main/run/d_mode/prepare_updated_b2_sources.py`, `main/core/regulatory/d_mode/dmode_reliable_selector.py`, `main/core/regulatory/common/` and the frozen source snapshot under `result_contain/AD_Dmode_4Cohort_20261024/source_snapshot/`.

1. RNA genes are harmonized by stable gene ID to the shared 36,601-gene dictionary.
2. For each fold, each SOURCE independently calculates HVG statistics using only its training donor cells, with equal total donor weighting. Top 2,000 per SOURCE are intersected. If the intersection has fewer than 256 genes, use the frozen equal-mean-rank fallback selecting 2,000. Fold target does not participate.
3. Full shared RNA gene matrix is normalized to CP10K and log1p before selected features are taken.
4. ATAC is harmonized by peak interval overlap mapping against a SOURCE-native interval union; TARGET peaks do not define the union. The existing scReLIFT-D Stable Regulatory Template selector (`B_D`) and Cell-specific Regulatory Field (`F`) equations are retained.
5. Important limitation from `DATA_ADAPTER_SCOPE.md`: canonical peak availability excludes peaks absent from the whole SOURCE, but existing `source_stability` and `fit_dmode_field` do not consume per-library peak availability. Within a SOURCE, a peak absent from a technical library is represented as structural zero. Do not claim library-aware missingness correction.
6. Jaccard is used in separate regulatory reproducibility analysis, not as the RNA–ATAC harmonization method.
7. Normalization, feature selection, peak union/mapping, `B_D`, `F`, scaler, imputation and model selection are SOURCE-only. Target data are transformed with frozen source state.

The preparation code directly records fold manifests, feature lists, selection audit and dependency hashes. Feature dimensions can vary by fold due to the frozen intersection/fallback and peak union; report the per-fold manifest values, not one universal number. Exact per-fold selected-gene / canonical-peak counts are **UNRESOLVED in this narrative** and available in `preprocessing/folds/fold*/COMPLETE.json` and `selector_manifest.json`.

## 5. Feature construction and harmonization

The Teacher receives harmonized RNA and gene-aligned Cell-specific Regulatory Field. Native peaks map to the fold SOURCE peak dictionary using genomic interval overlap; the source template maps regulatory peaks to candidate gene links in a GTF-derived TSS graph (radius and cap are implementation parameters in the frozen selector). `B_D` encodes stable shared regulatory links; `F` encodes per-cell regulatory activity projected into RNA gene space. Exact TSS radius, maximum candidate genes per peak, correlation threshold and edge selection settings must be copied from the matching frozen `selector_manifest.json`; they are not inferred here.

**VERIFIED**: RNA identity harmonization and source-only top-2,000 intersection/fallback, and the separation of Jaccard analysis from modality alignment. **PARTIALLY VERIFIED**: precise low-level formula parameter values pending direct extraction of every fold selector manifest.

## 6. Experimental split and validation protocol

| Fold | Held-out TARGET | Three SOURCEs | Reliable Probe units |
|---|---|---|---:|
| 1 | GSE214979 | GSE268609, SEAAD_MTG, GSE288063_GSE278576 | 47 |
| 2 | GSE268609 | GSE214979, SEAAD_MTG, GSE288063_GSE278576 | 49 |
| 3 | SEAAD_MTG | GSE214979, GSE268609, GSE288063_GSE278576 | 38 |
| 4 | GSE288063_GSE278576 | GSE214979, GSE268609, SEAAD_MTG | 52 |

One vector and one label per qualified donor/sample-group; do not repeat labels by cell count. GSE268609's “donor-level” prediction unit is technically a sample group and biological independence remains unverified. Unknown or non-P2 cells remain in annotation audit and are excluded from the five P2 blocks. New cohort uses author class/subclass mapping; no disease-label-driven mapping or transfer is used.

## 7. Model input/output and training procedure

For each fold × seed, train a three-SOURCE Teacher, select checkpoint using SOURCE validation only, initialize/distill an RNA-only Student with MSE, then freeze it. Full B2/scReLIFT representation and RNA-only AE Encoder each output 64-dimensional cell embeddings. No Student decoder and no Local-KL objective are part of the frozen protocol. Exact Teacher/Student layer widths and training hyperparameters must be read from the per-fold, per-seed `models/fold*/seed_*/{teacher,student}/manifest.json` and configs; this audit verified package completeness but did not transcribe all training parameter values. **PARTIALLY VERIFIED**.

### Donor P2 and classifier

For each donor/sample-group, average 64-D embeddings separately in the fixed order Excitatory, Inhibitory, Astrocytes, Oligodendrocytes, OPC. Concatenate to 320-D. No region reweighting or cell-count weighting across groups. Missing feature blocks are imputed by an equal-donor per-feature mean fitted on Probe-eligible SOURCE units only; then `StandardScaler` fits on SOURCE vectors only. TARGET uses the frozen mapping, imputer and scaler.

Linear Probe: `sklearn.linear_model.LogisticRegression`, L2, `C=0.1`, `solver=lbfgs`, `max_iter=1000`, `tol=1e-4`, `class_weight=balanced`, `fit_intercept=True`. Fixed probability cutoff = 0.5; no target-based threshold or hyperparameter search.

PCA-RNA64: SOURCE-only RNA scaler; equal-SOURCE subsample capped at 20,000 training cells per SOURCE; randomized sklearn PCA with 64 components and `random_state=seed`; same P2 aggregation, imputation/scaler, and Logistic Regression. Two-dimensional PCA is visualization only.

## 8. Evaluation metrics and statistical tests

Per fold and seed: AUROC, AUPRC, balanced accuracy, Macro-F1, sensitivity, specificity and accuracy. Fold/seed predictions and probabilities are retained. Formal aggregate: first equal-weight four-fold macro-average within each seed; then report mean ± sample SD (`ddof=1`) over three seeds. This is seed variability, not a donor-generalization confidence interval. No target metrics are used for selection.

The formal result table in `FINAL_RESULTS.md`:

| Method | AUROC | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| scReLIFT (historical `Full_B2`) | 0.672 ± 0.032 | 0.583 ± 0.012 | 0.516 ± 0.017 |
| RNA-only Encoder | 0.717 ± 0.046 | 0.642 ± 0.014 | 0.561 ± 0.023 |
| PCA-RNA64 | 0.633 ± 0.033 | 0.592 ± 0.025 | 0.513 ± 0.031 |

For GSE268609 as target, the report gives scReLIFT 0.700 ± 0.050 / 0.612 ± 0.082 / 0.595 ± 0.101; RNA-only 0.608 ± 0.038 / 0.537 ± 0.000 / 0.511 ± 0.000; PCA-RNA64 0.442 ± 0.058 / 0.388 ± 0.000 / 0.381 ± 0.000. These are cross-checks from formal report, not a license to replace figure data.

## 9. Exact parameters and random seeds

| Parameter | Value | Evidence |
|---|---|---|
| Seeds | 20020130, 20020131, 20020132 | `frozen_protocol.json`, verified |
| Fold count | 4 | protocol and 36 evaluation artifacts, verified |
| Per-cell embedding | 64 dimensions | final report and manifests, verified |
| P2 unit vector | 320 dimensions, five fixed classes | freeze and saved probe feature files, verified |
| Logistic Regression | L2; C 0.1; lbfgs; max_iter 1000; tol 1e-4; balanced; intercept | freeze and serialized estimators, verified |
| Threshold | 0.5 | freeze and probability integrity audit, verified |
| HVGs | per-SOURCE top 2,000; intersection if >=256 else equal-rank top 2,000 fallback | frozen code and fold audits, verified |
| Shared RNA dictionary | 36,601 genes | manifests, verified |
| Teacher/Student optimizer, learning rates, batch, epochs, layer widths, checkpoint patience | Read per run | exact global extraction not completed; PARTIALLY VERIFIED / cite run manifest |

## 10. Output files and reproducibility information

- `frozen_protocol.json`, `reports/FINAL_RESULTS.md`, `reports/FINAL_INTEGRITY_CHECK.json`, `reports/FINAL_DEPENDENCY_MANIFEST.json`, `metrics_per_fold_seed.csv`, `metrics_summary.csv`.
- For each of 36 fold × seed × method evaluations: `probe.joblib`, `predictions.csv`, `metrics.json`, `donor_p2_features.npz`, `probe_training_units.csv`, `p2_coverage.csv`.
- 12 fold × seed encoder training combinations, 72 `.pt` model files as reported by cleanup audit (teacher/student and RNA AE artifacts; inspect exact count by model manifest before citing).
- software versions: `disease/results/zpb/AD_FOURFOLD_P2_LINEAR_GSE288063_20261028_rev04/software_versions.json`: Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3, scikit-learn 1.7.2, PyTorch 2.4.1+cu121, h5py 3.16.0, AnnData 0.11.4, Scanpy 1.11.5, joblib 1.5.3; NVIDIA A30.

## 11. Verified facts

- Current formal cohort is exactly four groups and four LODO folds; GSE288063+GSE278576 stays combined.
- 36 evaluations (4 folds × 3 seeds × 3 representations) completed, with no seed selection or hyperparameter changes in comparison scope.
- Frozen Probe, cutoff, P2, source-only preprocessing and no target fitting have integrity evidence.
- GSE268609 comprises 13 technical groups, 8 AD + 5 HA, not verified independent biological donors.
- SEA-AD binary endpoint excludes ADNC-Low and combines Intermediate/High versus Not AD.

## 12. Unresolved issues and discrepancies

- GSE214979 exact label distribution and donor-level inclusion details should be transcribed from the formal donor manifest.
- Full sample-to-biological-donor verification for GSE268609 is not resolved; keep the technical-group unit label.
- G4 hidden biological donor overlap cannot be ruled out by donor-ID string audit alone; G4 labels are linked to series/source.
- Exact per-fold selected gene/peak dimensions and full Teacher/Student hyperparameters are in manifests but not copied into this summary; this is intentionally not inferred.
- Figure 5 provenance: `disease/results/zpb/AD_LODO_BIOLOGY_VALIDATION_20261029/07_publication_figures/FIGURE_DATA_AUDIT.md` audits the local publication-figure directory, whose visible outputs are named Figure_A–Figure_E. The available audit does not establish which manuscript Figure 5 panel maps to which exact input CSV, fold, or seed. Therefore the manuscript Figure 5 mapping is **UNRESOLVED** until the manuscript-panel crosswalk and plotting inputs are traced. Do not substitute the formal mean or a selected fold without verifying that mapping.
- Current formal rev04 biology uses cell-type/donor mean CP10K-log1p followed by Welch + BH, not raw-count pseudobulk.

### Formal downstream expression analysis (rev04)

Frozen prediction labels define predicted AD versus predicted Control groups, then donor/sample-group mean full-gene CP10K-log1p values are compared using Welch t-test and Benjamini–Hochberg. At least two units per predicted group are required; otherwise `NOT_EVALUABLE`. Effect is log2 ratio of mean donor-level CP10K with pseudocount 1. Test genes are detected in at least two target units and are not selected using true diagnosis. Enrichment uses GO BP 2025 and Reactome 2022, hypergeometric ORA + BH, background of tested genes mapped to each database; query FDR < 0.05 and |log2FC| > 0.25, up to 200 genes per direction; term size 10–2,000, FDR < 0.05, overlap >=3. This is exploratory and selection-dependent because predictions come from RNA.

### Later raw-count pseudobulk exploration (separate protocol)

`AD_FOLD2_PSEUDOBULK_3SEED_20261030/FOLD2_PSEUDOBULK_REPORT_ZH.md` is later exploratory analysis, not a replacement for frozen rev04 results. It sums raw integer counts by technical group × P2 cell type (minimum 20 cells), applies library-size CPM then log1p, Welch and BH. Each predicted group needs at least two eligible units. GO BP 2025 / Reactome 2022 use actual tested database-mapped genes as background; significant input FDR <0.05 and |log2FC| >=0.25; up to 200 per direction; term size 10–2,000; pathway FDR <0.05 and overlap >=3. Fold2 seed 20020132 Astrocytes has 542 DEGs, 11 GO, 31 Reactome terms, all reported Control-up. The analysis explicitly states it is exploratory and prediction-defined. Do not combine these with rev04 CP10K/log1p counts.

## 13. Obsolete protocol history

- Three-group / three-fold AD LODO rev03 is superseded by rev04 four-group/four-fold.
- GSE272082 and its old benchmark are retired; do not use in training, testing, DEG or enrichment.
- Old Group 4 cohorts must not replace the combined GSE288063 + GSE278576 group.
- Earlier cell-level CP10K/log1p downstream analysis and later raw-count pseudobulk are separate estimands; both can remain archived, neither overwrites the other.
- Do not claim model superiority from the selected Fold2 pseudobulk example or pathway count.
