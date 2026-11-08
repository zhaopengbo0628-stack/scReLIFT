# PBMC Cell-State / Subtype Discovery Protocol

Audit snapshot: 2026-11-04. This document records exploratory candidate-state analysis, not a claim that a new cell type has been established.

## 1. Experimental objective

Search cell-type-internal subgroups in PBMC LODO target embeddings, compare scReLIFT/Full B2 with PCA, assess stability, and characterize candidate differential RNA/ATAC signals. CD8 effector memory T cells (CD8 TEM) are one illustrative case among all searched candidates. **VERIFIED** from the current all-fold report.

## 2. Dataset inventory

Input set: four PBMC targets D1–D4, with full target cell counts 9,347, 6,317, 8,105 and 2,413. Candidate taxonomies contain 17, 26, 24 and 23 reference subclasses, respectively. Dataset identities and cell counts are listed in `01_PBMC_Protocol.md`. Current search retains 82 method × reference-class candidate groups, 179 candidate × seed partitions, and 537 full-gene comparisons. CD8 TEM illustrative cases: D1 622 reference cells (330/182 candidate A/B); D3 487 (258/135); D4 159 (90/46). **VERIFIED** from `final_report.md`.

## 3. Original data and processed file paths

- Current search package: `subtype/PBMC_AllCells_AllFold_CaseSearch_20261030/`.
- Input registry: `input/datasets.json`; label/input audit and hashes are in `audit/` and `DELIVERY_MANIFEST.json`.
- Candidate tables: `candidate_evidence_matrix.csv`, `candidate_seed_evidence_matrix.csv` and fold-local `analyses/D{1..4}/`.
- Full B2 and PCA embeddings/figures: `folds/D{1..4}/`.
- Older named case-only analyses remain under `subtype/pbmc_dmode_subtype_analysis_20261028/` and `results/PBMC_Dmode_Subtype_Publication_20261028/`; those are historical sources, not the current all-fold selection protocol.

## 4. Complete data-processing pipeline

1. Load each held-out PBMC target and its source-trained embedding plus author/harmonized reference labels. All-cell count and barcode alignment are audited; D2 has an author-annotation source distinct from 10x cohorts.
2. Generate candidate partitions using the frozen Task99 candidate search rule. For a reference class with `n` cells, `K = min(max(2*n_reference_classes, 30), 50)` (the variable denotes number of reference classes in the source rule). Each candidate cluster must have at least 30 cells and majority purity >=0.60; retain two largest eligible clusters for a reference class as candidate A/B. Only that class's member cells enter A/B; other cells remain in full-target plots and class-vs-rest analyses.
3. Score every candidate and fixed seed, then rank completeness/QC, UMAP comparison and cross-seed stability. All candidates and seeds are retained; the report says no candidates were removed for unfavorable evidence.
4. For selected candidate evidence, calculate differential expression, pathway enrichment and ATAC peak differences; these are same-target post-hoc analyses and are not independent validations.

Implementation references: `source/prepare_fold.py`, `source/score_candidates.py`, `source/build_case_data.py`, `source/collect_and_rank.py`, `source/write_final_report.py`, `source/plot_case_umap.py`, and `source/case_plot_common.py`. Exact final Figure 4 panel file mapping and dashed ellipse drawing call need separate plotting-manifest crosswalk; the current inventory confirms the three views, but exact panel-level input file list is **PARTIALLY VERIFIED**.

## 5. Feature construction and harmonization

RNA inputs use common 36,601-gene identity and fold/source-selected feature panels. Full B2 produces 64-D Student embeddings. PCA comparators use the same fold source gene panel, with source-fit PCA (source scaler fit then transform target) and target-fit PCA full-normalization sensitivity (descriptive, target-adapted). The final report explicitly cautions that the inherited subtype evaluation standardizes embeddings on the target and builds the full target graph/UMAP; therefore it does not prove fully inductive zero-shot separation.

ATAC uses the existing scReLIFT-D candidate template and target peaks for exploratory differential accessibility. Read-count/binary cached values are not to be described as raw fragments. Exact per-case interval test settings should be taken from fold analysis tables/manifests.

## 6. Experimental split and validation protocol

This is a target-side exploratory analysis within PBMC LODO, not a new train/test split of candidate cells for independent confirmation. Candidate assignment and UMAP are conditioned on the target dataset. Three fixed seeds used in the current all-cell analysis are 20020130, 20020131 and 20020132, but the downstream effective paired comparisons differ by candidate and fold. D1/D3/D4 CD8 TEM examples are highlighted; that is not proof of discovery across all PBMC datasets.

## 7. Model input/output and training procedure

Input is frozen scReLIFT cell embeddings and RNA counts; comparator PCA embeddings share cell identities. Output is candidate A/B membership, same-cell 2-D views, molecular comparisons and cross-seed stability evidence. This document does not add a Student decoder or Local-KL loss. Exact upstream Teacher/Student settings are the PBMC scReLIFT-D package-specific settings, not redefined here.

## 8. Evaluation metrics and statistical tests

### Partition and stability

- Cluster-to-reference candidate search: candidate K, minimum cluster size 30, purity >=0.60, two largest valid clusters.
- Cross-seed candidate membership Jaccard and shared-cell partition ARI/NMI are retained. For the paper's three-view illustration no universal Jaccard/ARI/NMI value applies; candidate-specific values are in the all-fold report. Previous case-only report values such as Jaccard 0.8553 / ARI 0.8773 / NMI 0.7850 are historical and must not be substituted for current D1/D3/D4 values.
- Same candidate members are compared across Full B2 and PCA in UMAP silhouette and balanced Purity@15; original standardized embeddings also use KMeans K=2, `n_init=100`. Because candidates originate from Full B2, these are post-selection descriptive recovery comparisons, not unbiased accuracy.

### Differential expression and enrichment

The all-fold report documents full 36,601-gene counts, full-library CP10K/log1p, two-sided asymptotic Mann–Whitney test, log2FC from mean CP10K with 1e-4 pseudocount, and BH over four folds × candidates × seeds × comparisons. DEG threshold FDR <0.05 and |log2FC| >0.25; large-effect marker requires |log2FC| >0.5 and detection in either group >=0.1. Full-ranked GSEA uses all mapped gene symbols and 10,000 permutations; expressed-gene-background GSEA and DEG-ORA are separate sensitivity families. GO BP 2025 and Reactome 2022 GMT versions/hashes are retained. Full-ranked GSEA for selected D1/D3/D4 CD8 TEM cases has no GO or Reactome term passing the global four-fold FDR; do not report sensitivity/ORA signals as if this main-analysis null were positive.

ATAC candidates undergo all-peak testing with fold/candidate/seed BH and a further detection-proportion difference >=0.05. No covariate adjustment for depth is documented; motif scanning background is unavailable. Same-cohort RNA, enrichment and ATAC are correlated evidence, not independent replications.

## 9. Exact parameters and random seeds

| Parameter | Value | Status |
|---|---|---|
| Search folds | D1–D4 target | VERIFIED |
| Seeds | 20020130, 20020131, 20020132 | VERIFIED for current full search |
| Candidate K | `min(max(2 * number_of_reference_classes, 30), 50)` | VERIFIED from report; confirm exact implementation line before code freeze |
| Minimum candidate cluster size | 30 cells | VERIFIED |
| Minimum majority purity | 0.60 | VERIFIED |
| Candidate clusters per reference class | 2 largest eligible | VERIFIED |
| DEG FDR / effect | <0.05 / >0.25 absolute log2FC | VERIFIED |
| Large-effect marker | >0.5 absolute log2FC and detection fraction >=0.1 in either group | VERIFIED |
| UMAP neighbors/min_dist/metric/init | Per embedding manifest | UNRESOLVED here; do not copy another dataset's values |
| Full B2/PCA exact UMAP seeds and color map for Figure 4 | Figure-source crosswalk needed | PARTIALLY VERIFIED |

## 10. Output files and reproducibility information

Current package preserves `DELIVERY_MANIFEST.json`, audit manifests, all fold candidate matrices, input hashes, RNA/ATAC DEG and pathway tables, UMAP/embedding artifacts, plot scripts, and final report. Use full-fold tables, not only the D1 case panel. Figure review artifacts are under `.aris/traces/paper-figure/2026-10-31_run01/`; they support final visual review but do not replace data provenance.

## 11. Verified facts

- Current analysis searches all four target folds and retains negative and PCA-favorable cases.
- CD8 TEM is an illustrative exploratory candidate, not a confirmed novel subtype.
- D1 has stronger marker/ATAC evidence in the report; D3/D4 show higher cross-seed stability in some measures.
- Complete-ranked GSEA is negative for GO and Reactome after global correction in selected CD8 TEM examples.
- Candidate/source selection bias and measured QC/depth associations are explicitly reported.

## 12. Unresolved issues and discrepancies

- Exact Figure 4 plotting script-to-panel input manifest, cell list, palette/color mapping and ellipse coordinates.
- Exact UMAP parameters by method/fold/seed, including whether any seed changes only clustering/UMAP or also preprocessing in every artifact.
- The current paper's selected Figure 4 donor/target/seed needs panel-level provenance; do not assume D1 seed 20020132 solely from the narrative.
- Fully inductive target processing is not established because target-fitted embedding StandardScaler and full-target UMAP/graph are part of the inherited analysis.
- Same-target annotations are used in candidate search; this is not completely label-free discovery.

## 13. Obsolete protocol history

- Earlier single-case CD8 TEM report and its Jaccard/ARI/NMI values are not the current four-fold all-candidate audit.
- The current report supersedes cherry-picked candidate-only summaries for describing the search. Candidate selection remains post-hoc, and all current candidate matrices should be retained.
- Do not describe the work as a discovery across all PBMC datasets or as a proven new cell type.
