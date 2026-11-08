# scReLIFT Experimental Protocol Index

Live-server follow-up: [05_LIVE_DATA_SCAN_20261104.md](05_LIVE_DATA_SCAN_20261104.md) and [machine-readable inventory](live_data_inventory_20261104.json) record the successful 2026-11-04 SSH scan of AD/PBMC/BMMC. Use that addendum for newly verified cohort counts, feature dimensions, SOURCE splits, and historical-path discrepancies. The remote-access limitation below describes the earlier audit only.

Audit snapshot: 2026-11-04. Scope is protocol reconstruction from the local, checksum-verified mirror `E:\my_research\zpb`. The live SSH share was not independently reachable during this audit, so server-only objects and any changes after the mirrored transfer remain outside verification. These documents are an audit record, not Supplementary Methods prose.

## Status vocabulary

- **VERIFIED**: supported by implementation and/or final artifacts in the local mirror.
- **PARTIALLY VERIFIED**: supported by one primary source or an existing audit, without a second independent check.
- **UNRESOLVED**: evidence is missing or inconsistent; do not fill by inference.

## Documents

| File | Scope | Main evidence |
|---|---|---|
| [01_PBMC_Protocol.md](01_PBMC_Protocol.md) | Four paired PBMC sources; scReLIFT-D LODO and scReLIFT-S Intra/Inter; baselines and metrics | `zpb/data/benchmark/PBMC/*/legacy_corrected_refs_v1/manifest.json`; `zpb/data/docs/UPDATED_FOUR_DATASET_INDEPENDENCE_AUDIT.md`; `zpb/main/docs/B2_THREE_GROUP_FINAL_AUDIT.md`; `zpb/main/result/s_mode/intra/62_smode_full_b2_pbmc_scoring_timing_5seed_20261014/manifest.json`; `zpb/result_contain/99_pbmc_lodo_dmode_20261024/metrics/summary_all_methods.csv` |
| [02_BMMC_Protocol.md](02_BMMC_Protocol.md) | GSE194122 data audit and four-Site Intra comparison; historical Inter distinction | `zpb/data/run/audit_bmmc.py`; `zpb/data/run/s_mode/prepare_bmmc_site_intra.py`; `zpb/comparison/result/BMMC_intra/TASK98_FINAL_REPORT.md`; `zpb/result_contain/96_bmmc_multiome_recovery/FINAL_REPORT.md` |
| [03_AD_LODO_Protocol.md](03_AD_LODO_Protocol.md) | Highest-priority formal four-group, four-fold LODO and exploratory downstream analyses | `zpb/disease/ACTIVE_AD_PROTOCOL_rev04.md`; `zpb/result_contain/AD_FOURFOLD_P2_LINEAR_GSE288063_20261028_rev04/frozen_protocol.json`; `.../reports/FINAL_RESULTS.md`; `.../reports/FINAL_INTEGRITY_CHECK.json`; `zpb/result_contain/AD_FOLD2_PSEUDOBULK_3SEED_20261030/FOLD2_PSEUDOBULK_REPORT_ZH.md` |
| [04_Subtype_Discovery_Protocol.md](04_Subtype_Discovery_Protocol.md) | PBMC all-cell candidate subtype search, CD8 TEM illustrative case, comparison and enrichment | `zpb/subtype/PBMC_AllCells_AllFold_CaseSearch_20261030/final_report.md`; `.../source/prepare_fold.py`; `.../source/score_candidates.py`; `.../source/build_case_data.py`; `.../source/case_plot_common.py`; `.../DELIVERY_MANIFEST.json` |

## Current protocol at a glance

| Area | Current finding | Status |
|---|---|---|
| PBMC cohort identity | PBMC-MO-001 through MO-004: D1 10x granulocyte-sorted Multiome; D2 GEO GSE196990 HDa; D3 10x unsorted 10k; D4 10x unsorted 3k. Cell counts 9,347 / 6,317 / 8,105 / 2,413; 36,601-gene aligned RNA dictionary in the audited scReLIFT-D data. | VERIFIED from identity manifests and independence audit |
| PBMC scReLIFT-D | Four-source LODO and three seeds are present in audit/result artifacts. The inspected `B2_THREE_GROUP_FINAL_AUDIT.md` is explicitly `pre_final` three-seed evidence, not a 15-seed final benchmark. | PARTIALLY VERIFIED; distinguish result packages |
| PBMC scReLIFT-S | Five seeds 20020130–20020134; scoring manifest describes 80 frozen-embedding evaluations, 65 timed and 15 missing timing. Current scored outputs are marked `pre_final`. | VERIFIED for this scoring package; timing incomplete |
| BMMC | GSE194122: 69,249 paired cells, 13,431 GEX genes, 116,490 ATAC peaks, 22 labels, 4 Sites, 10 donor IDs. Site Intra comparison: four Sites × four comparator methods × three seeds. | VERIFIED from data audit and Task98 report |
| AD | Formal four groups and four-fold LODO; 3 seeds; 36 model/representation evaluations, with one donor/sample-group-level probe sample per unit. GSE268609's 13 units are technical groups, not 13 independently verified donors. | VERIFIED for frozen rev04 package |
| PBMC subtype search | Four PBMC targets; all candidate cases retained. D1, D3, D4 CD8 TEM examples are exploratory states, not established novel cell types. | VERIFIED from current all-fold report |

## Conflicts and interpretation rules

1. Do not call PBMC scReLIFT-D “three-source only” a model constraint. It describes particular LODO folds; the model itself is not limited to three training sources.
2. PBMC results in the three-group final audit are explicitly a three-seed pre-final run. Do not present them as final 15-seed results.
3. BMMC has 10 distinct donor IDs and 13 Site–donor observations; Site1/Site2/Site4 share donor1 and Site1/Site3 share donor3. The four Site evaluations are not independent-donor or independent-study replications.
4. The formal AD rev04 biology summaries use donor/sample-group mean CP10K-log1p expression with Welch + BH. The 2026-10-30 Fold2 pseudobulk adds raw integer-count summation followed by CPM/log1p and is a separate exploratory analysis. Never merge counts or pathway totals across these protocols.
5. New names: scReLIFT, scReLIFT-D, scReLIFT-S, Stable Regulatory Template, Cell-specific Regulatory Field, Teacher–Student knowledge distillation, MSE, RNA-only Student, Palette-U, ECLARE-S. Preserve historical artifact identifiers in provenance.

## Audit coverage and limitations

The audit directly inspected the four PBMC identity manifests, PBMC independence and training audit reports, the scReLIFT-S scoring manifest/config, BMMC raw-data audit and per-Site result reports, AD frozen protocol/results/integrity/feature-preparation sources, and subtype analysis report/config/source files. The audited records cite at least 10 implementation scripts, 8 protocol/config/manifest files, and 4 dataset identity manifests; this is a bounded audit, not a line-by-line review of every model core or historical run. The transfer manifest reports 15,332 server files transferred and SHA256-verified at transfer time. Current remote state was not checked because SSH was unavailable.

See each protocol for exact unresolved items, evidence paths, and obsolete history. Missing values remain `UNRESOLVED`; no parameter is filled from memory.
