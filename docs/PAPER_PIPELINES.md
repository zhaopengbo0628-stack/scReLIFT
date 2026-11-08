# Paper pipeline map

Paths below are relative to `paper_snapshot/`. These files preserve original namespaces, absolute paths, hash checks, historical configuration and dataset-specific behavior. They are evidence and expert reproduction entry points; the portable CLI does not automatically rewrite historical files.

| Workflow | Original entry points |
| --- | --- |
| Model, losses, optimizer, resume | `main/core/run_b2_strict_lodo.py` |
| SOURCE harmonization and B_D/F | `main/run/d_mode/prepare_updated_b2_sources.py`; `main/core/regulatory/common/legacy_support/v5d/` |
| PBMC D-mode training/inference | `main/run/d_mode/run_updated_b2_validated.py`; `result_contain/99_pbmc_lodo_dmode_20261024/source/` |
| PBMC raw readers and source rebuild | `main/run/d_mode/unblock_updated_full_b2.py`; `main/run/d_mode/source_rebuild_updated_full_b2.py` |
| PBMC S-mode preparation | `data/code/s_mode/prepare.py`; `main/core/regulatory/s_mode/single_source.py` |
| PBMC S-mode training/scoring | `main/run/s_mode/train.py`, `execute.py`, `summarize.py` |
| BMMC author data audit | `data/run/audit_bmmc.py` |
| BMMC site preparation and scReLIFT | `data/run/s_mode/prepare_bmmc_site_intra.py`; `data/run/s_mode/run_bmmc_intra.py` |
| Comparator experiment adapters | `comparison/run/`; `comparison/configs/` |
| Causal/control ablations | `ablation/result/basic/36_b2_three_causal_ablations_20261003/source/`; `ablation/result/basic/38_b2_complete_controls_20261003/source/`; `ablation/result/basic/task38_corrected_pbmc002/` |
| Fusion / multiple Teacher ablations | `ablation/code/fusion/`; `ablation/code/three_teacher/`; `ablation/run/` |
| Formal AD four-fold rev04 | `result_contain/AD_FOURFOLD_P2_LINEAR_GSE288063_20261028_rev04/source/ad_pipeline.py` and `frozen_protocol.json` in its parent |
| AD preprocessing / raw cohort adapter | `ad_prepare.py`, `ad_new_cohort.py` in the rev04 source directory |
| AD Teacher–Student / RNA AE | `ad_train.py`, `ad_rna.py` in the rev04 source directory |
| AD donor/sample-group P2 probe | `ad_runtime.py`, `ad_evaluate.py` in the rev04 source directory |
| AD biology / final integrity | `ad_biology.py`, `ad_final_check.py`, `ad_verify_bundle.py` in the rev04 source directory |
| AD inherited implementation dependencies | `result_contain/AD_Dmode_4Cohort_LODO_20261026_rev02/queue/ad_fourfold_prepare.py`; `result_contain/AD_Dmode_4Cohort_20261024/queue/source_balanced_selector.py`; its `source_snapshot/` |
| PBMC all-cell model and subtype discovery | `subtype/PBMC_D1_AllCells_Dmode_20261030/source/`; `subtype/PBMC_AllCells_AllFold_CaseSearch_20261030/source/` |
| Subtype DEG / enrichment | `subtype/PBMC_D1_AllCells_Subtype_DEG_20261030/source/` |
| Regulatory edge/stability analysis | `model_dev/regulatory_biology_20261026/`; `regulatory/run_pbmc_biology.py` |
| Motif scan and corrected null | `regulatory/prepare_motif_inputs.py`, `run_fimo_scan.py`, `run_fimo_matched.py`, `run_motif_null_corrected.py`, `run_motif_null_strict.py`, `finalize_motif_correction.py` |

The AD rev04 adapter deliberately reuses selected older functions. Preserve these dependencies without interpreting their obsolete default cohorts as the active AD protocol. Active AD groups are GSE214979, GSE268609, SEAAD_MTG and the joint GSE288063_GSE278576 group. GSE272082 must not be used in new formal AD runs. The joint group must not be split into separate outer folds.

AD uses SOURCE-balanced regulatory weighting without the n_train multiplier, donor/sample-group handling, donor-balanced HVG preparation and a train-stage adapter that skips empty SOURCE validation partitions. Its five cell-type blocks each pool 64 dimensions into a 320-dimensional P2 vector. Imputation, StandardScaler and LogisticRegression fit SOURCE units only; C=0.1, balanced L2 probe, threshold 0.5. GSE268609's 13 technical groups are not 13 independently verified patients.

Source definitions, exact scientific caveats and unresolved historical differences are documented in `docs/protocol_audit/`. Those audit files predate packaging and may retain older display-name conventions. New descriptions use scReLIFT and ECLARE-S.
