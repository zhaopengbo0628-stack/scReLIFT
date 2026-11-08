# scReLIFT

Public source repository: [zhaopengbo0628-stack/scReLIFT](https://github.com/zhaopengbo0628-stack/scReLIFT).

**Single-cell Regulatory Latent Information Fusion and Transfer for RNA-only Analysis**

scReLIFT learns a stable signed peak–gene regulatory template from paired SOURCE RNA–ATAC data, builds cell-specific regulatory fields, fits a multimodal Teacher and distills its fused representation into a frozen RNA-only Student. TARGET inference uses RNA alone.

This repository contains a portable Python interface and checksum-traceable source snapshots from the paper experiments. The original model/training file is preserved; the installable package extracts its computational definitions verbatim. Dataset-specific pipelines are retained in `paper_snapshot/`. See [reproduction scope](docs/REPRODUCIBILITY.md) for executable workflows and full paper reruns.

## Layout

```text
src/screlift/       model, losses, training, preprocessing, inference, evaluation
configs/           active seed policy and portable input examples
examples/          command-line workflow
tests/             synthetic integration and scientific-invariant checks
paper_snapshot/    original PBMC/BMMC/AD/subtype/regulatory sources and configurations
docs/              data contract, paper pipeline map, reproduction scope
provenance/        source SHA256, extraction record, verification reports
environment.yml    observed CUDA environment specification
LICENSE            MIT
```

## Install

Linux with NVIDIA CUDA is the reference training platform. Python 3.10 and PyTorch 2.4.1/CUDA 12.1 were observed on the source server. CPU inference and forward/backward validation are supported; frozen training uses CUDA memory/timing/RNG functions.

```bash
conda env create -f environment.yml
conda activate screlift
screlift --help
```

Alternatively install a suitable PyTorch build first, then `python -m pip install -e .`. The environment pins observed core versions; it is not a complete lockfile for historical comparators or downstream analyses.

## Run

Prepare inputs according to [DATA_FORMAT.md](docs/DATA_FORMAT.md). Configuration paths resolve relative to the configuration file. No private server mount is required by the portable interface.

```bash
# One-source protocol, paired SOURCE counts and frozen splits
screlift prepare-s --config configs/s_mode.example.json --output outputs/source_cache
# Three-source LODO, SOURCE-only prepared directories
screlift prepare-d --config configs/d_mode.example.json --output outputs/fold_cache
screlift train --cache outputs/fold_cache --seed 20020130 --output outputs/model_20020130
screlift infer --bundle outputs/model_20020130 --rna data/target/rna_counts.npz --genes data/target/gene_ids.npy --cell-ids data/target/cell_ids.npy --output outputs/target_20020130 --device cpu
screlift evaluate --inference outputs/target_20020130 --labels data/target/labels.npy --label-cell-ids data/target/cell_ids.npy --fold target_D1 --output outputs/metrics_D1_20020130.json
```

Repeat for **20020130, 20020131, 20020132** and all intended folds. `screlift summarize --metrics <metric JSON files> --output outputs/summary.json` requires all three seeds per included fold, rejects duplicates/smoke runs, and reports mean/sample SD (`ddof=1`) and equal-fold macro averages by seed. The output lists included folds; partial-fold summaries are not full four-fold reproductions.

The clustering evaluator fits StandardScaler on frozen evaluation embeddings and uses the known label count for KMeans, followed by post-hoc Hungarian matching. This is an evaluation protocol, distinct from frozen RNA-only inference.

## Paper workflows

- [Paper pipeline map](docs/PAPER_PIPELINES.md): original PBMC, BMMC, AD, regulatory and subtype entry points.
- [Data availability](docs/DATA_AVAILABILITY.md): public dataset IDs, splits and reference assets. Raw data/checkpoints are not bundled.
- [Reproducibility](docs/REPRODUCIBILITY.md): unchanged computations, adaptations and validation scope.
- [Chinese guide](README_zh.md).

Official model display name: **scReLIFT**. D-mode/S-mode describe protocols. Historical names remain in original evidence. The comparator display name is **ECLARE-S**.

## Validation

```bash
python tests/smoke_pipeline.py
python tests/smoke_pipeline.py --integration-output /tmp/screlift_smoke --cuda
python scripts/verify_release.py
```

Synthetic tests do not establish agreement with paper scores. See `provenance/VALIDATION.md` for observed results.

## Citation and license

`CITATION.cff` records the manuscript authors, software title and public repository URL. A manuscript DOI has not yet been supplied. Original scReLIFT code is MIT licensed. Data and third-party projects retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
