# Portable input contract

Every matrix uses cells as rows. Row identity and frozen splits must be established before fitting. Supply the original protocol's QC/cohort selection and split arrays; this interface never invents a paper split. NumPy string arrays must use a Unicode dtype rather than pickled objects.

## S-mode source inputs

| File | Required content |
| --- | --- |
| `rna_counts.npz` | SciPy CSR raw nonnegative integer RNA counts, all validated genes before library normalization |
| `gene_ids.npy` | Stable gene IDs in RNA column order; version suffixes on ENSG IDs are removed and identical stable IDs merged |
| `atac_counts.npz` | Paired raw integer ATAC counts in the same cell order |
| `peak_names.npy` | Unique coordinates matching ATAC columns, e.g. `chr1:100000-100500` |
| `metadata.npz` | Integer arrays `train_index`, `val_index`, `test_index`; disjoint, covering every row exactly once |
| reference GTF | Gene annotations including stable gene IDs; GENCODE v32 is the paper reference |

The source configuration identifies each path. S-mode fits CP10K/log1p HVG scores, 2000 selected genes, RNA scaling, B_S and F statistics using SOURCE train only. Validation rows are transformed using fitted statistics. Test ATAC rows are not used, although the input sparse file contains the complete paired source cohort. Candidate edges use a 250 kb TSS window and at most 50 peaks per gene; stability uses ten 20% resamples.

Raw-data QC, paired-barcode intersection and dataset-specific annotation processing remain explicit prerequisites. See the original PBMC/BMMC preparation scripts in PAPER_PIPELINES.md.

## D-mode prepared source inputs

For each of the three SOURCE directories, provide `rna.npy` (float32 CP10K/log1p RNA before SOURCE standardization), `gene_ids.npy`, binary SciPy CSR `atac_binary.npz`, `peak_names.npy`, and `metadata.npz` with frozen train/val/test arrays. Feature selection must already have been fitted with SOURCE train only under the intended experiment protocol. Do not supply TARGET to this preparation command.

The portable D-mode adapter reuses the original common-gene harmonization, merged SOURCE peak universe, peak overlap mapping, cross-source selector and field computation. The inherited selector uses `omega_d proportional to n_train,d * Q_d`. This matches the PBMC historical core. Formal AD uses a different SOURCE-balanced weighting and must use its frozen AD-specific pipeline; do not substitute this adapter for AD preprocessing.

The prepared-directory contract is deliberately explicit: arbitrary gene selection is not guaranteed to reproduce the paper. Full-library RNA normalization must precede selected-gene slicing. TARGET must not define HVGs, the feature universe, B, F statistics, checkpoint choice or RNA scaling.

## RNA-only target inputs

Provide CSR `rna_counts.npz`, matching `gene_ids.npy`, and preferably a unique Unicode `cell_ids.npy` matching every row. The frozen Student maps stable gene IDs to its SOURCE dictionary, performs full-library CP10K/log1p and SOURCE standardization, then produces 64-dimensional embeddings. Missing genes become zero before SOURCE standardization and are reported. An entirely unmatched dictionary is rejected.

Labels are supplied only to `evaluate`, as a Unicode `labels.npy` plus `label-cell-ids` in exactly the saved inference order. The evaluator verifies saved embedding/cell-order hashes before reading labels. The clustering evaluation uses known K and evaluation-population StandardScaler as specified in the historical evaluator. Inference itself does not fit TARGET statistics.

## Cached inputs and relocation

Cache/run manifests record absolute resolved input paths and SHA256. Resume is allowed only for unchanged inputs and configuration. Move raw inputs before building a cache; moving or editing them afterward intentionally invalidates verification. Inference bundles contain portable weights plus preprocessing state and can be moved independently of training inputs.
