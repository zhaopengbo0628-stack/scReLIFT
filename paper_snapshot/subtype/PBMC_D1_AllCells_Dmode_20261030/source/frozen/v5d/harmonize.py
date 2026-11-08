from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .dataset_spec import DatasetSpec
from .regulatory_graph import parse_peak
from .utils import sha256_array


def _model_gene_ids(root: Path) -> np.ndarray:
    ids = np.load(root / "gene_ids.npy", allow_pickle=True).astype(str)
    index_path = root / "canonical_pbmc10k_gene_indices.npy"
    if index_path.exists():
        ids = ids[np.load(index_path).astype(np.int64)]
    return ids


def _model_gene_names(root: Path) -> np.ndarray:
    path = root / "gene_names.npy"
    if not path.exists():
        return _model_gene_ids(root)
    names = np.load(path, allow_pickle=True).astype(str)
    index_path = root / "canonical_pbmc10k_gene_indices.npy"
    if index_path.exists():
        names = names[np.load(index_path).astype(np.int64)]
    return names


def _rna(root: Path) -> np.ndarray:
    canonical = root / "rna_canonical_pbmc10k.npy"
    return np.load(canonical if canonical.exists() else root / "rna.npy", mmap_mode="r")


def _metadata(root: Path) -> dict[str, np.ndarray]:
    path = root / "metadata.npz"
    if not path.exists():
        path = root / "splits.npz"
    with np.load(path, allow_pickle=True) as meta:
        return {key: meta[key].copy() for key in meta.files}


def _overlap_mapping(source_peaks: np.ndarray, canonical_peaks: np.ndarray) -> tuple[np.ndarray, dict]:
    canonical_by_chrom: dict[str, list[tuple[int, int, int]]] = {}
    for index, name in enumerate(canonical_peaks):
        chrom, start, end = parse_peak(str(name))
        canonical_by_chrom.setdefault(chrom, []).append((start, end, index))
    for values in canonical_by_chrom.values():
        values.sort()

    source_by_chrom: dict[str, list[tuple[int, int, int]]] = {}
    for index, name in enumerate(source_peaks):
        chrom, start, end = parse_peak(str(name))
        source_by_chrom.setdefault(chrom, []).append((start, end, index))

    mapping = np.full(source_peaks.size, -1, dtype=np.int64)
    one_to_many = 0
    for chrom, source_values in source_by_chrom.items():
        canonical_values = canonical_by_chrom.get(chrom, [])
        active: list[tuple[int, int, int]] = []
        pointer = 0
        for start, end, source_index in sorted(source_values):
            while pointer < len(canonical_values) and canonical_values[pointer][0] < end:
                active.append(canonical_values[pointer])
                pointer += 1
            active = [value for value in active if value[1] > start]
            candidates = [value for value in active if value[0] < end]
            if candidates:
                overlaps = [min(end, value[1]) - max(start, value[0]) for value in candidates]
                mapping[source_index] = candidates[int(np.argmax(overlaps))][2]
                one_to_many += int(len(candidates) > 1)
    counts = np.bincount(mapping[mapping >= 0], minlength=canonical_peaks.size)
    return mapping, {
        "source_peaks": int(source_peaks.size),
        "mapped_peaks": int(np.count_nonzero(mapping >= 0)),
        "unmapped_peaks": int(np.count_nonzero(mapping < 0)),
        "canonical_peaks": int(canonical_peaks.size),
        "canonical_coverage": float(np.count_nonzero(counts) / max(canonical_peaks.size, 1)),
        "source_intervals_overlapping_multiple_canonical": int(one_to_many),
        "canonical_regions_with_multiple_source_peaks": int(np.count_nonzero(counts > 1)),
    }


def _canonical_peaks_from_sources(roots: list[Path]) -> np.ndarray:
    intervals: list[tuple[str, int, int]] = []
    for root in roots:
        for name in np.load(root / "peak_names.npy", allow_pickle=True).astype(str):
            intervals.append(parse_peak(str(name)))
    if not intervals:
        raise RuntimeError("No source peaks available for V5-D canonical peak universe")
    intervals.sort(key=lambda item: (item[0], item[1], item[2]))
    merged: list[list[object]] = []
    for chrom, start, end in intervals:
        if not merged or merged[-1][0] != chrom or start >= int(merged[-1][2]):
            merged.append([chrom, start, end])
        else:
            merged[-1][2] = max(int(merged[-1][2]), end)
    return np.asarray([f"chr{chrom}:{start}-{end}" for chrom, start, end in merged], dtype=str)


def _aggregate_binary(atac: sparse.csr_matrix, mapping: np.ndarray, canonical_count: int) -> sparse.csr_matrix:
    keep = mapping[atac.indices] >= 0
    rows = np.repeat(np.arange(atac.shape[0]), np.diff(atac.indptr))[keep]
    cols = mapping[atac.indices[keep]]
    values = np.ones(rows.size, dtype=np.float32)
    result = sparse.coo_matrix((values, (rows, cols)), shape=(atac.shape[0], canonical_count)).tocsr()
    result.data[:] = 1.0
    return result


def prepare_harmonized(
    specs: list[DatasetSpec],
    outdir: Path,
    feature_reference_names: list[str] | None = None,
    held_out_target_name: str | None = None,
) -> dict:
    spec_by_name = {spec.name: spec for spec in specs}
    if feature_reference_names is None:
        reference_specs = specs
    else:
        missing = [name for name in feature_reference_names if name not in spec_by_name]
        if missing:
            raise ValueError(f"Unknown feature reference datasets: {missing}")
        reference_specs = [spec_by_name[name] for name in feature_reference_names]
    if not reference_specs:
        raise ValueError("At least one feature reference dataset is required")
    if held_out_target_name is not None and held_out_target_name in {spec.name for spec in reference_specs}:
        raise ValueError("Held-out target must not define the V5-D feature universe")

    roots = [spec.data_dir for spec in specs]
    gene_sets = [_model_gene_ids(root) for root in roots]
    reference_roots = [spec.data_dir for spec in reference_specs]
    reference_gene_sets = [_model_gene_ids(root) for root in reference_roots]
    common = sorted(set(reference_gene_sets[0]).intersection(*(set(values) for values in reference_gene_sets[1:])))
    if not common:
        raise RuntimeError("No common model genes across V5-D feature reference datasets")
    common_genes = np.asarray(common, dtype=str)
    common_names = common_genes.copy()

    canonical_peaks = _canonical_peaks_from_sources(reference_roots)
    peak_widths = np.asarray([parse_peak(name)[2] - parse_peak(name)[1] for name in canonical_peaks], dtype=np.int64)
    outdir.mkdir(parents=True, exist_ok=False)
    np.save(outdir / "common_gene_ids.npy", common_genes)
    np.save(outdir / "common_gene_names.npy", common_names)
    np.save(outdir / "canonical_peak_names.npy", canonical_peaks)
    dataset_rows = []
    for spec, root, genes in zip(specs, roots, gene_sets):
        meta = _metadata(root)
        source_rna = np.asarray(_rna(root), dtype=np.float32)
        gene_lookup = {gene: index for index, gene in enumerate(genes)}
        gene_index = np.asarray([gene_lookup.get(gene, -1) for gene in common_genes], dtype=np.int64)
        present = gene_index >= 0
        rna = np.zeros((source_rna.shape[0], common_genes.size), dtype=np.float32)
        if np.any(present):
            rna[:, present] = source_rna[:, gene_index[present]]
        atac = sparse.load_npz(root / "atac_binary.npz").tocsr().astype(np.float32)
        source_peaks = np.load(root / "peak_names.npy", allow_pickle=True).astype(str)
        mapping, mapping_manifest = _overlap_mapping(source_peaks, canonical_peaks)
        peak_counts = np.bincount(mapping[mapping >= 0], minlength=canonical_peaks.size)
        canonical_peak_available = peak_counts > 0
        canonical_atac = _aggregate_binary(atac, mapping, canonical_peaks.size)
        target = outdir / spec.name
        target.mkdir()
        np.save(target / "rna.npy", rna)
        np.save(target / "canonical_peak_available.npy", canonical_peak_available)
        sparse.save_npz(target / "atac_binary.npz", canonical_atac)
        for key in ("train_index", "val_index", "test_index"):
            if key not in meta:
                raise RuntimeError(f"{spec.name} metadata is missing {key}")
        np.savez_compressed(
            target / "metadata.npz",
            train_index=meta["train_index"].astype(np.int64),
            val_index=meta["val_index"].astype(np.int64),
            test_index=meta["test_index"].astype(np.int64),
            labels=meta.get("labels", np.asarray([], dtype=np.int64)),
        )
        np.save(target / "peak_mapping.npy", mapping)
        row = {
            "name": spec.name,
            "source_dir": str(root),
            "cells": int(rna.shape[0]),
            "genes_original": int(genes.size),
            "common_genes": int(common_genes.size),
            "mapped_common_genes": int(np.count_nonzero(present)),
            "missing_common_genes": int(np.count_nonzero(~present)),
            "common_gene_coverage": float(np.mean(present)),
            "gene_mapping_sha256": sha256_array(gene_index),
            "peaks_original": int(source_peaks.size),
            "structural_available_peaks": int(np.count_nonzero(canonical_peak_available)),
            "structural_available_peak_fraction": float(np.mean(canonical_peak_available)),
            "train_cells": int(meta["train_index"].size),
            "val_cells": int(meta["val_index"].size),
            "test_cells": int(meta["test_index"].size),
            **mapping_manifest,
            "peak_mapping_sha256": sha256_array(mapping),
        }
        (target / "peak_mapping_manifest.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        dataset_rows.append(row)
    manifest = {
        "dataset_count": len(specs),
        "feature_reference_datasets": [spec.name for spec in reference_specs],
        "held_out_target_dataset": held_out_target_name,
        "target_excluded_from_feature_universe": held_out_target_name is not None and held_out_target_name not in {spec.name for spec in reference_specs},
        "common_gene_count": int(common_genes.size),
        "canonical_peak_count": int(canonical_peaks.size),
        "canonical_peak_reference": "deterministic_merged_union_of_source_peaks",
        "canonical_peak_source_order_invariant": True,
        "canonical_peak_width_median": float(np.median(peak_widths)),
        "canonical_peak_width_q90": float(np.percentile(peak_widths, 90)),
        "canonical_peak_width_max": int(peak_widths.max()),
        "canonical_peak_interval_semantics": "half-open; adjacent intervals are not merged",
        "common_gene_ids_sha256": sha256_array(common_genes),
        "canonical_peak_names_sha256": sha256_array(canonical_peaks),
        "datasets": dataset_rows,
    }
    (outdir / "gene_harmonization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (outdir / "peak_harmonization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
