from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py
import numpy as np
from scipy import sparse

from experiment_registry import PBMC10K_DATASET, PBMC10K_DATASET_LABEL, PBMC10K_HISTORICAL_VERSION, canonical_run_id
from diagnostics.fit_train_tfidf_lsi import tfidf_from_train_idf
from diagnostics.prepare_pbmc10k_multiome import (
    choose_dataset,
    filter_cells,
    find_pbmc_candidates,
    intersect_barcodes,
    label_inventory_from_h5ad,
    normalize_log1p,
    read_10x_h5,
    split_labels,
)
from graph_attention.topology import build_tss_graph, parse_peak


GENE_ID = re.compile(r'gene_id "([^"]+)"')
GENE_NAME = re.compile(r'gene_name "([^"]+)"')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=Path(r"E:\my_research\data\PBMC10k_model_v1"))
    p.add_argument("--results-dir", type=Path, default=Path("results/pbmc10k_e3_r1_20260807"))
    p.add_argument("--num-genes", type=int, default=2000)
    p.add_argument("--window-bp", type=int, default=150_000)
    p.add_argument("--max-candidates", type=int, default=25)
    return p.parse_args()


def read_10x_h5_with_ids(path: Path):
    rna, atac, barcodes, gene_names, peak_names, report = read_10x_h5(path)
    with h5py.File(path, "r") as h:
        ft = np.asarray([x.decode() for x in h["matrix/features/feature_type"][:]])
        ids = np.asarray([x.decode().split(".")[0] for x in h["matrix/features/id"][:]])
    return rna, atac, barcodes, gene_names, ids[ft == "Gene Expression"], peak_names, report


def read_tss_by_ensembl(path: Path) -> tuple[dict[str, tuple[str, int]], dict[str, tuple[str, int]]]:
    by_id, by_name = {}, {}
    with path.open("rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            gene_id = GENE_ID.search(fields[8])
            gene_name = GENE_NAME.search(fields[8])
            start, end = int(fields[3]), int(fields[4])
            tss = (fields[0], end - 1 if fields[6] == "-" else start - 1)
            if gene_id:
                by_id.setdefault(gene_id.group(1).split(".")[0], tss)
            if gene_name:
                by_name.setdefault(gene_name.group(1), tss)
    return by_id, by_name


def hvg_train_indices(raw: sparse.csr_matrix, train_index: np.ndarray, candidate: np.ndarray, n: int) -> np.ndarray:
    log = normalize_log1p(raw[:, candidate])
    x = log[train_index].tocsr()
    mean = np.asarray(x.mean(axis=0)).ravel()
    mean_sq = np.asarray(x.power(2).mean(axis=0)).ravel()
    var = np.maximum(mean_sq - mean * mean, 0.0)
    score = var / np.maximum(mean, 1e-6)
    return candidate[np.argsort(score, kind="stable")[-n:]]


def graph_connected_gene_indices(gene_keys: np.ndarray, tss: dict[str, tuple[str, int]], peak_names: np.ndarray, window_bp: int) -> np.ndarray:
    by_chrom: dict[str, list[int]] = {}
    for peak in peak_names:
        chrom, start, end = parse_peak(str(peak))
        by_chrom.setdefault(chrom, []).append((start + end) // 2)
    for positions in by_chrom.values():
        positions.sort()
    eligible = []
    import bisect

    for index, gene in enumerate(gene_keys):
        location = tss.get(str(gene))
        if location is None:
            continue
        chrom, position = location
        positions = by_chrom.get(chrom, [])
        left = bisect.bisect_left(positions, position - window_bp)
        if left < len(positions) and positions[left] <= position + window_bp:
            eligible.append(index)
    return np.asarray(eligible, dtype=np.int64)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    for d in ("preprocessing", "graph"):
        (args.results_dir / d).mkdir(parents=True, exist_ok=True)

    inventory = find_pbmc_candidates()
    dataset_dir, h5_path, h5ad_path = choose_dataset(inventory)
    rna_raw, atac_raw, barcodes, gene_names, gene_ids, peak_names, structure = read_10x_h5_with_ids(h5_path)
    obs, label_inv = label_inventory_from_h5ad(h5ad_path)
    paired_idx, labels_str, pairing = intersect_barcodes(barcodes, obs)
    if labels_str is None or not label_inv["supervised_upper_bound_available"]:
        raise RuntimeError("No verified cell_type labels; stop supervised PBMC transfer.")

    rna_raw = rna_raw[paired_idx].tocsr()
    atac_raw = atac_raw[paired_idx].tocsr()
    keep_qc, qc, filter_manifest = filter_cells(rna_raw, atac_raw.astype(bool).astype(np.float32).tocsr())
    labels_str = labels_str[keep_qc]
    rna_raw = rna_raw[keep_qc].tocsr()
    atac_raw = atac_raw[keep_qc].tocsr().astype(bool).astype(np.float32)
    train, val, test, split = split_labels(labels_str)
    keep_label = split["keep"]
    rna_raw = rna_raw[keep_label].tocsr()
    atac_raw = atac_raw[keep_label].tocsr()
    labels = split["labels"]

    old_meta = np.load(Path(r"E:\my_research\data\PBMC10k_multiome_upper_bound\metadata.npz"), allow_pickle=True)
    if not (np.array_equal(train, old_meta["train_index"]) and np.array_equal(val, old_meta["val_index"]) and np.array_equal(test, old_meta["test_index"])):
        raise RuntimeError("PBMC model split does not match upper-bound split.")
    if not np.array_equal(labels, old_meta["labels"]):
        raise RuntimeError("PBMC model labels do not match upper-bound labels.")

    gtf_path = next(dataset_dir.rglob("*.gtf"))
    tss_id, tss_name = read_tss_by_ensembl(gtf_path)
    mapped = np.asarray([gid in tss_id for gid in gene_ids], dtype=bool)
    if mapped.mean() < 0.90:
        mapped = np.asarray([name in tss_name for name in gene_names], dtype=bool)
        tss = {name: tss_name[name] for name in gene_names[mapped]}
        gene_keys = gene_names
        mapping_mode = "gene_symbol_fallback"
    else:
        tss = {gid: tss_id[gid] for gid in gene_ids[mapped]}
        gene_keys = gene_ids
        mapping_mode = "ensembl_id"
    if mapped.mean() < 0.90:
        raise RuntimeError(f"GTF mapped fraction too low: {mapped.mean():.4f}")

    candidate = graph_connected_gene_indices(gene_keys, tss, peak_names, args.window_bp)
    selected_genes = hvg_train_indices(rna_raw, train, candidate, args.num_genes)
    selected_keys = gene_keys[selected_genes]
    graph, selected_peak_rel = build_tss_graph(selected_keys, peak_names, tss, args.window_bp, args.max_candidates)

    rna = normalize_log1p(rna_raw)[:, selected_genes].toarray().astype(np.float32)
    rna_counts = rna_raw[:, selected_genes].astype(np.float32).tocsr()
    atac = atac_raw[:, selected_peak_rel].astype(np.float32).tocsr()
    train_atac = atac[train]
    df = np.asarray(train_atac.sum(axis=0)).ravel().astype(np.float32)
    idf = np.log1p(float(train.size) / (1.0 + df)).astype(np.float32)
    tfidf = tfidf_from_train_idf(atac, idf)
    train_values = tfidf[train].data
    scale = float(np.median(train_values[train_values > 0])) if np.any(train_values > 0) else 1.0
    tfidf = (tfidf / max(scale, 1e-8)).tocsr().astype(np.float32)
    peak_mean = np.asarray(tfidf[train].mean(axis=0)).ravel().astype(np.float32)

    np.save(args.outdir / "rna.npy", rna)
    sparse.save_npz(args.outdir / "rna_counts.npz", rna_counts)
    sparse.save_npz(args.outdir / "atac_binary.npz", atac)
    sparse.save_npz(args.outdir / "atac_tfidf.npz", tfidf)
    np.save(args.outdir / "atac_tfidf_idf.npy", idf)
    np.save(args.outdir / "atac_tfidf_peak_mean.npy", peak_mean)
    np.save(args.outdir / "labels.npy", labels)
    np.save(args.outdir / "label_names.npy", split["label_names"])
    np.save(args.outdir / "barcodes.npy", barcodes[paired_idx][keep_qc][keep_label])
    np.save(args.outdir / "gene_names.npy", gene_names[selected_genes])
    np.save(args.outdir / "gene_ids.npy", gene_ids[selected_genes])
    np.save(args.outdir / "peak_names.npy", peak_names[selected_peak_rel])
    graph.save(args.outdir / "graph.npz")
    np.savez_compressed(
        args.outdir / "metadata.npz",
        labels=labels,
        label_names=split["label_names"],
        train_index=train,
        val_index=val,
        test_index=test,
        cell_names=barcodes[paired_idx][keep_qc][keep_label],
        gene_names=gene_names[selected_genes],
        gene_ids=gene_ids[selected_genes],
        peak_names=peak_names[selected_peak_rel],
    )

    graph_manifest = {
        "reference": str(gtf_path),
        "genome": "GRCh38",
        "mapping_mode": mapping_mode,
        "genes_total": int(gene_ids.size),
        "genes_mapped": int(mapped.sum()),
        "selected_genes": int(selected_genes.size),
        "peaks_total": int(peak_names.size),
        "graph_covered_peaks": int(selected_peak_rel.size),
        "candidate_edges": int(graph.candidate_mask.sum()),
        "mean_candidates_per_gene": float(graph.candidate_mask.sum(axis=1).mean()),
        "median_candidates_per_gene": float(np.median(graph.candidate_mask.sum(axis=1))),
        "genes_with_zero_peaks": 0,
        "distance_bp": {
            "mean": float(graph.distance_bp[graph.candidate_mask].mean()),
            "median": float(np.median(graph.distance_bp[graph.candidate_mask])),
            "p95": float(np.percentile(graph.distance_bp[graph.candidate_mask], 95)),
        },
        "window_bp": args.window_bp,
        "max_candidates": args.max_candidates,
        "prior": "GLUE (1 + distance_kb)^-0.75",
    }
    rna_manifest = {
        "hvg_algorithm": "train-only log-normalized variance/mean among GTF-mapped genes",
        "hvg_count": int(selected_genes.size),
        "train_only": True,
        "gene_symbol_saved": True,
        "ensembl_id_saved": True,
    }
    data_manifest = {
        "dataset": PBMC10K_DATASET,
        "dataset_label": PBMC10K_DATASET_LABEL,
        "version": PBMC10K_HISTORICAL_VERSION,
        "run_id_example": canonical_run_id("P0", 20020130, version=PBMC10K_HISTORICAL_VERSION),
        "dataset_dir": str(dataset_dir),
        "cells": int(labels.size),
        "rna_genes": int(rna.shape[1]),
        "atac_peaks": int(atac.shape[1]),
        "source_cells_filtered_h5": int(structure["num_cells"]),
        "source_genes": int(structure["num_genes"]),
        "source_peaks": int(structure["num_peaks"]),
        "train_cells": int(train.size),
        "val_cells": int(val.size),
        "test_cells": int(test.size),
        "split_reused_from_upper_bound": True,
        "atac_binary": "count > 0",
        "atac_tfidf": "train-fitted IDF, median positive train TF-IDF scaling",
    }
    (args.results_dir / "preprocessing" / "rna_preprocessing_manifest.json").write_text(json.dumps(rna_manifest, indent=2), encoding="utf-8")
    (args.results_dir / "preprocessing" / "data_manifest.json").write_text(json.dumps(data_manifest, indent=2), encoding="utf-8")
    (args.results_dir / "preprocessing" / "qc_summary.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    (args.results_dir / "preprocessing" / "qc_filter_manifest.json").write_text(json.dumps(filter_manifest, indent=2), encoding="utf-8")
    (args.results_dir / "graph" / "graph_manifest.json").write_text(json.dumps(graph_manifest, indent=2), encoding="utf-8")
    print(json.dumps({"cells": int(labels.size), "genes": int(rna.shape[1]), "peaks": int(atac.shape[1]), "graph_edges": int(graph.candidate_mask.sum())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
