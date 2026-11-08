from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from diagnostics.fit_train_tfidf_lsi import tfidf_from_train_idf
from .regulatory_graph import build_candidate_graph


def normalize_log1p(rna: sparse.csr_matrix) -> sparse.csr_matrix:
    totals = np.asarray(rna.sum(axis=1)).ravel().astype(np.float32)
    norm = sparse.diags(1e4 / np.maximum(totals, 1.0)).dot(rna).tocsr()
    norm.data = np.log1p(norm.data)
    return norm


class V5SData:
    def __init__(self, root: Path, selection_root: Path | None = None):
        self.root = Path(root)
        self.rna = np.load(self.root / "rna.npy", mmap_mode="r")
        self.atac = sparse.load_npz(self.root / "atac_tfidf.npz").tocsr()
        with np.load(self.root / "metadata.npz", allow_pickle=True) as meta:
            self.train = meta["train_index"].astype(np.int64)
            self.val = meta["val_index"].astype(np.int64)
            self.test = meta["test_index"].astype(np.int64)
        self.graph = np.load(self.root / "regulatory_graph.npz")
        selection_root = self.root if selection_root is None else Path(selection_root)
        self.b_reg = np.load(selection_root / "B_reg.npy", mmap_mode="r")
        self.gene_confidence = np.load(selection_root / "gene_confidence.npy")
        self.atac_train_mean = np.load(self.root / "atac_train_mean.npy")
        self.train_mean = np.load(self.root / "rna_train_mean.npy")
        self.train_std = np.load(self.root / "rna_train_std.npy")

    def atac_rows(self, rows: np.ndarray) -> np.ndarray:
        return self.atac[rows].toarray().astype(np.float32)


def prepare_dataset(raw_h5: Path, historical_dir: Path, gtf: Path, outdir: Path, window_bp: int, max_candidates: int) -> dict:
    from prepare_pbmc10k_model_v1 import read_10x_h5_with_ids, read_tss_by_ensembl

    outdir.mkdir(parents=True, exist_ok=False)
    with np.load(historical_dir / "metadata.npz", allow_pickle=True) as old:
        cells = old["cell_names"].astype(str)
        gene_names = old["gene_names"].astype(str)
        gene_ids = old["gene_ids"].astype(str)
        train = old["train_index"].astype(np.int64)
        val = old["val_index"].astype(np.int64)
        test = old["test_index"].astype(np.int64)
        labels = old["labels"].astype(np.int64)
        label_names = old["label_names"]
    rna_raw, atac_raw, barcodes, _, raw_gene_ids, raw_peak_names, report = read_10x_h5_with_ids(raw_h5)
    cell_lookup = {value: i for i, value in enumerate(barcodes.astype(str))}
    gene_lookup = {value: i for i, value in enumerate(raw_gene_ids.astype(str))}
    missing_cells = [value for value in cells if value not in cell_lookup]
    missing_genes = [value for value in gene_ids if value not in gene_lookup]
    if missing_cells or missing_genes:
        raise RuntimeError(f"Raw H5 alignment failed: missing_cells={len(missing_cells)}, missing_genes={len(missing_genes)}")
    cell_rows = np.asarray([cell_lookup[value] for value in cells], dtype=np.int64)
    gene_rows = np.asarray([gene_lookup[value] for value in gene_ids], dtype=np.int64)
    rna_raw = rna_raw[cell_rows][:, gene_rows].tocsr()
    atac_raw = atac_raw[cell_rows].tocsr().astype(bool).astype(np.float32)
    tss_by_id, _ = read_tss_by_ensembl(gtf)
    tss = {gene_id: tss_by_id[gene_id] for gene_id in gene_ids if gene_id in tss_by_id}
    if len(tss) != len(gene_ids):
        raise RuntimeError(f"GTF is missing {len(gene_ids) - len(tss)} model genes")
    graph, selected_peaks = build_candidate_graph(gene_ids, raw_peak_names.astype(str), tss, window_bp, max_candidates)
    atac = atac_raw[:, selected_peaks].tocsr()
    train_atac = atac[train]
    df = np.asarray(train_atac.sum(axis=0)).ravel().astype(np.float32)
    idf = np.log1p(float(train.size) / (1.0 + df)).astype(np.float32)
    atac_tfidf = tfidf_from_train_idf(atac, idf).tocsr()
    rna = normalize_log1p(rna_raw).toarray().astype(np.float32)
    rna_mean = rna[train].mean(axis=0).astype(np.float32)
    rna_std = rna[train].std(axis=0).astype(np.float32)
    sparse.save_npz(outdir / "atac_binary.npz", atac)
    sparse.save_npz(outdir / "atac_tfidf.npz", atac_tfidf)
    np.save(outdir / "rna.npy", rna)
    np.save(outdir / "rna_train_mean.npy", rna_mean)
    np.save(outdir / "rna_train_std.npy", rna_std)
    atac_train_mean = np.asarray(atac_tfidf[train].mean(axis=0)).ravel().astype(np.float32)
    np.save(outdir / "atac_train_mean.npy", atac_train_mean)
    np.save(outdir / "gene_names.npy", gene_names)
    np.save(outdir / "gene_ids.npy", gene_ids)
    np.save(outdir / "peak_names.npy", raw_peak_names.astype(str)[selected_peaks])
    np.savez_compressed(outdir / "metadata.npz", cell_names=cells, gene_names=gene_names, gene_ids=gene_ids, peak_names=raw_peak_names.astype(str)[selected_peaks], labels=labels, label_names=label_names, train_index=train, val_index=val, test_index=test)
    np.savez_compressed(outdir / "regulatory_graph.npz", candidate_index=graph["candidate_index"], candidate_mask=graph["candidate_mask"], distance_bp=graph["distance_bp"], peak_names=raw_peak_names.astype(str)[selected_peaks])
    manifest = {
        "dataset": "PBMC10k",
        "raw_h5_report": report,
        "cells": int(cells.size),
        "model_genes": int(gene_ids.size),
        "train_cells": int(train.size),
        "val_cells": int(val.size),
        "test_cells": int(test.size),
        "candidate_peaks": int(selected_peaks.size),
        "window_bp": int(window_bp),
        "max_candidates_per_gene": int(max_candidates),
        "historical_split_reused": True,
        "labels_used_for_preparation": False,
        "test_used_for_preparation": False,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
