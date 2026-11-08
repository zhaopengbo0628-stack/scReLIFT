from __future__ import annotations

import numpy as np
from scipy import sparse


def fit_dmode_field(
    atac_binary: sparse.csr_matrix,
    b_reg: sparse.spmatrix,
    train_index: np.ndarray,
    batch_size: int = 256,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    if atac_binary.shape[1] != b_reg.shape[0]:
        raise ValueError("ATAC and B_D peak dimensions differ")
    train_atac = atac_binary[train_index].tocsr()
    document_frequency = np.asarray(train_atac.sum(axis=0)).ravel().astype(np.float32)
    idf = np.log1p(float(train_index.size) / (1.0 + document_frequency)).astype(np.float32)
    depth = np.asarray(atac_binary.sum(axis=1)).ravel().astype(np.float32)
    tfidf = sparse.diags(1.0 / np.maximum(depth, 1.0)).dot(atac_binary).multiply(idf).tocsr()
    peak_mean = np.asarray(tfidf[train_index].mean(axis=0)).ravel().astype(np.float32)
    field_mean_atac = np.asarray(peak_mean @ b_reg).ravel().astype(np.float32)
    fields = []
    for start in range(0, tfidf.shape[0], batch_size):
        values = tfidf[start:start + batch_size] @ b_reg
        if sparse.issparse(values):
            values = values.toarray()
        fields.append(np.asarray(values, dtype=np.float32) - field_mean_atac)
    field = np.concatenate(fields)
    field_mean = field[train_index].mean(axis=0).astype(np.float32)
    field_std = field[train_index].std(axis=0).astype(np.float32)
    representation = ((field - field_mean) / (field_std + 1e-6)).astype(np.float32)
    return representation, {
        "idf": idf,
        "peak_mean": peak_mean,
        "field_mean_atac": field_mean_atac,
        "field_mean": field_mean,
        "field_std": field_std,
    }, {
        "input": "canonical binary ATAC with source-train TF-IDF",
        "shared_graph": "B_D",
        "normalization": "source-train per-gene mean/std",
        "gene_confidence_used": False,
        "labels_used": False,
        "trainable_parameters": 0,
        "cells": int(atac_binary.shape[0]),
        "peaks": int(atac_binary.shape[1]),
        "genes": int(b_reg.shape[1]),
        "nonzero_edges": int(b_reg.nnz),
        "zero_train_field_std": int(np.count_nonzero(field_std < 1e-6)),
    }
