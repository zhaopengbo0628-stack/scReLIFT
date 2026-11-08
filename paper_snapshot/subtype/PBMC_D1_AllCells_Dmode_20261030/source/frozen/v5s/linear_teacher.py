from __future__ import annotations

import numpy as np
from scipy import sparse


def compute_linear_teacher(
    atac_tfidf: sparse.csr_matrix,
    b_reg: np.ndarray,
    train_index: np.ndarray,
    gene_confidence: np.ndarray,
    row_permutation: np.ndarray | None = None,
    batch_size: int = 128,
) -> tuple[np.ndarray, dict]:
    permutation = (
        np.arange(atac_tfidf.shape[0], dtype=np.int64)
        if row_permutation is None
        else np.asarray(row_permutation, dtype=np.int64)
    )
    peak_mean = np.asarray(atac_tfidf[permutation[train_index]].mean(axis=0)).ravel().astype(np.float32)
    outputs = []
    for start in range(0, atac_tfidf.shape[0], batch_size):
        rows = np.arange(start, min(start + batch_size, atac_tfidf.shape[0]))
        centered = atac_tfidf[permutation[rows]] - peak_mean
        outputs.append(np.asarray(centered @ b_reg, dtype=np.float32))
    field = np.concatenate(outputs, axis=0)
    train_field = field[train_index]
    mean = train_field.mean(axis=0).astype(np.float32)
    std = train_field.std(axis=0).astype(np.float32)
    z = ((field - mean) / (std + 1e-6)).astype(np.float32)
    z *= np.sqrt(np.asarray(gene_confidence, dtype=np.float32))[None, :]
    return z, {
        "input": "train-fitted TF-IDF ATAC",
        "centering": "per-peak mean on train rows after condition permutation",
        "normalization": "per-gene train mean/std",
        "b_reg_shape": [int(x) for x in b_reg.shape],
        "nonzero_b_reg": int(np.count_nonzero(np.abs(b_reg) > 1e-8)),
        "train_field_std_zero_count": int(np.count_nonzero(std < 1e-6)),
        "gene_confidence_applied_after_train_normalization": True,
    }
