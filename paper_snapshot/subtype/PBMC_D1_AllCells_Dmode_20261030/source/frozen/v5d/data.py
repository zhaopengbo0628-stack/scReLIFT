from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse


class V5DData:
    def __init__(self, root: Path, name: str):
        self.name = name
        self.root = Path(root) / name
        self.rna = np.load(self.root / "rna.npy", mmap_mode="r")
        self.atac = sparse.load_npz(self.root / "atac_binary.npz").tocsr()
        availability_path = self.root / "canonical_peak_available.npy"
        if availability_path.exists():
            self.canonical_peak_available = np.load(availability_path).astype(bool)
        else:
            self.canonical_peak_available = np.asarray(self.atac.getnnz(axis=0) > 0).ravel()
        with np.load(self.root / "metadata.npz", allow_pickle=True) as meta:
            self.train = meta["train_index"].astype(np.int64)
            self.val = meta["val_index"].astype(np.int64)
            self.test = meta["test_index"].astype(np.int64)
        if self.rna.shape[0] != self.atac.shape[0]:
            raise RuntimeError(f"{name}: RNA and ATAC cell counts differ")
        if self.canonical_peak_available.shape != (self.atac.shape[1],):
            raise RuntimeError(f"{name}: canonical peak availability mask has the wrong shape")
        splits = np.concatenate((self.train, self.val, self.test))
        if np.unique(splits).size != splits.size:
            raise RuntimeError(f"{name}: train/val/test splits overlap")
        if splits.size and (splits.min() < 0 or splits.max() >= self.rna.shape[0]):
            raise RuntimeError(f"{name}: split index is out of range")

    def atac_rows(self, rows: np.ndarray) -> sparse.csr_matrix:
        return self.atac[rows].tocsr()

    def manifest(self) -> dict:
        return json.loads((self.root / "peak_mapping_manifest.json").read_text(encoding="utf-8"))
