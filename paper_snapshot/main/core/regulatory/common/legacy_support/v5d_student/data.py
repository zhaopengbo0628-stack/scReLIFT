from __future__ import annotations

from pathlib import Path

import numpy as np


class RNAOnlySource:
    def __init__(self, harmonized_root: Path, name: str):
        self.name = name
        self.root = Path(harmonized_root) / name
        self.rna = np.load(self.root / "rna.npy", mmap_mode="r")
        with np.load(self.root / "metadata.npz", allow_pickle=True) as metadata:
            self.train = metadata["train_index"].astype(np.int64)
            self.val = metadata["val_index"].astype(np.int64)
            self.test = metadata["test_index"].astype(np.int64)
        combined = np.concatenate((self.train, self.val, self.test))
        if np.unique(combined).size != combined.size:
            raise RuntimeError(f"{name}: train/val/test splits overlap")

    def rows(self, split: str) -> np.ndarray:
        return getattr(self, split)


def fit_dataset_balanced_normalization(sources: list[RNAOnlySource]) -> tuple[np.ndarray, np.ndarray]:
    means = []
    variances = []
    for source in sources:
        values = np.asarray(source.rna[source.train], dtype=np.float32)
        means.append(values.mean(axis=0))
        variances.append(values.var(axis=0))
    dataset_means = np.stack(means)
    global_mean = dataset_means.mean(axis=0).astype(np.float32)
    global_variance = np.mean(np.stack(variances) + (dataset_means - global_mean) ** 2, axis=0)
    return global_mean, np.sqrt(global_variance).astype(np.float32)


def normalized_rows(source: RNAOnlySource, rows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(source.rna[rows], dtype=np.float32) - mean) / (std + 1e-6)).astype(np.float32)
