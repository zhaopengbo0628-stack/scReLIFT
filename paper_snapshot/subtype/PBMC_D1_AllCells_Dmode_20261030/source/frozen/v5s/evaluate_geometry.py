from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, confusion_matrix, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .geometry import cosine_similarity


def evaluate_test_geometry(values: np.ndarray, test_labels: np.ndarray | None, rna_reference: np.ndarray, seed: int, k: int = 15) -> dict[str, float]:
    similarity = cosine_similarity(values)
    rna_similarity = cosine_similarity(rna_reference)
    off_diagonal = ~np.eye(values.shape[0], dtype=bool)
    left, right = similarity[off_diagonal], rna_similarity[off_diagonal]
    result = {
        "geometry_pearson": float(np.corrcoef(left, right)[0, 1]),
        "geometry_spearman": float(spearmanr(left, right).statistic),
        "cells": int(values.shape[0]),
        "dimensions": int(values.shape[1]),
    }
    if test_labels is None or test_labels.size == 0:
        return result
    distances = 1.0 - similarity
    np.fill_diagonal(distances, np.nan)
    same = test_labels[:, None] == test_labels[None, :]
    same[np.eye(test_labels.size, dtype=bool)] = False
    different = ~same
    different[np.eye(test_labels.size, dtype=bool)] = False
    same_micro = float(np.nanmean(distances[same]))
    same_by_type = [
        np.nanmean(distances[same & (test_labels[:, None] == value)])
        for value in np.unique(test_labels)
        if np.any(same & (test_labels[:, None] == value))
    ]
    different_mean = float(np.nanmean(distances[different]))
    scaled = StandardScaler().fit_transform(values)
    clusters = KMeans(n_clusters=np.unique(test_labels).size, n_init=20, random_state=seed).fit_predict(scaled)
    contingency = confusion_matrix(test_labels, clusters)
    label_index, cluster_index = linear_sum_assignment(-contingency)
    mapping = {cluster: label for label, cluster in zip(label_index, cluster_index)}
    prediction = np.asarray([mapping[cluster] for cluster in clusters])
    neighbors = NearestNeighbors(n_neighbors=min(k + 1, values.shape[0]), metric="cosine").fit(values)
    neighbor_index = neighbors.kneighbors(return_distance=False)[:, 1:]
    result.update({
        "ari": float(adjusted_rand_score(test_labels, clusters)),
        "nmi": float(normalized_mutual_info_score(test_labels, clusters)),
        "cluster_acc": float(accuracy_score(test_labels, prediction)),
        "knn_purity_at_15": float(np.mean(test_labels[neighbor_index] == test_labels[:, None])),
        "same_type_distance_micro": same_micro,
        "same_type_distance_macro": float(np.nanmean(same_by_type)),
        "different_type_distance": different_mean,
        "separation_ratio": different_mean / max(same_micro, 1e-8),
        "silhouette": float(silhouette_score(values, test_labels, metric="cosine")),
    })
    return result


def evaluate_saved_test_geometry(data_dir: Path, representation: np.ndarray, rna_reference: np.ndarray, seed: int) -> dict[str, float]:
    with np.load(data_dir / "metadata.npz", allow_pickle=True) as metadata:
        test_index = metadata["test_index"].astype(np.int64)
        labels = metadata.get("labels", np.asarray([], dtype=np.int64))
        test_labels = labels.astype(np.int64)[test_index] if labels.size == metadata["train_index"].size + metadata["val_index"].size + metadata["test_index"].size else None
    return evaluate_test_geometry(representation, test_labels, rna_reference, seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--rna-reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20020130)
    args = parser.parse_args()
    with np.load(args.data_dir / "metadata.npz", allow_pickle=True) as metadata:
        test_index = metadata["test_index"].astype(np.int64)
    representation = np.load(args.representation)
    rna_reference = np.load(args.rna_reference)
    if representation.shape[0] != test_index.size or rna_reference.shape[0] != test_index.size:
        raise RuntimeError("Evaluation arrays must contain test cells in split order")
    with np.load(args.data_dir / "metadata.npz", allow_pickle=True) as metadata:
        test_labels = metadata["labels"].astype(np.int64)[test_index]
    result = evaluate_test_geometry(representation, test_labels, rna_reference, args.seed)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
