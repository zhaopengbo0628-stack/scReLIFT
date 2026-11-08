from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def _correlations(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    pearson = float(np.corrcoef(left, right)[0, 1]) if left.std() > 0 and right.std() > 0 else float("nan")
    spearman = float(spearmanr(left, right).statistic) if left.size > 1 else float("nan")
    return pearson, spearman


def evaluate_geometry(
    representation: np.ndarray,
    rna_reference: np.ndarray,
    labels: np.ndarray | None,
    seed: int,
    k: int = 15,
) -> dict[str, float]:
    similarity = _normalize(representation) @ _normalize(representation).T
    reference_similarity = _normalize(rna_reference) @ _normalize(rna_reference).T
    off_diagonal = ~np.eye(representation.shape[0], dtype=bool)
    pearson, spearman = _correlations(similarity[off_diagonal], reference_similarity[off_diagonal])
    result: dict[str, float] = {
        "geometry_pearson_to_rna": pearson,
        "geometry_spearman_to_rna": spearman,
        "cells": int(representation.shape[0]),
        "dimensions": int(representation.shape[1]),
    }
    if labels is None:
        return result

    labels = np.asarray(labels)
    distances = 1.0 - similarity
    diagonal = np.eye(labels.size, dtype=bool)
    same = (labels[:, None] == labels[None]) & ~diagonal
    different = (labels[:, None] != labels[None]) & ~diagonal
    same_micro = float(np.mean(distances[same]))
    same_values = [
        np.mean(distances[(labels[:, None] == value) & same])
        for value in np.unique(labels)
        if np.any((labels[:, None] == value) & same)
    ]
    same_macro = float(np.mean(same_values)) if same_values else float("nan")
    different_mean = float(np.mean(distances[different]))
    scaled = StandardScaler().fit_transform(representation)
    clusters = KMeans(n_clusters=np.unique(labels).size, n_init=20, random_state=seed).fit_predict(scaled)
    contingency = contingency_matrix(labels, clusters)
    label_rows, cluster_columns = linear_sum_assignment(-contingency)
    unique_labels = np.unique(labels)
    mapping = {cluster: unique_labels[label] for label, cluster in zip(label_rows, cluster_columns)}
    predicted = np.asarray([mapping[value] for value in clusters])
    neighbor_count = min(k + 1, representation.shape[0])
    neighbors = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine").fit(representation)
    neighbor_index = neighbors.kneighbors(representation, return_distance=False)[:, 1:]
    result.update({
        "ari": float(adjusted_rand_score(labels, clusters)),
        "nmi": float(normalized_mutual_info_score(labels, clusters)),
        "cluster_acc": float(np.mean(predicted == labels)),
        "knn_purity_at_15": float(np.mean(labels[neighbor_index] == labels[:, None])),
        "same_type_distance_micro": same_micro,
        "same_type_distance_macro": same_macro,
        "different_type_distance": different_mean,
        "separation_ratio": different_mean / max(same_micro, 1e-8),
        "silhouette": float(silhouette_score(representation, labels, metric="cosine")),
    })
    return result


def load_split_labels(dataset_root: Path, rows: np.ndarray) -> np.ndarray | None:
    with np.load(dataset_root / "metadata.npz", allow_pickle=True) as metadata:
        if "labels" not in metadata or metadata["labels"].size == 0:
            return None
        return metadata["labels"][rows]
