"""Shared posthoc evaluator for the prepared B2 benchmarks."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                             balanced_accuracy_score, f1_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

KMEANS = dict(init="k-means++", n_init=100, random_state=20020130,
              max_iter=1000, tol=1e-4, algorithm="lloyd")


def evaluate_frozen_embedding(embedding: np.ndarray, labels: np.ndarray) -> tuple[dict, np.ndarray]:
    x = np.asarray(embedding, dtype=np.float64)
    y = np.asarray(labels)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(y) < 16:
        raise ValueError("embedding/label shapes invalid or too small")
    if not np.isfinite(x).all() or len(np.unique(y)) < 2:
        raise ValueError("embedding must be finite and labels must have >=2 classes")
    xs = StandardScaler().fit_transform(x)
    clusters = KMeans(n_clusters=len(np.unique(y)), **KMEANS).fit_predict(xs)
    truth, truth_ids = np.unique(y, return_inverse=True)
    _, cluster_ids = np.unique(clusters, return_inverse=True)
    table = np.zeros((len(truth), len(np.unique(cluster_ids))), dtype=np.int64)
    np.add.at(table, (truth_ids, cluster_ids), 1)
    rows, cols = linear_sum_assignment(-table)
    mapping = dict(zip(cols, rows))
    matched = np.asarray([mapping[c] for c in cluster_ids])
    nn = NearestNeighbors(n_neighbors=min(16, len(y)), metric="euclidean").fit(xs)
    neighbors = nn.kneighbors(return_distance=False)[:, 1:]
    purity = float(np.mean(truth_ids[neighbors] == truth_ids[:, None]))
    metrics = {
        "ARI": float(adjusted_rand_score(truth_ids, cluster_ids)),
        "NMI": float(normalized_mutual_info_score(truth_ids, cluster_ids)),
        "ACC": float(accuracy_score(truth_ids, matched)),
        "Macro_F1": float(f1_score(truth_ids, matched, average="macro", zero_division=0)),
        "Balanced_Accuracy": float(balanced_accuracy_score(truth_ids, matched)),
        "kNN_Purity_at_15": purity,
        "Cell_Type_Silhouette": float(silhouette_score(xs, truth_ids)),
        "known_k": int(len(truth)),
        "labels_loaded_after_embedding_frozen": True,
        "scaler_fit_on_frozen_target_embedding": True,
        "kmeans": KMEANS,
    }
    return metrics, xs.astype(np.float32)
