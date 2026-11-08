from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def teacher_alignment_metrics(
    student_embedding: np.ndarray,
    teacher_embedding: np.ndarray,
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    student = _normalize(student_embedding)
    teacher = _normalize(teacher_embedding)
    if student.shape[0] != teacher.shape[0]:
        raise ValueError("Student and Teacher embeddings must contain the same cells")
    cells = student.shape[0]
    if cells < 3:
        return {
            "geometry_rmse": float("nan"),
            "ranking_spearman": float("nan"),
            **{f"teacher_recall_at_{k}": float("nan") for k in ks},
            "teacher_top10_mean_rank": float("nan"),
            "teacher_top10_median_rank": float("nan"),
        }
    student_sim = student @ student.T
    teacher_sim = teacher @ teacher.T
    mask = ~np.eye(cells, dtype=bool)
    metrics = {
        "geometry_rmse": float(np.sqrt(np.mean((student_sim[mask] - teacher_sim[mask]) ** 2))),
        "ranking_spearman": float(spearmanr(student_sim[mask], teacher_sim[mask]).statistic),
    }
    student_order = np.argsort(-student_sim, axis=1)
    teacher_order = np.argsort(-teacher_sim, axis=1)
    top_rank_k = min(10, cells - 1)
    recall_values = {k: [] for k in ks}
    top10_ranks = []
    for anchor in range(cells):
        s_order = student_order[anchor][student_order[anchor] != anchor]
        t_order = teacher_order[anchor][teacher_order[anchor] != anchor]
        rank = np.empty(cells, dtype=np.int64)
        rank[s_order] = np.arange(1, cells, dtype=np.int64)
        for k in ks:
            actual_k = min(k, cells - 1)
            recall_values[k].append(
                len(set(t_order[:actual_k]).intersection(s_order[:actual_k])) / max(actual_k, 1)
            )
        top10_ranks.extend(rank[t_order[:top_rank_k]].tolist())
    for k, values in recall_values.items():
        metrics[f"teacher_recall_at_{k}"] = float(np.mean(values))
    metrics["teacher_top10_mean_rank"] = float(np.mean(top10_ranks))
    metrics["teacher_top10_median_rank"] = float(np.median(top10_ranks))
    return metrics
