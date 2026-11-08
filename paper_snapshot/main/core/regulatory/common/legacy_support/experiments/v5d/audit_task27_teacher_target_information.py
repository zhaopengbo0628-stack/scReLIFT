from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

TEMPERATURES = (0.01, 0.02, 0.05, 0.07, 0.10)


def summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)), "p10": float(np.quantile(values, 0.1)), "p90": float(np.quantile(values, 0.9))}


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = scores.astype(np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    values = np.exp(scaled)
    return values / values.sum(axis=1, keepdims=True)


def audit_bank(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as values:
        scores = values["teacher_similarity"].astype(np.float64)
        local_k = int(values["local_k"])
    ordered = np.sort(scores, axis=1)
    means, stds = scores.mean(axis=1), scores.std(axis=1)
    raw = {
        "score_mean": summary(means), "score_std": summary(stds),
        "score_range": summary(ordered[:, -1] - ordered[:, 0]),
        "top1_top2_margin": summary(ordered[:, -1] - ordered[:, -2]),
        "top1_median_margin": summary(ordered[:, -1] - np.median(scores, axis=1)),
        "local_mean": summary(scores[:, :local_k].mean(axis=1)),
        "random_mean": summary(scores[:, local_k:].mean(axis=1)),
        "local_random_gap": summary(scores[:, :local_k].mean(axis=1) - scores[:, local_k:].mean(axis=1)),
        "coefficient_of_variation": summary(stds / np.maximum(np.abs(means), 1e-12)),
        "candidate_sum": summary(scores.sum(axis=1)),
    }
    log_width = math.log(scores.shape[1])
    q_by_temperature = {}
    for temperature in TEMPERATURES:
        q = softmax(scores, temperature)
        entropy = -np.sum(q * np.log(np.maximum(q, 1e-300)), axis=1)
        ordered_q = np.sort(q, axis=1)
        q_by_temperature[f"{temperature:.2f}"] = {
            "entropy": summary(entropy),
            "normalized_entropy": summary(entropy / log_width),
            "information_above_uniform": summary(log_width - entropy),
            "effective_candidate_count": summary(np.exp(entropy)),
            "top1_probability": summary(ordered_q[:, -1]),
            "top5_probability_mass": summary(ordered_q[:, -5:].sum(axis=1)),
            "local_probability_mass": summary(q[:, :local_k].sum(axis=1)),
            "random_probability_mass": summary(q[:, local_k:].sum(axis=1)),
        }
    return {
        "path": str(path), "anchors": int(scores.shape[0]), "candidate_count": int(scores.shape[1]), "local_k": local_k,
        "candidate_scores_sum_to_one": bool(np.allclose(scores.sum(axis=1), 1.0, atol=1e-5)),
        "raw_scores": raw, "q_by_temperature": q_by_temperature,
    }


def aggregate(records: list[dict[str, object]], temperature: float) -> dict[str, float]:
    key = f"{temperature:.2f}"
    weights = np.asarray([record["anchors"] for record in records], dtype=np.float64)

    def weighted(path: tuple[str, ...]) -> float:
        values = []
        for record in records:
            value = record
            for part in path:
                value = value[part]
            values.append(float(value))
        return float(np.average(values, weights=weights))

    result = {
        "anchors": int(weights.sum()),
        "raw_score_std": weighted(("raw_scores", "score_std", "mean")),
        "raw_local_random_gap": weighted(("raw_scores", "local_random_gap", "mean")),
        "candidate_sum": weighted(("raw_scores", "candidate_sum", "mean")),
    }
    for field in ("entropy", "normalized_entropy", "information_above_uniform", "effective_candidate_count", "top1_probability", "top5_probability_mass", "local_probability_mass", "random_probability_mass"):
        result[field] = weighted(("q_by_temperature", key, field, "mean"))
    return result


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task26-dir", type=Path, required=True)
    parser.add_argument("--task10-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    dmode: dict[str, dict[str, dict[str, object]]] = {}
    for path in sorted(args.task26_dir.glob("target_D*/pair_banks/*_val.npz")):
        dmode.setdefault(path.parents[1].name, {})[path.name.removesuffix("_val.npz")] = audit_bank(path)
    if len(dmode) != 4 or any(len(values) != 3 for values in dmode.values()):
        raise RuntimeError("Expected 4 Task26 folds with 3 source validation banks each")
    task10 = {path.parents[1].name: audit_bank(path) for path in sorted(args.task10_dir.glob("D*/seed_20020130/val_pair_bank.npz"))}
    if len(task10) != 4:
        raise RuntimeError("Expected 4 Task10 validation banks")
    fold_summary = {fold: aggregate(list(values.values()), 0.07) for fold, values in dmode.items()}
    all_dmode = [record for values in dmode.values() for record in values.values()]
    temperature_summary = {f"{temperature:.2f}": aggregate(all_dmode, temperature) for temperature in TEMPERATURES}
    task10_summary = aggregate(list(task10.values()), 0.05)
    actual, cold = temperature_summary["0.07"], temperature_summary["0.01"]
    information_values = np.asarray([row["information_above_uniform"] for row in fold_summary.values()])
    weak = actual["normalized_entropy"] >= 0.98
    scale_sensitive = cold["normalized_entropy"] <= actual["normalized_entropy"] - 0.01
    ordering = actual["raw_local_random_gap"] > 0 and actual["raw_score_std"] > 0
    dataset_dependent = information_values.max() >= 2 * max(information_values.min(), 1e-12)
    information = "DATASET_DEPENDENT" if dataset_dependent else ("WEAK" if weak else "STRONG")
    if weak and ordering and scale_sensitive:
        flatness, bottleneck = "TEMPERATURE_SCALE_MISMATCH", "TEACHER_TARGET_SCALE_MISMATCH"
    elif weak and not ordering:
        flatness, bottleneck = "PAIR_BANK_NOT_DISCRIMINATIVE", "PAIR_BANK_INFORMATION_BOTTLENECK"
    elif weak:
        flatness, bottleneck = "COMBINATION", "TEACHER_INFORMATION_WEAK_AT_CANDIDATE_LEVEL"
    else:
        flatness, bottleneck = "UNCERTAIN", "UNRESOLVED"
    report = {
        "task_id": 27, "experiment": "D-mode Teacher target information content audit", "training_performed": False, "gpu_used": False,
        "teacher_retrained": False, "student_trained": False, "snf_recomputed": False, "B_or_F_recomputed": False, "labels_used": False,
        "source_commit": args.source_commit or git_commit(args.repo), "temperatures": list(TEMPERATURES), "task26_actual_temperature": 0.07, "task10_actual_temperature": 0.05,
        "dmode_records": dmode, "dmode_fold_summary": fold_summary, "dmode_temperature_summary": temperature_summary,
        "task10_records": task10, "task10_summary": task10_summary,
        "task10_candidate_scores_sum_to_one": all(record["candidate_scores_sum_to_one"] for record in task10.values()),
        "task26_candidate_scores_sum_to_one": all(record["candidate_scores_sum_to_one"] for record in all_dmode),
        "classification": {
            "teacher_target_information": information, "target_flatness_source": flatness, "primary_bottleneck": bottleneck,
            "d1_more_informative_than_other_folds": bool(fold_summary["target_D1"]["information_above_uniform"] > np.mean([fold_summary[name]["information_above_uniform"] for name in ("target_D2", "target_D3", "target_D4")])),
            "safe_to_train_next_task": "NO",
        },
    }
    (args.outdir / "TASK27_TEACHER_TARGET_INFORMATION_AUDIT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.outdir / "manifest.json").write_text(json.dumps({key: report[key] for key in ("task_id", "experiment", "training_performed", "gpu_used", "teacher_retrained", "student_trained", "snf_recomputed", "B_or_F_recomputed", "labels_used", "source_commit")}, indent=2), encoding="utf-8")
    print(json.dumps(report["classification"], indent=2))


if __name__ == "__main__":
    main()
