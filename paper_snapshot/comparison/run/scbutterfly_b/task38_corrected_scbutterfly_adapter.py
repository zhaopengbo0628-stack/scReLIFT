import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score, adjusted_rand_score,
                             balanced_accuracy_score, f1_score, precision_recall_fscore_support,
                             silhouette_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path("/home/u202522802045/share/zpb")
OUT = ROOT / "task38_corrected_pbmc002_ablation"
SOURCE = ROOT / "result_contain/34_strict_d_scbutterfly_goal2_20261002/source/run_goal2_strict_scbutterfly.py"
sys.path.insert(0, str(ROOT / "dmode_dev/experiments/v5d"))
spec = importlib.util.spec_from_file_location("task38_scbutterfly", SOURCE)
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)
sc.PREP = OUT
sc.GOAL1 = OUT
sc.strict.GOAL1 = OUT
sc.branches.GOAL1 = OUT
_branch_read = sc.branches.read


def branch_read(path):
    result = _branch_read(path)
    path = Path(path)
    if path.name == "peak_harmonization_manifest.json":
        fold = path.parent.parent.name
        cache = _branch_read(OUT / "cache" / fold / "manifest.json")
        references = set(result.get("feature_reference_datasets", []))
        assert result.get("dataset_count") == 3
        assert result.get("held_out_target_dataset") is None
        assert references == set(cache["sources"])
        assert cache["target"] not in references
        result["target_excluded_from_feature_universe"] = True
    return result


sc.branches.read = branch_read
FOLDS = [f"target_D{i}" for i in range(1, 5)]
SEEDS = [20020130, 20020131, 20020132]
EVALUATOR = dict(init="k-means++", n_init=100, random_state=20020130,
                 max_iter=1000, tol=1e-4, algorithm="lloyd")


def labels_after_frozen(fold):
    manifest = _branch_read(OUT / "cache" / fold / "manifest.json")
    target = manifest["target"]
    processed = next((ROOT / "DATA").glob(f"{target}*/processed"))
    indices = np.load(OUT / "cache" / fold / "target_test_index.npy")
    with (processed / "label.csv").open(newline="", encoding="utf-8") as stream:
        labels = np.asarray([row["cell_type"] for row in csv.DictReader(stream)], dtype=str)[indices]
    return np.unique(labels, return_inverse=True)[1], labels


def evaluate_unit(fold, seed):
    unit = OUT / "scbutterfly" / fold / f"seed_{seed}"
    manifest = sc.branches.read(unit / "manifest.json")
    prediction = unit / "predicted_atac.npy"
    assert sc.strict.sha(prediction) == manifest["prediction_sha256"]
    x = np.load(prediction, mmap_mode="r")
    frozen_hash = sc.strict.sha(prediction)
    labels, raw_labels = labels_after_frozen(fold)
    standardized = StandardScaler().fit_transform(x)
    clusters = KMeans(n_clusters=len(np.unique(labels)), **EVALUATOR).fit_predict(standardized)
    size = len(np.unique(labels))
    table = np.zeros((size, size), dtype=np.int64)
    np.add.at(table, (labels, clusters), 1)
    label_idx, cluster_idx = linear_sum_assignment(-table)
    mapping = dict(zip(cluster_idx, label_idx))
    matched = np.asarray([mapping.get(value, -1) for value in clusters])
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, matched, labels=np.arange(size), zero_division=0)
    neighbors = NearestNeighbors(n_neighbors=16, metric="euclidean").fit(standardized)
    indices = neighbors.kneighbors(standardized, return_distance=False)[:, 1:]
    metrics = dict(
        condition="scButterfly", fold=fold, seed=seed,
        ari=float(adjusted_rand_score(labels, clusters)),
        nmi=float(adjusted_mutual_info_score(labels, clusters)),
        acc_hungarian=float(accuracy_score(labels, matched)),
        macro_f1_hungarian=float(f1_score(labels, matched, average="macro", zero_division=0)),
        balanced_accuracy_hungarian=float(balanced_accuracy_score(labels, matched)),
        knn_purity_at_15=float(np.mean(labels[indices] == labels[:, None])),
        cell_type_silhouette=float(silhouette_score(standardized, labels)),
        weighted_f1_hungarian=float(f1_score(labels, matched, average="weighted", zero_division=0)),
        macro_precision_hungarian=float(precision.mean()), macro_recall_hungarian=float(recall.mean()),
        per_class=[dict(cell_type=str(name), precision=float(precision[i]), recall=float(recall[i]),
                        f1=float(f1[i]), support=int(support[i])) for i, name in enumerate(np.unique(raw_labels))],
        evaluator=EVALUATOR, EVALUATION_TIME_UNSUPERVISED_SCALING=True,
        KNOWN_K_EVALUATION=True, target_atac_used=False, labels_loaded_after_embedding_frozen=True,
        target_prediction_sha256=frozen_hash)
    assert sc.strict.sha(prediction) == frozen_hash
    sc.strict.write(unit / "metrics.json", metrics)
    np.save(unit / "target_embedding_standardized.npy", standardized.astype(np.float32))
    sc.strict.write(unit / "embedding_manifest.json", dict(
        target_embedding_source=str(prediction), target_embedding_sha256=frozen_hash,
        standardized_embedding_sha256=sc.strict.sha(unit / "target_embedding_standardized.npy"),
        target_atac_used=False, labels_accessed_after_prediction_freeze=True))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    sc.prepare()
    if args.prepare_only:
        print("CORRECTED_SCBUTTERFLY_CACHE_PASS", flush=True)
        return
    if args.smoke_only:
        import fcntl
        with (ROOT / ".benchmark_a30.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            sc.run_unit(OUT / "scbutterfly_smoke", FOLDS[0], SEEDS[0], True)
        print("CORRECTED_SCBUTTERFLY_SMOKE_PASS", flush=True)
        return
    rows = []
    import fcntl
    with (ROOT / ".benchmark_a30.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for seed in SEEDS:
            for fold in FOLDS:
                started = time.perf_counter()
                sc.run_unit(OUT / "scbutterfly", fold, seed, False)
                metrics = evaluate_unit(fold, seed)
                metrics["total_wall_seconds"] = time.perf_counter() - started
                sc.strict.write(OUT / "scbutterfly" / fold / f"seed_{seed}/metrics.json", metrics)
                rows.append(metrics)
                sc.strict.write(OUT / "scbutterfly" / "progress.json", dict(
                    status="RUNNING", completed_units=len(rows), total_units=12, last_fold=fold,
                    last_seed=seed, updated_at=time.time()))
                print("CORRECTED_SCBUTTERFLY_COMPLETE", fold, seed, metrics["ari"], metrics["nmi"], flush=True)
                torch.cuda.empty_cache()
    sc.strict.write(OUT / "scbutterfly" / "progress.json", dict(
        status="COMPLETE", completed_units=len(rows), total_units=12, updated_at=time.time()))


if __name__ == "__main__":
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
