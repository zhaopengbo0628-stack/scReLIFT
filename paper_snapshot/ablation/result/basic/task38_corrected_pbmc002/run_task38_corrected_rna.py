from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             balanced_accuracy_score, f1_score, precision_recall_fscore_support,
                             silhouette_score)
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/u202522802045/share/zpb")
OUT = ROOT / "task38_corrected_pbmc002_ablation"
STRICT_PATH = ROOT / "result_contain/goal2_strict_preparation_20261002/source/run_strict_lodo_two_step.py"
SEEDS = (20020130, 20020131, 20020132)
EVALUATOR = dict(init="k-means++", n_init=100, random_state=20020130,
                 max_iter=1000, tol=1e-4, algorithm="lloyd")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load_strict():
    sys.path[:0] = [str(ROOT / "dmode_dev/experiments/v5d"),
                    str(ROOT / "V5/Teacher-S10x20_Student-64D-KL/source")]
    spec = importlib.util.spec_from_file_location("task38_strict", STRICT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def labels_for(strict, fold):
    mapping = {"PBMC-MO-001": 1, "PBMC-MO-002": 2,
               "PBMC-MO-003": 3, "PBMC-MO-004": 4}
    target_key, _ = strict.parent.FOLDS[fold]
    name = strict.parent.DATASET_NAMES[target_key]
    processed = next((ROOT / "DATA").glob(f"{name}*/processed"))
    index = np.load(OUT / "cache" / fold / "target_test_index.npy")
    with (processed / "label.csv").open(newline="", encoding="utf-8") as stream:
        labels = np.asarray([row["cell_type"] for row in csv.DictReader(stream)], dtype=str)[index]
    return processed, index, np.unique(labels, return_inverse=True)[1]


def train_model(model, train, val, seed, unit, epochs, batch, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_tensor = torch.from_numpy(train)
    val_tensor = torch.from_numpy(val).to(device)
    generator = np.random.default_rng(seed)
    history, best, best_epoch = [], float("inf"), 0
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "epoch": 0, "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()},
                "kind": "initial"}, unit / "checkpoints/initial.pt")
    for epoch in range(1, epochs + 1):
        model.train()
        order = generator.permutation(len(train))
        total = 0.0
        for start in range(0, len(order), batch):
            target = train_tensor[order[start:start + batch]].to(device)
            keep = (torch.rand_like(target) > 0.15).float()
            loss = torch.nn.functional.mse_loss(model(target * keep), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite source train loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(target)
        model.eval()
        with torch.no_grad():
            val_loss = float(torch.nn.functional.mse_loss(model(val_tensor), val_tensor).cpu())
        row = {"epoch": epoch, "train_reconstruction": total / len(train),
               "source_val_reconstruction": val_loss}
        history.append(row)
        state = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "epoch": epoch, "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                 "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()},
                 "source_val_reconstruction": val_loss}
        torch.save(state, unit / "checkpoints/latest.pt")
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            torch.save(state, unit / "checkpoints/best.pt")
        write(unit / "training_history.json", history)
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epochs, "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()},
                "source_val_reconstruction": history[-1]["source_val_reconstruction"],
                "kind": "final"}, unit / "checkpoints/final.pt")
    best_state = torch.load(unit / "checkpoints/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
    return model.eval(), history, best_epoch


def evaluate(z, labels):
    standardized = StandardScaler().fit_transform(z)
    cluster = KMeans(n_clusters=len(np.unique(labels)), **EVALUATOR).fit_predict(standardized)
    n = len(np.unique(labels))
    table = np.zeros((n, n), dtype=np.int64)
    np.add.at(table, (labels, cluster), 1)
    label_idx, cluster_idx = linear_sum_assignment(-table)
    mapping = dict(zip(cluster_idx, label_idx))
    matched = np.asarray([mapping.get(i, -1) for i in cluster])
    p, r, f, support = precision_recall_fscore_support(labels, matched, labels=np.arange(n), zero_division=0)
    neighbors = NearestNeighbors(n_neighbors=16, metric="euclidean").fit(standardized)
    indices = neighbors.kneighbors(standardized, return_distance=False)[:, 1:]
    purity = float(np.mean(labels[indices] == labels[:, None]))
    return {
        "ari": float(adjusted_rand_score(labels, cluster)),
        "nmi": float(adjusted_mutual_info_score(labels, cluster)),
        "acc_hungarian": float(np.mean(matched == labels)),
        "macro_f1_hungarian": float(f1_score(labels, matched, average="macro", zero_division=0)),
        "balanced_accuracy_hungarian": float(balanced_accuracy_score(labels, matched)),
        "weighted_f1_hungarian": float(f1_score(labels, matched, average="weighted", zero_division=0)),
        "macro_precision_hungarian": float(p.mean()),
        "macro_recall_hungarian": float(r.mean()),
        "per_class": [{"class_index": i, "precision": float(p[i]), "recall": float(r[i]),
                        "f1": float(f[i]), "support": int(support[i])} for i in range(n)],
        "cell_type_silhouette": float(silhouette_score(standardized, labels)),
        "knn_purity_at_15": purity,
        "evaluator": EVALUATOR,
        "EVALUATION_TIME_UNSUPERVISED_SCALING": True,
        "KNOWN_K_EVALUATION": True,
    }, standardized.astype(np.float32)


def run_unit(strict, fold, seed, smoke=False):
    from v5s.denoise import DenoisingAutoencoder
    device = torch.device("cuda")
    cache = OUT / "cache" / fold
    cache_manifest = strict.verify_cache(cache)
    sources = strict.sources_for(argparse.Namespace(out=OUT), fold)
    end_to_end_start = time.perf_counter()
    mean, std = strict._normalization(sources)
    train = np.concatenate([strict.normalized_rows(s, s.train, mean, std) for s in sources])
    val = np.concatenate([strict.normalized_rows(s, s.val, mean, std) for s in sources])
    target = np.load(cache / "target_rna_test.npy", mmap_mode="r")
    unit = (OUT / "smoke_rna" / fold / f"seed_{seed}") if smoke else (OUT / "rna" / fold / f"seed_{seed}")
    if (unit / "manifest.json").exists():
        m = json.loads((unit / "manifest.json").read_text())
        assert sha(unit / "embeddings/target_embedding_raw.npy") == m["target_embedding_sha256"]
        return
    unit.mkdir(parents=True, exist_ok=True)
    (unit / "checkpoints").mkdir(exist_ok=True)
    (unit / "embeddings").mkdir(exist_ok=True)
    (unit / "preprocessing").mkdir(exist_ok=True)
    (unit / "evaluation").mkdir(exist_ok=True)
    (unit / "logs").mkdir(exist_ok=True)
    preprocessing_seconds = time.perf_counter() - end_to_end_start
    config = {"condition": "Strict RNA Baseline", "architecture": [train.shape[1], 512, 64, 512, train.shape[1]],
              "activation": "ReLU", "input_mask_rate": 0.15, "loss": "MSE reconstruction",
              "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.0001,
              "epochs": 1 if smoke else 20, "batch_size": 256, "seed": seed,
              "checkpoint_selection": "source-validation reconstruction only",
              "normalization": "dataset-balanced SOURCE train mean/std",
              "target_atac_used": False, "target_labels_used_for_training_or_selection": False,
              "target_cache_manifest_sha256": sha(cache / "manifest.json"),
              "run_tier": "smoke_only" if smoke else "pre_final"}
    write(unit / "config.json", config)
    write(unit / "data_manifest.json", cache_manifest)
    np.savez_compressed(unit / "preprocessing/source_rna_stats.npz", mean=mean, std=std)
    model = DenoisingAutoencoder(train.shape[1], 512, 64).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, history, best_epoch = train_model(model, train[:2048] if smoke else train,
                                             val[:512] if smoke else val, seed, unit,
                                             1 if smoke else 20, 256, device)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    torch.cuda.synchronize()
    infer_start = time.perf_counter()
    ztarget = strict.infer(model, target[:64] if smoke else target, mean, std)
    torch.cuda.synchronize()
    infer_seconds = time.perf_counter() - infer_start
    np.save(unit / "embeddings/target_embedding_raw.npy", ztarget)
    target_digest = sha(unit / "embeddings/target_embedding_raw.npy")
    standardized = StandardScaler().fit_transform(ztarget).astype(np.float32)
    np.save(unit / "embeddings/target_embedding_standardized.npy", standardized)
    inference_frozen = {"target_embedding_sha256": target_digest,
                        "target_cell_order_sha256": sha(cache / "target_cells.tsv"),
                        "target_rna_sha256": sha(cache / "target_rna_test.npy"),
                        "target_atac_accessed": False, "target_labels_accessed": False,
                        "frozen_before_label_access": True}
    write(unit / "embedding_manifest.json", inference_frozen)
    for split, rows in (("train", [s.train for s in sources]), ("val", [s.val for s in sources])):
        chunks = []
        with torch.no_grad():
            for s, indices in zip(sources, rows):
                x = strict.normalized_rows(s, indices, mean, std)
                chunks.append(model.encoder(torch.from_numpy(x).to(device)).cpu().numpy())
        np.save(unit / f"embeddings/source_{split}_embedding_raw.npy", np.concatenate(chunks).astype(np.float32))
    source_base = Path(cache_manifest["source_harmonized_root"])
    dataset_name_to_key = {name: key for key, name in strict.parent.DATASET_NAMES.items()}
    for split, attr in (("train", "train"), ("val", "val")):
        with (unit / f"embeddings/source_{split}_cells.tsv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["source", "source_row", "cell_id"])
            for s in sources:
                key = dataset_name_to_key[s.name]
                prepared = next((ROOT / "DATA").glob(f"{strict.parent.DATASET_NAMES[key]}*/processed"))
                barcode_file = prepared / "barcodes.npy"
                barcodes = np.load(barcode_file, allow_pickle=True).astype(str) if barcode_file.exists() else np.asarray((prepared / "barcodes.txt").read_text().splitlines(), dtype=str)
                rows_idx = getattr(s, attr)
                for row in rows_idx:
                    writer.writerow([s.name, int(row), barcodes[int(row)]])
    source_rows = sum(len(s.train) + len(s.val) for s in sources)
    with (unit / "embeddings/target_cells.tsv").open("w", encoding="utf-8") as stream:
        stream.write((cache / "target_cells.tsv").read_text(encoding="utf-8"))
    metrics = {}
    if not smoke:
        processed, index, labels = labels_for(strict, fold)
        metrics, _ = evaluate(ztarget, labels)
        metrics.update({"target_labels_loaded_after_embedding_freeze": True,
                        "target_label_file_sha256": sha(processed / "label.csv"),
                        "target_test_index_sha256": sha(cache / "target_test_index.npy")})
    write(unit / "metrics.json", metrics)
    timing = {"preprocessing_seconds": preprocessing_seconds, "training_seconds": train_seconds,
          "target_inference_seconds": infer_seconds, "total_model_train_seconds": train_seconds,
          "total_end_to_end_seconds": time.perf_counter() - end_to_end_start,
          "gpu_hours": train_seconds / 3600, "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
          "gpu_reserved_peak_mb": torch.cuda.max_memory_reserved() / 2**20,
          "gpu": torch.cuda.get_device_name(0), "gpu_vram_mb": torch.cuda.get_device_properties(0).total_memory / 2**20,
          "cuda_version": torch.version.cuda, "pytorch_version": torch.__version__,
          "precision": "float32", "batch_size": 256}
    write(unit / "timing.json", timing)
    write(unit / "parameter_count.json", {"total_params": parameter_count, "trainable_params": parameter_count})
    (unit / "environment.txt").write_text(
        f"{platform.platform()}\nPython {sys.version}\nTorch {torch.__version__}\nNumPy {np.__version__}\n"
        f"CUDA {torch.version.cuda}\nGPU {torch.cuda.get_device_name(0)}\n", encoding="utf-8")
    with (unit / "git_info.txt").open("w", encoding="utf-8") as stream:
        stream.write("Working-tree experiment source; code hashes are recorded in manifest.\n")
    embeddings = {str(p.relative_to(unit)): sha(p) for p in (unit / "embeddings").glob("*") if p.is_file()}
    write(unit / "embedding_manifest.json", {**inference_frozen, "files_sha256": embeddings,
          "source_train_rows": source_rows, "source_val_rows": sum(len(s.val) for s in sources)})
    checkpoints = {str(p.relative_to(unit)): sha(p) for p in (unit / "checkpoints").glob("*.pt")}
    write(unit / "checkpoint_manifest.json", checkpoints)
    write(unit / "manifest.json", {"status": "SMOKE_COMPLETE" if smoke else "EMBEDDING_FROZEN",
          "fold": fold, "seed": seed, "best_epoch": best_epoch, "target_embedding_sha256": target_digest,
          "cache_manifest_sha256": sha(cache / "manifest.json"), "checkpoint_count": len(checkpoints),
          "resources": timing, "total_params": parameter_count, "embedding_files_sha256": embeddings,
          "run_tier": config["run_tier"]})
    print("RNA_SMOKE" if smoke else "RNA_COMPLETE", fold, seed, metrics.get("ari"), metrics.get("nmi"), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()
    strict = load_strict()
    torch.set_num_threads(4)
    if args.stage == "smoke":
        run_unit(strict, "target_D1", 20020130, True)
        return
    for seed in SEEDS:
        for fold in strict.parent.FOLDS:
            run_unit(strict, fold, seed, False)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
