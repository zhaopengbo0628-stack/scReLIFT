from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import fields
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import sparse
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import run_task31_dmode_spectral_landmark_global as parent
from shared_rna_backbone_quality_audit import _normalization, _train_shared_rna
from v5d_student.data import RNAOnlySource, normalized_rows
from v5d_student.config import V5DStudentConfig
from v5d_student.pair_bank import load_pair_bank

SEEDS = (20020130, 20020131, 20020132)
SOURCE_ROOT = "pre_task14_dmode_source_quality_20260926"
OLD_PARENT = "31_dmode_spectral_landmark_global_20260930"
FLAGS = {"EVALUATION_TIME_UNSUPERVISED_SCALING": True, "KNOWN_K_EVALUATION": True,
         "target_labels_used_for_training_or_selection": False, "target_atac_accessed": False,
         "run_tier": "pre_final_user_declared_3seed"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def raw_h5_rna(path, cell_names, genes):
    """Read RNA entries only; never fetch ATAC values from matrix/data."""
    digest = hashlib.sha256()
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        features = matrix["features"]
        rna_rows = np.flatnonzero(features["feature_type"][:].astype(str) == "Gene Expression")
        ids = [x.split(".")[0] for x in features["id"][rna_rows].astype(str)]
        lookup = {x: i for i, x in enumerate(ids)}
        wanted = np.asarray([lookup.get(x, -1) for x in genes], dtype=np.int64)
        cells = {x: i for i, x in enumerate(matrix["barcodes"][:].astype(str))}
        columns = [cells[x] for x in cell_names]
        indptr = matrix["indptr"][:]
        feature_to_rna = np.full(int(matrix["shape"][0]), -1, dtype=np.int64)
        feature_to_rna[rna_rows] = np.arange(rna_rows.size)
        values = np.zeros((len(columns), len(genes)), dtype=np.float32)
        totals = np.zeros(len(columns), dtype=np.float32)
        digest.update("\n".join(ids).encode())
        for row, column in enumerate(columns):
            start, stop = int(indptr[column]), int(indptr[column + 1])
            indices = matrix["indices"][start:stop]
            positions = np.flatnonzero(feature_to_rna[indices] >= 0)
            counts = matrix["data"][start + positions].astype(np.float32)
            rna_indices = feature_to_rna[indices[positions]]
            digest.update(cell_names[row].encode())
            digest.update(rna_indices.tobytes())
            digest.update(counts.tobytes())
            totals[row] = counts.sum(dtype=np.float32)
            full = np.zeros(rna_rows.size, dtype=np.float32)
            np.add.at(full, rna_indices, counts)
            present = wanted >= 0
            values[row, present] = full[wanted[present]]
    return np.log1p(values * (1e4 / np.maximum(totals, 1.0))[:, None]), int((wanted < 0).sum()), digest.hexdigest()


def raw_npz_rna(path, test, genes):
    with np.load(path, allow_pickle=True) as payload:
        counts = sparse.csr_matrix((payload["data"], payload["indices"], payload["indptr"]), shape=tuple(payload["shape"]))[test]
        ids = payload["gene_ids"].astype(str)
    lookup = {x.split(".")[0]: i for i, x in enumerate(ids)}
    wanted = np.asarray([lookup.get(x, -1) for x in genes], dtype=np.int64)
    totals = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    values = np.zeros((len(test), len(genes)), dtype=np.float32)
    present = wanted >= 0
    values[:, present] = counts[:, wanted[present]].toarray().astype(np.float32)
    return np.log1p(values * (1e4 / np.maximum(totals, 1.0))[:, None]), int((wanted < 0).sum()), sha(path)


def verify_cache(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["cache_sha256"].items():
        assert sha(directory / name) == expected, name
    for name, expected in manifest["source_files_sha256"].items():
        assert sha(name) == expected, name
    return manifest


def sources_for(args, fold):
    manifest = verify_cache(args.out / "cache" / fold)
    return [RNAOnlySource(Path(manifest["source_harmonized_root"]), x) for x in manifest["sources"]]


def cache(args):
    for index, (fold, (target, sources)) in enumerate(parent.FOLDS.items()):
        old = args.root / "result_contain" / SOURCE_ROOT / fold
        out = args.out / "cache" / fold
        if (out / "manifest.json").exists():
            verify_cache(out)
            continue
        genes_path = old / "harmonized/common_gene_ids.npy"
        genes = np.load(genes_path, allow_pickle=True).astype(str)
        assert len(genes) == (443, 449, 523, 593)[index]
        target_name = parent.DATASET_NAMES[target]
        with np.load(old / "harmonized" / target_name / "metadata.npz", allow_pickle=True) as meta:
            test = meta["test_index"].astype(np.int64)
        processed = next((args.root / "DATA").glob(f"{target_name}*/processed"))
        if index == 0:
            raw = next((processed.parent / "raw").glob("*.h5"))
            barcodes = np.load(processed / "barcodes.npy", allow_pickle=True).astype(str)[test]
            values, missing, raw_hash = raw_h5_rna(raw, barcodes, genes)
            scope = "RNA genes/counts for target test barcodes only; mixed H5 ATAC bytes not hashed"
        else:
            raw = processed / "rna.npz"
            values, missing, raw_hash = raw_npz_rna(raw, test, genes)
            scope = "RNA-only NPZ file"
        assert missing == 0 and np.isfinite(values).all(), fold
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "target_rna_test.npy", values.astype(np.float32))
        np.save(out / "target_test_index.npy", test)
        np.save(out / "gene_ids.npy", genes)
        source_files = {}
        for key in sources:
            directory = old / "harmonized" / parent.DATASET_NAMES[key]
            for name in ("rna.npy", "metadata.npz"):
                source_files[str(directory / name)] = sha(directory / name)
        write(out / "manifest.json", {
            "protocol": "STRICT_SOURCE_ONLY_ZERO_SHOT", "fold": fold, "target": target_name,
            "sources": [parent.DATASET_NAMES[x] for x in sources],
            "source_harmonized_root": str(old / "harmonized"),
            "gene_universe": "intersection of three SOURCE preselected HVG lists; target excluded",
            "gene_count": len(genes), "gene_universe_source": str(genes_path),
            "source_files_sha256": source_files, "raw_target_source": str(raw),
            "raw_target_sha256": raw_hash, "raw_target_hash_scope": scope,
            "target_rows": len(test), "genuinely_absent_genes": missing,
            "target_transform": "library size 10000 using ALL raw RNA genes per cell; log1p; SOURCE-gene mapping",
            "target_hvg_used": False, "target_population_statistics_fitted": False,
            "target_atac_accessed": False, "target_labels_accessed": False,
            "cohort_caveat": "existing predeclared cohort/test rows retained; historical cohort QC/splits not newly label-free or ATAC-free",
            "cache_sha256": {x: sha(out / x) for x in ("target_rna_test.npy", "target_test_index.npy", "gene_ids.npy")}})
        print("CACHE_COMPLETE", fold, len(test), len(genes), missing, flush=True)


@torch.no_grad()
def infer(model, values, mean, std, raw=False):
    result = []
    for start in range(0, len(values), 512):
        batch = ((np.asarray(values[start:start + 512], np.float32) - mean) / (std + 1e-6)).astype(np.float32)
        tensor = torch.from_numpy(batch).cuda()
        output = model.forward_raw(tensor) if raw else model.encoder(tensor)
        result.append(output.cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def resources(start, inference_time, cells, model):
    torch.cuda.synchronize()
    return {"wall_seconds": time.perf_counter() - start, "inference_seconds": inference_time,
            "inference_cells_per_second": cells / max(inference_time, 1e-8),
            "parameters": sum(x.numel() for x in model.parameters()),
            "trainable_parameters": sum(x.numel() for x in model.parameters() if x.requires_grad),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "gpu": torch.cuda.get_device_name()}


def baseline(args):
    for fold in parent.FOLDS:
        sources = sources_for(args, fold)
        mean, std = _normalization(sources)
        train = np.concatenate([normalized_rows(x, x.train, mean, std) for x in sources])
        val = np.concatenate([normalized_rows(x, x.val, mean, std) for x in sources])
        target = np.load(args.out / "cache" / fold / "target_rna_test.npy", mmap_mode="r")
        for seed in SEEDS:
            out = args.out / "rna" / fold / f"seed_{seed}"
            if (out / "manifest.json").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            model, history = _train_shared_rna(train, val, 512, 64, 0.15, 20, 256, seed, torch.device("cuda"))
            torch.save({"model_state_dict": model.state_dict(), "feature_count": train.shape[1],
                        "seed": seed, "rna_mean": mean, "rna_std": std}, out / "shared_rna_best.pt")
            torch.cuda.synchronize()
            inference_start = time.perf_counter()
            embedding = infer(model, target, mean, std)
            np.save(out / "z_target_test.npy", embedding)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - inference_start
            write(out / "config.json", {"architecture": [train.shape[1], 512, 64, 512, train.shape[1]],
                  "activation": "ReLU", "mask_rate": 0.15, "loss": "MSE", "optimizer": "AdamW",
                  "lr": 1e-3, "weight_decay": 1e-4, "epochs": 20, "batch_size": 256,
                  "checkpoint_selection": "SOURCE validation reconstruction",
                  "normalization": "dataset-balanced SOURCE-train mean/std", "seed": seed})
            write(out / "training_history.json", history)
            write(out / "manifest.json", {**FLAGS, "status": "EMBEDDING_FROZEN", "fold": fold, "seed": seed,
                  "target_labels_accessed": False, "cache_manifest_sha256": sha(args.out / "cache" / fold / "manifest.json"),
                  "checkpoint_sha256": sha(out / "shared_rna_best.pt"), "embedding_sha256": sha(out / "z_target_test.npy"),
                  "best_epoch": min(history, key=lambda x: x["val_reconstruction"])["epoch"],
                  "resources": resources(start, elapsed, len(target), model)})
            print("RNA_COMPLETE", fold, seed, flush=True)
            del model
            torch.cuda.empty_cache()


def load_global(path):
    with np.load(path) as data:
        return parent.GlobalTarget(data["landmarks"].copy(), data["values"].copy(), float(data["mean"]), float(data["std"]), {})


def snapshot(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {"count": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9)), "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def diagnose(model, sources, mean, std, config, banks, targets, training):
    before = snapshot(model)
    gradient, ratios = [], []
    near = hit = elements = 0
    params = [p for p in model.adapter.parameters() if p.requires_grad]
    model.eval()
    for source in sources:
        for split in (source.train, source.val):
            for start in range(0, len(split), 512):
                values = torch.from_numpy(normalized_rows(source, split[start:start + 512], mean, std)).cuda()
                with torch.no_grad():
                    r = model.backbone(values)
                    bounded = torch.tanh(model.adapter(r))
                    delta = model.bound * bounded
                    ratios.extend((delta.norm(dim=1) / (r.norm(dim=1) + 1e-8)).cpu().tolist())
                    near += int((bounded.abs() >= .90).sum())
                    hit += int((bounded.abs() >= .99).sum())
                    elements += bounded.numel()
        rng = np.random.default_rng(20020130 + int(source.name[-3:]))
        for _ in range(4):
            anchors = rng.integers(0, banks[source.name].anchors, size=min(config.batch_anchor, banks[source.name].anchors))
            local, global_loss = parent._batch_losses(model, source, banks[source.name], targets[source.name], anchors,
                                                     mean, std, config, torch.device("cuda"))
            gl = torch.cat([g.flatten() for g in torch.autograd.grad(local, params, retain_graph=True)])
            gg = torch.cat([g.flatten() for g in torch.autograd.grad(global_loss, params)])
            nl, ng = float(gl.norm()), float(gg.norm())
            gradient.append({"source": source.name, "cosine": float(F.cosine_similarity(gl[None], gg[None])),
                             "local_norm": nl, "global_norm": ng,
                             "normalized_global_over_local_norm": (ng / training["global_initial_scale"]) / max(nl / training["local_initial_scale"], 1e-12)})
    assert before == snapshot(model), "Diagnostic changed parameters"
    return {"parameter_updates": False, "model_sha256_before_and_after": before,
            "bound": "elementwise 0.10*tanh; near>=0.90, hit>=0.99",
            "near_bound_fraction": near / elements, "bound_hit_fraction": hit / elements,
            "delta_over_scaffold_norm": distribution(ratios), "gradient_records": gradient}


def d_parent(args):
    for fold in parent.FOLDS:
        out = args.out / "parent" / fold / "seed_20020130"
        if (out / "manifest.json").exists():
            continue
        sources = sources_for(args, fold)
        legacy = args.root / "result_contain" / OLD_PARENT / fold
        old_config = json.loads((legacy / "seed_20020130/config.json").read_text())
        config = V5DStudentConfig(**{f.name: old_config[f.name] for f in fields(V5DStudentConfig) if f.name in old_config})
        saved = torch.load(args.out / "rna" / fold / "seed_20020130/shared_rna_best.pt", map_location="cpu", weights_only=False)
        from v5s.denoise import DenoisingAutoencoder
        dae = DenoisingAutoencoder(saved["feature_count"], 512, 64)
        dae.load_state_dict(saved["model_state_dict"])
        mean, std = saved["rna_mean"], saved["rna_std"]
        np.testing.assert_array_equal(mean, _normalization(sources)[0])
        np.testing.assert_array_equal(std, _normalization(sources)[1])
        train_banks, val_banks, train_targets, val_targets, used_files = {}, {}, {}, {}, {}
        for source in sources:
            bank_dir = args.root / "result_contain/26_dmode_rna_scaffold_bounded_residual_20260929" / fold / "pair_banks"
            tp, vp = bank_dir / f"{source.name}_train.npz", bank_dir / f"{source.name}_val.npz"
            train_banks[source.name], val_banks[source.name] = parent._project_topology_embeddings(load_pair_bank(tp), load_pair_bank(vp), 64)
            for split, store in (("train", train_targets), ("val", val_targets)):
                path = legacy / "global_targets" / f"{source.name}_{split}.npz"
                store[source.name] = load_global(path)
                used_files[str(path)] = sha(path)
            used_files[str(tp)], used_files[str(vp)] = sha(tp), sha(vp)
        weights_path = args.root / "result_contain" / SOURCE_ROOT / fold / "dataset_weights.npz"
        weights = np.load(weights_path)["omega"]
        dataset_weights = {s.name: float(w) for s, w in zip(sources, weights)}
        used_files[str(weights_path)] = sha(weights_path)
        out.mkdir(parents=True, exist_ok=True)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model, training = parent._train(sources, train_banks, val_banks, train_targets, val_targets, mean, std, config,
                                         20020130, torch.device("cuda"), dataset_weights, dae.encoder)
        for name, key in (("best_by_topology.pt", "best_topology_state_dict"), ("best_by_total_normalized_loss.pt", "best_total_state_dict")):
            torch.save({"model_state_dict": training.pop(key), "rna_mean": mean, "rna_std": std, "config": old_config}, out / name)
        write(out / "config.json", old_config)
        write(out / "training.json", training)
        torch.cuda.synchronize()
        inference_start = time.perf_counter()
        target = np.load(args.out / "cache" / fold / "target_rna_test.npy", mmap_mode="r")
        embedding = infer(model, target, mean, std, raw=True)
        np.save(out / "z_target_test.npy", embedding)
        torch.cuda.synchronize()
        inference_time = time.perf_counter() - inference_start
        validation = []
        for source in sources:
            z = infer(model, source.rna[source.val], mean, std, raw=True)
            np.save(out / f"{source.name}_z_val.npy", z)
            validation.append({"source": source.name, "topology": parent._topology_score(z, val_banks[source.name]),
                               "global_teacher_fidelity": parent._global_metric(z, val_targets[source.name])})
        write(out / "source_validation_metrics.json", validation)
        write(out / "diagnostics.json", diagnose(model, sources, mean, std, config, train_banks, train_targets, training))
        write(out / "manifest.json", {**FLAGS, "status": "EMBEDDING_FROZEN", "fold": fold, "seed": 20020130,
              "source_artifact_rebuilds": False, "source_artifact_hashes": used_files,
              "parent_training_function_source": str(Path(parent.__file__)), "parent_training_source_sha256": sha(parent.__file__),
              "cache_manifest_sha256": sha(args.out / "cache" / fold / "manifest.json"),
              "backbone_sha256": sha(args.out / "rna" / fold / "seed_20020130/shared_rna_best.pt"),
              "embedding_sha256": sha(out / "z_target_test.npy"), "target_labels_accessed": False,
              "resources": resources(start, inference_time, len(target), model)})
        print("PARENT_COMPLETE", fold, flush=True)
        del model, dae
        torch.cuda.empty_cache()


def teacher_sanity(args):
    path = args.root / "result_contain/pre_task31_dmode_global_target_audit_20260930/TASK31_STATIC_GLOBAL_TARGET_AUDIT.json"
    audit = json.loads(path.read_text())
    records = []
    for item in audit["records"]:
        files = item["source_files"]
        assert sha(files["full_teacher"]) == files["full_teacher_sha256"]
        assert sha(files["old_local_kl_pair_bank"]) == files["pair_bank_sha256"]
        graph = np.load(files["full_teacher"], mmap_mode="r")
        bank = load_pair_bank(Path(files["old_local_kl_pair_bank"]))
        k = min(15, len(graph) - 1)
        scores = np.asarray(graph).copy()
        np.fill_diagonal(scores, -np.inf)
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        preservation = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(top, bank.partner_index)]))
        records.append({"fold": item["fold"], "source": item["dataset"], "hashes_match": True,
                        "top15_preservation_rechecked": preservation, "cached_overlap": item["current_local_kl_overlap"],
                        "cached_spectral_rank": item["spectral_rank_energy"], "cached_relation": item["relation_target"]})
    return {"source_audit_sha256": sha(path), "checks": records,
            "method": "recheck source Teacher/pair-bank hashes and Top15; reuse hash-bound spectral audit, no eigensystem rebuild",
            "signal_criterion": "all connected, finite landmark targets, centered relation effective rank>=4",
            "pairbank_criterion": "mean low-frequency subspace similarity<0.90; geometric contraction, not proof of dominant failure cause"}


def evaluate(args, include_parent=True):
    from v5d.evaluate_teacher import evaluate_geometry
    rows = []
    methods = (("rna", SEEDS), ("parent", (20020130,))) if include_parent else (("rna", SEEDS),)
    for method, seeds in methods:
        for fold, (target_key, _) in parent.FOLDS.items():
            for seed in seeds:
                out = args.out / method / fold / f"seed_{seed}"
                manifest = json.loads((out / "manifest.json").read_text())
                assert sha(out / "z_target_test.npy") == manifest["embedding_sha256"]
                embedding = np.load(out / "z_target_test.npy")
                test = np.load(args.out / "cache" / fold / "target_test_index.npy")
                processed = next((args.root / "DATA").glob(f"{parent.DATASET_NAMES[target_key]}*/processed"))
                label_path = processed / "label.csv"
                with label_path.open(newline="", encoding="utf-8") as handle:
                    labels = np.asarray([row["cell_type"] for row in csv.DictReader(handle)], dtype=str)[test]
                metrics = evaluate_geometry(embedding, embedding, labels, seed)
                write(out / "target_metrics.json", {**metrics, **FLAGS, "labels_loaded_after_embedding_saved": True,
                      "label_file_sha256": sha(label_path)})
                rows.append({"method": method, "fold": fold, "seed": seed, "ari": metrics["ari"], "nmi": metrics["nmi"]})
    br = [x for x in rows if x["method"] == "rna"]
    seed_means = [{"seed": s, **{m: float(np.mean([x[m] for x in br if x["seed"] == s])) for m in ("ari", "nmi")}} for s in SEEDS]
    summary = {"per_unit": rows, "RNA_seed_fold_means": seed_means,
               "RNA_mean_std": {m: {"mean": float(np.mean([x[m] for x in seed_means])), "std_ddof1": float(np.std([x[m] for x in seed_means], ddof=1))} for m in ("ari", "nmi")},
               "RNA_per_fold_mean_std": {f: {m: {"mean": float(np.mean([x[m] for x in br if x["fold"] == f])), "std_ddof1": float(np.std([x[m] for x in br if x["fold"] == f], ddof=1))} for m in ("ari", "nmi")} for f in parent.FOLDS},
               "D_mean": {m: float(np.mean([x[m] for x in rows if x["method"] == "parent"])) for m in ("ari", "nmi")} if include_parent else None,
               "legacy_rna_ari_descriptive_only": 0.43938163, "legacy_D_ari_descriptive_only": 0.4663946902330116,
               "legacy_used_for_tuning": False}
    write(args.out / "SUMMARY.json", summary)
    with (args.out / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not include_parent:
        print("RNA_EVALUATION_COMPLETE", json.dumps(summary["RNA_mean_std"]), flush=True)
        return
    sanity = teacher_sanity(args)
    write(args.out / "TEACHER_PAIRBANK_SANITY.json", sanity)
    capacity, source_metrics, gradients, trajectory = [], [], [], []
    for fold in parent.FOLDS:
        directory = args.out / "parent" / fold / "seed_20020130"
        diag = json.loads((directory / "diagnostics.json").read_text())
        metrics = json.loads((directory / "source_validation_metrics.json").read_text())
        training = json.loads((directory / "training.json").read_text())
        fit = float(np.mean([x["global_teacher_fidelity"]["normalized_distance_pearson"] for x in metrics]))
        balance = float(np.median([x["normalized_global_over_local_norm"] for x in diag["gradient_records"]]))
        healthy = .1 <= balance <= 10 and all(x["global_norm"] > 1e-6 for x in diag["gradient_records"])
        reproduced = diag["near_bound_fraction"] >= .8 and diag["bound_hit_fraction"] >= .7 and fit < .4 and healthy
        capacity.append({"fold": fold, "reproduced": reproduced, "global_fit": fit, "gradient_norm_ratio_median": balance,
                         "near_bound": diag["near_bound_fraction"], "bound_hit": diag["bound_hit_fraction"],
                         "delta_over_R": diag["delta_over_scaffold_norm"]})
        source_metrics.extend(metrics)
        gradients.extend(diag["gradient_records"])
        trajectory.append({"fold": fold, "best_topology_epoch": training["best_topology_epoch"],
                           "first": training["history"][0], "last": training["history"][-1]})
    reproduced_count = sum(x["reproduced"] for x in capacity)
    capacity_flag = "YES" if reproduced_count == 4 else "NO" if reproduced_count == 0 else "MIXED"
    subspace = [x["cached_overlap"]["leading_low_frequency_subspace_similarity_full_vs_pairbank"] for x in sanity["checks"]]
    preserved = all(x["cached_relation"]["finite"] and x["cached_relation"]["effective_rank_centered_distance_matrix"] >= 4 and x["cached_overlap"]["full_teacher_components"] == 1 for x in sanity["checks"])
    decision = {
        "STRICT_RNA_BASELINE_3SEED": summary["RNA_mean_std"], "STRICT_D_PARENT_SEED30": summary["D_mean"],
        "STRICT_VS_LEGACY_DELTA": {"RNA_ARI": summary["RNA_mean_std"]["ari"]["mean"] - .43938163,
                                  "D_ARI": summary["D_mean"]["ari"] - .4663946902330116},
        "CAPACITY_BOTTLENECK_REPRODUCED": capacity_flag,
        "PAIRBANK_BOTTLENECK_REPRODUCED": "YES" if np.mean(subspace) < .9 else "NO",
        "GLOBAL_TEACHER_SIGNAL_PRESERVED": "YES" if preserved else "NO",
        "CAPACITY_BRANCH_AUTHORIZED": "YES" if capacity_flag == "YES" and preserved else "NO",
        "OT_BRANCH_STILL_RELEVANT": "YES" if np.mean(subspace) < .9 and preserved else "NO",
        "branch_training_launched": False, "capacity_by_fold": capacity, "source_metrics": source_metrics,
        "gradient_cosine": distribution([x["cosine"] for x in gradients]),
        "gradient_fraction_negative": float(np.mean([x["cosine"] < 0 for x in gradients])),
        "normalized_gradient_norm_ratio": distribution([x["normalized_global_over_local_norm"] for x in gradients]),
        "pairbank_subspace_similarity": distribution(subspace), "trajectory": trajectory,
        "pairbank_claim_scope": "hash-preserved geometric information contraction; not proof it is the primary strict-parent causal bottleneck",
        "predeclared_capacity_gate": "all 4 folds near>=0.8, hit>=0.7, Pearson<0.4, norm ratio 0.1..10, nonzero global gradients",
    }
    write(args.out / "FINAL_ADJUDICATION.json", decision)
    print("EVALUATION_COMPLETE", json.dumps(summary["RNA_mean_std"]), json.dumps(summary["D_mean"]), flush=True)
    print("FINAL_DECISION", json.dumps({k: v for k, v in decision.items() if k.isupper()}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/u202522802045/share/zpb"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("cache", "rna", "evaluate_rna", "parent", "evaluate"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, str(args.root / "V5/Teacher-S10x20_Student-64D-KL/source"))
    args.out.mkdir(parents=True, exist_ok=True)
    {"cache": cache, "rna": baseline, "evaluate_rna": lambda a: evaluate(a, False), "parent": d_parent, "evaluate": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
