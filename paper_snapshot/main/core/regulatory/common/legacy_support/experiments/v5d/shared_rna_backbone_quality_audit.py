from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v5d.data import V5DData


DATASET_NAMES = {
    "D1": "PBMC-MO-001",
    "D2": "PBMC-MO-002",
    "D3": "PBMC-MO-003",
    "D4": "PBMC-MO-004",
}
LABEL_PATHS = {
    "D1": "DATA/PBMC-MO-001__10X__GranSorted__10k__Multiome/processed/label.csv",
    "D2": "DATA/PBMC-MO-002__10X__GranSorted__10k__Multiome/processed/label.csv",
    "D3": "DATA/PBMC-MO-003__10X__Unsorted__10k__Multiome/processed/label.csv",
    "D4": "DATA/PBMC-MO-004__10X__Unsorted__3k__Multiome/processed/label.csv",
}
FOLDS = {
    "target_D1": ("D1", ("D2", "D3", "D4")),
    "target_D2": ("D2", ("D1", "D3", "D4")),
    "target_D3": ("D3", ("D1", "D2", "D4")),
    "target_D4": ("D4", ("D1", "D2", "D3")),
}


def _labels(path: Path, rows: np.ndarray) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as handle:
        values = np.asarray([row["cell_type"] for row in csv.DictReader(handle)], dtype=str)
    return values[rows]


def _normalization(sources: list[V5DData]) -> tuple[np.ndarray, np.ndarray]:
    train = [np.asarray(source.rna[source.train], dtype=np.float32) for source in sources]
    means = np.stack([values.mean(axis=0) for values in train])
    mean = means.mean(axis=0).astype(np.float32)
    variance = np.mean(
        np.stack([values.var(axis=0) for values in train]) + (means - mean) ** 2,
        axis=0,
    )
    return mean, np.sqrt(variance).astype(np.float32)


def _rows(source: V5DData, rows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(source.rna[rows], dtype=np.float32) - mean) / (std + 1e-6)).astype(np.float32)


def _l2(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def _purity(q: np.ndarray, labels: np.ndarray, k: int = 15) -> dict[str, object]:
    q = np.asarray(q, dtype=np.float32).copy()
    np.fill_diagonal(q, 0.0)
    k = min(k, q.shape[0] - 1)
    neighbors = np.argpartition(-q, k - 1, axis=1)[:, :k]
    scores = np.take_along_axis(q, neighbors, axis=1)
    same = labels[:, None] == labels[neighbors]
    top = same.mean(axis=1)
    weights = scores / np.maximum(scores.sum(axis=1, keepdims=True), 1e-12)
    per_type = {str(label): float(top[labels == label].mean()) for label in np.unique(labels)}
    return {
        "top15_purity": float(top.mean()),
        "macro_purity": float(np.mean(list(per_type.values()))),
        "q_weighted_purity": float(np.sum(weights * same, axis=1).mean()),
        "per_cell_type_purity": per_type,
    }


def _snf_q(rna: np.ndarray, field: np.ndarray) -> np.ndarray:
    import snf

    views = [
        snf.make_affinity(_l2(values), metric="sqeuclidean", K=20, mu=0.5, normalize=True)
        for values in (rna.astype(np.float32), field.astype(np.float32))
    ]
    fused = np.asarray(snf.snf(*views, K=20, t=20, alpha=1.0), dtype=np.float32)
    np.fill_diagonal(fused, 0.0)
    return fused / np.maximum(fused.sum(axis=1, keepdims=True), 1e-12)


def _train_shared_rna(
    values: np.ndarray,
    validation_values: np.ndarray,
    hidden_dim: int,
    latent_dim: int,
    mask_rate: float,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    from v5s.denoise import DenoisingAutoencoder

    torch.manual_seed(seed)
    model = DenoisingAutoencoder(values.shape[1], hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best_state = None
    best_val = float("inf")
    history = []
    train_tensor = torch.from_numpy(values)
    val_tensor = torch.from_numpy(validation_values).to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(values.shape[0])
        train_total = 0.0
        for start in range(0, order.size, batch_size):
            target = train_tensor[order[start:start + batch_size]].to(device)
            keep = (torch.rand_like(target) > mask_rate).float()
            prediction = model(target * keep)
            loss = torch.nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Shared RNA DAE loss is not finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach().cpu()) * target.shape[0]
        model.eval()
        with torch.no_grad():
            val_loss = float(torch.nn.functional.mse_loss(model(val_tensor), val_tensor).cpu())
        record = {"epoch": epoch, "train_loss": train_total / values.shape[0], "val_reconstruction": val_loss}
        history.append(record)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No shared RNA DAE checkpoint selected")
    model.load_state_dict(best_state)
    return model.eval(), history


@torch.no_grad()
def _encode(model: torch.nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    output = []
    tensor = torch.from_numpy(values)
    for start in range(0, values.shape[0], 512):
        output.append(model.encoder(tensor[start:start + 512].to(device)).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/u202522802045/share/zpb"))
    parser.add_argument("--audit-dir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/pre_task14_dmode_source_quality_20260926"))
    parser.add_argument("--outdir", type=Path, default=Path("/home/u202522802045/share/zpb/result_contain/pre_task14_dmode_shared_rna_audit_20260926"))
    parser.add_argument("--v5-source", type=Path, default=Path("/home/u202522802045/share/zpb/V5/Teacher-S10x20_Student-64D-KL/source"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20020130)
    args = parser.parse_args()
    if args.v5_source.name == "source" and not args.v5_source.exists():
        args.v5_source = Path("/home/u202522802045/share/zpb/V5/Teacher-S10x20_Student-64D-KL/source")
    if str(args.v5_source) not in sys.path:
        sys.path.insert(0, str(args.v5_source))
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    reports = {}
    for fold_name, (target_key, source_keys) in FOLDS.items():
        fold_dir = args.audit_dir / fold_name
        sources = [V5DData(fold_dir / "harmonized", DATASET_NAMES[key]) for key in source_keys]
        mean, std = _normalization(sources)
        train_values = np.concatenate([_rows(source, source.train, mean, std) for source in sources])
        val_values = np.concatenate([_rows(source, source.val, mean, std) for source in sources])
        model, history = _train_shared_rna(train_values, val_values, 512, 64, 0.15, args.epochs, 256, args.seed, device)
        out_fold = args.outdir / fold_name
        out_fold.mkdir()
        torch.save({"model_state_dict": model.state_dict(), "seed": args.seed, "latent_dim": 64, "feature_count": int(train_values.shape[1])}, out_fold / "shared_rna_best.pt")
        source_reports = {}
        for source in sources:
            values = _rows(source, source.val, mean, std)
            z = _encode(model, values, device)
            np.save(out_fold / f"{source.name}_z_rna_shared_val.npy", z)
            f_val = np.load(fold_dir / f"{source.name}_F_val.npy")
            labels_path = args.root / LABEL_PATHS[next(key for key, value in DATASET_NAMES.items() if value == source.name)]
            labels = _labels(labels_path, source.val)
            rna_q = _l2(z) @ _l2(z).T
            teacher_q = _snf_q(z, f_val)
            source_reports[source.name] = {
                "RNA_shared": _purity(rna_q, labels),
                "Teacher_SNF": _purity(teacher_q, labels),
                "train_cells": int(source.train.size),
                "val_cells": int(source.val.size),
            }
        report = {
            "target": DATASET_NAMES[target_key],
            "sources": [DATASET_NAMES[key] for key in source_keys],
            "seed": args.seed,
            "device": str(device),
            "shared_rna_feature_count": int(train_values.shape[1]),
            "training": {"objective": "RNA reconstruction only", "mask_rate": 0.15, "epochs": args.epochs, "checkpoint_selection": "source validation reconstruction", "labels_used": False, "target_used": False},
            "source_quality": source_reports,
            "history": history,
        }
        (out_fold / "manifest.json").write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
        reports[fold_name] = report
    (args.outdir / "SUMMARY.json").write_text(json.dumps(_json_safe({"training_performed": True, "student_training_performed": False, "task14_reserved": False, "folds": reports}), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
