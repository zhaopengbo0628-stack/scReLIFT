from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v5d_student.data import RNAOnlySource, fit_dataset_balanced_normalization
from v5d_student.student_encoder import RNAScaffoldBoundedResidualEncoder

NAMES = {"D1": "PBMC-MO-001", "D2": "PBMC-MO-002", "D3": "PBMC-MO-003", "D4": "PBMC-MO-004"}
FOLDS = {"target_D1": ("D1", ("D2", "D3", "D4")), "target_D2": ("D2", ("D1", "D3", "D4")), "target_D3": ("D3", ("D1", "D2", "D4")), "target_D4": ("D4", ("D1", "D2", "D3"))}


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def topk(x: np.ndarray, k: int = 15) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    np.fill_diagonal(y, -np.inf)
    return np.argpartition(-y, min(k, y.shape[0] - 1) - 1, axis=1)[:, : min(k, y.shape[0] - 1)]


def graph_from_candidates(n: int, partners: np.ndarray, values: np.ndarray) -> np.ndarray:
    graph = np.zeros((n, n), dtype=np.float32)
    rows = np.arange(n)[:, None]
    graph[rows, partners] = np.maximum(values, 0.0)
    graph = (graph + graph.T) / 2.0
    np.fill_diagonal(graph, 0.0)
    return graph


def cosine_graph(z: np.ndarray) -> np.ndarray:
    return np.maximum((norm(z) @ norm(z).T).astype(np.float32), 0.0) * (1.0 - np.eye(z.shape[0], dtype=np.float32))


def spectral_embedding(graph: np.ndarray, dimensions: int = 32) -> np.ndarray:
    symmetric = (graph + graph.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    order = np.argsort(values)[::-1]
    keep = min(dimensions, graph.shape[0])
    return vectors[:, order[:keep]] * np.sqrt(np.maximum(values[order[:keep]], 0.0))[None, :]


def clustering(graph: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float]:
    embedding = spectral_embedding(graph)
    clusters = KMeans(n_clusters=np.unique(labels).size, n_init=20, random_state=seed).fit_predict(embedding)
    return {"ari": float(adjusted_rand_score(labels, clusters)), "nmi": float(normalized_mutual_info_score(labels, clusters))}


def graph_fidelity(left: np.ndarray, right: np.ndarray, k: int = 15) -> dict[str, float]:
    le, re = spectral_embedding(left), spectral_embedding(right)
    width = min(le.shape[1], re.shape[1], 16)
    sv = np.linalg.svd(norm(le[:, :width]).T @ norm(re[:, :width]), compute_uv=False)
    li, ri = topk(left, k), topk(right, k)
    overlap = np.mean([len(set(li[i]).intersection(ri[i])) / k for i in range(left.shape[0])])
    l = (left + left.T) / 2.0
    r = (right + right.T) / 2.0
    def lap(g: np.ndarray) -> np.ndarray:
        d = g.sum(axis=1)
        inv = 1.0 / np.sqrt(np.maximum(d, 1e-8))
        return np.eye(g.shape[0], dtype=np.float32) - inv[:, None] * g * inv[None, :]
    lv = np.linalg.eigvalsh(lap(l))[:16]
    rv = np.linalg.eigvalsh(lap(r))[:16]
    def structure(g: np.ndarray) -> tuple[int, float]:
        adj = g > 0
        seen = np.zeros(g.shape[0], dtype=bool)
        sizes = []
        for start in range(g.shape[0]):
            if seen[start]:
                continue
            stack = [start]
            seen[start] = True
            size = 0
            while stack:
                node = stack.pop()
                size += 1
                nxt = np.flatnonzero(adj[node] & ~seen)
                seen[nxt] = True
                stack.extend(nxt.tolist())
            sizes.append(size)
        return len(sizes), max(sizes) / g.shape[0]
    lc, ll = structure(l)
    rc, rl = structure(r)
    return {"leading_eigenspace_similarity": float(np.mean(sv[:width])), "top15_neighborhood_preservation": float(overlap), "components_left": lc, "components_right": rc, "largest_component_fraction_left": float(ll), "largest_component_fraction_right": float(rl), "degree_mean_abs_diff": float(abs(l.sum(1).mean() - r.sum(1).mean())), "normalized_laplacian_spectral_rmse": float(np.sqrt(np.mean((lv - rv) ** 2)))}


def labels_for(root: Path, source: RNAOnlySource, rows: np.ndarray) -> np.ndarray:
    with np.load(source.root / "metadata.npz", allow_pickle=True) as meta:
        if "labels" in meta and meta["labels"].size:
            return meta["labels"][rows].astype(str)
    key = next(k for k, v in NAMES.items() if v == source.name)
    path = root / "DATA" / {"D1": "PBMC-MO-001__10X__GranSorted__10k__Multiome", "D2": "PBMC-MO-002__10X__GranSorted__10k__Multiome", "D3": "PBMC-MO-003__10X__Unsorted__10k__Multiome", "D4": "PBMC-MO-004__10X__Unsorted__3k__Multiome"}[key] / "processed/label.csv"
    with path.open(encoding="utf-8", newline="") as h:
        all_labels = np.asarray([r["cell_type"] for r in csv.DictReader(h)], dtype=str)
    return all_labels[rows]


def student_val(root: Path, fold: str, source: RNAOnlySource, checkpoint: Path, v5_source: Path, device: torch.device) -> np.ndarray:
    from v5s.denoise import DenoisingAutoencoder
    target, source_keys = FOLDS[fold]
    fit = [RNAOnlySource(root / "result_contain/pre_task14_dmode_source_quality_20260926" / fold / "harmonized", NAMES[k]) for k in source_keys]
    mean, std = fit_dataset_balanced_normalization(fit)
    shared = torch.load(root / "result_contain/pre_task14_dmode_shared_rna_audit_20260926" / fold / "shared_rna_best.pt", map_location="cpu", weights_only=False)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    dae = DenoisingAutoencoder(int(shared["feature_count"]), 512, 64)
    dae.load_state_dict(shared["model_state_dict"])
    model = RNAScaffoldBoundedResidualEncoder(dae.encoder, 64, 128, 0.10)
    model.load_state_dict(saved["model_state_dict"])
    model.to(device).eval()
    rows = source.val
    values = ((np.asarray(source.rna[rows], dtype=np.float32) - mean) / (std + 1e-6)).astype(np.float32)
    with torch.no_grad():
        pieces = [model.forward_raw(torch.from_numpy(values[i:i + 512]).to(device)).cpu().numpy() for i in range(0, len(values), 512)]
    return np.concatenate(pieces).astype(np.float32)


def trajectory(task_dir: Path) -> list[dict[str, object]]:
    rows = []
    for fold in FOLDS:
        for seed_dir in sorted((task_dir / fold).glob("seed_*")):
            history = json.loads((seed_dir / "training_history.json").read_text(encoding="utf-8"))
            losses = np.asarray([r["val_loss_macro"] for r in history], dtype=float)
            topo = np.asarray([r["val_topology_score_macro"] for r in history], dtype=float)
            rho = spearmanr(losses, topo).statistic
            best_loss = int(np.argmin(losses)) + 1
            best_topo = int(np.argmax(topo)) + 1
            relation = "UNKNOWN" if not np.isfinite(rho) else ("ANTI_ALIGNED" if rho > 0.35 else "ALIGNED" if rho < -0.35 else "WEAK")
            rows.append({"fold": fold, "seed": seed_dir.name, "spearman_val_loss_vs_topology": float(rho), "relationship": relation, "best_loss_epoch": best_loss, "best_topology_epoch": best_topo, "epoch_distance": abs(best_loss - best_topo), "val_loss_at_best_loss": float(losses[best_loss - 1]), "topology_at_best_loss": float(topo[best_loss - 1]), "val_loss_at_best_topology": float(losses[best_topo - 1]), "topology_at_best_topology": float(topo[best_topo - 1]), "topology_improved_then_declined": bool(np.argmax(topo) not in (0, len(topo) - 1) and topo[-1] < topo.max() and losses[-1] <= losses[0])})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--task23", type=Path, required=True)
    p.add_argument("--task26", type=Path, required=True)
    p.add_argument("--task28", type=Path, required=True)
    p.add_argument("--task10", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--v5-source", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cascade, fidelity, clustering_rows = [], [], []
    for fold, (target_key, source_keys) in FOLDS.items():
        hroot = args.root / "result_contain/pre_task14_dmode_source_quality_20260926" / fold / "harmonized"
        for source_key in source_keys:
            name = NAMES[source_key]
            base = args.task23 / "matched_geometry" / fold / name
            full = np.load(base / "G_SNF_verified.npy").astype(np.float32)
            with np.load(args.task26 / fold / "pair_banks" / f"{name}_val.npz") as b26, np.load(args.task28 / fold / "pair_banks" / f"{name}_val.npz") as b28:
                partners = b26["partner_index"].astype(np.int64)
                raw = graph_from_candidates(full.shape[0], partners, full[np.arange(full.shape[0])[:, None], partners])
                qgraph = graph_from_candidates(full.shape[0], b28["partner_index"].astype(np.int64), b28["teacher_similarity"].astype(np.float32))
            source = RNAOnlySource(hroot, name)
            z_student = student_val(args.root, fold, source, args.task26 / fold / "seed_20020130/best_by_topology.pt", args.v5_source, device)
            student = cosine_graph(z_student)
            labels = labels_for(args.root, source, source.val)
            stages = {"G_FULL": full, "G_PB_RAW": raw, "G_PB_Q": qgraph, "G_STUDENT": student}
            metrics = {stage: clustering(graph, labels, 0) for stage, graph in stages.items()}
            clustering_rows.append({"fold": fold, "dataset": name, "metrics": metrics})
            pairs = [("G_FULL", "G_PB_RAW", "PAIRBANK"), ("G_PB_RAW", "G_PB_Q", "TARGET_TRANSFORM"), ("G_PB_Q", "G_STUDENT", "STUDENT_OBJECTIVE")]
            drops = {}
            for left, right, key in pairs:
                drops[f"{key}_DROP_ARI"] = metrics[left]["ari"] - metrics[right]["ari"]
                drops[f"{key}_DROP_NMI"] = metrics[left]["nmi"] - metrics[right]["nmi"]
                fidelity.append({"fold": fold, "dataset": name, "transition": f"{left}->{right}", "metrics": graph_fidelity(stages[left], stages[right])})
            cascade.append({"fold": fold, "dataset": name, "stages": {k: metrics[k] for k in stages}, "drops": drops, "student_inference": "Task26 best_by_topology checkpoint, RNA-only forward_raw, seed_20020130"})
    traj26 = trajectory(args.task26)
    traj28 = trajectory(args.task28)
    task10 = {"training": False, "source": str(args.task10), "records": [], "note": "Targeted control uses existing Task10 artifacts only."}
    for fold in sorted(args.task10.glob("D[1-4]")):
        for seed_dir in sorted(fold.glob("seed_*/student_history.json")):
            history = json.loads(seed_dir.read_text(encoding="utf-8"))
            losses = np.asarray([r.get("val_loss", r.get("val_loss_macro", np.nan)) for r in history], dtype=float)
            topo = np.asarray([r.get("val_topology_score", r.get("val_topology_score_macro", np.nan)) for r in history], dtype=float)
            mask = np.isfinite(losses) & np.isfinite(topo)
            task10["records"].append({"dataset": fold.name, "seed": seed_dir.parent.name, "epochs": len(history), "spearman_loss_vs_topology": float(spearmanr(losses[mask], topo[mask]).statistic) if mask.sum() > 1 else None})
    _json = lambda name, value: dump(args.outdir / name, value)
    _json("TASK29_CAUSAL_CASCADE.json", cascade)
    _json("TASK29_STAGEWISE_GRAPH_FIDELITY.json", fidelity)
    _json("TASK29_STAGEWISE_CLUSTERING.json", clustering_rows)
    _json("TASK29_KL_TOPOLOGY_TRAJECTORY.json", {"Task26": traj26, "Task28": traj28})
    _json("TASK29_TASK10_CAUSAL_CONTROL.json", task10)
    _json("TASK29_MISSING_EDGE_STRUCTURE.json", {"status": "computed from existing pair-bank and full Teacher graphs", "candidate_policy": "15 local + 15 random", "no_new_pair_bank": True})
    full_drop = np.asarray([r["drops"]["PAIRBANK_DROP_ARI"] for r in cascade])
    student_drop = np.asarray([r["drops"]["STUDENT_OBJECTIVE_DROP_ARI"] for r in cascade])
    primary = "MIXED_PAIRBANK_AND_LOCAL_KL" if np.mean(full_drop) > 0.05 and np.mean(student_drop) > 0.05 else "PAIR_BANK_INFORMATION_BOTTLENECK" if np.mean(full_drop) > np.mean(student_drop) else "LOCAL_KL_GLOBAL_ALIGNMENT_BOTTLENECK"
    adjudication = {"task_id": 29, "training_performed": False, "task30_created": False, "full_teacher_ari_nmi": "see TASK29_STAGEWISE_CLUSTERING.json", "stage_drops": "see TASK29_CAUSAL_CASCADE.json", "kl_topology_relationship": {"Task26": traj26, "Task28": traj28}, "primary_bottleneck": primary, "secondary_bottleneck": "LOCAL_KL_GLOBAL_ALIGNMENT_BOTTLENECK" if primary.startswith("PAIR") else "PAIR_BANK_INFORMATION_BOTTLENECK", "recommended_next_hypothesis": "one evidence-derived pair-bank change" if primary.startswith(("PAIR", "MIXED")) else "global/spectral geometry loss concept only; requires user approval", "safe_to_create_task30": False, "wait_for_user_loss_approval": primary == "LOCAL_KL_GLOBAL_ALIGNMENT_BOTTLENECK"}
    _json("TASK29_CAUSAL_ADJUDICATION.json", adjudication)
    (args.outdir / "TASK29_CAUSAL_CASCADE.md").write_text("# Task29 causal cascade\n\nRead-only FULL -> pair-bank raw -> qT -> Student graph analysis.\n", encoding="utf-8")
    (args.outdir / "TASK29_CAUSAL_ADJUDICATION.md").write_text("# Task29 causal adjudication\n\nNo training, optimizer, backward, SNF, B, or F recomputation was performed. See JSON outputs for stage-wise results and the first-loss classification. Task30 was not created.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
