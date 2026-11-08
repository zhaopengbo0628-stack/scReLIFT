import argparse
import csv
import fcntl
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT = Path("/home/u202522802045/share/zpb")
OUT = ROOT / "task38_corrected_pbmc002_ablation"
ENV = ROOT / "result_contain/38_b2_complete_controls_20261003/source"
V5D = ROOT / "result_contain/36_b2_three_causal_ablations_20261003/source/experiments/v5d"
sys.path.insert(0, str(V5D))
spec = importlib.util.spec_from_file_location("task38_controls", ENV / "run_b2_complete_controls.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
b, a = c.b, c.a
FOLDS = [f"target_D{i}" for i in range(1, 5)]
SEEDS = [20020130, 20020131, 20020132]


def labels_after_frozen(fold):
    manifest = b.read(OUT / "cache" / fold / "manifest.json")
    target = manifest["target"]
    processed = next((ROOT / "DATA").glob(f"{target}*/processed"))
    indices = np.load(OUT / "cache" / fold / "target_test_index.npy")
    with (processed / "label.csv").open(newline="", encoding="utf-8") as stream:
        labels = np.asarray([row["cell_type"] for row in csv.DictReader(stream)], dtype=str)[indices]
    return np.unique(labels, return_inverse=True)[1]


def build_unscreened(fold):
    destination = OUT / "fields" / fold
    if (destination / "manifest.json").exists():
        for path, digest in b.read(destination / "manifest.json")["hashes"].items():
            assert b.sha(path) == digest, path
        return
    cm = b.read(OUT / "cache" / fold / "manifest.json")
    hroot = Path(cm["source_harmonized_root"])
    root = hroot.parent
    genes = np.load(root / "gene_ids.npy")
    peaks = np.load(hroot / "canonical_peak_names.npy")
    gtf = ROOT.parent.parent / "glue_156478/graph_files/gencode.v32.primary_assembly.annotation.gtf.gz"
    graph, _ = a.build_candidate_graph(genes, peaks, a.read_tss(gtf), 250000, 50)
    mask, index = graph["candidate_mask"], graph["candidate_index"]
    edge_gene = np.repeat(np.arange(len(genes)), index.shape[1])[mask.ravel()]
    edge_peak = index[mask]
    full = sparse.load_npz(root / "B_D.npz").tocsc()
    score, sign = np.load(root / "R_D.npy"), np.load(root / "sign_D.npy")
    selected = np.load(root / "selected.npy").astype(bool)
    assert len(score) == len(edge_gene)

    def mapped(values):
        values = values.astype(np.float32).copy()
        for gene in range(len(genes)):
            positions = edge_gene == gene
            denominator = np.abs(values[positions]).sum()
            if denominator > 0:
                values[positions] /= denominator
        nz = np.abs(values) > 1e-8
        return sparse.coo_matrix((values[nz], (edge_peak[nz], edge_gene[nz])), shape=full.shape).tocsc()

    selected_difference = mapped(np.where(selected, sign * score, 0)) - full
    assert not selected_difference.nnz or np.abs(selected_difference.data).max() < 1e-7
    control = mapped(sign * score)
    assert control.shape == full.shape and control.nnz >= full.nnz
    destination.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(destination / "B_UNSCREENED.npz", control)
    np.save(destination / "candidate_edge_gene.npy", edge_gene)
    np.save(destination / "candidate_edge_peak.npy", edge_peak)
    inputs = [root / name for name in ["B_D.npz", "R_D.npy", "sign_D.npy", "selected.npy", "gene_ids.npy"]]
    inputs += [hroot / "canonical_peak_names.npy", gtf]
    for name in cm["sources"]:
        source_dir = hroot / name
        with np.load(source_dir / "metadata.npz", allow_pickle=True) as metadata:
            train, val = metadata["train_index"], metadata["val_index"]
        assert not np.intersect1d(train, val).size
        values = sparse.load_npz(source_dir / "atac_binary.npz")[np.r_[train, val]]
        field, state, diagnostics = a.fit_dmode_field(values, control, np.arange(len(train)))
        np.save(destination / f"{name}_F_train.npy", field[:len(train)])
        np.save(destination / f"{name}_F_val.npy", field[len(train):])
        np.savez(destination / f"{name}_field_state.npz", **state)
        b.write(destination / f"{name}_field_diagnostics.json", diagnostics)
        inputs += [source_dir / "metadata.npz", source_dir / "atac_binary.npz"]
    files = [path for path in destination.iterdir() if path.is_file()]
    hashes = {str(path): b.sha(path) for path in files + inputs if path.is_file()}
    b.write(destination / "manifest.json", dict(hashes=hashes, target_read=False, full_reconstruction_pass=True,
        candidate_edges=len(edge_gene), full_nnz=full.nnz, unscreened_nnz=control.nnz, genes=len(genes),
        operation="remove selected mask only; preserve sign_D*R_D and gene L1, TF-IDF and SOURCE-train standardization",
        directly_matched=True, limitation=c.POLICY["A3_limit"]))


def setup():
    b.GOAL1 = OUT
    b.labels_after_frozen = labels_after_frozen
    for fold in FOLDS:
        sources, _, _, _ = b.load_sources(fold, OUT)
        probe_dir = OUT / "reference_probes" / fold
        probe_dir.mkdir(parents=True, exist_ok=True)
        for source in sources:
            for split in ["train", "val"]:
                path = probe_dir / f"{source['name']}_F_{split}.npy"
                field = source[split]["field"]
                if path.exists():
                    assert np.array_equal(np.load(path), field), path
                else:
                    np.save(path, field)
        stats = np.load(OUT / "rna" / fold / "seed_20020130/preprocessing/source_rna_stats.npz")
        path = OUT / "rna" / fold / "seed_20020130/shared_rna_best.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"rna_mean": stats["mean"], "rna_std": stats["std"]}, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert np.array_equal(loaded["rna_mean"], stats["mean"])
        assert np.array_equal(loaded["rna_std"], stats["std"])
    for fold in FOLDS:
        build_unscreened(fold)


def smoke():
    device = torch.device("cuda")
    for condition in ["RNA_ONLY", "SHUFFLED_F", "UNSCREENED"]:
        a.configure(condition)
        sources, _, _, _ = c.control_sources(condition, FOLDS[0], SEEDS[0], OUT)
        b.seed_all(SEEDS[0])
        width = sources[0]["train"]["rna"].shape[1]
        model = b.Teacher(width, sources[0]["train"]["field"].shape[1])
        if condition == "RNA_ONLY":
            model = a.RNATeacher(model)
        model = model.to(device)
        batch = next(b.batches(sources, np.random.default_rng(SEEDS[0])))
        rna, field = b.batch_tensor(batch, "rna", device), b.batch_tensor(batch, "field", device)
        losses = a.losses(model, rna, field)
        loss = sum(losses)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        student = model.rna_encoder
        with torch.no_grad():
            target = model(rna, field)[2]
        kd = torch.nn.functional.mse_loss(student(rna), target)
        assert torch.isfinite(kd)
        print("CONTROL_SMOKE_PASS", condition, float(loss.detach()), float(kd), flush=True)
        del model, student
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", choices=["RNA_ONLY", "SHUFFLED_F", "UNSCREENED"],
                        default=["RNA_ONLY", "SHUFFLED_F", "UNSCREENED"])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    setup()
    if args.prepare_only:
        print("CORRECTED_CONTROL_PREPARATION_PASS", flush=True)
        return
    b.teacher_losses, b.save_checkpoint = a.losses, a.save_checkpoint
    a.sources_for = c.control_sources
    lock_path = ROOT / ".benchmark_a30.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.smoke_only:
            smoke()
            print("CORRECTED_CONTROL_SMOKE_PASS", flush=True)
            return
        for condition in args.conditions:
            for seed in SEEDS:
                for fold in FOLDS:
                    a.configure(condition)
                    row = a.run_unit(condition, fold, seed, OUT, torch.device("cuda"))
                    print("CORRECTED_CONTROL_COMPLETE", condition, fold, seed,
                          row.get("ari"), row.get("nmi"), flush=True)
                    torch.cuda.empty_cache()
    print("CORRECTED_CONTROLS_COMPLETE", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
