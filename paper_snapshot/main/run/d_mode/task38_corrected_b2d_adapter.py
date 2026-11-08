import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

ROOT = Path("/home/u202522802045/share/zpb")
OUT = ROOT / "task38_corrected_pbmc002_ablation" / "b2d"
SOURCE = ROOT / "result_contain/37_b2_distillability_20261003/source/experiments/v5d/run_b2_distillability.py"
FULL = ROOT / "final_dmode_b2full_updated_pbmc002"
sys.path.insert(0, str(SOURCE.parent))
spec = importlib.util.spec_from_file_location("task38_b2d", SOURCE)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
d.b.GOAL1 = ROOT / "task38_corrected_pbmc002_ablation"
d.PARENT = FULL
d.RULES = dict(d.RULES, student_seeds=[20020130, 20020131], run_tier="pre_final_corrected_2seed")
d.b.CONFIG = dict(d.BASE, method="B2-D", run_tier="pre_final_corrected_2seed")

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def local_status(out, value, details=None):
    write(out / "status.json", {"status": value, "details": details, "result_path": str(out)})

d.status = local_status

def labels_after_frozen(fold):
    manifest = d.b.read(d.b.GOAL1 / "cache" / fold / "manifest.json")
    target = manifest["target"]
    processed = next((ROOT / "DATA").glob(f"{target}*/processed"))
    indices = __import__("numpy").load(d.b.GOAL1 / "cache" / fold / "target_test_index.npy")
    import csv
    with (processed / "label.csv").open(newline="", encoding="utf-8") as stream:
        labels = __import__("numpy").asarray([row["cell_type"] for row in csv.DictReader(stream)], dtype=str)[indices]
    return __import__("numpy").unique(labels, return_inverse=True)[1]

d.b.labels_after_frozen = labels_after_frozen

def prepare():
    refs = {}
    for fold in d.b.FOLDS:
        checkpoint = FULL / fold / "seed_20020130/checkpoints/teacher/best.pt"
        manifest = checkpoint.parent / "manifest.json"
        assert checkpoint.exists() and manifest.exists()
        refs[fold] = {"path": str(checkpoint), "sha256": d.b.sha(checkpoint)}
    OUT.mkdir(parents=True, exist_ok=True)
    source_manifest = d.b.GOAL1 / "data_manifest.json"
    write(OUT / "data_manifest.json", d.b.read(source_manifest))
    protocol = dict(d.RULES, parent_checkpoints=refs,
                    corrected_data_root=str(d.b.GOAL1),
                    parent=str(FULL),
                    source_path=str(SOURCE),
                    code_hashes={str(SOURCE): d.b.sha(SOURCE), str(Path(d.b.__file__)): d.b.sha(d.b.__file__)})
    if (OUT / "protocol.json").exists():
        assert d.b.read(OUT / "protocol.json") == protocol
    else:
        write(OUT / "protocol.json", protocol)
    local_status(OUT, "PREPARED", "corrected source cache and corrected Full B2 Teacher30 checkpoints verified")

def evaluate_rows(rows):
    audit = d.b.read(d.b.GOAL1 / "FULL_B2_REUSE_AUDIT.json")
    lookup = {(r["fold"], int(r["seed"])): r for r in audit["units"]}
    for row in rows:
        ref = lookup[(row["fold"], int(row["seed"]))]
        row["delta_vs_full_b2_ari"] = row["ari"] - ref["ari"]
        row["delta_vs_full_b2_nmi"] = row["nmi"] - ref["nmi"]
    return rows

def run():
    device = torch.device("cuda")
    d.b.save_checkpoint = d.snapshot
    local_status(OUT, "RUNNING", "corrected B2-D Step1 and Step2")
    folds = []
    for fold in d.b.FOLDS:
        folds.append(d.step1_fold(fold, OUT, device))
        torch.cuda.empty_cache()
    write(OUT / "STEP1_ADJUDICATION.json", {"DISTILLABILITY_OPERATOR_VALID": "YES" if all(x["DISTILLABILITY_OPERATOR_VALID"] == "YES" for x in folds) else "NO", "folds": folds})
    rows = []
    for seed in d.RULES["student_seeds"]:
        for fold in d.b.FOLDS:
            unit = OUT / "step2" / fold / f"seed_{seed}"
            if (unit / "z_target_test.npy").exists() and not (unit / "metrics.json").exists():
                embedding = __import__("numpy").load(unit / "z_target_test.npy")
                labels = d.b.labels_after_frozen(fold)
                digest = d.b.sha(unit / "z_target_test.npy")
                row = dict(fold=fold, seed=seed, embedding_sha256=digest, **d.b.evaluate(embedding, labels))
                d.b.write(unit / "metrics.json", row)
                rows.append(row)
            else:
                rows.append(d.step2_fold(fold, seed, OUT, device))
            torch.cuda.empty_cache()
    rows = evaluate_rows(rows)
    for seed in d.RULES["student_seeds"]:
        seed_rows = [x for x in rows if int(x["seed"]) == seed]
        write(OUT / f"SEED_{seed}_ADJUDICATION.json", {
            "seed": seed, "rows": seed_rows,
            "mean_ari": sum(x["ari"] for x in seed_rows) / len(seed_rows),
            "mean_nmi": sum(x["nmi"] for x in seed_rows) / len(seed_rows),
            "mean_delta_vs_full_b2_ari": sum(x["delta_vs_full_b2_ari"] for x in seed_rows) / len(seed_rows)
        })
    write(OUT / "FINAL_REPORT.json", {"TASK_ID": 37, "status": "COMPLETE", "seeds": d.RULES["student_seeds"], "step1": folds, "rows": rows, "checkpoints": len(list(OUT.rglob("*.pt")))})
    local_status(OUT, "COMPLETED", "corrected B2-D 2-seed run complete")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "smoke", "run"], required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
        return
    for path, digest in d.b.read(OUT / "protocol.json")["code_hashes"].items():
        assert d.b.sha(path) == digest
    if args.stage == "smoke":
        with (ROOT / ".benchmark_a30.lock").open("a") as lock:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
            d.smoke(OUT, torch.device("cuda"))
        return
    import fcntl
    with (ROOT / ".benchmark_a30.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run()

if __name__ == "__main__":
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
