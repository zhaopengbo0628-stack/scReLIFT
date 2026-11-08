from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adjudicate_task29_causal import FOLDS, NAMES, RNAOnlySource, clustering, labels_for
from audit_task29_capacity import fidelity_to_full, old_pairbank, sparse_topk


def graph_mass(graph: np.ndarray) -> float:
    return float(np.maximum(graph, 0.0).sum())


def recovery(practical: float, old: float, oracle: float, *, lower_is_better: bool = False) -> float | None:
    if lower_is_better:
        denominator = old - oracle
        numerator = old - practical
    else:
        denominator = oracle - old
        numerator = practical - old
    if abs(denominator) < 1e-12:
        return None
    return float(numerator / denominator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task23", type=Path, required=True)
    parser.add_argument("--task26", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)

    records: list[dict[str, object]] = []
    for fold, (_, source_keys) in FOLDS.items():
        harmonized = args.root / "result_contain/pre_task14_dmode_source_quality_20260926" / fold / "harmonized"
        for source_key in source_keys:
            name = NAMES[source_key]
            base = args.task23 / "matched_geometry" / fold / name
            full = np.load(base / "G_SNF_verified.npy").astype(np.float32)
            source = RNAOnlySource(harmonized, name)
            labels = labels_for(args.root, source, source.val)
            old = old_pairbank(full, args.task26 / fold / "pair_banks" / f"{name}_val.npz")
            top30 = sparse_topk(full, 30)
            k60_oracle = sparse_topk(full, 60)
            practical = sparse_topk(full, 60)
            if not np.array_equal(k60_oracle, practical):
                raise RuntimeError(f"K60 practical rule is not an exact oracle reproduction: {fold}/{name}")
            graphs = {
                "FULL_TEACHER": full,
                "OLD_PAIRBANK_RAW": old,
                "TOP30_REJECTED": top30,
                "K60_SPECTRAL_ORACLE": k60_oracle,
                "K60_PRACTICAL_RAW": practical,
            }
            row = {"fold": fold, "dataset": name, "nodes": int(full.shape[0]), "policies": {}}
            for policy, graph in graphs.items():
                metrics = {"identity": True} if policy == "FULL_TEACHER" else fidelity_to_full(full, graph)
                row["policies"][policy] = {
                    "graph_fidelity_to_full": metrics,
                    "posthoc_clustering": clustering(graph, labels, 0),
                    "candidate_mass_fraction": graph_mass(graph) / max(graph_mass(full), 1e-12),
                    "nonzero_directed": int(np.count_nonzero(graph)),
                }
            row["practical_equals_k60_oracle"] = True
            records.append(row)

    policy_names = ("OLD_PAIRBANK_RAW", "TOP30_REJECTED", "K60_SPECTRAL_ORACLE", "K60_PRACTICAL_RAW")
    summary: dict[str, dict[str, float | bool | int]] = {}
    for policy in policy_names:
        rows = [record["policies"][policy] for record in records]
        fidelity = [row["graph_fidelity_to_full"] for row in rows]
        summary[policy] = {
            "records": len(rows),
            "mean_principal_angle_cosine": float(np.mean([x["principal_angle_mean_cosine"] for x in fidelity])),
            "mean_teacher_top15_neighborhood_preservation": float(np.mean([x["teacher_top15_neighborhood_preservation"] for x in fidelity])),
            "mean_normalized_laplacian_spectral_rmse": float(np.mean([x["normalized_laplacian_spectral_rmse"] for x in fidelity])),
            "all_connected": bool(all(x["components_right"] == 1 for x in fidelity)),
            "mean_largest_component_fraction": float(np.mean([x["largest_component_fraction_right"] for x in fidelity])),
            "mean_candidate_mass_fraction": float(np.mean([row["candidate_mass_fraction"] for row in rows])),
            "mean_ari": float(np.mean([row["posthoc_clustering"]["ari"] for row in rows])),
            "mean_nmi": float(np.mean([row["posthoc_clustering"]["nmi"] for row in rows])),
        }

    old = summary["OLD_PAIRBANK_RAW"]
    oracle = summary["K60_SPECTRAL_ORACLE"]
    practical = summary["K60_PRACTICAL_RAW"]
    recovery_summary = {
        "principal_angle_similarity": recovery(practical["mean_principal_angle_cosine"], old["mean_principal_angle_cosine"], oracle["mean_principal_angle_cosine"]),
        "top15_neighborhood_preservation": recovery(practical["mean_teacher_top15_neighborhood_preservation"], old["mean_teacher_top15_neighborhood_preservation"], oracle["mean_teacher_top15_neighborhood_preservation"]),
        "normalized_laplacian_fidelity": recovery(practical["mean_normalized_laplacian_spectral_rmse"], old["mean_normalized_laplacian_spectral_rmse"], oracle["mean_normalized_laplacian_spectral_rmse"], lower_is_better=True),
        "source_ari": recovery(practical["mean_ari"], old["mean_ari"], oracle["mean_ari"]),
        "source_nmi": recovery(practical["mean_nmi"], old["mean_nmi"], oracle["mean_nmi"]),
    }
    top15_drop = old["mean_teacher_top15_neighborhood_preservation"] - practical["mean_teacher_top15_neighborhood_preservation"]
    gates = {
        "principal_angle_materially_improves_over_old": practical["mean_principal_angle_cosine"] > old["mean_principal_angle_cosine"],
        "laplacian_rmse_improves_over_old": practical["mean_normalized_laplacian_spectral_rmse"] < old["mean_normalized_laplacian_spectral_rmse"],
        "top15_preservation_not_collapsed": top15_drop <= 0.02,
        "connectivity_valid": practical["all_connected"],
        "posthoc_directionally_consistent": practical["mean_ari"] >= old["mean_ari"] and practical["mean_nmi"] >= old["mean_nmi"],
    }
    result = {
        "task_id": 29,
        "training_performed": False,
        "teacher_retrained": False,
        "student_trained": False,
        "snf_recomputed": False,
        "labels_used_for_selection": False,
        "labels_used_only_posthoc": True,
        "k60_oracle_selection_rule": "exact per-anchor top-60 Full Teacher affinity, diagonal excluded",
        "k60_oracle_graph_semantics": "Full SNF affinity; row-wise top-K; arithmetic-mean union symmetrization; zero diagonal",
        "k60_oracle_edge_budget_semantics": "60 directed candidates per anchor before union symmetrization",
        "practical_rule": "exact per-anchor Teacher top-60 neighbors with exact Full Teacher weights",
        "practical_rule_is_oracle_reproduction": True,
        "summary": summary,
        "recovery_vs_k60_ceiling": recovery_summary,
        "top15_preservation_gate": {"absolute_drop": float(top15_drop), "max_tolerated_drop": 0.02, "interpretation": "material collapse gate uses a predeclared 0.02 absolute tolerance"},
        "cpu_pass": {**gates, "passed": all(gates.values())},
        "records": records,
    }
    (args.outdir / "TASK29_PRACTICAL_K60_ORACLE.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# Task29 practical K60 pair-bank oracle",
        "",
        "No training, Teacher retraining, B/F recomputation, or SNF recomputation was performed.",
        "",
        "K60_ORACLE_SELECTION_RULE: exact per-anchor top-60 Full Teacher affinity",
        "K60_ORACLE_GRAPH_SEMANTICS: row-wise top-K + arithmetic-mean union symmetrization + zero diagonal",
        "K60_ORACLE_EDGE_BUDGET_SEMANTICS: 60 directed candidates per anchor before union",
        "PRACTICAL_RULE: exact Teacher top-60 with exact Full Teacher weights",
        "PRACTICAL_RULE_IS_ORACLE_REPRODUCTION: YES",
        "",
        "| Policy | Principal cosine | Top15 preservation | Laplacian RMSE | Connected | ARI | NMI |",
        "|---|---:|---:|---:|:---:|---:|---:|",
    ]
    for policy in ("OLD_PAIRBANK_RAW", "TOP30_REJECTED", "K60_SPECTRAL_ORACLE", "K60_PRACTICAL_RAW"):
        row = summary[policy]
        lines.append(f"| {policy} | {row['mean_principal_angle_cosine']:.4f} | {row['mean_teacher_top15_neighborhood_preservation']:.4f} | {row['mean_normalized_laplacian_spectral_rmse']:.4f} | {row['all_connected']} | {row['mean_ari']:.4f} | {row['mean_nmi']:.4f} |")
    lines.extend(["", f"CPU_PASS: {'YES' if result['cpu_pass']['passed'] else 'NO'}", "", "Recovery vs K60 ceiling:"])
    for key, value in recovery_summary.items():
        lines.append(f"- {key}: {value:.4f}" if value is not None else f"- {key}: null")
    (args.outdir / "TASK29_PRACTICAL_K60_ORACLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
