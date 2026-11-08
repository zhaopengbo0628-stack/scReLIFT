from __future__ import annotations

import json
import re
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np


PEAK_PATTERN = re.compile(r"^(?P<chrom>[^:_]+)[:_](?P<start>\d+)[-](?P<end>\d+)$")


def normalize_chromosome(value: str) -> str:
    chrom = str(value)
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    return "MT" if chrom == "M" else chrom


def parse_peak(value: str) -> tuple[str, int, int]:
    match = PEAK_PATTERN.match(str(value))
    if not match:
        raise ValueError(f"Invalid peak coordinate: {value}")
    return normalize_chromosome(match["chrom"]), int(match["start"]), int(match["end"])


def build_candidate_graph(genes: np.ndarray, peak_names: np.ndarray, tss: dict[str, tuple[str, int]], window_bp: int, max_candidates: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    by_chrom: dict[str, list[tuple[int, int]]] = {}
    for peak_index, peak_name in enumerate(peak_names):
        chrom, start, end = parse_peak(str(peak_name))
        by_chrom.setdefault(chrom, []).append(((start + end) // 2, peak_index))
    for values in by_chrom.values():
        values.sort()
    candidates = []
    for gene in map(str, genes):
        if gene not in tss:
            raise ValueError(f"Missing TSS for {gene}")
        chrom, position = tss[gene]
        chrom = normalize_chromosome(chrom)
        values = by_chrom.get(chrom, [])
        positions = [item[0] for item in values]
        left = bisect_left(positions, position - window_bp)
        right = bisect_right(positions, position + window_bp)
        nearest = sorted(((abs(midpoint - position), peak_index) for midpoint, peak_index in values[left:right]))[:max_candidates]
        candidates.append(nearest)
    selected = np.asarray(sorted({peak for values in candidates for _, peak in values}), dtype=np.int64)
    remap = {old: new for new, old in enumerate(selected.tolist())}
    index = np.zeros((len(genes), max_candidates), dtype=np.int64)
    mask = np.zeros_like(index, dtype=bool)
    distance = np.zeros_like(index, dtype=np.float32)
    for gene_index, values in enumerate(candidates):
        for column, (distance_value, old_peak) in enumerate(values):
            index[gene_index, column] = remap[old_peak]
            mask[gene_index, column] = True
            distance[gene_index, column] = distance_value
    return {"candidate_index": index, "candidate_mask": mask, "distance_bp": distance}, selected


def build_b_reg(
    data_dir: Path,
    selection_dir: Path,
    folds: int,
    seed: int,
    predictability_epsilon: float = 0.1,
    gene_confidence_epsilon: float = 0.1,
) -> dict:
    from .edge_selector import select_soft_edges

    result = select_soft_edges(data_dir, folds, seed, predictability_epsilon, gene_confidence_epsilon)
    selection_dir.mkdir(parents=True, exist_ok=False)
    np.save(selection_dir / "B_reg.npy", result["B_reg"])
    np.save(selection_dir / "fold_rho.npy", result["fold_rho"])
    np.save(selection_dir / "median_rho.npy", result["median_rho"])
    np.save(selection_dir / "sign_stability.npy", result["sign_stability"])
    np.save(selection_dir / "rho_mad.npy", result["rho_mad"])
    np.save(selection_dir / "rho_iqr.npy", result["rho_iqr"])
    np.save(selection_dir / "Q_S.npy", result["Q_S"])
    np.save(selection_dir / "gene_confidence.npy", result["gene_confidence"])
    (selection_dir / "edges.json").write_text(json.dumps(result["edges"], indent=2), encoding="utf-8")
    (selection_dir / "edge_statistics.json").write_text(json.dumps(result["statistics"], indent=2), encoding="utf-8")
    (selection_dir / "manifest.json").write_text(json.dumps(result["manifest"], indent=2), encoding="utf-8")
    np.savez_compressed(selection_dir / "regulatory_graph.npz", **result["graph"])
    return result["manifest"]
