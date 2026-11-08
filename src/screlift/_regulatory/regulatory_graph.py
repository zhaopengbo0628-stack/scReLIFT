from __future__ import annotations

import re
import gzip
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np


PEAK_PATTERN = re.compile(r"^(?P<chrom>[^:_-]+)(?:[:_]|-)(?P<start>\d+)-(?P<end>\d+)$")
GENE_ID = re.compile(r'gene_id "([^"]+)"')


def normalize_chromosome(value: str) -> str:
    value = str(value)
    value = value[3:] if value.lower().startswith("chr") else value
    return "MT" if value == "M" else value


def parse_peak(value: str) -> tuple[str, int, int]:
    match = PEAK_PATTERN.match(str(value))
    if not match:
        raise ValueError(f"Invalid peak coordinate: {value}")
    return normalize_chromosome(match["chrom"]), int(match["start"]), int(match["end"])


def read_tss(path: Path) -> dict[str, tuple[str, int]]:
    result: dict[str, tuple[str, int]] = {}
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            match = GENE_ID.search(fields[8])
            if match is None:
                continue
            position = int(fields[4]) - 1 if fields[6] == "-" else int(fields[3]) - 1
            result.setdefault(match.group(1).split(".")[0], (fields[0], position))
    return result


def build_candidate_graph(
    gene_ids: np.ndarray,
    peak_names: np.ndarray,
    tss: dict[str, tuple[str, int]],
    window_bp: int,
    max_candidates: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    by_chrom: dict[str, list[tuple[int, int]]] = {}
    for peak_index, peak_name in enumerate(peak_names):
        chrom, start, end = parse_peak(str(peak_name))
        by_chrom.setdefault(chrom, []).append(((start + end) // 2, peak_index))
    for values in by_chrom.values():
        values.sort()

    candidates = []
    for gene in map(str, gene_ids):
        gene_key = gene.split(".")[0]
        if gene_key not in tss:
            raise ValueError(f"Missing TSS for common gene {gene}")
        chrom, position = tss[gene_key]
        values = by_chrom.get(normalize_chromosome(chrom), [])
        positions = [item[0] for item in values]
        left = bisect_left(positions, position - window_bp)
        right = bisect_right(positions, position + window_bp)
        nearest = sorted(
            ((abs(midpoint - position), peak_index) for midpoint, peak_index in values[left:right])
        )[:max_candidates]
        candidates.append(nearest)

    selected = np.asarray(sorted({peak for values in candidates for _, peak in values}), dtype=np.int64)
    index = np.zeros((len(gene_ids), max_candidates), dtype=np.int64)
    mask = np.zeros_like(index, dtype=bool)
    distance = np.zeros_like(index, dtype=np.float32)
    for gene_index, values in enumerate(candidates):
        for column, (distance_value, old_peak) in enumerate(values):
            index[gene_index, column] = old_peak
            mask[gene_index, column] = True
            distance[gene_index, column] = distance_value
    return {"candidate_index": index, "candidate_mask": mask, "distance_bp": distance}, selected
