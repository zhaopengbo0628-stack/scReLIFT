"""Stage GEO paired counts without fitting population preprocessing or a model."""
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

root = Path(sys.argv[1])
processed = root / "processed"
export = processed / "rds_export"
labels_dir = root / "labels"
logs = root / "logs"
for directory in (labels_dir, logs):
    directory.mkdir(exist_ok=True)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metadata(modality, accession):
    name = f"{accession}_{modality}_Metadata_Final_Multiome.txt"
    with (root / "raw" / name / name).open() as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = ["cell_id"] + next(reader)
        rows = []
        for row in reader:
            assert len(row) == len(header)
            rows.append(dict(zip(header, row)))
    result = {row["cell_id"]: row for row in rows}
    assert len(result) == len(rows)
    return result


rna_meta = metadata("RNA", "GSE196988")
atac_meta = metadata("ATAC", "GSE196987")
all_cells = (export / "rna.cells").read_text().splitlines()
assert len(set(all_cells)) == len(all_cells)
assert set(all_cells) == set(rna_meta) == set(atac_meta)
cells = [cell for cell in all_cells if rna_meta[cell]["sample"] == "HDa"]
assert len(cells) == 6317
assert all(atac_meta[c]["sample"] == "HDa" for c in cells)
assert all(rna_meta[c]["Sex"] == "Female" and rna_meta[c]["Age"] == "36"
           and rna_meta[c]["Status"] == "Healthy" for c in cells)
assert all(rna_meta[c]["predicted.celltype.l2"] == atac_meta[c]["predicted.celltype.l2"] for c in cells)
mapping = {
    "B naive": "Naive B", "B memory": "Memory B",
    "B intermediate": "Intermediate B", "CD8 TEM_1": "CD8 TEM",
    "CD8 TEM_2": "CD8 TEM", "cDC1": "cDC", "cDC2": "cDC",
    "NK_CD56bright": "NK", "NK Proliferating": "NK",
}
original = [rna_meta[c]["predicted.celltype.l2"] for c in cells]
merged = [mapping.get(label, label) for label in original]
names, encoded = np.unique(merged, return_inverse=True)


def write_labels(path, cell_ids, values):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["barcode", "cell_type"])
        writer.writerows(zip(cell_ids, values))


write_labels(labels_dir / "labels_original.csv", cells, original)
write_labels(processed / "label.csv", cells, merged)
np.save(processed / "barcodes.npy", np.asarray(cells))
(processed / "barcodes.txt").write_text("\n".join(cells) + "\n")
np.save(processed / "labels.npy", encoded)
np.save(processed / "label_names.npy", names)
(labels_dir / "label_mapping.json").write_text(json.dumps({
    "source_column": "predicted.celltype.l2",
    "rule": "Merge equivalent subtype names; preserve unmatched types. No invented CD8 TEM split.",
    "mapping": {label: mapping.get(label, label) for label in sorted(set(original))},
    "unannotated_or_doublet_policy": "preserved and explicitly labeled; no label-based cell exclusion",
}, indent=2) + "\n")
with (processed / "cell_metadata.csv").open("w", newline="") as handle:
    fields = ["cell_id", "sample", "Sex", "Age", "Status", "predicted.celltype.l1", "predicted.celltype.l2", "MarkerAnnotations"]
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rna_meta[c] for c in cells)

shapes = {}
feature_arrays = {}
for modality in ("rna", "atac"):
    features = (export / f"{modality}.features").read_text().splitlines()
    ids = (export / f"{modality}.cells").read_text().splitlines()
    assert len(set(ids)) == len(ids) and set(ids) == set(all_cells)
    data = np.fromfile(export / f"{modality}.values", dtype="<i4")
    indices = np.fromfile(export / f"{modality}.indices", dtype="<i4")
    indptr = np.fromfile(export / f"{modality}.indptr", dtype="<i4")
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(len(ids), len(features)))
    matrix.sort_indices()
    if ids != cells:
        lookup = {cell: i for i, cell in enumerate(ids)}
        matrix = matrix[[lookup[cell] for cell in cells]]
    assert matrix.min() >= 0 and np.all(np.asarray(matrix.sum(axis=1)).ravel() > 0)
    sparse.save_npz(processed / f"{modality}_counts.npz", matrix)
    if modality == "atac":
        features = [re.sub(r"^(chr[^-]+)-(\d+)-(\d+)$", r"\1:\2-\3", value) for value in features]
        assert all(re.fullmatch(r"chr[^:]+:\d+-\d+", value) for value in features)
        matrix.data = np.ones(matrix.nnz, dtype=np.float32)
    sparse.save_npz(processed / f"{modality}.npz", matrix)
    key = "gene_names" if modality == "rna" else "peak_names"
    np.save(processed / f"{key}.npy", np.asarray(features))
    feature_arrays[key] = np.asarray(features)
    shapes[modality] = list(matrix.shape)
    print(modality, matrix.shape, matrix.nnz, "saved", flush=True)
    del matrix, data, indices, indptr

# An unlabeled, reproducible split for internal use; LODO must define source folds separately.
order = np.random.default_rng(20020130).permutation(len(cells))
cut1, cut2 = int(len(cells) * .7), int(len(cells) * .85)
splits = {"train_index": order[:cut1], "val_index": order[cut1:cut2], "test_index": order[cut2:]}
for name, index in splits.items():
    np.save(processed / f"{name}.npy", index)
np.savez_compressed(processed / "metadata.npz", labels=encoded, label_names=names,
                    cell_names=np.asarray(cells), **feature_arrays, **splits)

# Separate common-granularity evaluation files leave existing datasets' labels intact.
for other in root.parent.iterdir():
    path = other / "processed" / "label.csv"
    if not path.is_file():
        continue
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    write_labels(other / "processed" / "label_harmonized.csv",
                 [r["barcode"] for r in rows],
                 [mapping.get(r["cell_type"], r["cell_type"]) for r in rows])

source_files = sorted(p for p in (root / "raw").rglob("*") if p.is_file())
source_hashes = {str(p.relative_to(root)): digest(p) for p in source_files}
identity = {
    "dataset_id": "PBMC-MO-002", "canonical_identity": "GEO:GSE196990:GSE196988+GSE196987:HDa",
    "official_title": "Durable Alterations of Hematopoiesis and Chromatin post-COVID-19",
    "provider": "NCBI GEO", "platform": "10x Genomics Multiome (GEO-distributed)",
    "official_urls": [f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}" for acc in ("GSE196990", "GSE196988", "GSE196987")],
    "analysis_version": "not specified", "cell_count": len(cells), "cell_types": len(names),
    "subset_rule": "RNA metadata sample == HDa; synchronize both matrices and labels by full cell ID",
    "donor": {"sample": "HDa", "sex": "Female", "age_years": 36, "status": "Healthy"},
    "genome_build": "hg38",
    "cohort_provenance": "Published post-QC paired cohort; original authors used RNA and ATAC QC and clustering. This is not an ATAC-free target cohort construction.",
    "cohort_provenance_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5906327",
    "modalities": ["RNA", "ATAC"], "sample_count": len(set(rna_meta[c]["sample"] for c in cells)),
    "status_counts": dict(Counter(rna_meta[c]["Status"] for c in cells)),
    "source_files": list(source_hashes), "source_sha256": source_hashes,
    "legacy_directory_name": root.name,
    "legacy_identity_valid": False,
    "identity_note": "Directory retained for compatibility; this replaces the removed 10x sorted dataset.",
    "label_source": "GSE196988 RNA metadata predicted.celltype.l2; paired ATAC annotations verified identical",
}
(root / "DATASET_IDENTITY.json").write_text(json.dumps(identity, indent=2) + "\n")
outputs = {str(p.relative_to(root)): digest(p) for directory in (processed, labels_dir)
           for p in directory.iterdir() if p.is_file()}
report = {
    "status": "COMPLETED", "matrix_shapes": shapes, "paired_cells": len(cells),
    "rna_atac_metadata_label_agreement": True, "original_label_counts": dict(Counter(original)),
    "merged_label_counts": dict(Counter(merged)), "source_sha256": source_hashes,
    "output_sha256": outputs, "population_normalization_fitted": False,
    "hvg_selected": False, "pca_fitted": False, "training_performed": False,
    "rna_npz_semantics": "raw counts; complete gene universe",
    "atac_npz_semantics": "binary accessibility; raw counts separately preserved",
    "split": "unstratified 70/15/15 with seed20020130; not an experiment",
    "evaluation_labels": "label_harmonized.csv for all four datasets; common naming, unequal class presence allowed",
    "independence": "independent GEO study provenance; expression fingerprint duplication not yet tested",
}
(logs / "processing_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: report[k] for k in ("status", "matrix_shapes", "paired_cells", "merged_label_counts")}), flush=True)
