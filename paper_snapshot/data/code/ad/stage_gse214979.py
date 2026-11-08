"""Verify and stage author-QC paired counts; no annotation alignment or model fitting."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

import h5py
import numpy as np


AUTHOR_COMMIT = 'fabe6e4d4166f6e028e4124ffb10956c068ccac2'
AUTHOR_CODE = f'https://github.com/aanderson54/scMultiomics_AD/blob/{AUTHOR_COMMIT}/docs/index.Rmd'
PAPER = 'https://doi.org/10.1016/j.xgen.2023.100263'
QC_RULES = [('nFeature_RNA', '>', 200), ('nFeature_RNA', '<', 10000),
            ('percent.mt', '<', 5), ('nucleosome_signal', '<', 2),
            ('TSS.enrichment', '>', 2), ('dbl', '<', 3.5)]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def qc_pass(row):
    failures = []
    for field, operator, limit in QC_RULES:
        value = float(row[field])
        passed = np.isfinite(value) and (value > limit if operator == '>' else value < limit)
        if not passed:
            failures.append(f'{field}{operator}{limit}')
    return failures


def scan_matrix(matrix):
    """Read every sparse count/index in bounded cell chunks, including column integrity."""
    nfeatures, ncells = map(int, matrix['shape'][:])
    ptr = matrix['indptr'][:]
    nnz = len(matrix['data'])
    if (len(ptr) != ncells + 1 or ptr[0] != 0 or ptr[-1] != nnz
            or len(matrix['indices']) != nnz or (np.diff(ptr) < 0).any()):
        raise ValueError('Invalid sparse pointers or dimensions')
    types = matrix['features/feature_type'][:].astype(str)
    if len(types) != nfeatures or set(types) != {'Gene Expression', 'Peaks'}:
        raise ValueError('Unexpected feature types or dimensions')
    is_rna = types == 'Gene Expression'
    counts = np.zeros((ncells, 4), dtype=np.int64)
    stored_zeros = 0
    for first in range(0, ncells, 256):
        last = min(first + 256, ncells)
        lo, hi = int(ptr[first]), int(ptr[last])
        indices = matrix['indices'][lo:hi]
        values = matrix['data'][lo:hi]
        if (not np.isfinite(values).all() or (values < 0).any()
                or (values != np.floor(values)).any()):
            raise ValueError('Nonfinite, negative, or noninteger raw counts')
        if ((indices < 0) | (indices >= nfeatures)).any():
            raise ValueError('Out-of-bounds feature index')
        columns = np.repeat(np.arange(last - first), np.diff(ptr[first:last + 1]))
        if ((columns[1:] == columns[:-1]) & (indices[1:] <= indices[:-1])).any():
            raise ValueError('Duplicate or unsorted feature indices within a cell')
        stored_zeros += int((values == 0).sum())
        for offset, keep in [(0, is_rna[indices]), (2, ~is_rna[indices])]:
            # Integer accumulation preserves raw counts exactly; no dense expression matrix.
            np.add.at(counts[first:last, offset], columns[keep], values[keep])
            np.add.at(counts[first:last, offset + 1], columns[keep & (values > 0)], 1)
        if first % 16384 == 0:
            print(f'SCAN {last}/{ncells}', flush=True)
    if (counts[:, [0, 2]] == 0).any():
        raise ValueError('Cell has empty RNA or ATAC')
    return counts, stored_zeros


def stage(raw, output, commit):
    started = time.monotonic()
    h5 = raw / 'GSE214979_filtered_feature_bc_matrix.h5'
    metadata = raw / 'GSE214979_cell_metadata.csv.gz'
    with gzip.open(metadata, 'rt', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    failures = Counter(reason for row in rows for reason in qc_pass(row))
    if failures:
        raise ValueError(f'Author-QC violations require review: {dict(failures)}')
    if any(row['id'] in ('', 'Other', 'unassigned', 'doublet') for row in rows):
        raise ValueError('Missing or unresolved donor assignment')
    donors = defaultdict(list)
    for row in rows:
        if row['Status'] not in ('AD', 'Ctrl'):
            raise ValueError('Unexpected disease status')
        expected = "Alzheimer's" if row['Status'] == 'AD' else 'Unaffected'
        if row['Diagnosis'] != expected:
            raise ValueError('Status/Diagnosis disagreement')
        donors[row['id']].append(row)
    for donor, records in donors.items():
        for field in ('Status', 'Age', 'Sex', 'structure', 'Repository'):
            if len({r[field] for r in records}) != 1:
                raise ValueError(f'Inconsistent donor metadata: {donor}/{field}')
    with h5py.File(h5, 'r', rdcc_nbytes=1 << 20) as handle:
        matrix = handle['matrix']
        cells = matrix['barcodes'][:].astype(str)
        if len(set(cells)) != len(cells) or [r[''] for r in rows] != cells.tolist():
            raise ValueError('Metadata and paired RNA/ATAC barcodes differ')
        features = matrix['features']
        types = features['feature_type'][:].astype(str)
        ids = features['id'][:].astype(str)
        names = features['name'][:].astype(str)
        genome = features['genome'][:].astype(str)
        if set(genome) != {'GRCh38'}:
            raise ValueError('Unexpected reference genome')
        rna_index, atac_index = np.flatnonzero(types == 'Gene Expression'), np.flatnonzero(types == 'Peaks')
        if (len(cells), len(rna_index), len(atac_index)) != (105332, 36601, 150614):
            raise ValueError('Dimensions disagree with this frozen author cohort')
        for selected in (rna_index, atac_index):
            if len(set(ids[selected])) != len(selected) or any(not g for g in ids[selected]):
                raise ValueError('Duplicate or missing feature ID')
        peaks = []
        for peak in ids[atac_index]:
            match = re.fullmatch(r'([^:]+):(\d+)-(\d+)', peak)
            if match is None or int(match[2]) >= int(match[3]):
                raise ValueError(f'Invalid peak interval: {peak}')
            peaks.append(match.groups())
        counts, stored_zeros = scan_matrix(matrix)
        nnz = len(matrix['data'])
    input_hashes = {p.name: sha(p) for p in (h5, metadata)}
    output.mkdir(parents=True, exist_ok=False)
    for source in (h5, metadata):
        destination = output / source.name
        shutil.copy2(source, destination)
        if sha(destination) != input_hashes[source.name]:
            raise ValueError('Staged file differs from raw input')
    np.save(output / 'rna_feature_index.npy', rna_index)
    np.save(output / 'atac_feature_index.npy', atac_index)
    np.save(output / 'raw_cell_index.npy', np.arange(len(cells), dtype=np.int64))
    np.save(output / 'qc_pass_mask.npy', np.ones(len(cells), dtype=bool))
    (output / 'barcodes.txt').write_text('\n'.join(cells) + '\n', encoding='utf-8')
    with (output / 'rna_features.tsv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['raw_feature_index', 'gene_id', 'gene_name', 'genome'])
        writer.writerows((int(i), ids[i], names[i], genome[i]) for i in rna_index)
    with (output / 'atac_peaks.bed').open('w', newline='', encoding='utf-8') as stream:
        csv.writer(stream, delimiter='\t').writerows(peaks)
    with (output / 'cells.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['barcode', 'raw_cell_index', 'donor_id', 'disease_status',
                         'author_predicted_id', 'author_subcluster', 'author_cluster_id'])
        writer.writerows((r[''], i, r['id'], r['Status'], r['predicted.id'],
                          r['subs'], r['seurat_clusters']) for i, r in enumerate(rows))
    with (output / 'qc_per_cell.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['barcode', 'raw_rna_counts', 'raw_rna_detected_genes',
                         'raw_atac_peak_counts', 'raw_atac_detected_peaks',
                         'author_nFeature_RNA', 'author_percent_mt', 'author_nucleosome_signal',
                         'author_TSS_enrichment', 'author_doublet_density', 'qc_pass'])
        writer.writerows([r[''], *map(int, counts[i]), r['nFeature_RNA'], r['percent.mt'],
                          r['nucleosome_signal'], r['TSS.enrichment'], r['dbl'], True]
                         for i, r in enumerate(rows))
    with (output / 'donors.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['donor_id', 'Status', 'Age', 'Sex', 'structure', 'Repository', 'cells'])
        writer.writerows([donor, *[records[0][f] for f in ('Status', 'Age', 'Sex', 'structure', 'Repository')],
                          len(records)] for donor, records in sorted(donors.items()))
    report = dict(status='AUTHOR_QC_AND_PAIRED_COUNTS_VERIFIED', input_cells=len(cells),
                  retained_cells=len(cells), removed_cells=0, qc_failures=dict(failures),
                  author_qc=[dict(field=f, operator=o, limit=v) for f, o, v in QC_RULES],
                  qc_basis='Published metadata, author code lines 184-188 and 212-218; no new thresholds',
                  full_sparse_scan=True, sparse_nnz=nnz, stored_zeros=stored_zeros,
                  zero_rna_cells=int((counts[:, 0] == 0).sum()),
                  zero_atac_cells=int((counts[:, 2] == 0).sum()),
                  totals=dict(rna=int(counts[:, 0].sum()), atac=int(counts[:, 2].sum())),
                  disease_cells=dict(Counter(r['Status'] for r in rows)),
                  disease_donors=dict(Counter(rs[0]['Status'] for rs in donors.values())),
                  original_annotations={f: dict(Counter(r[f] for r in rows))
                                        for f in ('predicted.id', 'subs', 'seurat_clusters')},
                  missing_metadata={f: sum(r[f] in ('', 'NA') for r in rows) for f in fields if f},
                  raw_vs_author_rna_feature_count_mismatches=int(sum(
                      int(r['nFeature_RNA']) != counts[i, 1] for i, r in enumerate(rows))),
                  raw_vs_author_rna_count_mismatches=int(sum(
                      int(r['nCount_RNA']) != counts[i, 0] for i, r in enumerate(rows))),
                  warnings=['Author analysis removes PMI-associated genes before RNA QC; raw counts preserved.',
                            'Disease status is donor-level diagnosis, not measured per-cell disease state.',
                            'Disease, donor age, tissue/repository labels are associated; audit before prediction.',
                            'Author predicted.id is retained as author reference-based annotation, not new ground truth.',
                            'No annotation harmonization, new doublet inference, split, model preprocessing or training.'])
    write_json(output / 'QC_REPORT.json', report)
    manifest = dict(accession='GSE214979', super_series='GSE214637', species='Homo sapiens',
                    technology='10x Genomics single-nucleus Multiome RNA+ATAC', reference='GRCh38',
                    tissue='Postmortem prefrontal cortex; author metadata structure retained',
                    genes=len(rna_index), atac_features=len(atac_index), atac_representation='called peaks',
                    matrix=h5.name, matrix_layout='features_by_cells_10x_CSC',
                    rna_feature_index='rna_feature_index.npy', atac_feature_index='atac_feature_index.npy',
                    metadata=metadata.name, barcode_order='exact author H5 and metadata order',
                    author_code=AUTHOR_CODE, paper=PAPER, source_commit=commit,
                    source_script_sha256=sha(__file__), input_hashes=input_hashes,
                    annotations_harmonized=False, normalized=False, hvg_selected=False,
                    split_created=False, model_trained=False, qc_report='QC_REPORT.json',
                    author_filtered_cohort_reused=True, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    manifest['output_hashes'] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    write_json(output / 'manifest.json', manifest)
    # Read-back checks include the full counts hash and exact metadata/index correspondence.
    if (output / 'barcodes.txt').read_text().splitlines() != cells.tolist():
        raise ValueError('Saved barcode order changed')
    for filename, expected in [('rna_feature_index.npy', rna_index), ('atac_feature_index.npy', atac_index)]:
        if not np.array_equal(np.load(output / filename), expected):
            raise ValueError('Saved modality index changed')
    print(json.dumps(dict(status=report['status'], retained_cells=len(cells),
                         output=str(output), elapsed_seconds=time.monotonic() - started)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    stage(args.raw, args.output, args.commit)
