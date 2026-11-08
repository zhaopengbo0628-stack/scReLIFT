from __future__ import annotations

import csv
import argparse
import gzip
import hashlib
import itertools
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import h5py
from scipy import sparse


ROOT = Path('/home/u202522802045/share/zpb')
DATA = ROOT / 'DATA'
OUT = ROOT / 'final_dmode_b2full_updated_pbmc002'
IDS = {
    'D1': DATA / 'PBMC-MO-001__10X__GranSorted__10k__Multiome',
    'D2': DATA / 'PBMC-MO-002__10X__GranSorted__10k__Multiome',
    'D3': DATA / 'PBMC-MO-003__10X__Unsorted__10k__Multiome',
    'D4': DATA / 'PBMC-MO-004__10X__Unsorted__3k__Multiome',
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_sparse(path: Path) -> sparse.csr_matrix:
    with np.load(path, allow_pickle=True) as z:
        if {'data', 'indices', 'indptr', 'shape'} <= set(z.files):
            return sparse.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape'])).tocsr()
    return sparse.load_npz(path).tocsr()


def gene_ids(dataset: str) -> np.ndarray:
    root = IDS[dataset] / 'processed'
    for name in ('gene_ids.npy', 'gene_names.npy'):
        path = root / name
        if path.exists():
            values = np.load(path, allow_pickle=True).astype(str)
            if dataset != 'D2' or np.all(np.char.startswith(values, 'ENSG')):
                return values
            gtf = ROOT.parent.parent / 'glue_156478/graph_files/gencode.v32.primary_assembly.annotation.gtf.gz'
            name_to_id = {}
            with gzip.open(gtf, 'rt') as stream:
                for line in stream:
                    if line.startswith('#'):
                        continue
                    fields = line.rstrip().split('\t')
                    if len(fields) < 9 or fields[2] != 'gene':
                        continue
                    attrs = dict(re.findall(r'(gene_id|gene_name) "([^"]+)"', fields[8]))
                    if 'gene_id' in attrs and 'gene_name' in attrs:
                        name_to_id.setdefault(attrs['gene_name'], attrs['gene_id'].split('.')[0])
            mapped = np.asarray([name_to_id.get(value, f'SYMBOL:{value}') for value in values], dtype=str)
            return mapped
    with np.load(root / 'rna.npz', allow_pickle=True) as z:
        for key in ('gene_ids', 'gene_names'):
            if key in z.files:
                return z[key].astype(str)
    raise FileNotFoundError(f'{dataset}: gene IDs')


def rna(dataset: str) -> sparse.csr_matrix:
    root = IDS[dataset] / 'processed'
    for name in ('rna_counts.npz', 'rna.npz'):
        if (root / name).exists():
            return load_sparse(root / name)
    raise FileNotFoundError(f'{dataset}: RNA')


def barcodes(dataset: str) -> list[str]:
    root = IDS[dataset] / 'processed'
    return (root / 'barcodes.txt').read_text().splitlines()


def sparse_fingerprints(x: sparse.csr_matrix, genes: np.ndarray, canonical: np.ndarray) -> list[str]:
    x = x.copy()
    x.sum_duplicates()
    x.eliminate_zeros()
    if np.any(x.data < 0) or not np.all(x.data == np.floor(x.data)):
        raise ValueError('Fingerprint input must be nonnegative integer raw counts')
    lookup = {g: i for i, g in enumerate(canonical)}
    cols = np.asarray([lookup[g] for g in genes], dtype='<u8')
    gene_digest = hashlib.sha256('\n'.join(canonical).encode()).digest()
    out = []
    for row in range(x.shape[0]):
        lo, hi = x.indptr[row], x.indptr[row + 1]
        order = np.argsort(cols[x.indices[lo:hi]], kind='stable')
        payload = np.column_stack((cols[x.indices[lo:hi]][order], x.data[lo:hi][order].astype('<u8')))
        out.append(hashlib.sha256(gene_digest + payload.tobytes()).hexdigest())
    return out


def raw_rna(dataset: str):
    directory = IDS[dataset]
    cells = barcodes(dataset)
    if dataset == 'D2':
        reference = next((IDS['D3'] / 'raw').glob('*feature_bc_matrix.h5'))
        with h5py.File(reference, 'r') as handle:
            features = handle['matrix/features']
            mask = features['feature_type'][:].astype(str) == 'Gene Expression'
            ids = features['id'][:].astype(str)[mask]
            names = features['name'][:].astype(str)[mask]
        seen = set()
        unique_names = []
        reserved = set(names)
        for name in names:
            value, suffix = name, 1
            while value in seen:
                value = f'{name}.{suffix}'
                suffix += 1
                if value in reserved:
                    continue
            unique_names.append(value)
            seen.add(value)
        supplied = np.load(directory / 'processed/gene_names.npy').astype(str)
        if not np.array_equal(supplied, np.asarray(unique_names)):
            raise ValueError('D2 gene symbols do not match the complete 10x reference order')
        path = directory / 'processed/rna_counts.npz'
        return load_sparse(path), ids, path, {'gene_mapping': 'all 36601 ordered symbols exactly verified against 10x feature reference, including unique suffixes', 'reference': str(reference), 'reference_sha256': sha(reference)}
    path = next((directory / 'raw').glob('*feature_bc_matrix.h5'))
    with h5py.File(path, 'r') as handle:
        matrix = handle['matrix']
        mask = matrix['features/feature_type'][:].astype(str) == 'Gene Expression'
        ids = matrix['features/id'][:].astype(str)[mask]
        feature_map = np.full(mask.size, -1, dtype=np.int64)
        feature_map[mask] = np.arange(mask.sum())
        lookup = {cell: i for i, cell in enumerate(matrix['barcodes'][:].astype(str))}
        pointers = matrix['indptr'][:]
        values, columns, indptr = [], [], [0]
        for cell in cells:
            row = lookup[cell]
            lo, hi = int(pointers[row]), int(pointers[row + 1])
            indices = matrix['indices'][lo:hi]
            keep = mask[indices]
            positions = np.flatnonzero(keep) + lo
            data = matrix['data'][positions] if positions.size else np.asarray([], np.int64)
            values.append(data)
            columns.append(feature_map[indices[keep]])
            indptr.append(indptr[-1] + data.size)
        counts = sparse.csr_matrix((np.concatenate(values), np.concatenate(columns), np.asarray(indptr)), shape=(len(cells), ids.size))
    return counts, ids, path, {'gene_mapping': 'RNA feature IDs from original H5; processed barcodes mapped by exact ID; ATAC values not read'}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def audit() -> dict:
    records = {}
    loaded = {key: raw_rna(key) for key in IDS}
    all_genes = sorted(set().union(*(set(item[1]) for item in loaded.values())))
    canonical = np.asarray(all_genes, dtype=str)
    for key in IDS:
        x, genes, path, provenance = loaded[key]
        assert x.shape[1] == genes.size == len(set(genes))
        fps = sparse_fingerprints(x, genes, canonical)
        counts = Counter(fps)
        records[key] = {
            'cell_count': int(x.shape[0]), 'gene_count': int(x.shape[1]),
            'gene_ids_sha256': hashlib.sha256('\n'.join(genes).encode()).hexdigest(),
            'canonical_gene_union_count': len(canonical), 'barcodes': barcodes(key),
            'fingerprints': fps, 'within_duplicate_excess': int(x.shape[0] - len(counts)),
            'within_duplicate_cells': int(sum(n for n in counts.values() if n > 1)),
            'rna_path': str(path), 'rna_sha256': sha(path), 'provenance': provenance,
        }
    pairs = []
    for a, b in itertools.combinations(IDS, 2):
        ca, cb = Counter(records[a]['fingerprints']), Counter(records[b]['fingerprints'])
        shared = ca.keys() & cb.keys()
        na, nb = sum(ca[k] for k in shared), sum(cb[k] for k in shared)
        pairs.append({'dataset_A': a, 'dataset_B': b, 'n_shared_profiles': len(shared),
                      'n_exact_duplicate_cells_A': na, 'n_exact_duplicate_cells_B': nb,
                      'fraction_A': na / records[a]['cell_count'], 'fraction_B': nb / records[b]['cell_count'],
                      'barcode_overlap': len(set(records[a]['barcodes']) & set(records[b]['barcodes']))})
    identities = {}
    for key, directory in IDS.items():
        identity = json.loads((directory / 'DATASET_IDENTITY.json').read_text())
        identities[key] = {
            'dataset_id': identity.get('dataset_id'), 'canonical_identity': identity.get('canonical_identity'),
            'platform': identity.get('platform'), 'technology': identity.get('technology'),
            'donor': identity.get('donor', identity.get('donor_physiological_status')),
            'cell_count': records[key]['cell_count'], 'label_count': identity.get('label_count', identity.get('cell_types')),
            'batch_count': identity.get('batch_count'), 'official_urls': identity.get('official_urls', []),
            'identity_sha256': sha(directory / 'DATASET_IDENTITY.json'),
        }
    reuse = []
    old = ROOT / 'result_contain/pre_task14_dmode_source_quality_20260926'
    for fold in ('target_D1', 'target_D2', 'target_D3', 'target_D4'):
        manifest = old / fold / 'fold_manifest.json'
        reuse.append({'fold': fold, 'old_manifest_exists': manifest.exists(),
                      'status': 'REBUILD_REQUIRED',
                      'reason': 'NEW_D2 source dependency' if fold != 'target_D2' else 'No exact source/code/config/feature/candidate hash match has been demonstrated; target_D2 sources do not include D2.'})
    passed = not any(row['n_shared_profiles'] for row in pairs) and not any(row['within_duplicate_cells'] for row in records.values())
    write_json(ROOT / 'UPDATED_FOUR_DATASET_INDEPENDENCE_AUDIT.json', {
        'status': 'PASS_NO_EXACT_RNA_COUNT_DUPLICATES' if passed else 'FAIL_DUPLICATES', 'training_performed': False,
        'FOUR_DATASET_INDEPENDENCE': 'PASS' if passed else 'FAIL',
        'scope': 'current paired cohorts; complete raw RNA features, not 2000-gene model projections; distinct dataset provenance does not prove donor independence',
        'canonical_fingerprint': 'SHA256(union-gene-order digest + sorted sparse gene-index/count pairs)',
        'datasets': records, 'pairs': pairs,
    })
    write_json(ROOT / 'UPDATED_DATASET_PROVENANCE_TABLE.json', {'datasets': identities, 'source': 'current DATA/DATASET_IDENTITY.json'})
    write_json(ROOT / 'UPDATED_FULLB2_REUSE_AUDIT.json', {'status': 'REBUILD_REQUIRED', 'rows': reuse, 'historical_files_modified': False})
    with (ROOT / 'UPDATED_PAIRWISE_CELL_FINGERPRINTS.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset', 'canonical_identity', 'barcode', 'gene_order_sha256', 'raw_count_fingerprint', 'rna_file_sha256'])
        for key, record in records.items():
            for cell, fingerprint in zip(record['barcodes'], record['fingerprints']):
                writer.writerow([key, identities[key]['canonical_identity'], cell, record['gene_ids_sha256'], fingerprint, record['rna_sha256']])
    with (ROOT / 'UPDATED_PAIRWISE_DUPLICATE_MATRIX.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    lines = ['# Updated four-dataset independence audit', '', f'FOUR_DATASET_INDEPENDENCE = {"PASS" if passed else "FAIL"}', '', 'Complete raw RNA gene universe; provenance retained separately from the comparable count hash.', '', '| Pair | Exact count duplicates A / B | Fraction A / B | Barcode overlap |', '| --- | --- | --- | --- |']
    for row in pairs:
        lines.append(f"| {row['dataset_A']}-{row['dataset_B']} | {row['n_exact_duplicate_cells_A']} / {row['n_exact_duplicate_cells_B']} | {row['fraction_A']} / {row['fraction_B']} | {row['barcode_overlap']} |")
    for key, record in records.items():
        lines.append(f"\n{key}: {record['cell_count']} cells; {record['gene_count']} raw genes; within-cohort duplicate cells {record['within_duplicate_cells']}; input {record['rna_path']}; SHA256 {record['rna_sha256']}.")
    (ROOT / 'UPDATED_FOUR_DATASET_INDEPENDENCE_AUDIT.md').write_text('\n'.join(lines) + '\n')
    return {'datasets': records, 'pairs': pairs, 'canonical_genes': canonical.tolist()}


def blocked_report(result):
    audit_record = json.loads((ROOT / 'UPDATED_FOUR_DATASET_INDEPENDENCE_AUDIT.json').read_text())
    reuse = json.loads((ROOT / 'UPDATED_FULLB2_REUSE_AUDIT.json').read_text())
    for row in reuse['rows']:
        row['status'] = 'REBUILD_REQUIRED'
        row['reason'] = 'NEW_D2 source dependency' if row['fold'] != 'target_D2' else 'Exact source/code/config/feature/candidate hash match not demonstrated; sources do not include D2.'
    write_json(ROOT / 'UPDATED_FULLB2_REUSE_AUDIT.json', reuse)
    with (ROOT / 'UPDATED_FULLB2_REUSE_AUDIT.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reuse['rows'][0]))
        writer.writeheader()
        writer.writerows(reuse['rows'])
    provenance = json.loads((ROOT / 'UPDATED_DATASET_PROVENANCE_TABLE.json').read_text())
    rows = []
    for key, directory in IDS.items():
        identity = json.loads((directory / 'DATASET_IDENTITY.json').read_text())
        annotation = {}
        if (directory / 'IDENTITY.yaml').exists():
            for line in (directory / 'IDENTITY.yaml').read_text().splitlines():
                if ': ' in line and not line.startswith(' '):
                    name, value = line.split(': ', 1)
                    annotation[name] = value
        atac = next(path for path in [directory / 'processed/atac_counts.npz', directory / 'processed/atac.npz'] if path.exists())
        rows.append(dict(dataset=key, canonical_dataset_id=identity['canonical_identity'],
            source_accession=identity.get('tenx_dataset_slug', identity['canonical_identity']),
            donor=json.dumps(identity.get('donor'), ensure_ascii=True) if identity.get('donor') else annotation.get('Donor', 'UNKNOWN'),
            library_identity='UNKNOWN; dataset-level release identity only',
            sequencing_platform=identity.get('platform'),
            processing_version=identity.get('chemistry_or_reference', 'GEO exported HDa raw counts'),
            rna_file=result['datasets'][key]['rna_path'], rna_sha256=result['datasets'][key]['rna_sha256'],
            atac_file=str(atac), atac_sha256=sha(atac),
            paired_cells=result['datasets'][key]['cell_count'],
            raw_gene_count=result['datasets'][key]['gene_count'],
            official_url=annotation.get('Official URL', ';'.join(identity.get('official_urls', [])))))
    with (ROOT / 'UPDATED_DATASET_PROVENANCE_TABLE.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    provenance['verified_current_files'] = rows
    provenance['NEW_D2_PROVENANCE'] = 'PASS'
    provenance['claim_limit'] = 'Distinct GEO HDa study identity, not an alternate release of current 10x D1/D3/D4; physical library IDs and donor independence not proven.'
    write_json(ROOT / 'UPDATED_DATASET_PROVENANCE_TABLE.json', provenance)
    gate = dict(FOUR_DATASET_INDEPENDENCE=audit_record['FOUR_DATASET_INDEPENDENCE'],
        NEW_D2_PROVENANCE='PASS', SOURCE_ARTIFACT_REBUILD='FAIL', TARGET_RNA_ONLY_DRYRUN='FAIL',
        FULL_B2_METHOD_DRIFT='YES', UPDATED_RUNNER_READY='NO', FORMAL_RUN_STATUS='NOT_STARTED',
        status='BLOCKED_STEP1', seeds_completed=[], folds_completed=[], folds_rebuilt=[], folds_reused=[],
        B_D_REBUILD_STATUS='INCOMPLETE_INVALID_CANDIDATE', F_REBUILD_STATUS='NOT_COMPLETE',
        target_atac_loaded_in_aborted_candidate=True, training_performed=False,
        candidate_staging_valid=False, checkpoints_created=0, target_embeddings_created=0,
        MULTI_TEACHER_READY_WITH_NEW_D2='NOT_VALIDATED', FUSION_READY_WITH_NEW_D2='NOT_VALIDATED',
        UMAP_READY=False, run_tier='pre_final_user_explicit_3seeds',
        authorized_seeds=[20020130, 20020131, 20020132],
        blockers=[
            'Candidate staging stores raw SOURCE counts/all genes, whereas frozen Task35 consumes source-HVG-intersection log-normalized RNA and SOURCE-train mean/std; no validated equivalence.',
            'Candidate prepare_harmonized loads held-out target ATAC; aborted artifacts must not be reused.',
            'Copied pre-Task14 source-quality runner is not an actual neural Full B2 training runner; --smoke did not implement a bounded smoke.',
            'Candidate source preprocessing state and full code/config/feature/candidate hash lineage are not complete.'
        ],
        authoritative_teacher_runner=str(ROOT / 'result_contain/35_b2_shared_regulatory_teacher_dense_kd_20261003/source/run_b2_strict_lodo.py'),
        authoritative_strict_input_runner=str(ROOT / 'dmode_dev/experiments/v5d/run_strict_lodo_two_step.py'),
        historical_results_modified=False, registry_modified=False)
    write_json(OUT / '_benchmark_meta/STEP1_GATE.json', gate)
    staging_manifest = OUT / 'source_input_manifest.json'
    value = json.loads(staging_manifest.read_text())
    value.update(status='INVALID_FOR_FROZEN_FULL_B2', training_allowed=False, gate=str(OUT / '_benchmark_meta/STEP1_GATE.json'))
    write_json(staging_manifest, value)
    report = ['# Updated Full B2 preflight gate', '', 'STATUS: BLOCKED_STEP1. No neural training performed.', '',
        '| Gate | Result |', '| --- | --- |']
    for key in ['FOUR_DATASET_INDEPENDENCE', 'NEW_D2_PROVENANCE', 'SOURCE_ARTIFACT_REBUILD', 'TARGET_RNA_ONLY_DRYRUN', 'FULL_B2_METHOD_DRIFT', 'UPDATED_RUNNER_READY', 'FORMAL_RUN_STATUS']:
        report.append(f'| {key} | {gate[key]} |')
    report.extend(['', '## Blockers', ''] + [f'- {item}' for item in gate['blockers']])
    report.extend(['', '## Evidence', '', f"- Frozen Teacher/Student: {gate['authoritative_teacher_runner']}",
        f"- Frozen source-HVG-intersection and strict RNA inference: {gate['authoritative_strict_input_runner']}",
        '- Candidate source_rebuild_updated_full_b2.py originally passed all four datasets to prepare_harmonized; the helper reads atac_binary.npz for every spec.',
        '- Raw counts audit supersedes the earlier invalid 2000-gene projection audit. Same barcode strings alone are not duplicate evidence.',
        '- D1/D3/D4 donor/library independence is not established solely by their release slugs. Donor descriptions match age/sex but do not prove identical or different donors.',
        '- No completed B_D/F folds, Teacher checkpoints, Student checkpoints, embeddings or metrics. No formal or ablation run started.',
        '- STOP as required by Step1 gate; core method and historical results are unchanged.'])
    (OUT / '_benchmark_meta/UPDATED_FULLB2_REPORT.md').write_text('\n'.join(report) + '\n')
    print(json.dumps(gate, indent=2))


def stage_inputs(audit_result: dict) -> Path:
    stage = OUT / 'source_inputs'
    stage.mkdir(parents=True, exist_ok=False)
    for key, directory in IDS.items():
        root = stage / key
        root.mkdir()
        proc = directory / 'processed'
        x = rna(key)
        genes = gene_ids(key)
        peaks_path = proc / 'peak_names.npy'
        atac_path = next(p for p in (proc / 'atac_binary.npz', proc / 'atac.npz', proc / 'atac_counts.npz') if p.exists())
        np.save(root / 'rna.npy', x.toarray().astype(np.float32))
        np.save(root / 'gene_ids.npy', genes)
        np.save(root / 'gene_names.npy', genes)
        if peaks_path.exists():
            peaks = np.load(peaks_path, allow_pickle=True).astype(str)
        else:
            with np.load(proc / 'atac.npz', allow_pickle=True) as z:
                peaks = z['peak_ids'].astype(str)
        np.save(root / 'peak_names.npy', peaks)
        atac = load_sparse(atac_path).tocsr().astype(np.float32)
        atac.data[:] = 1.0
        sparse.save_npz(root / 'atac_binary.npz', atac)
        for name in ('train_index.npy', 'val_index.npy', 'test_index.npy'):
            source = proc / name
            if not source.exists():
                with np.load(proc / 'metadata.npz', allow_pickle=True) as meta:
                    np.save(root / name, meta[name])
            else:
                np.save(root / name, np.load(source))
        np.savez(root / 'metadata.npz', train_index=np.load(root / 'train_index.npy'),
                 val_index=np.load(root / 'val_index.npy'), test_index=np.load(root / 'test_index.npy'))
    write_json(OUT / 'source_input_manifest.json', {'status': 'STAGED', 'source_inputs': str(stage), 'training_performed': False})
    return stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.report_only:
        blocked_report(json.loads((ROOT / 'UPDATED_FOUR_DATASET_INDEPENDENCE_AUDIT.json').read_text()))
        return
    if args.audit_only:
        result = audit()
        blocked_report(result)
        print(json.dumps({'status': 'RAW_RNA_AUDIT_COMPLETE', 'pairs': result['pairs']}, indent=2))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    result = audit()
    stage_inputs(result)
    print(json.dumps({'status': 'AUDITED_AND_STAGED', 'output': str(OUT), 'pairs': result['pairs']}, indent=2))


if __name__ == '__main__':
    main()
