"""Author-QC SEA-AD paired counts only; no model preprocessing or training."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata.experimental import read_elem


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(16 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def extract(matrix, rows, output):
    shape = tuple(map(int, matrix.attrs['shape']))
    if matrix.attrs['encoding-type'] != 'csr_matrix':
        raise ValueError('Expected cells-by-features CSR')
    ptr = matrix['indptr'][:]
    nnz = len(matrix['data'])
    if (len(ptr) != shape[0]+1 or ptr[0] != 0 or ptr[-1] != nnz
            or np.any(np.diff(ptr) < 0) or len(matrix['indices']) != nnz):
        raise ValueError('Invalid sparse matrix structure')
    totals, detected = np.zeros(len(rows), dtype=np.int64), np.zeros(len(rows), dtype=np.int64)
    out_ptr = np.zeros(len(rows)+1, dtype=np.int64)
    with h5py.File(output, 'w') as handle:
        group = handle.create_group('X')
        group.attrs.update({'encoding-type': 'csr_matrix', 'encoding-version': '0.1.0',
                            'shape': (len(rows), shape[1])})
        data = group.create_dataset('data', (0,), maxshape=(None,), dtype=matrix['data'].dtype,
                                    chunks=True, compression='gzip', compression_opts=1)
        indices = group.create_dataset('indices', (0,), maxshape=(None,), dtype=matrix['indices'].dtype,
                                       chunks=True, compression='gzip', compression_opts=1)
        # Read original CSR sequentially; validate every count, export only selected rows.
        for first in range(0, shape[0], 256):
            last = min(first+256, shape[0])
            lo, hi = int(ptr[first]), int(ptr[last])
            values, features = matrix['data'][lo:hi], matrix['indices'][lo:hi]
            if (not np.all(np.isfinite(values)) or np.any(values < 0)
                    or np.any(values != np.floor(values))):
                raise ValueError('Nonfinite, negative or noninteger raw counts')
            if np.any((features < 0) | (features >= shape[1])):
                raise ValueError('Feature index out of range')
            positions = np.flatnonzero((rows >= first) & (rows < last))
            segments, feature_segments = [], []
            for position in positions:
                row = rows[position]
                start, end = int(ptr[row]-lo), int(ptr[row+1]-lo)
                x, ix = values[start:end], features[start:end]
                if len(np.unique(ix)) != len(ix):
                    raise ValueError('Duplicate feature within selected cell')
                totals[position], detected[position] = int(x.sum()), int(np.count_nonzero(x))
                if totals[position] == 0:
                    raise ValueError('Empty selected cell')
                out_ptr[position+1] = len(x)
                segments.append(x); feature_segments.append(ix)
            if segments:
                x, ix = np.concatenate(segments), np.concatenate(feature_segments)
                offset = len(data)
                data.resize((offset+len(x),)); indices.resize((offset+len(ix),))
                data[offset:] = x; indices[offset:] = ix
            if first % 16384 == 0:
                print('SCAN', output.name, last, '/', shape[0], flush=True)
        # rows must be ascending so exported order and indptr agree.
        if np.any(np.diff(rows) <= 0):
            raise ValueError('Export rows are not strictly ascending')
        np.cumsum(out_ptr, out=out_ptr)
        if out_ptr[-1] != len(data):
            raise ValueError('Export CSR pointers differ from data length')
        group.create_dataset('indptr', data=out_ptr)
    return totals, detected, shape, nnz


def stage(raw, output, commit):
    report = json.loads((raw/'PAIRING_REPORT.json').read_text())
    if not report['permitted_full_download'] or any(report['identity_mismatches'].values()):
        raise ValueError('Paired identities not verified')
    inputs = {}
    for key in ('rna', 'atac'):
        name = report[key]
        state = json.loads((raw/(name+'.download.json')).read_text())
        if state['status'] != 'COMPLETE' or (raw/name).stat().st_size != state['size_bytes']:
            raise ValueError('Full raw download incomplete')
        inputs[name] = sha(raw/name)
        if inputs[name] != state['sha256']:
            raise ValueError('Raw SHA256 mismatch')
    pairs = pd.read_csv(raw/'PAIRED_CELLS.csv').sort_values('rna_row').reset_index(drop=True)
    if len(pairs) != report['paired_cells'] or not pairs.cell_id.is_unique:
        raise ValueError('Paired mapping incomplete or duplicated')
    if not pairs.identity_consistent.all():
        raise ValueError('Paired mapping contains identity conflict')
    output.mkdir(parents=True, exist_ok=False)
    with h5py.File(raw/report['rna'], 'r') as rna, h5py.File(raw/report['atac'], 'r') as atac:
        obs_r = read_elem(rna['obs']).iloc[pairs.rna_row.to_numpy()].copy()
        obs_a = read_elem(atac['obs']).iloc[pairs.atac_row.to_numpy()].copy()
        if not np.array_equal(obs_r.index.astype(str), pairs.cell_id):
            raise ValueError('RNA cell order mismatch')
        if not np.array_equal(obs_a.index.astype(str), pairs.cell_id):
            raise ValueError('ATAC cell order mismatch')
        if (not np.all(obs_r['method'].astype(str) == '10xMulti')
                or not np.all(obs_a['method'].astype(str) == '10xMulti')
                or not np.array_equal(obs_r['Donor ID'].astype(str), obs_a['Donor ID'].astype(str))):
            raise ValueError('Paired method or donor mismatch')
        genes = read_elem(rna['var'])
        peaks = read_elem(atac['var'])
        if not genes['gene_ids'].is_unique or not peaks.index.is_unique:
            raise ValueError('Duplicate gene or peak ID; requires review')
        if set(genes.genome.astype(str)) != {'GRCh38'}:
            raise ValueError('Unexpected RNA reference genome')
        intervals = []
        for peak in peaks.index.astype(str):
            match = re.fullmatch(r'([^:]+):(\d+)-(\d+)', peak)
            if match is None:
                match = re.fullmatch(r'(chr[^_]+)_(\d+)_(\d+)', peak)
            if match is None or int(match[2]) >= int(match[3]):
                raise ValueError('Unverified ATAC interval format: '+peak)
            intervals.append(match.groups())
        obs_r.to_csv(output/'cells_rna_author.csv', index_label='cell_id')
        obs_a.to_csv(output/'cells_atac_author.csv', index_label='cell_id')
        genes.to_csv(output/'rna_features.tsv', sep='\t', index_label='gene_name')
        peaks.to_csv(output/'atac_features.tsv', sep='\t', index_label='peak_id')
        pd.DataFrame(intervals).to_csv(output/'atac_peaks.bed', sep='\t', index=False, header=False)
        # ATAC rows sorted for sequential export; map them back to RNA order explicitly.
        order = np.argsort(pairs.atac_row.to_numpy())
        rt, rd, rs, rn = extract(rna['layers/UMIs'], pairs.rna_row.to_numpy(), output/'rna_counts.h5')
        at, ad, ats, an = extract(atac['X'], pairs.atac_row.to_numpy()[order], output/'atac_counts.h5')
        # Reorder exported ATAC row ranges on disk, never densify.
        sorted_path = output/'atac_counts.h5'
        reordered_path = output/'atac_counts_reordered.h5'
        with h5py.File(sorted_path, 'r') as source, h5py.File(reordered_path, 'w') as target:
            src, dst = source['X'], target.create_group('X')
            dst.attrs.update(dict(src.attrs))
            ptr = src['indptr'][:]; inverse = np.argsort(order)
            pointers = np.r_[0, np.cumsum(np.diff(ptr)[inverse])].astype(np.int64)
            dst.create_dataset('indptr', data=pointers)
            for field in ('data', 'indices'):
                values = dst.create_dataset(field, (int(pointers[-1]),), dtype=src[field].dtype,
                    chunks=True, compression='gzip', compression_opts=1)
                for i, j in enumerate(inverse):
                    values[pointers[i]:pointers[i+1]] = src[field][ptr[j]:ptr[j+1]]
        sorted_path.unlink(); reordered_path.rename(sorted_path)
        at, ad = at[np.argsort(order)], ad[np.argsort(order)]
    pairs.to_csv(output/'paired_cell_index.csv', index=False)
    (output/'barcodes.txt').write_text('\n'.join(pairs.cell_id)+'\n')
    pd.DataFrame(dict(cell_id=pairs.cell_id, rna_counts=rt, rna_genes=rd, atac_counts=at,
        atac_features=ad, author_final_qc_intersection=True)).to_csv(output/'qc_per_cell.csv', index=False)
    for field in ('Class', 'Subclass', 'Supertype'):
        pd.crosstab(obs_r['Donor ID'], obs_r[field]).to_csv(output/('donor_'+field.lower()+'_counts.csv'))
    donor_fields = [f for f in ('Age at Death', 'Sex', 'Gender', 'Cognitive Status',
                    'Overall AD neuropathological Change', 'Braak', 'CERAD score', 'APOE Genotype') if f in obs_r]
    for field in donor_fields:
        if (obs_r.groupby('Donor ID', observed=True)[field].nunique(dropna=False) > 1).any():
            raise ValueError('Inconsistent donor metadata: '+field)
    donors = obs_r.groupby('Donor ID', observed=True)[donor_fields].first()
    donors['cells'] = obs_r.groupby('Donor ID', observed=True).size()
    donors.to_csv(output/'donors.csv')
    qc = dict(status='AUTHOR_FINAL_QC_PAIRED_COUNTS_VERIFIED', retained_cells=len(pairs),
        donors=len(donors), rna_genes=rs[1], atac_features=ats[1],
        raw_rna_shape=rs, raw_atac_shape=ats, scanned_rna_nnz=rn, scanned_atac_nnz=an,
        qc_basis='Intersection of author final-nuclei RNA20260622 and ATAC20241206, both 10xMulti',
        added_thresholds=False, annotation_harmonized=False, disease_binary_defined=False,
        author_rna_classes={f: obs_r[f].value_counts().to_dict() for f in ('Class','Subclass','Supertype')},
        warnings=['Cross-version taxonomy/QC differ; preserve both annotations.',
                  'Disease metadata is donor-level; no AD/Control recoding or split frozen.'])
    write_json(output/'QC_REPORT.json', qc)
    manifest = dict(status='COMPLETE', source_commit=commit, script_sha256=sha(__file__),
        raw_path=str(raw), input_sha256=inputs, count_order='same paired_cell_index order in both modalities',
        counts_layout='cells_by_features_CSR_HDF5_X', author_qc=True,
        model_preprocessing=False, hvg=False, normalized=False, split_frozen=False, training=False,
        output_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()})
    write_json(output/'manifest.json', manifest)
    (output/'README.md').write_text('Author-QC paired SEA-AD counts; see manifest.json and QC_REPORT.json.\n'
        'RNA uses raw UMIs; ATAC uses verified integer X. No normalization, HVG, split or training.\n')
    print(json.dumps(qc), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    stage(args.raw, args.output, args.commit)
