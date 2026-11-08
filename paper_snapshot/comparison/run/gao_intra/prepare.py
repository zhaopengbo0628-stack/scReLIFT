"""Audit frozen Gao data; create paired train/val-only format views, without refitting."""
import argparse
import csv
import fcntl
import os
import re
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from common import ROOT, CACHE, BENCH, SOURCE, SEEDS, TIER, METHODS, read, write, sha


def paired_csr(path, rows):
    """Extract only selected CSR rows from compressed arrays; never construct Test ATAC."""
    with np.load(path) as arrays:
        pointer = arrays['indptr']
        shape = arrays['shape']
    rows = np.asarray(rows)
    sizes = np.diff(pointer)[rows]
    output_pointer = np.r_[0, np.cumsum(sizes)]
    output = {}
    # Zip seek discards compressed gaps; Test values are never decoded as model arrays.
    with zipfile.ZipFile(path) as archive:
        for name in ['indices', 'data']:
            with archive.open(name+'.npy') as stream:
                version = np.lib.format.read_magic(stream)
                header = np.lib.format._read_array_header(stream, version)
                dtype = header[2]
                start = stream.tell()
                values = np.empty(int(output_pointer[-1]), dtype=dtype)
                order = np.argsort(rows)
                for j in order:
                    left, right = int(pointer[rows[j]]), int(pointer[rows[j]+1])
                    stream.seek(start+left*dtype.itemsize)
                    raw = stream.read((right-left)*dtype.itemsize)
                    values[output_pointer[j]:output_pointer[j+1]] = np.frombuffer(raw, dtype=dtype)
                output[name] = values
    return sparse.csr_matrix((output['data'], output['indices'], output_pointer), shape=(len(rows), int(shape[1])))


def initialize():
    registry = ROOT/'result_contain/task_registry.csv'
    with (ROOT/'result_contain/task_registry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with registry.open() as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        used = {int(row['task_id']) for row in rows}
        for parent in [ROOT/'result_contain', ROOT/'comparison/result', ROOT/'main/result/s_mode/intra']:
            for p in parent.iterdir():
                match = re.match(r'(\d+)_', p.name)
                if match:
                    used.add(int(match[1]))
        task = max(used | {83})+1
        out = ROOT/'result_contain'/f'{task}_gao_intra_8methods_3seed'
        out.mkdir(exist_ok=False)
        plan = dict(task_id=task, result_root=str(out), seeds=SEEDS, run_tier=TIER,
                    methods=METHODS, expected_main_units=24, expected_pca_units=3,
                    pause_granularity='method x seed; current atomic unit finishes before pause/stop',
                    server_git_commit=None, server_git_status='project has no .git; execution hashes authoritative')
        write(out/'RUN_PLAN.json', plan)
        row = dict.fromkeys(fields, '')
        row.update(task_id=str(task), status='PREFLIGHT', experiment_name=out.name,
            model_version='Gao S-mode intra 8 methods + RNA PCA', result_path=str(out),
            run_script_path=str(ROOT/'comparison/run/gao_intra/manage.py'), datasets='Gao P56 VIS',
            seeds=','.join(map(str, SEEDS)), launcher='direct_detached', created_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        rows.append(row)
        temporary = registry.with_name(registry.name+'.gao.tmp')
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(registry)
    print(out, flush=True)


def prepare(out):
    audit = {}
    for p in [CACHE/'manifest.json', BENCH/'manifest.json']:
        for name, expected in read(p)['hashes'].items():
            source = Path(name)
            actual = sha(source)
            if actual != expected:
                historical = ROOT/'main/result/s_mode/intra'/CACHE.parent.name/'source'/source.name
                assert source.suffix == '.py' and historical.exists() and sha(historical) == expected, name
                audit[name] = dict(historical_sha256=expected, current_sha256=actual,
                                   historical_source=str(historical), change='later dataset adapter extension; frozen data unchanged')
    cells = np.load(SOURCE/'barcodes.npy').astype(str)
    meta = pd.read_csv(BENCH/'metadata.csv', keep_default_na=False)
    assert np.array_equal(cells, meta.barcode.to_numpy(str)) and len(set(cells)) == 35194
    split = {s: np.load(CACHE/f'{s}_index.npy') for s in ['train', 'val', 'test']}
    original = np.load(BENCH/'splits.npz')
    assert all(np.array_equal(idx, original[s+'_index']) for s, idx in split.items())
    assert [len(split[s]) for s in ['train', 'val', 'test']] == [24635, 5279, 5280]
    assert np.array_equal(np.sort(np.concatenate(list(split.values()))), np.arange(len(cells)))
    assert meta.iloc[split['test']].subclass_label.nunique() == 25
    assert meta.age_label.eq('P56').all() and meta.donor_name.nunique() == 4 and meta.library_prep.nunique() == 6
    view = out/'inputs'
    view.mkdir(exist_ok=False)
    ntrain = len(split['train'])
    paired = np.r_[split['train'], split['val']]
    atac = paired_csr(SOURCE/'atac_counts.npz', paired)
    sparse.save_npz(view/'atac_train_val_counts.npz', atac)
    columns = np.load(CACHE/'hvg_native_columns.npy')
    rna = sparse.load_npz(SOURCE/'rna_counts.npz').tocsr()[:, columns]
    sparse.save_npz(view/'rna_hvg_counts.npz', rna)
    np.save(view/'train_val_original_indices.npy', paired)
    (view/'barcodes.txt').write_text('\n'.join(cells)+'\n')
    for name in ['gene_ids.npy', 'peak_names.npy', 'train_index.npy', 'val_index.npy', 'test_index.npy']:
        shutil.copy2(CACHE/name, view/name)
    counts_binary = atac[:ntrain].astype(bool).astype(np.float32)
    selected = np.asarray(counts_binary.sum(0)).ravel() >= np.ceil(.005*ntrain)
    write(out/'DATASET_AND_SPLIT_MANIFEST.json', dict(status='VERIFIED', cells=len(cells),
        split_counts={k:len(v) for k,v in split.items()}, donors=4, libraries=6, test_subclasses=25,
        barcode_sha256=sha(SOURCE/'barcodes.npy'), split_sha256=sha(BENCH/'splits.npz'),
        metadata_sha256=sha(BENCH/'metadata.csv'), frozen_HVG_sha256=sha(CACHE/'hvg.csv'),
        cache_hash_audit=audit, paired_atac_rows=len(paired), paired_atac_excludes_test=True,
        pbmc_prevalence_selected_peaks=int(selected.sum()), test_labels_for_training=False,
        reference_assumption='Task80 project GRCm38/gencode M25; author exact release unresolved',
        author_peak_calling='Author subclass-by-age peak universe reused, as Task80; not train-only peak calling',
        hashes={str(p):sha(p) for p in list(view.iterdir())+[CACHE/'manifest.json', BENCH/'manifest.json']}))
    print('INPUTS_READY', selected.sum(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--initialize', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    initialize() if args.initialize else prepare(args.output)
