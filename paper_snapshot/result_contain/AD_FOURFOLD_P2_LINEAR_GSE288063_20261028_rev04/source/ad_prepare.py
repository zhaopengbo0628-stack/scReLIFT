"""CPU-only two-holdout cache adapter. Historical scientific modules stay frozen."""
from __future__ import annotations

import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path('/home/u202522802045/share/zpb')
OLD = ROOT / 'result_contain/AD_Dmode_4Cohort_LODO_20261026_rev02'
OUT = ROOT / 'result_contain/AD_FOURFOLD_P2_LINEAR_GSE288063_20261028_rev04/preprocessing'
NAME = 'GSE288063_GSE278576'
NAMES = ['GSE214979','GSE268609','SEAAD_MTG',NAME]
FOLDS = {f'fold{i+1}':(n,[s for s in NAMES if s!=n]) for i,n in enumerate(NAMES)}
EXPECTED = {'GSE214979':(105332,15),'GSE268609':(51246,13),'SEAAD_MTG':(113019,24),NAME:(54999,10)}
LABELS = {'GSE214979':'disease_status','GSE268609':'Group','SEAAD_MTG':'Overall AD neuropathological Change',NAME:'disease_status'}
POS = {'AD', 'EOAD', 'sEOAD', 'Intermediate', 'High'}
NEG = {'HA', 'Ctrl', 'Control', 'Not AD'}
spec = importlib.util.spec_from_file_location('frozen_fourfold_prep', OLD / 'queue/ad_fourfold_prepare.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
# Split lookup and progress writes belong to this run, not the historical run.
legacy.OUT, legacy.QUEUE = OUT, OUT / 'queue'
import ad_new_cohort as new_cohort
legacy.source_defs = lambda: new_cohort.source_defs(legacy)


def event(stage, **extra):
    record = dict(stage=stage, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  pid=os.getpid(), **extra)
    print(json.dumps(record), flush=True)
    legacy.write(OUT / 'queue/cpu_status.json', record)
    with (OUT / 'queue/events.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')


def atomic_array(path, values):
    tmp = path.with_name(path.stem + '.tmp.npy')
    np.save(tmp, values)
    tmp.replace(path)


def validate_dense(path, shape, nonnegative=False):
    arr = np.load(path, mmap_mode='r')
    assert arr.shape == shape and arr.dtype == np.float32, (path, arr.shape, shape)
    for start in range(0, shape[0], 4096):
        block = arr[start:start + 4096]
        assert np.isfinite(block).all(), str(path)
        if nonnegative:
            assert (block >= 0).all(), str(path)
    return arr


def prepare_rna(reader, dest, selected):
    path = dest / 'rna.npy'
    marker = dest / 'RNA_COMPLETE.json'
    if marker.exists():
        assert json.loads(marker.read_text())['sha256'] == legacy.sha(path)
    else:
        tmp = dest / 'rna.building.npy'
        a = legacy.build_dense(reader, tmp, selected)
        del a
        validate_dense(tmp, (reader.rows.size, selected.size), True)
        tmp.replace(path)
        legacy.write(marker, dict(sha256=legacy.sha(path), rows=int(reader.rows.size),
                                  genes=int(selected.size), normalization='CP10K_full_36601_log1p'))
    return validate_dense(path, (reader.rows.size, selected.size), True)


def prepare_atac(reader, native, canon, dest, cohort):
    marker = dest / 'ATAC_COMPLETE.json'
    path = dest / 'atac_binary.npz'
    if marker.exists():
        for p, h in json.loads(marker.read_text())['hashes'].items():
            assert legacy.sha(p) == h, p
        b = sparse.load_npz(path)
        mapping = np.load(dest / 'peak_mapping.npy')
    else:
        mapping = legacy.map_peaks(native, canon)
        assert mapping.shape == (len(native),) and (mapping >= 0).all()
        tmpdir = dest / '_atac_build'
        tmpdir.mkdir(exist_ok=True)
        b = legacy.build_atac(reader, mapping, len(canon), tmpdir)
        assert b.shape == (reader.rows.size, len(canon))
        assert np.isfinite(b.data).all() and (b.data == 1).all()
        (tmpdir / 'atac_binary.npz').replace(path)
        tmpdir.rmdir()
        atomic_array(dest / 'peak_mapping.npy', mapping)
        atomic_array(dest / 'canonical_peak_available.npy',
                     np.bincount(mapping, minlength=len(canon)) > 0)
        if cohort == NAME:
            masks = np.stack([
                np.bincount(mapping[mp], minlength=len(canon)) > 0
                for mp in reader.mappings
            ])
            atomic_array(dest / 'library_peak_available.npy', masks)
        outputs = [path, dest / 'peak_mapping.npy', dest / 'canonical_peak_available.npy']
        if cohort == NAME:
            outputs.append(dest / 'library_peak_available.npy')
        legacy.write(marker, dict(hashes={str(p): legacy.sha(p) for p in outputs},
                                  zero_atac_cells=int((b.getnnz(axis=1) == 0).sum()),
                                  binary=True, absent_peak_is_unobserved=True))
    assert b.shape == (reader.rows.size, len(canon))
    assert np.isfinite(b.data).all() and (b.data == 1).all()
    assert (b.getnnz(axis=1) > 0).all(), 'Zero ATAC rows require audit'
    return b


def input_paths(defs):
    paths=set()
    for rna,atac,*_ in defs.values():
        for reader in (rna,atac):
            parts=reader.parts if isinstance(reader,legacy.Concat) else [reader]
            paths.update(p.path for p in parts)
    paths.update([new_cohort.BUNDLE/'download_manifest.json',new_cohort.BUNDLE/'verification_report_rev04.json',
        ROOT/'data/benchmark/AD/GSE214979/author_qc_v1_20261012_bundle/post_qc/cells.csv',
        ROOT/'result_contain/AD_Dmode_4Cohort_20261024/data_audit/seaad_source3_eligible_counts/cells.csv.gz',
        ROOT/'data/raw/AD/GSE288063/GSE288063_metadata_humanAD.tsv.gz',
        ROOT/'data/reference/genome_annotation/Homo_sapiens/GRCh38/gencode_32/gencode.v32.primary_assembly.annotation.gtf.gz'])
    return sorted(paths)

def check_genes(genes):
    assert genes.size == len(set(genes)) == 36601
    assert all(g.startswith('ENSG') and '.' not in g for g in genes)
    p2 = ROOT / 'data/staging/AD/GSE268609/verified_csr_20261024/rna_counts.h5ad'
    _, v2 = legacy.read_obs(p2)
    p3 = ROOT / 'result_contain/AD_Dmode_Goal_20261024/rev02_three_source_seaad_holdout/preprocessing/seaad_paired_counts/rna_features.tsv'
    g3 = pd.read_csv(p3, sep='\t').gene_ids.astype(str).to_numpy()
    # Legacy source_defs checks every retired_legacy_cohort library's exact gene order.
    assert set(v2['id'].astype(str)) == set(genes) == set(g3)


def units_table(name, meta, tr, va, unit):
    label = LABELS[name]
    assert not meta[unit].isna().any() and not meta[label].isna().any()
    assert meta.groupby(unit)[label].nunique().max() == 1
    assert set(meta[label].astype(str)) <= POS | NEG, (name, meta[label].unique())
    assert not set(tr) & set(va) and len(tr) + len(va) == len(meta)
    assert not set(meta.iloc[tr][unit].astype(str)) & set(meta.iloc[va][unit].astype(str))
    tab = meta.groupby(unit, sort=True)[label].agg(['first', 'size']).reset_index()
    tab.columns = ['unit_id', 'original_label', 'cells']
    tab['unit_id'] = tab.unit_id.astype(str)
    tab['binary_label'] = tab.original_label.isin(POS).astype(int)
    tab['cohort'] = name
    tab['unit_type'] = 'sample_group' if name == 'GSE268609' else 'donor'
    val_units = set(meta.iloc[va][unit].astype(str))
    tab['representation_split'] = np.where(tab.unit_id.isin(val_units), 'val', 'train')
    return tab


def save_cohort(dest, meta, roster, native, tr, va, libraries=None):
    """Resume immutable metadata without rewriting gzip/npz container timestamps."""
    marker=dest / "COHORT_COMPLETE.json"
    if marker.exists():
        record=json.loads(marker.read_text())
        for path,digest in record["hashes"].items():assert legacy.sha(path)==digest,path
        assert record["cells"]==len(meta) and record["units"]==len(roster)
        previous=pd.read_csv(dest / "cells.csv.gz",low_memory=False)
        assert previous.barcode.astype(str).tolist()==meta.barcode.astype(str).tolist()
        assert previous.count_file.astype(str).tolist()==meta.count_file.astype(str).tolist()
        assert previous.count_row.tolist()==meta.count_row.tolist()
        splits=np.load(dest / "metadata.npz")
        assert np.array_equal(splits["train_index"],tr) and np.array_equal(splits["val_index"],va)
        return
    meta.to_csv(dest / "cells.csv.gz",index=False)
    roster.to_csv(dest / "evaluation_units.csv",index=False)
    np.savez(dest / "metadata.npz",train_index=tr,val_index=va)
    atomic_array(dest / "native_peak_names.npy",native)
    paths=[dest / name for name in ["cells.csv.gz","evaluation_units.csv","metadata.npz","native_peak_names.npy"]]
    if libraries is not None:
        pd.DataFrame({"library_index":np.arange(len(libraries)),"library_id":libraries}).to_csv(dest / "library_order.csv",index=False)
        paths.append(dest / "library_order.csv")
    legacy.write(marker,dict(cells=len(meta),units=len(roster),hashes={str(p):legacy.sha(p) for p in paths}))


def setup():
    (OUT/'queue').mkdir(parents=True,exist_ok=True);(OUT/'provenance').mkdir(exist_ok=True)
    for name in ('GSE214979','SEAAD_MTG'):
        src=OLD/'queue'/f'{name}_donor_split.csv';dst=OUT/'queue'/src.name
        if dst.exists():assert legacy.sha(src)==legacy.sha(dst)
        else:shutil.copyfile(src,dst)
    protocol=json.loads((OUT.parent/'frozen_protocol.json').read_text())
    legacy.write(OUT/'queue/protocol.json',protocol)
    for p in [Path(__file__),Path(legacy.__file__),Path(new_cohort.__file__),
              ROOT/'result_contain/AD_Dmode_4Cohort_20261024/queue/source_balanced_selector.py',
              ROOT/'main/core/regulatory/common/legacy_support/v5d/dmode_reliable_selector.py',
              ROOT/'main/core/regulatory/common/legacy_support/v5d/dmode_teacher.py',
              ROOT/'main/core/regulatory/common/legacy_support/v5d/regulatory_graph.py']:
        q=OUT/'provenance'/p.name
        if q.exists():assert legacy.sha(p)==legacy.sha(q)
        else:shutil.copyfile(p,q)
    return protocol

def main():
    setup()
    event('LOAD_FROZEN_INPUTS_AND_AUTHOR_QC_REPLACEMENT')
    genes, defs = legacy.source_defs()
    check_genes(genes)
    rosters, stats, manifest = {}, {}, []
    for name, (rna, atac, meta, native, tr, va, unit) in defs.items():
        assert (len(meta), meta[unit].nunique()) == EXPECTED[name]
        assert rna.rows.size == atac.rows.size == len(meta)
        if not isinstance(rna,legacy.Concat):
            assert np.array_equal(rna.rows, atac.rows)
            meta['count_row'], meta['count_file'] = rna.rows, str(rna.path)
        assert meta.barcode.astype(str).is_unique if 'barcode' in meta else meta.index.is_unique
        rosters[name] = units_table(name, meta, tr, va, unit)
        dest = OUT / 'cohorts' / name
        dest.mkdir(parents=True, exist_ok=True)
        libs=None
        if name == NAME:
            libs = list(dict.fromkeys(meta.library_id.astype(str)))
            assert len(libs) == len(atac.parts) == 11
        save_cohort(dest,meta,rosters[name],native,tr,va,libs)
        manifest.append(dict(cohort=name, cells=len(meta), units=int(meta[unit].nunique()),
                             positive_units=int(rosters[name].binary_label.sum()),
                             negative_units=int((rosters[name].binary_label == 0).sum()),
                             train_cells=len(tr), val_cells=len(va), native_peaks=len(native),
                             unit_type='sample_group' if name == 'GSE268609' else 'donor'))
        event('COHORT_QC_VERIFIED', **manifest[-1])
    pd.DataFrame(manifest).to_csv(OUT / 'cohort_summary.csv', index=False)
    event('HASH_FROZEN_INPUTS')
    hashfile = OUT / 'provenance/input_hashes.json'
    previous = json.loads(hashfile.read_text()) if hashfile.exists() else {}
    identities = {}
    for path in input_paths(defs):
        st = path.stat()
        old = previous.get(str(path), {})
        assert not old or (old['bytes'] == st.st_size and old['mtime_ns'] == st.st_mtime_ns), path
        identities[str(path)] = dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns,
                                     sha256=old.get('sha256') or legacy.sha(path))
        legacy.write(hashfile, identities)
    # Every cohort is SOURCE in at least one requested experiment. These statistics
    # are not passed into a fold where that cohort is TARGET.
    (OUT / 'hvg').mkdir(exist_ok=True)
    for name, (rna, _, meta, _, tr, va, unit) in defs.items():
        path = OUT / 'hvg' / f'{name}.csv'
        marker = path.with_suffix('.COMPLETE.json')
        if not marker.exists():
            path.unlink(missing_ok=True)
        event('SOURCE_TRAIN_HVG_STATS', cohort=name)
        ids = meta.iloc[tr][unit].astype(str)
        w = np.zeros(len(meta))
        counts = ids.value_counts()
        w[tr] = 1 / (ids.map(counts).to_numpy() * len(counts))
        assert np.isclose(w.sum(), 1) and (w[va] == 0).all()
        if marker.exists():
            assert legacy.sha(path) == json.loads(marker.read_text())['sha256']
        stats[name] = legacy.norm_stats(rna, genes, w, path)
        assert np.array_equal(stats[name].gene_id.to_numpy(), genes)
        assert stats[name].top2000.sum() == 2000
        legacy.write(marker, dict(sha256=legacy.sha(path), train_cells=len(tr),
                                  train_units=len(counts), source_train_only=True, labels_used=False))
    reports = []
    for fold, (target, sources) in FOLDS.items():
        fd = OUT / 'folds' / fold
        fd.mkdir(parents=True, exist_ok=True)
        assert target not in sources and len(set(sources)) == 3
        complete = fd / 'COMPLETE.json'
        if complete.exists():
            data = json.loads(complete.read_text())
            for p, digest in data['hashes'].items():
                assert legacy.sha(p) == digest, p
            reports.append(data['summary'])
            event('FOLD_VERIFIED_ALREADY_COMPLETE', fold=fold)
            continue
        event('FOLD_START', fold=fold, target=target, sources=sources)
        top = [set(stats[n].loc[stats[n].top2000, 'gene_id']) for n in sources]
        inter = sorted(set.intersection(*top))
        rank = np.mean([stats[n].set_index('gene_id').loc[genes].hvg_rank.to_numpy() / len(genes)
                        for n in sources], axis=0)
        selected = (np.flatnonzero(np.isin(genes, inter)) if len(inter) >= 256
                    else np.lexsort((genes, rank))[:2000])
        gene_ids = genes[selected]
        atomic_array(fd / 'gene_ids.npy', gene_ids)
        pd.DataFrame({'gene_id': inter}).to_csv(fd / 'source_hvg_intersection.csv', index=False)
        legacy.write(fd / 'hvg_audit.json', dict(sources=sources, target=target,
                     intersection=len(inter), selected=len(selected), target_used_for_fit=False,
                     fallback_used=len(inter) < 256, normalization_genes=len(genes)))
        intervals = sorted(legacy.parse_peak(p) for n in sources for p in defs[n][3])
        merged = []
        for chrom, start, end in intervals:
            assert 0 <= start < end
            if not merged or merged[-1][0] != chrom or start >= merged[-1][2]:
                merged.append([chrom, start, end])
            else:
                merged[-1][2] = max(merged[-1][2], end)
        canon = np.asarray([f'chr{c}:{s}-{e}' for c, s, e in merged], dtype=str)
        atomic_array(fd / 'canonical_peak_names.npy', canon)
        event('SOURCE_FEATURES_FROZEN', fold=fold, genes=len(gene_ids),
              intersection=len(inter), peaks=len(canon))
        datasets = []
        for name in sources:
            rna, atac, meta, native, tr, va, unit = defs[name]
            dest = fd / name
            dest.mkdir(exist_ok=True)
            event('SOURCE_RNA_ATAC_CACHE', fold=fold, cohort=name)
            rr = prepare_rna(rna, dest, selected)
            b = prepare_atac(atac, native, canon, dest, name)
            np.savez(dest / 'metadata.npz', train_index=tr, val_index=va,
                     test_index=np.array([], dtype=np.int64))
            datasets.append(SimpleNamespace(name=name, rna=rr, atac=b, train=tr, val=va,
                            canonical_peak_available=np.load(dest / 'canonical_peak_available.npy')))
        # Target cache exposes no train/val subset and never builds target ATAC/F.
        td = fd / target
        td.mkdir(exist_ok=True)
        event('TARGET_RNA_FROZEN_TRANSFORM_ONLY', fold=fold, cohort=target)
        target_rna = prepare_rna(defs[target][0], td, selected)
        np.savez(td / 'metadata.npz', train_index=np.array([], dtype=np.int64),
                 val_index=np.array([], dtype=np.int64), test_index=np.arange(len(target_rna)))
        rosters[target].to_csv(td / 'target_evaluation_units.csv', index=False)
        probe = pd.concat([rosters[n] for n in sources], ignore_index=True)
        assert set(probe.cohort) == set(sources) and target not in set(probe.cohort)
        assert len(probe) == sum(EXPECTED[n][1] for n in sources)
        probe.to_csv(fd / 'source_probe_units_after_encoder_freeze.csv', index=False)
        tss = legacy.read_tss(ROOT / 'data/reference/genome_annotation/Homo_sapiens/GRCh38/gencode_32/gencode.v32.primary_assembly.annotation.gtf.gz')
        graph, _ = legacy.build_candidate_graph(gene_ids, canon, tss, 250000, 50)
        graph['peak_names'] = canon
        np.savez_compressed(fd / 'candidate_graph.npz', **graph)
        event('STABLE_REGULATORY_SELECTION', fold=fold,
              candidate_edges=int(graph['candidate_mask'].sum()))
        import v5d.dmode_reliable_selector as selector_module
        original_stability = selector_module._source_stability
        def logged_stability(data, *args, **kwargs):
            event('STABILITY_SOURCE_START', fold=fold, cohort=data.name, train_cells=len(data.train))
            result = original_stability(data, *args, **kwargs)
            event('STABILITY_SOURCE_COMPLETE', fold=fold, cohort=data.name)
            return result
        selector_module._source_stability = logged_stability
        try:
            selector = legacy.load_selector(ROOT / 'main/core/regulatory/common/legacy_support')
            sel = selector(datasets, graph, gene_ids, seed=20020130)
        finally:
            selector_module._source_stability = original_stability
        bd = sel['B_D']
        assert bd.nnz > 0 and bd.shape == (len(canon), len(gene_ids))
        assert np.isfinite(bd.data).all()
        l1 = np.asarray(abs(bd).sum(axis=0)).ravel()
        assert np.all(np.isclose(l1, 1, atol=1e-5) | (l1 == 0))
        sparse.save_npz(fd / 'B_D.npz', bd)
        edges = bd.tocoo()
        pd.DataFrame(dict(gene_id=gene_ids[edges.col], peak=canon[edges.row],
                          weight=edges.data)).to_csv(fd / 'source_regulatory_edges.csv', index=False)
        diagnostics = {k: np.asarray(v) for k, v in sel.items()
                       if k not in ('B_D', 'manifest')}
        np.savez_compressed(fd / 'selector_diagnostics.npz', **diagnostics)
        legacy.write(fd / 'selector_manifest.json', sel['manifest'])
        for ds in datasets:
            event('SOURCE_TRAIN_FIELD_FIT', fold=fold, cohort=ds.name)
            field, params, fm = legacy.fit_dmode_field(ds.atac, bd, ds.train)
            path = fd / ds.name
            atomic_array(path / 'field.npy', field)
            validate_dense(path / 'field.npy', (len(ds.rna), len(gene_ids)))
            np.savez_compressed(path / 'field_state.npz', **params)
            legacy.write(path / 'field_manifest.json', fm)
            del field, params
        summary = dict(experiment=fold, target=target, sources='+'.join(sources),
                       genes=len(gene_ids), hvg_intersection=len(inter), canonical_peaks=len(canon),
                       candidate_edges=int(graph['candidate_mask'].sum()), regulatory_edges=bd.nnz,
                       source_cells=sum(len(d.rna) for d in datasets), target_cells=len(target_rna),
                       source_probe_units=len(probe), target_units=len(rosters[target]),
                       target_used_for_fit=False, disease_labels_used_for_representation=False,
                       status='CPU_COMPLETE', p2_status='PENDING', gpu_training_started=False)
        # All output files and code/config identities are covered by completion hashes.
        event('VERIFY_AND_HASH_FOLD_OUTPUTS', fold=fold)
        files = [p for p in fd.rglob('*') if p.is_file() and p != complete]
        files += list((OUT / 'provenance').glob('*'))
        files += list((OUT / 'queue').glob('*split.csv')) + [OUT / 'queue/protocol.json']
        files += [OUT / 'cohort_summary.csv']
        for name in sources + [target]:
            files += [p for p in (OUT / 'cohorts' / name).glob('*') if p.is_file()]
        for name in sources:
            files += [OUT / 'hvg' / f'{name}.csv', OUT / 'hvg' / f'{name}.COMPLETE.json']
        legacy.write(complete, dict(summary=summary, hashes={str(p): legacy.sha(p) for p in files}))
        reports.append(summary)
        pd.DataFrame(reports).to_csv(OUT / 'preprocessing_summary.csv', index=False)
        event('FOLD_COMPLETE', **summary)
        del datasets, rr, b, target_rna, sel, bd, diagnostics, graph, selector
        gc.collect()
    legacy.write(OUT / 'CPU_COMPLETE.json', dict(status='COMPLETE', experiments=list(FOLDS),
                 seeds=legacy.SEEDS, std_ddof=1, GPU_training_started=False,
                 hashes={str(OUT / 'folds' / f / 'COMPLETE.json'):
                         legacy.sha(OUT / 'folds' / f / 'COMPLETE.json') for f in FOLDS}))
    event('COMPLETED', manifest=str(OUT / 'CPU_COMPLETE.json'))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        event('FAILED', error=repr(exc), traceback=traceback.format_exc())
        raise

