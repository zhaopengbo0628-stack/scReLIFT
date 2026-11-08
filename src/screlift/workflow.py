"""Portable orchestration around the frozen scReLIFT computations."""
from __future__ import annotations

import copy
import json
import platform
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
import torch
from threadpoolctl import threadpool_limits

from . import _frozen_core as core
from .preprocessing import (normalize_counts, canonical_genes, select_hvg,
                            build_bs, build_field, map_rna, finite)
from ._regulatory.regulatory_graph import build_candidate_graph, read_tss
from ._regulatory.harmonize import prepare_harmonized
from ._regulatory.dataset_spec import DatasetSpec
from ._regulatory.data import V5DData
from ._regulatory.dmode_reliable_selector import build_source_only_b
from ._regulatory.dmode_teacher import fit_dmode_field

SEEDS = (20020130, 20020131, 20020132)
DEFAULT_CONFIG = copy.deepcopy(core.CONFIG)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def resolve(base, name):
    path = Path(name).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()

def new_directory(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Output already exists; use a new directory: {path}')
    path.mkdir(parents=True)
    return path

def hashes(paths):
    return {str(Path(p).resolve()): core.sha(p) for p in paths if Path(p).is_file()}

def validate_splits(path, rows):
    with np.load(path, allow_pickle=False) as saved:
        result = {}
        for key in ('train', 'val', 'test'):
            values = saved[key + '_index']
            if values.ndim != 1 or values.dtype.kind not in 'iu':
                raise ValueError(f'{key} split must be a one-dimensional integer array')
            result[key] = values.astype(np.int64)
    joined = np.concatenate(list(result.values()))
    if not len(result['train']) >= 2 or not len(result['val']):
        raise ValueError('At least two training rows and one validation row are required')
    if not np.array_equal(np.sort(joined), np.arange(rows)):
        raise ValueError('Frozen splits must be disjoint and cover every row exactly once')
    return result

def prepare_s(config_path, output):
    """Fit S-mode preprocessing using source train only; preserve supplied splits."""
    config_path = Path(config_path).resolve()
    cfg, base = read(config_path), config_path.parent
    names = ('rna_counts', 'gene_ids', 'atac_counts', 'peak_names', 'splits', 'gtf')
    paths = {k: resolve(base, cfg[k]) for k in names}
    counts = sparse.load_npz(paths['rna_counts']).tocsr()
    genes = np.load(paths['gene_ids'], allow_pickle=False).astype(str)
    finite(counts, 'raw RNA')
    if (counts.data < 0).any() or (counts.data != np.floor(counts.data)).any():
        raise ValueError('RNA must contain nonnegative integer counts')
    index = validate_splits(paths['splits'], counts.shape[0])
    counts, genes, inverse = canonical_genes(counts, genes)
    normalized = normalize_counts(counts[index['train']])
    hvg_count = int(cfg.get('hvg_count', 2000))
    if not cfg.get('smoke_only', False) and hvg_count != 2000:
        raise ValueError('Paper S-mode requires 2000 HVGs; alternate counts are smoke-only')
    hvg, scores = select_hvg(normalized, hvg_count)
    rna_train = normalized[:, hvg].toarray()
    rna_val = normalize_counts(counts[index['val']])[:, hvg].toarray()
    paired_rows = np.concatenate([index['train'], index['val']])
    # Sparse transport contains the paired source. Test ATAC rows are never used.
    atac_all = sparse.load_npz(paths['atac_counts']).tocsr()
    peaks = np.load(paths['peak_names'], allow_pickle=False).astype(str)
    if atac_all.shape != (counts.shape[0], len(peaks)) or len(set(peaks)) != len(peaks):
        raise ValueError('ATAC dimensions or peak identifiers differ')
    atac = atac_all[paired_rows].copy()
    del atac_all
    finite(atac, 'source train/val ATAC')
    if (atac.data < 0).any() or (atac.data != np.floor(atac.data)).any():
        raise ValueError('ATAC must contain nonnegative integer counts')
    atac = atac.astype(bool).astype(np.float32)
    selected = genes[hvg]
    tss = read_tss(paths['gtf'])
    mapped = np.flatnonzero([g.split('.')[0] in tss for g in selected])
    graph = dict(candidate_index=np.zeros((hvg_count, 50), np.int64),
                 candidate_mask=np.zeros((hvg_count, 50), bool),
                 distance_bp=np.zeros((hvg_count, 50), np.float32))
    small, _ = build_candidate_graph(selected[mapped], peaks, tss, 250000, 50)
    for key in graph:
        graph[key][mapped] = small[key]
    graph['peak_names'] = peaks
    matrix, diagnostics = build_bs(rna_train, atac[:len(index['train'])], graph)
    if matrix.nnz == 0:
        raise ValueError('No effective regulatory evidence; inspect inputs rather than fabricate edges')
    field, state, report = build_field(atac, matrix, np.arange(len(index['train'])))
    out = new_directory(output)
    np.save(out / 'gene_ids.npy', selected)
    np.save(out / 'peak_names.npy', peaks)
    np.save(out / 'hvg_scores.npy', scores)
    np.savez_compressed(out / 'rna_state.npz', mean=rna_train.mean(0), std=rna_train.std(0),
                        hvg_index=hvg, raw_gene_inverse=inverse)
    np.savez_compressed(out / 'field_state.npz', **state)
    np.savez_compressed(out / 'candidate_graph.npz', **graph)
    np.savez_compressed(out / 'regulatory_stability.npz', **diagnostics)
    sparse.save_npz(out / 'B_S.npz', matrix)
    for split, values in [('train', rna_train), ('val', rna_val)]:
        np.save(out / f'rna_{split}.npy', values.astype(np.float32))
        np.save(out / f'{split}_index.npy', index[split])
    np.save(out / 'test_index.npy', index['test'])
    np.save(out / 'F_train.npy', field[:len(index['train'])])
    np.save(out / 'F_val.npy', field[len(index['train']):])
    write(out / 'manifest.json', dict(mode='S', source=cfg['source'],
          smoke_only=bool(cfg.get('smoke_only', False)), genes=len(selected),
          source_train_only_fitting=True, labels_used=False, test_atac_used=False,
          field_report=report, hashes=hashes(list(paths.values()) + list(out.iterdir()) + [config_path])))
    return out

def prepare_d(config_path, output):
    """Re-use paper harmonization/selector on three source-only prepared directories."""
    config_path = Path(config_path).resolve()
    cfg, base = read(config_path), config_path.parent
    sources = cfg['sources']
    names = [s['name'] for s in sources]
    if len(names) != 3 or len(set(names)) != 3 or cfg['target'] in names:
        raise ValueError('This paper adapter requires three distinct sources excluding target')
    specs = [DatasetSpec(s['name'], resolve(base, s['prepared_dir'])) for s in sources]
    dependencies = [config_path, resolve(base, cfg['gtf'])]
    for spec in specs:
        genes = np.load(spec.data_dir / 'gene_ids.npy', allow_pickle=False)
        rna = np.load(spec.data_dir / 'rna.npy', mmap_mode='r', allow_pickle=False)
        if rna.ndim != 2 or rna.shape[1] != len(genes) or len(set(genes)) != len(genes):
            raise ValueError(f'{spec.name}: invalid SOURCE prepared RNA/gene dictionary')
        if not np.isfinite(rna).all() or (rna < 0).any():
            raise ValueError('Prepared RNA must be nonnegative CP10K-log1p, before standardization')
        validate_splits(spec.data_dir / 'metadata.npz', len(rna))
        atac = sparse.load_npz(spec.data_dir / 'atac_binary.npz')
        if atac.shape[0] != len(rna) or not np.isin(atac.data, [0, 1]).all():
            raise ValueError('Prepared ATAC must be binary with paired SOURCE rows')
        dependencies.extend(p for p in spec.data_dir.iterdir() if p.is_file())
    out = new_directory(output)
    prepare_harmonized(specs, out / 'harmonized', feature_reference_names=names,
                       held_out_target_name=cfg['target'])
    genes = np.load(out / 'harmonized/common_gene_ids.npy')
    peaks = np.load(out / 'harmonized/canonical_peak_names.npy')
    graph, _ = build_candidate_graph(genes, peaks, read_tss(resolve(base, cfg['gtf'])), 250000, 50)
    graph['peak_names'] = peaks
    data = [V5DData(out / 'harmonized', name) for name in names]
    selection = build_source_only_b(data, graph, genes, seed=20020130)
    if selection['B_D'].nnz == 0:
        raise ValueError('No effective shared regulatory evidence')
    sparse.save_npz(out / 'B_D.npz', selection['B_D'])
    np.save(out / 'gene_ids.npy', genes)
    np.savez_compressed(out / 'candidate_graph.npz', **graph)
    for key, value in selection.items():
        if isinstance(value, np.ndarray):
            np.save(out / f'{key}.npy', value)
        elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
            np.savez_compressed(out / f'{key}.npz', **dict(zip(names, value)))
    for source in data:
        field, state, report = fit_dmode_field(source.atac, selection['B_D'], source.train)
        np.save(out / f'{source.name}_F_train.npy', field[source.train])
        np.save(out / f'{source.name}_F_val.npy', field[source.val])
        np.savez_compressed(out / f'{source.name}_field_state.npz', **state)
        write(out / f'{source.name}_field_report.json', report)
    write(out / 'manifest.json', dict(mode='D', sources=names, target=cfg['target'],
          smoke_only=bool(cfg.get('smoke_only', False)), source_train_only_fitting=True,
          target_used=False, labels_used=False, selection=selection['manifest'],
          hashes=hashes(dependencies + [p for p in out.rglob('*') if p.is_file()])))
    return out

def load_cache(cache):
    cache = Path(cache).resolve()
    manifest = read(cache / 'manifest.json')
    # Fail closed on input changes; do not silently regenerate frozen evidence.
    for name, expected in manifest['hashes'].items():
        if core.sha(name) != expected:
            raise ValueError(f'Cache/input hash mismatch: {name}')
    if manifest['mode'] == 'S':
        with np.load(cache / 'rna_state.npz') as saved:
            mean, std = saved['mean'], saved['std']
        src = dict(name=manifest['source'])
        for split in ('train', 'val'):
            raw = np.load(cache / f'rna_{split}.npy')
            src[split] = dict(rna=((raw-mean)/(std+1e-6)).astype(np.float32),
                              raw_nonzero=raw != 0, field=np.load(cache / f'F_{split}.npy'))
        sources = [src]
    else:
        sources = []
        for name in manifest['sources']:
            source = V5DData(cache / 'harmonized', name)
            sources.append(dict(name=name, raw=source.rna, train_index=source.train, val_index=source.val))
        blocks = [s['raw'][s['train_index']] for s in sources]
        means = np.stack([x.mean(0) for x in blocks])
        mean = means.mean(0).astype(np.float32)
        std = np.sqrt(np.mean(np.stack([x.var(0) for x in blocks])+(means-mean)**2, axis=0)).astype(np.float32)
        for s in sources:
            for split in ('train', 'val'):
                raw = s['raw'][s[split + '_index']]
                s[split] = dict(rna=((raw-mean)/(std+1e-6)).astype(np.float32),
                               raw_nonzero=raw != 0, field=np.load(cache / f"{s['name']}_F_{split}.npy"))
    return sources, mean, std, manifest

def train(cache, output, seed, smoke=False):
    if seed not in SEEDS:
        raise ValueError(f'Active seeds are exactly {SEEDS}')
    if not torch.cuda.is_available():
        raise RuntimeError('Frozen training uses CUDA timing/RNG. Use a CUDA GPU; CPU inference is supported.')
    cache = Path(cache).resolve()
    sources, mean, std, manifest = load_cache(cache)
    if manifest.get('smoke_only') and not smoke:
        raise ValueError('A smoke-only cache cannot be used for a formal run')
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = hashes([cache / 'manifest.json'] + list(Path(__file__).parent.rglob('*.py')))
    inputs.update(manifest['hashes'])
    core.CONFIG = copy.deepcopy(DEFAULT_CONFIG)
    core.CONFIG.update(balanced_batch_counts=[256] if len(sources) == 1 else [86, 85, 85],
                       run_tier='smoke' if smoke else 'release_adapter',
                       validation='single Source validation' if len(sources) == 1 else 'equal dataset means')
    if smoke:
        core.CONFIG.update(epochs=2, patience=2)
    cfg = dict(seed=seed, mode=manifest['mode'], configuration=core.CONFIG, input_hashes=inputs)
    if (out / 'run.json').exists() and read(out / 'run.json') != cfg:
        raise ValueError('Existing run has different inputs/configuration; use a new directory')
    write(out / 'run.json', cfg)
    device = torch.device('cuda:0')
    core.seed_all(seed)
    teacher = core.Teacher(len(mean), len(mean)).to(device)
    teacher = core.train_stage(teacher, sources, out / 'teacher', seed, device, True, inputs)
    teacher.requires_grad_(False)
    for s in sources:
        for split in ('train', 'val'):
            s[split]['teacher_latent'] = core.forward_numpy(teacher, s[split]['rna'], device, s[split]['field'])[2]
    if len(sources) == 1:
        latent = sources[0]['train']['teacher_latent']
        mu, sigma = latent.mean(0), latent.std(0)
    else:
        mu = np.mean([s['train']['teacher_latent'].mean(0) for s in sources], axis=0)
        variance = np.mean([s['train']['teacher_latent'].var(0)+(s['train']['teacher_latent'].mean(0)-mu)**2 for s in sources], axis=0)
        sigma = np.sqrt(variance).astype(np.float32)
    for s in sources:
        s.update(latent_mean=mu, latent_std=sigma, raw_zero_standardized=-mean/(std+1e-6))
    inputs = dict(inputs, **hashes([out / 'teacher/best.pt']))
    core.seed_all(seed)
    student = copy.deepcopy(teacher.rna_encoder).requires_grad_(True).to(device)
    student = core.train_stage(student, sources, out / 'student', seed, device, False, inputs)
    np.savez_compressed(out / 'preprocessing.npz', gene_ids=np.load(cache / 'gene_ids.npy'),
                        mean=mean, std=std, latent_mean=mu, latent_std=sigma)
    # This bundle is a weights-only dictionary plus separate NumPy preprocessing state.
    temporary = out / 'student_weights.tmp'
    torch.save({k: v.detach().cpu() for k, v in student.state_dict().items()}, temporary)
    temporary.replace(out / 'student_weights.pt')
    write(out / 'model_manifest.json', dict(model='scReLIFT', mode=manifest['mode'], seed=seed,
          status='SMOKE_COMPLETE' if smoke else 'TRAINING_COMPLETE',
          target_used=False, labels_used=False,
          preprocessing_sha256=core.sha(out / 'preprocessing.npz'),
          weights_sha256=core.sha(out / 'student_weights.pt'),
          environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                           platform=platform.platform(), cuda=torch.version.cuda)))
    return out

def infer(bundle, rna_path, genes_path, output, cell_ids=None, device='cpu'):
    bundle = Path(bundle).resolve()
    record = read(bundle / 'model_manifest.json')
    for name, key in [('preprocessing.npz', 'preprocessing_sha256'), ('student_weights.pt', 'weights_sha256')]:
        if core.sha(bundle / name) != record[key]:
            raise ValueError('Inference bundle hash mismatch: ' + name)
    with np.load(bundle / 'preprocessing.npz', allow_pickle=False) as saved:
        wanted, mean, std = saved['gene_ids'], saved['mean'], saved['std']
    counts = sparse.load_npz(rna_path).tocsr()
    finite(counts, 'raw target RNA')
    if (counts.data < 0).any() or (counts.data != np.floor(counts.data)).any():
        raise ValueError('TARGET RNA must contain nonnegative integer counts before gene merging')
    if counts.shape[0] == 0:
        raise ValueError('RNA input has no cells')
    ids = np.load(genes_path, allow_pickle=False).astype(str)
    values, hit = map_rna(counts, ids, wanted, mean, std)
    if not hit.any():
        raise ValueError('No stable gene IDs match the trained model')
    if cell_ids:
        cells = np.load(cell_ids, allow_pickle=False).astype(str)
        if cells.shape != (len(values),) or len(set(cells)) != len(cells):
            raise ValueError('Cell IDs must match RNA rows and be unique')
    else:
        cells = np.asarray([f'row_{i}' for i in range(len(values))])
    model = core.Encoder(len(mean)).to(device)
    model.load_state_dict(torch.load(bundle / 'student_weights.pt', map_location=device, weights_only=True))
    embedding = core.forward_numpy(model, values, torch.device(device))
    finite(embedding, 'RNA-only Student embedding')
    out = new_directory(output)
    np.save(out / 'embedding.npy', embedding)
    np.save(out / 'cell_ids.npy', cells)
    write(out / 'manifest.json', dict(model='scReLIFT', mode=record['mode'], seed=record['seed'],
          training_status=record['status'], cells=len(values), dimensions=64,
          gene_hit_count=int(hit.sum()), gene_missing_ids=wanted[~hit].tolist(),
          target_atac_used=False, target_fit=False, target_adaptation=False,
          embedding_sha256=core.sha(out / 'embedding.npy'), cell_ids_sha256=core.sha(out / 'cell_ids.npy'),
          input_hashes=hashes([rna_path, genes_path, bundle / 'model_manifest.json'] + ([cell_ids] if cell_ids else []))))
    return out

def evaluate(inference, labels_path, label_cells_path, output, fold):
    inference = Path(inference)
    manifest = read(inference / 'manifest.json')
    if core.sha(inference / 'embedding.npy') != manifest['embedding_sha256']:
        raise ValueError('Embedding changed after it was frozen')
    if core.sha(inference / 'cell_ids.npy') != manifest['cell_ids_sha256']:
        raise ValueError('Cell order changed after inference')
    cells = np.load(inference / 'cell_ids.npy', allow_pickle=False).astype(str)
    label_cells = np.load(label_cells_path, allow_pickle=False).astype(str)
    if not np.array_equal(cells, label_cells):
        raise ValueError('Evaluation label cell order does not match frozen inference rows')
    labels = np.load(labels_path, allow_pickle=False).astype(str)
    embedding = np.load(inference / 'embedding.npy')
    if labels.shape != (len(embedding),) or len(np.unique(labels)) < 2:
        raise ValueError('Labels must be a one-dimensional, aligned array with at least two classes')
    _, y = np.unique(labels, return_inverse=True)
    with threadpool_limits(limits=1):
        metrics = core.evaluate(embedding, y)
    result = dict(model='scReLIFT', mode=manifest['mode'], fold=fold, seed=manifest['seed'],
                  training_status=manifest['training_status'], metrics=metrics,
                  embedding_sha256=manifest['embedding_sha256'],
                  labels_sha256=core.sha(labels_path), label_cells_sha256=core.sha(label_cells_path))
    write(output, result)
    return result

def summarize(paths, output):
    rows = [read(p) for p in paths]
    seen = set()
    for row in rows:
        key = (row['model'], row['mode'], row['fold'], row['seed'])
        if key in seen or row['seed'] not in SEEDS:
            raise ValueError('Duplicate experiment or unexpected seed: ' + str(key))
        if row.get('training_status') != 'TRAINING_COMPLETE':
            raise ValueError('Only formal trained models can enter paper summaries')
        seen.add(key)
    if len({(r['model'], r['mode']) for r in rows}) != 1:
        raise ValueError('Summarize one model/protocol at a time')
    folds = sorted({r['fold'] for r in rows})
    if any({r['seed'] for r in rows if r['fold'] == fold} != set(SEEDS) for fold in folds):
        raise ValueError('Each included fold must have all three active seeds')
    numeric = [k for k, v in rows[0]['metrics'].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    summary = {}
    for fold in folds + ['equal_fold_macro']:
        per_seed = []
        for seed in SEEDS:
            values = [r for r in rows if r['seed'] == seed and (fold == 'equal_fold_macro' or r['fold'] == fold)]
            per_seed.append(dict(seed=seed, **{m: float(np.mean([r['metrics'][m] for r in values])) for m in numeric}))
        summary[fold] = dict(per_seed=per_seed, mean={m: float(np.mean([r[m] for r in per_seed])) for m in numeric},
                            sample_sd={m: float(np.std([r[m] for r in per_seed], ddof=1)) for m in numeric})
    result = dict(model=rows[0]['model'], mode=rows[0]['mode'], seeds=list(SEEDS), ddof=1,
                  included_folds=folds, folds_and_macro=summary, input_hashes=hashes(paths))
    write(output, result)
    return result
