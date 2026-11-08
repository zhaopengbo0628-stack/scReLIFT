"""Independent donor-prediction experiment; reuse frozen Full B2 training unchanged."""
import argparse
import csv
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from donor_probe import SEEDS, PCA_DIMS, REGULARIZATION, balanced_split, cell_indices
from donor_probe import identify_metadata, pool, fit_probe, metrics
from ad import sha, write

ROOT = Path('/home/u202522802045/share/zpb')
TASK61 = ROOT/'result_contain/61_smode_full_b2_ad_donor_5seed_20261013'
BENCHMARK = ROOT/'data/benchmark/AD/GSE214979/author_qc_v1_20261012'


def audit():
    manifest = json.loads((BENCHMARK/'manifest.json').read_text())
    for name in ('matrix', 'metadata'):
        path = BENCHMARK/manifest[name]
        if sha(path) != manifest['input_hashes'][path.name]:
            raise ValueError(f'Raw input hash mismatch: {path}')
    with h5py.File(BENCHMARK/manifest['matrix'], 'r') as handle:
        m = handle['matrix']
        barcodes = m['barcodes'][:].astype(str)
        types = m['features/feature_type'][:].astype(str)
        dimensions = m['shape'][:]
        for name, kind in [('rna', 'Gene Expression'), ('atac', 'Peaks')]:
            np.testing.assert_array_equal(np.flatnonzero(types == kind),
                                          np.load(BENCHMARK/f'{name}_feature_index.npy'))
        if int(dimensions[1]) != len(barcodes):
            raise ValueError('H5 dimensions/barcodes differ')
    frame = pd.read_csv(BENCHMARK/manifest['metadata'], dtype=str)
    cells, donors, fields = identify_metadata(frame, barcodes)
    old = pd.read_csv(BENCHMARK/'cells.csv', dtype=str)
    for new, prior in [('barcode', 'barcode'), ('donor', 'donor_id'), ('cell_type', 'author_predicted_id')]:
        np.testing.assert_array_equal(cells[new], old[prior])
    np.testing.assert_array_equal(cells.disease, old.disease_status.map({'Ctrl': 0, 'AD': 1}))
    qc = json.loads((BENCHMARK/'QC_REPORT.json').read_text())
    if len(cells) != 105332 or qc['zero_rna_cells'] or qc['zero_atac_cells']:
        raise ValueError('Paired cohort changed')
    return cells, donors, dict(fields=fields, matrix_shape=dimensions.tolist(),
        cells=len(cells), genes=int(sum(types == 'Gene Expression')), peaks=int(sum(types == 'Peaks')),
        modality_pairing='one shared H5 barcode axis; exact metadata reindex; no barcode trimming',
        duplicate_barcodes=0, missing_metadata_barcodes=0, discarded_cells=0,
        raw_hashes=manifest['input_hashes'], author_qc=qc['author_qc'], full_sparse_scan_reused=qc['full_sparse_scan'],
        qc_report_sha256=sha(BENCHMARK/'QC_REPORT.json'))


def outer_validator(rows, split):
    full = np.sort(np.concatenate(list(split.values())))
    if not np.array_equal(full, np.arange(len(rows))):
        raise ValueError('Missing or overlapping cells')
    fit = {rows[int(i)]['donor_id'] for i in np.r_[split['train'], split['val']]}
    test = {rows[int(i)]['donor_id'] for i in split['test']}
    if fit & test or len(fit) != 11 or len(test) != 4:
        raise ValueError('Outer donor isolation failed')
    for s in split:
        if {rows[int(i)]['disease_status'] for i in split[s]} != {'AD', 'Ctrl'}:
            raise ValueError('Both diagnoses required')


def register(source, commit):
    from register_ad import active_predecessors
    from queue_ad import process_identity
    cells, donors, evidence = audit()
    donors, split_evidence = balanced_split(donors)
    indices = cell_indices(cells, donors)
    prior = json.loads((TASK61/'source/protocol.json').read_text())
    tests = set(donors.loc[donors.split == 'test', 'donor'])
    exposed = tests & set(prior['donors']['train'] + prior['donors']['val'])
    actual_dim = int(np.load(TASK61/'models/AD/seed_20020130/train_student_embedding.npy', mmap_mode='r').shape[1])
    registry = ROOT/'result_contain/task_registry.csv'
    with (ROOT/'result_contain/task_registry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with registry.open(newline='') as handle:
            reader = csv.DictReader(handle); columns, rows = reader.fieldnames, list(reader)
        used = {int(r['task_id']) for r in rows}
        for directory in (ROOT/'slum', ROOT/'result_contain'):
            for p in directory.iterdir():
                match = re.match(r'(?:task)?(\d+)(?:_|\b)', p.name, re.I)
                if match:
                    used.add(int(match[1]))
        task = max(used)+1
        name = f'{task}_smode_ad_donor_prediction_5seed_20261014'
        out = ROOT/'result_contain'/name
        snapshot = out/'source'; snapshot.mkdir(parents=True, exist_ok=False)
        for filename in ('ad_donor_prediction.py', 'donor_probe.py', 'test_donor_probe.py', 'ad.py'):
            shutil.copy2(source/filename, snapshot/filename)
        for filename in ('frozen_train.py', 'register_ad.py', 'queue_ad.py'):
            shutil.copy2(TASK61/'source'/filename, snapshot/filename)
        split = ROOT/f'data/benchmark/AD/GSE214979/donor_prediction_split_v1_task{task}'
        split.mkdir(exist_ok=False)
        for s, ix in indices.items():
            np.save(split/f'{s}_index.npy', ix)
        donors.to_csv(out/'donor_split.csv', index=False)
        donors.drop(columns='split').to_csv(out/'donor_metadata_summary.csv', index=False)
        donors[['donor', 'disease', 'split']].to_csv(split/'donor_split.csv', index=False)
        cells.to_csv(out/'cells.csv', index=False)
        train_donors = set(donors.loc[donors.split == 'train', 'donor'])
        types = sorted(cells.loc[cells.donor.isin(train_donors), 'cell_type'].unique())
        if not set(cells.cell_type).issubset(types):
            raise ValueError('Test-only cell types require an explicit protocol decision')
        config = json.loads((TASK61/'source/config.json').read_text())
        predecessors, historical = active_predecessors(rows)
        protocol = dict(task_id=task, user_request='Task62 AD donor prediction; registry Task62 already occupied',
            method='Full B2 S-mode + frozen donor Linear Probe',
            scope='Single-dataset preliminary experiment; no cross-source stability claim',
            run_tier='pre_final', seeds=SEEDS, latent_dim=actual_dim, cell_types=types,
            donors=donors[['donor', 'disease', 'split']].to_dict('records'),
            outer_train_donors=11, outer_test_donors=4, split_evidence=split_evidence,
            inner_model_validation='10% fixed cells per outer-training donor; same eleven donors remain outer-train; no disease/type-based cell selection',
            split_counts={s: len(ix) for s, ix in indices.items()},
            source_fit='train cells only: HVG, RNA scaler, B_S, TF-IDF, F scaler; original validation loss checkpoint rule',
            test='frozen Student RNA only; no test statistics for input preprocessing',
            task61_checkpoint_reuse=False, task61_exposed_test_donors=sorted(exposed),
            checkpoint_reason='new outer split; prior validation counts as exposure, not just gradient training',
            cell_type_order_source='author predicted.id observed in outer-training donors only',
            missing_type='NaN block; impute per-feature equal-donor mean inside each LODO training fold; no all-missing columns allowed',
            probe=dict(PCA=list(PCA_DIMS), C=list(REGULARIZATION), penalty='L2', threshold=0.5,
                selection='pooled training-donor LODO AUROC; ties smaller PCA then smaller C',
                steps='inner-fold mean imputation -> StandardScaler -> PCA(full) -> LogisticRegression',
                representation_scope='fixed unsupervised representation trained on outer-train donors; inner LODO evaluates the probe only',
                labels='Control=0, AD=1; used only donor stratification and supervised probe'),
            research_diagnostics='WARNING_ONLY', architecture_changed=False, adaptation=False,
            selection_data='metadata covariates only; no Task61/Task62 disease prediction or clustering scores',
            git_commit=commit, data_audit=evidence,
            risks=['15 donors, fixed 4-donor test; seed repetitions are not independent cohorts',
                   'age, region, repository, sex and cell composition may confound diagnosis',
                   'Task61 already evaluated some donors; exploratory, not a new blind disease cohort',
                   'author predicted.id is reference-based annotation, not independently validated ground truth'])
        write(snapshot/'protocol.json', protocol)
        write(out/'data_audit.json', evidence)
        dependencies = {p: h for p, h in config['code_hashes'].items() if not p.startswith(str(TASK61/'source'))}
        dependencies.update({str(p): sha(p) for p in snapshot.glob('*.py')})
        dependencies[str(snapshot/'protocol.json')] = sha(snapshot/'protocol.json')
        dependencies.update({str(p): sha(p) for p in split.iterdir()})
        prefix = ROOT/'slum'/name
        config.update(task_id=task, result=str(out), cache=str(ROOT/f'main/cache/AD/GSE214979/donor_prediction_v1_task{task}'),
            split=str(split), protocol_path=str(snapshot/'protocol.json'), git_commit=commit,
            predecessor_tasks=[int(r['task_id']) for r in predecessors],
            predecessor_processes=[dict(pid=int(r['pid']), start_ticks=process_identity(r['pid']))
                for r in predecessors if r.get('pid') and process_identity(r['pid'])],
            excluded_historical_registry_entries=historical, exitcode=str(prefix)+'.exitcode', code_hashes=dependencies)
        write(snapshot/'config.json', config)
        write(out/'manifest.json', dict(status='REGISTERED_PENDING_VERIFICATION', **protocol))
        (out/'README.md').write_text('# AD donor-level disease prediction\n\nSingle-dataset, 5-seed pre_final. See source/protocol.json. Task61 is read-only.\n')
        now = datetime.now(timezone.utc).isoformat()
        row = dict.fromkeys(columns, '')
        row.update(task_id=str(task), status='REGISTERED_PENDING_VERIFICATION', experiment_name=name,
            model_version='Full-B2-S-AD-donor-LinearProbe', result_path=str(out),
            run_script_path=str(snapshot/'ad_donor_prediction.py'), stdout_path=str(prefix)+'.out',
            stderr_path=str(prefix)+'.err', pid_path=str(prefix)+'.pid', exitcode_path=str(prefix)+'.exitcode',
            launcher='direct_nohup_queue', scheduler='none', launch_method='single_GPU_two_CPU_threads',
            created_at=now, updated_at=now, git_commit=commit, datasets='GSE214979', seeds=','.join(map(str, SEEDS)))
        rows.append(row)
        temporary = registry.with_suffix('.tmp')
        with temporary.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, columns); writer.writeheader(); writer.writerows(rows)
            handle.flush(); os.fsync(handle.fileno())
        temporary.replace(registry)
    print(json.dumps(dict(task_id=task, config=str(snapshot/'config.json'), result=str(out),
        donors=protocol['donors'], dim=actual_dim, checkpoint_exposure=sorted(exposed)), indent=2), flush=True)


def infer(config, runner, unit, cache):
    import torch
    import ad
    from main.core.regulatory.s_mode.single_source import map_rna, finite
    out = Path(config['result'])
    cells = pd.read_csv(out/'cells.csv', dtype={'donor': str})
    donors = pd.read_csv(out/'donor_split.csv', dtype={'donor': str})
    protocol = json.loads(Path(config['protocol_path']).read_text())
    checkpoint = unit/'checkpoints/student/best.pt'
    training = json.loads((unit/'training_manifest.json').read_text())
    if sha(checkpoint) != training['hashes'][str(checkpoint)]:
        raise ValueError('Corrupt Student checkpoint')
    student = runner.b.Encoder(len(np.load(cache/'gene_ids.npy'))).to('cuda:0')
    student.load_state_dict(torch.load(checkpoint, map_location='cuda:0', weights_only=False)['model'])
    student.eval().requires_grad_(False)
    before = sha(checkpoint)
    state = np.load(cache/'rna_state.npz')
    rna_ix = np.load(BENCHMARK/'rna_feature_index.npy')
    raw = BENCHMARK/'GSE214979_filtered_feature_bc_matrix.h5'
    with h5py.File(raw, 'r') as handle:
        ids = handle['matrix/features/id'][:].astype(str)[rna_ix]
    z = np.empty((len(cells), protocol['latent_dim']), np.float32)
    started = time.perf_counter()
    for donor in donors.donor:
        ix = np.flatnonzero(cells.donor.eq(donor))
        counts = ad.read_counts(raw, ix, rna_ix)  # test reads RNA values only
        x, hit = map_rna(counts, ids, np.load(cache/'gene_ids.npy'), state['mean'], state['std'])
        if not hit.all():
            raise ValueError('Selected RNA gene missing')
        values = runner.b.forward_numpy(student, x, torch.device('cuda:0'))
        if values.shape != (len(ix), z.shape[1]):
            raise ValueError('Latent dimension mismatch')
        z[ix] = values
    finite(z, 'RNA-only donor embeddings')
    if before != sha(checkpoint) or any(p.requires_grad for p in student.parameters()):
        raise ValueError('Student changed during inference')
    inference_seconds = time.perf_counter()-started
    np.save(unit/'cell_embeddings.npy', z)
    features, missing = pool(z, cells, donors, protocol['cell_types'])
    pd.DataFrame(missing).to_csv(unit/'cell_type_availability.csv', index=False)
    np.savez_compressed(unit/'donor_embeddings.npz', donor_ids=donors.donor.to_numpy(dtype=str),
                        cell_types=np.asarray(protocol['cell_types']), **features)
    fit = donors.split.eq('train').to_numpy(); test = ~fit
    records, predictions, chosen = [], [], {}
    import joblib
    for scheme, x in features.items():
        probability, selected, cv, model = fit_probe(x[fit], donors.disease.to_numpy()[fit], x[test])
        chosen[scheme] = dict(selected=selected, candidates=cv)
        records.append(dict(seed=int(unit.name.split('_')[-1]), scheme=scheme,
            **metrics(donors.disease.to_numpy()[test], probability), PCA=selected['pca_dim'], C=selected['C']))
        for row, p in zip(donors[test].to_dict('records'), probability):
            predictions.append(dict(seed=records[-1]['seed'], scheme=scheme, donor=row['donor'],
                true_label=int(row['disease']), ad_probability=float(p), prediction=int(p >= 0.5), cells=int(row['cells'])))
        joblib.dump(model, unit/f'{scheme}_probe.joblib')
    write(unit/'probe_cv.json', chosen)
    pd.DataFrame(records).to_csv(unit/'probe_metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(unit/'probe_predictions.csv', index=False)
    write(unit/'donor_prediction_manifest.json', dict(status='COMPLETE', seed=records[0]['seed'],
        student_checkpoint_sha256=before, student_frozen=True, test_atac_used=False, target_fit=False,
        donor_order=donors.donor.tolist(), dimensions={k: list(v.shape) for k, v in features.items()},
        latent_dim=z.shape[1], inference_seconds=inference_seconds,
        scope='all 15 donors RNA-only after freezing; probe fit eleven training donors only',
        hashes={str(p): sha(p) for p in unit.iterdir() if p.is_file() and p.name != 'donor_prediction_manifest.json'}))
    print('DONOR_PROBE_COMPLETE', unit.name, records, flush=True)


def summarize(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    completed = sorted(out.glob('models/AD/seed_*/donor_prediction_manifest.json'))
    if not completed:
        return
    frames, predictions = [], []
    from ad import verify_hashes
    for p in completed:
        verify_hashes(json.loads(p.read_text())['hashes'])
        frames.append(pd.read_csv(p.parent/'probe_metrics.csv'))
        predictions.append(pd.read_csv(p.parent/'probe_predictions.csv', dtype={'donor': str}))
    rows = pd.concat(frames, ignore_index=True)
    rows.to_csv(out/'probe_metrics_per_seed.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out/'probe_predictions.csv', index=False)
    fields = ['AUROC', 'Balanced_Accuracy', 'Macro_F1', 'Sensitivity', 'Specificity']
    summary = rows.groupby('scheme')[fields].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(out/'probe_summary.csv')
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), layout='constrained')
    colors = ['#4878A8', '#3C8D7B', '#C28A42']
    for ax, field in zip(axes, fields[:3]):
        for j, scheme in enumerate(('P1', 'P2', 'P3')):
            values = rows.loc[rows.scheme == scheme, field].to_numpy()
            ax.bar(j, values.mean(), yerr=values.std(ddof=1) if len(values)>1 else 0,
                   width=0.6, color=colors[j], capsize=3)
            ax.scatter(j+np.linspace(-0.12, 0.12, len(values)), values, s=12, color='black', zorder=3)
        ax.set(xticks=[0, 1, 2], xticklabels=['P1', 'P2', 'P3'], ylim=(0, 1.08), title=field.replace('_', ' '))
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(out/'probe_comparison.png', dpi=300); plt.close(fig)
    protocol = json.loads((out/'source/protocol.json').read_text())
    lines = ['# AD donor prediction: exploratory fixed holdout', '',
        f"Completed seeds: {len(completed)}/5. Fixed 11 train / 4 test donors. Single-dataset preliminary experiment.",
        '', 'Student retrained; architecture/loss/checkpoint rule unchanged; frozen RNA-only inference.',
        f"Latent dimension: {protocol['latent_dim']}; types: {protocol['cell_types']}.",
        'Missing type blocks: training-fold mean imputation before scaler/PCA. No disease-based cell selection.',
        'Probe CV is donor LODO with a fixed unsupervised outer-train representation; not nested representation retraining.',
        '', '| Scheme | AUROC | Balanced Accuracy | Macro-F1 | Sensitivity | Specificity |',
        '| --- | --- | --- | --- | --- | --- |']
    for scheme, row in summary.iterrows():
        lines.append('| '+scheme+' | '+' | '.join(f"{row[f+'_mean']:.4f} ± {row[f+'_std']:.4f}" for f in fields)+' |')
    lines += ['', '## Donors', '', donors_markdown(pd.read_csv(out/'donor_split.csv', dtype={'donor': str})),
        '', '## Interpretation limits', '', *protocol['risks'],
        '', 'P2 is predeclared primary. P1/P3 comparison is exploratory; do not select a method using these four test donors.',
        'Seed variance measures repeated training on the same split, not independent patient-cohort generalization.',
        'Reusable: validated metadata/barcode audit, pooling, fold-local imputation/scaling/PCA/probe, metrics and provenance. New datasets require identity/annotation/GTF checks and source-only retraining.',
        '', 'See data_audit.json, source/protocol.json, per-seed manifests and source/Git commit for provenance.']
    (out/'experiment_report.md').write_text('\n'.join(lines)+'\n')


def donors_markdown(donors):
    cols = ['donor', 'disease', 'age', 'sex', 'cells', 'region', 'repository', 'split']
    return '\n'.join(['| '+' | '.join(cols)+' |', '| '+' | '.join(['---']*len(cols))+' |'] +
        ['| '+' | '.join(str(r[c]) for c in cols)+' |' for r in donors.to_dict('records')])


def worker(config):
    import ad
    import torch
    from threadpoolctl import threadpool_limits
    ad.verify_hashes(config['code_hashes'])
    runner = ad.runtime(Path(config['result'])/'source')
    out = Path(config['result'])
    torch.set_num_threads(2)
    with threadpool_limits(2):
        cache, manifest = ad.prepare(config, runner, split_validator=outer_validator)
        runner.train('AD', SEEDS[0], out, cache, manifest, smoke=True, evaluation_targets=())
        write(out/'verification.json', dict(cache_check='PASS', smoke='PASS', outer_donor_disjoint=True,
            model='unchanged Full B2; single-dataset B_S', research_diagnostics='WARNING_ONLY'))
        for seed in SEEDS:
            runner.train('AD', seed, out, cache, manifest, evaluation_targets=())
            unit = out/'models/AD'/f'seed_{seed}'
            if (unit/'donor_prediction_manifest.json').exists():
                ad.verify_hashes(json.loads((unit/'donor_prediction_manifest.json').read_text())['hashes'])
            else:
                infer(config, runner, unit, cache)
            summarize(out)
            if seed == SEEDS[0]:
                write(out/'first_seed_verification.json', dict(seed=seed, status='PASS',
                    donor_predictions=12, schemes=3, test_donors=4, remaining_seeds_authorized=True))
        if len(list(out.glob('models/AD/seed_*/donor_prediction_manifest.json'))) != 5:
            raise ValueError('Missing seed')


def queue(config_path):
    from queue_ad import dependencies_ready, update
    config = json.loads(config_path.read_text())
    try:
        update(config, 'QUEUED_AFTER_EXISTING_TASKS')
        while True:
            with (ROOT/'result_contain/task_registry.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            if dependencies_ready(config, rows):
                break
            time.sleep(30)
        with (ROOT/'.benchmark_a30.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                time.sleep(30)
            update(config, 'RUNNING_PREPROCESS_SMOKE_AND_5SEED')
            env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                       NUMEXPR_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='0')
            subprocess.run([config['python'], '-B', str(config_path.parent/'ad_donor_prediction.py'),
                            '--stage', 'worker', '--config', str(config_path)], cwd=ROOT, env=env, check=True)
            update(config, 'COMPLETED')
            Path(config['exitcode']).write_text('0\n')
    except Exception as error:
        update(config, 'BLOCKED_OR_FAILED', error=repr(error))
        Path(config['exitcode']).write_text('1\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['register', 'queue', 'worker'], required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--commit')
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    if args.stage == 'register':
        if not args.commit or len(args.commit) != 40:
            raise ValueError('Full verified Git commit required')
        register(args.source, args.commit)
    elif args.stage == 'queue':
        queue(args.config)
    else:
        worker(json.loads(args.config.read_text()))
