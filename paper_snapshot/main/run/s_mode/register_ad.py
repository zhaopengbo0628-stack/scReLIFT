"""Reserve an immutable AD task and donor split under the existing server registry lock."""
import argparse
import csv
import fcntl
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ad import SEEDS, donor_split, sha, write
from queue_ad import process_identity

ROOT = Path('/home/u202522802045/share/zpb')
PYTHON = '/home/u202522802045/envs/graph_attention/bin/python'
BENCHMARK = ROOT/'data/benchmark/AD/GSE214979/author_qc_v1_20261012'


def active_predecessors(rows):
    active, historical = [], []
    for row in rows:
        if 'RUNNING' not in row['status'] and not row['status'].startswith('QUEUED'):
            continue
        pid = row.get('pid')
        exitcode = Path(row['exitcode_path']) if row.get('exitcode_path') else None
        if ('RUNNING' in row['status'] and pid and process_identity(pid) is None
                and exitcode is not None and exitcode.is_file()):
            historical.append(dict(task_id=int(row['task_id']), registry_status=row['status'],
                pid=pid, process_present=False, exitcode=exitcode.read_text().strip(),
                exitcode_path=str(exitcode), exitcode_sha256=sha(exitcode),
                reason='terminated historical process; stale RUNNING registry, not a live queue'))
        else:
            active.append(row)
    return active, historical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--resume-result', type=Path)
    args = parser.parse_args()
    if len(args.commit) != 40:
        raise ValueError('Full source Git commit required')
    with (BENCHMARK/'cells.csv').open() as handle:
        cells = list(csv.DictReader(handle))
    split, donors = donor_split(cells)
    if len(cells) != 105332:
        raise ValueError('Frozen cohort changed')
    registry = ROOT/'result_contain/task_registry.csv'
    with (ROOT/'result_contain/task_registry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with registry.open(newline='') as handle:
            reader = csv.DictReader(handle); columns, rows = reader.fieldnames, list(reader)
        used = {int(r['task_id']) for r in rows}
        for directory in (ROOT/'slum', ROOT/'result_contain'):
            for path in directory.iterdir():
                match = re.match(r'(?:task)?(\d+)(?:_|\b)', path.name, re.I)
                if match:
                    used.add(int(match[1]))
        number = int(args.resume_result.name.split('_')[0]) if args.resume_result else max(used)+1
        name = f'{number}_smode_full_b2_ad_donor_5seed_20261013'
        out = ROOT/'result_contain'/name
        if args.resume_result:
            matches = [r for r in rows if r['task_id']==str(number)]
            if args.resume_result.resolve() != out.resolve() or any(
                    r['status']!='REGISTERED_PENDING_VERIFICATION' or r['result_path']!=str(out) for r in matches):
                raise ValueError('Only an inactive own reservation can be repaired')
            if json.loads((out/'manifest.json').read_text())['status'] != 'REGISTERED_PENDING_VERIFICATION':
                raise ValueError('Cannot repair an active or completed experiment')
            rows = [r for r in rows if r['task_id'] != str(number)]
        source = out/'source'
        source.mkdir(parents=True, exist_ok=bool(args.resume_result))
        for filename in ('ad.py', 'queue_ad.py', 'register_ad.py', 'test_ad_smode.py'):
            shutil.copy2(args.source/filename, source/filename)
        shutil.copy2(args.source/'train.py', source/'frozen_train.py')
        shutil.copy2(ROOT/'main/run/s_mode/tests/test_smode_full_b2.py', source/'test_smode_full_b2.py')
        split_root = ROOT/f'data/benchmark/AD/GSE214979/donor_split_v1_20261013_task{number}'
        split_root.mkdir(exist_ok=bool(args.resume_result))
        for s, ix in split.items():
            p = split_root/f'{s}_index.npy'
            if p.exists():
                np.testing.assert_array_equal(np.load(p), ix)
            else:
                np.save(p, ix)
        protocol = dict(task_id=number, method='Full B2 S-mode', run_tier='pre_final', seeds=SEEDS,
            benchmark=str(BENCHMARK), benchmark_manifest_sha256=sha(BENCHMARK/'manifest.json'),
            cells=105332, donor_counts={s: len(ds) for s, ds in donors.items()}, donors=donors,
            split_counts={s: len(ix) for s, ix in split.items()}, split_seed=SEEDS[0],
            split_rule='disease-stratified donor permutation; 1 donor/state test, 1 donor/state val, remainder train (11/2/2)',
            split_limitation='single held-out donor per disease; cell-level scores are not donor-level clinical accuracy',
            disease_counts={s: {d: sum(cells[int(i)]['disease_status']==d for i in ix) for d in ('AD','Ctrl')}
                            for s, ix in split.items()},
            source_train_fits=['HVG', 'RNA scaler', 'B_S', 'ATAC IDF', 'F scaler', 'Teacher', 'Student'],
            validation='held-out donors; original label-free stage loss checkpoint selection',
            test='held-out donors; frozen Student RNA-only', label_use='donor split stratification and posthoc evaluation only',
            primary='AD/Ctrl KMeans K=2', secondary='unaltered author predicted.id; known K',
            clustering='unchanged Task35 evaluator; test-embedding StandardScaler remains posthoc evaluation only',
            metric_mapping='ACC/F1/Balanced ACC are posthoc Hungarian-matched KMeans, no classifier',
            confounding='AD/Ctrl age, tissue region and repository differ; no batch correction or causal claim',
            qc='reuse Task51 author QC, no new filter/cohort change', reference='GRCh38; existing gencode v32 GTF',
            research_quality_diagnostics='Warning only; never discard a seed', adaptation=False,
            no_test_atac=True, git_commit=args.commit,
            split_hashes={str(p):sha(p) for p in split_root.glob('*_index.npy')},
            resources='serial single NVIDIA A30; 2 CPU threads; defer preprocessing and smoke behind all predecessors')
        protocol_path = source/'protocol.json'
        write(protocol_path, protocol)
        write(split_root/'manifest.json', protocol)
        predecessors, historical = active_predecessors(rows)
        processes = []
        for r in predecessors:
            pid = r.get('pid')
            ticks = process_identity(pid) if pid else None
            if ticks:
                processes.append(dict(pid=int(pid), start_ticks=ticks))
        prefix = ROOT/'slum'/name
        dependencies = [ROOT/'main/core/run_b2_strict_lodo.py',
            ROOT/'main/core/regulatory/s_mode/single_source.py', ROOT/'data/code/s_mode/prepare.py',
            ROOT/'main/run/prepare_updated_b2_sources.py', ROOT/'main/run/unblock_updated_full_b2.py',
            ROOT/'dmode_dev/v5d/dmode_reliable_selector.py', ROOT/'dmode_dev/v5d/dmode_teacher.py',
            ROOT/'dmode_dev/v5d/regulatory_graph.py', ROOT/'dmode_dev/v5d/data.py',
            ROOT/'dmode_dev/v5d/harmonize.py', ROOT/'dmode_dev/v5d/dataset_spec.py',
            ROOT/'V5/v5_final_20260906/v5s/data.py']
        config = dict(root=str(ROOT), task_id=number, result=str(out), benchmark=str(BENCHMARK),
            cache=str(ROOT/f'main/cache/AD/GSE214979/smode_donor_v1_task{number}'), split=str(split_root),
            protocol_path=str(protocol_path), python=PYTHON, seeds=SEEDS, git_commit=args.commit,
            gtf='/home/u202522802045/glue_156478/graph_files/gencode.v32.primary_assembly.annotation.gtf.gz',
            predecessor_tasks=[int(r['task_id']) for r in predecessors], predecessor_processes=processes,
            excluded_historical_registry_entries=historical,
            exitcode=str(prefix)+'.exitcode',
            code_hashes={str(p):sha(p) for p in dependencies+list(source.glob('*.py'))})
        write(source/'config.json', config)
        write(out/'manifest.json', dict(status='REGISTERED_PENDING_VERIFICATION', **protocol,
            source_hashes=config['code_hashes'], config_sha256=sha(source/'config.json')))
        text = '# Full B2 S-mode AD donor-held-out experiment\n\n'
        text += 'Five fixed seeds, pre_final. Primary disease clustering; secondary original author cell types.\n\n'
        text += 'No annotation harmonization, disease classifier, model change or test ATAC. Quality diagnostics warn only.\n\n'
        text += f'Predecessor tasks: {config["predecessor_tasks"]}; stale exitcodes do not release the queue.\n\n'
        text += json.dumps(protocol, indent=2)+'\n'
        (out/'README.md').write_text(text)
        now = datetime.now(timezone.utc).isoformat()
        row = dict.fromkeys(columns, '')
        row.update(task_id=str(number), status='REGISTERED_PENDING_VERIFICATION', experiment_name=name,
            model_version='Full-B2-S-AD-donor-heldout', result_path=str(out),
            run_script_path=str(source/'queue_ad.py'), stdout_path=str(prefix)+'.out', stderr_path=str(prefix)+'.err',
            pid_path=str(prefix)+'.pid', exitcode_path=str(prefix)+'.exitcode', launcher='direct_nohup_queue',
            scheduler='none', launch_method='after_existing_tasks_single_GPU_two_CPU_threads',
            created_at=now, updated_at=now, git_commit=args.commit, datasets='GSE214979', seeds=','.join(map(str, SEEDS)))
        rows.append(row)
        temporary = registry.with_suffix('.tmp')
        with temporary.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, columns); writer.writeheader(); writer.writerows(rows)
            handle.flush(); os.fsync(handle.fileno())
        temporary.replace(registry)
    print(json.dumps(dict(task_id=number, result=str(out), config=str(source/'config.json'), prefix=str(prefix),
                         donor_counts=protocol['donor_counts'], split_counts=protocol['split_counts'],
                         disease_counts=protocol['disease_counts'], predecessors=config['predecessor_tasks'])))


if __name__ == '__main__':
    main()
