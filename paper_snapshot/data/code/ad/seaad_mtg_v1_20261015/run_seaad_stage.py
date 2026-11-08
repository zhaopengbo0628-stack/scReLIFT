"""Reserve one data-only task, download official H5ADs and build paired benchmark."""
import argparse
import csv
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/u202522802045/share/zpb')
REGISTRY = ROOT/'result_contain/task_registry.csv'
RAW = ROOT/'data/raw/AD/SEAAD_MTG/original_rna20260622_atac20241206'
OUTPUT = ROOT/'data/benchmark/AD/SEAAD_MTG/author_qc_paired_v1_20261015'


def registry(change, reserve=False):
    with (ROOT/'result_contain/task_registry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with REGISTRY.open(newline='') as stream:
            reader = csv.DictReader(stream); fields = reader.fieldnames; rows = list(reader)
        if reserve:
            numbers = [int(r['task_id']) for r in rows if r['task_id'].isdigit()]
            for directory in ('result_contain', 'slum'):
                for path in (ROOT/directory).iterdir():
                    match = re.match(r'^(?:task)?(\d+)[_\W]', path.name)
                    if match: numbers.append(int(match[1]))
            task = max(numbers)+1
            name = f'{task}_seaad_mtg_paired_benchmark_20261015'
            result = ROOT/'result_contain'/name
            result.mkdir()
            change.update(task_id=str(task), experiment_name=name, result_path=str(result))
            row = dict.fromkeys(fields, ''); rows.append(row)
        else:
            row = next(r for r in rows if r['task_id'] == change['task_id'])
        row.update({k: v for k, v in change.items() if k in fields})
        row['updated_at'] = datetime.now(timezone.utc).isoformat()
        temporary = REGISTRY.with_suffix('.csv.tmp')
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)
        temporary.replace(REGISTRY)
        return dict(row)


def main(commit):
    if OUTPUT.exists():
        raise FileExistsError('Existing benchmark preserved; inspect it before rerun')
    row = registry(dict(status='RUNNING_DOWNLOAD', model_version='SEAAD_author_QC_no_training',
        datasets='SEAAD_MTG_10xMulti_RNA20260622_ATAC20241206', git_commit=commit,
        scheduler='none', launcher='direct_nohup', launch_method='CPU2_network_only_no_GPU',
        pid=str(os.getpid()), created_at=datetime.now(timezone.utc).isoformat()), reserve=True)
    result = Path(row['result_path']); task = row['task_id']
    (result/'source').mkdir(); (result/'logs').mkdir()
    for file in Path(__file__).parent.glob('*.py'):
        shutil.copy2(file, result/'source'/file.name)
    (result/'README.md').write_text('SEA-AD MTG author-QC paired benchmark construction. No training.\n'
        'Status and exceptions: status.json. Final benchmark is usable only with COMPLETE manifest.\n')
    (RAW/'SERVER_STATUS.json').write_text(json.dumps(dict(task_id=task, result=str(result),
        raw=str(RAW), benchmark=str(OUTPUT), status='RUNNING_DOWNLOAD'), indent=2))
    try:
        # Copied Windows resume records are evidence only; their partial matrices remain local.
        for file in RAW.glob('*.download.json'):
            state = json.loads(file.read_text())
            if state.get('status') != 'COMPLETE' and not (RAW/file.name.removesuffix('.download.json')).exists() and not (RAW/(file.name.removesuffix('.download.json')+'.part')).exists():
                file.rename(file.with_suffix('.json.local_prior'))
        for script, arguments, status in [
            ('download_seaad.py', ['--raw', str(RAW)], 'RUNNING_DOWNLOAD'),
            ('stage_seaad.py', ['--raw', str(RAW), '--output', str(OUTPUT), '--commit', commit], 'RUNNING_QC')]:
            registry(dict(task_id=task, status=status))
            (result/'status.json').write_text(json.dumps(dict(task_id=task,status=status,benchmark=str(OUTPUT))))
            with (result/'logs'/(script+'.out')).open('w') as out, (result/'logs'/(script+'.err')).open('w') as err:
                subprocess.run([sys.executable, '-u', str(result/'source'/script), *arguments],
                    stdout=out, stderr=err, check=True)
        for name in ('QC_REPORT.json', 'manifest.json', 'donors.csv'):
            shutil.copy2(OUTPUT/name, result/name)
        shutil.copy2(RAW/'SOURCE.md', OUTPUT/'SOURCE.md')
        registry(dict(task_id=task, status='COMPLETED', completed_at=datetime.now(timezone.utc).isoformat()))
        status = dict(task_id=task,status='COMPLETE',benchmark=str(OUTPUT),raw=str(RAW),result=str(result))
        (result/'status.json').write_text(json.dumps(status,indent=2))
        (RAW/'SERVER_STATUS.json').write_text(json.dumps(status,indent=2))
        (result/'exitcode').write_text('0\n')
    except Exception as error:
        registry(dict(task_id=task,status='FAILED_HARD_STOP'))
        status = dict(task_id=task,status='FAILED_HARD_STOP',error=repr(error),benchmark=str(OUTPUT))
        (result/'status.json').write_text(json.dumps(status,indent=2))
        (RAW/'SERVER_STATUS.json').write_text(json.dumps(status,indent=2))
        (result/'exitcode').write_text('1\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    main(parser.parse_args().commit)
