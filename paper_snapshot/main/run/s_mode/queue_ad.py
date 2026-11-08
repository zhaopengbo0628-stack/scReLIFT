"""Wait behind a frozen predecessor chain; never signal or modify another job."""
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ad import sha, verify_hashes, write


def process_identity(pid):
    try:
        value = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return value[19] if value[0] != 'Z' else None
    except FileNotFoundError:
        return None


def dependencies_ready(config, rows):
    if any(process_identity(p['pid']) == p['start_ticks'] for p in config['predecessor_processes']):
        return False
    for task in config['predecessor_tasks']:
        matches = [r for r in rows if r['task_id'] == str(task)]
        if len(matches) != 1:
            raise RuntimeError(f'Predecessor Task{task} registry missing/ambiguous')
        status = matches[0]['status']
        if status.startswith('COMPLETED'):
            continue
        if status.startswith(('FAILED', 'BLOCKED', 'SKIPPED')):
            raise RuntimeError(f'Predecessor Task{task}: {status}; no automatic bypass')
        pid = matches[0].get('pid')
        if pid and process_identity(pid) is not None:
            return False
        # A queued predecessor must not be bypassed after its parent chain dies.
        raise RuntimeError(f'Predecessor Task{task} is {status} but its chain is absent')
    return True


def update(config, status, **extra):
    root, out = Path(config['root']), Path(config['result'])
    now = datetime.now(timezone.utc).isoformat()
    path = out/'manifest.json'
    value = json.loads(path.read_text())
    value.update(status=status, updated_at=now, pid=os.getpid(), **extra)
    write(path, value)
    registry = root/'result_contain/task_registry.csv'
    with (root/'result_contain/task_registry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with registry.open(newline='') as handle:
            reader = csv.DictReader(handle); columns, rows = reader.fieldnames, list(reader)
        row = next(r for r in rows if r['task_id'] == str(config['task_id']))
        row.update(status=status, updated_at=now, pid=str(os.getpid()))
        if status == 'COMPLETED':
            row['completed_at'] = now
        temporary = registry.with_suffix('.tmp')
        with temporary.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, columns); writer.writeheader(); writer.writerows(rows)
            handle.flush(); os.fsync(handle.fileno())
        temporary.replace(registry)


def main():
    config_path = Path(sys.argv[1]); config = json.loads(config_path.read_text())
    root, out = Path(config['root']), Path(config['result'])
    exit_path = Path(config['exitcode'])
    try:
        verify_hashes(config['code_hashes'])
        update(config, 'QUEUED_AFTER_EXISTING_TASKS', predecessor_tasks=config['predecessor_tasks'])
        print('QUEUED', config['task_id'], 'after', config['predecessor_tasks'], flush=True)
        while True:
            with (root/'result_contain/task_registry.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            if dependencies_ready(config, rows):
                break
            time.sleep(30)
        with (root/'.benchmark_a30.lock').open('a') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(30)
            # Other jobs may not honor the shared lock; wait for GPU users instead of killing them.
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                           text=True).strip():
                time.sleep(30)
            update(config, 'RUNNING_PREPROCESS_SMOKE_AND_5SEED')
            verify_hashes(config['code_hashes'])
            env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                       NUMEXPR_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='0')
            command = [config['python'], '-B', str(config_path.parent/'ad.py'), '--config', str(config_path)]
            subprocess.run(command, env=env, check=True, cwd=root)
            update(config, 'COMPLETED')
            exit_path.write_text('0\n')
    except Exception as error:
        update(config, 'BLOCKED_OR_FAILED', error=repr(error))
        exit_path.write_text('1\n')
        raise


if __name__ == '__main__':
    main()
