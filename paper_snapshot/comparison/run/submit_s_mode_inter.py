"""Register and launch the PBMC five-seed inter continuation exactly once."""
import csv
import fcntl
import os
import shutil
import subprocess
from datetime import datetime,timezone
from pathlib import Path

from pbmc_intra_common import ROOT,SEEDS,read,write,sha
from s_mode_inter import KEYS,NAMES,BASE,PROTOCOL,INPUTS


def main():
    queue=ROOT/'comparison/docs/SMODE_INTER_5SEED_QUEUE_20261014'
    queue.mkdir(exist_ok=False)
    methods=['multigrate','cobolt','palette','eclare','scmomer','b2']
    envs={'multigrate':'multivi_scvi','cobolt':'graph_attention','palette':'graph_attention',
          'eclare':'scmomer_eclare_sintra','scmomer':'scmomer_eclare_sintra','b2':'scmomer_eclare_sintra'}
    tasks=[]
    registry=ROOT/'result_contain/task_registry.csv'
    with (ROOT/'result_contain/task_registry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with registry.open() as handle:
            reader=csv.DictReader(handle); fields=reader.fieldnames; rows=list(reader)
        first=max(int(r['task_id']) for r in rows)+1
        now=datetime.now(timezone.utc).isoformat()
        for offset,method in enumerate(methods):
            number=first+offset
            name=f'{number}_smode_{method}_pbmc_inter_5seed_20261014'
            out=ROOT/'result_contain'/name
            out.mkdir(exist_ok=False)
            for folder in ['checkpoints','evaluations','metrics','logs','source']: (out/folder).mkdir()
            old=next((ROOT/'result_contain').glob(f'{62 if method=="b2" else 52+offset}_*'))
            for source in KEYS:
                for seed in SEEDS if method=='b2' else SEEDS[:3]:
                    previous=old/'checkpoints'/source/f'seed_{seed}'
                    unit=out/'checkpoints'/source/f'seed_{seed}'; unit.mkdir(parents=True)
                    # Existing files stay immutable. Hard links avoid duplicating model storage.
                    for path in previous.iterdir():
                        if path.is_file(): os.link(path,unit/path.name)
                    if method=='b2':
                        shutil.copy2(BASE/source/'gene_ids.npy',unit/'gene_ids.npy')
                        write(unit/'manifest.json',dict(status='COMPLETE',reused_from=str(previous)))
                    write(unit/'checkpoint_reuse.json',dict(source_unit=str(previous),
                        hashes={p.name:sha(p) for p in previous.iterdir() if p.is_file()}))
            if method=='b2':
                shutil.copy2(old/'metrics/regulatory_graphs_inherited.csv',out/'metrics/regulatory_graphs_inherited.csv')
                for path in (old/'evaluations').glob('D*/seed_*/D*/manifest.json'):
                    record=read(path)
                    if record['family']!='inter' or record['inference_seconds'] is None: continue
                    destination=out/'evaluations'/path.parent.relative_to(old/'evaluations')
                    shutil.copytree(path.parent,destination)
            files=[ROOT/f'comparison/run/{f}' for f in ['s_mode_inter.py','queue_s_mode_inter.py',
                   'submit_s_mode_inter.py','pbmc_intra_common.py']]+[PROTOCOL]
            if method!='b2':
                files += [ROOT/'comparison/run/s_mode_intra.py',ROOT/'comparison/run/neural_s_mode_intra.py',
                          ROOT/f'comparison/configs/{method}/s_mode_pbmc_intra.json',
                          ROOT/f'comparison/run/{method}/s_mode_intra.py']
                if method in ['scmomer','eclare']: files.append(ROOT/f'comparison/run/{method}/s_mode_adapter.py')
                config=read(ROOT/f'comparison/configs/{method}/s_mode_pbmc_intra.json')
                historical=read(old/'config.json')
                assert all(historical.get(k)==v for k,v in config.items() if k not in ['seeds','seed_policy'])
                write(out/'source/frozen_config.json',config)
            else: files.append(ROOT/'main/core/run_b2_strict_lodo.py')
            for path in files:
                dest=out/'source'/path.relative_to(ROOT); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,dest)
            task=dict(task_id=number,method=method,method_name=NAMES[method],result_path=str(out),
                python=f'/home/u202522802045/envs/{envs[method]}/bin/python',device='cpu' if method=='palette' else 'cuda',
                datasets=KEYS,seeds=SEEDS,source_task=str(old),source_sha256={str(p):sha(p) for p in files})
            tasks.append(task)
            write(out/'manifest.json',dict(**task,status='QUEUED',run_tier='pre_final',expected_evaluations=60,
                protocol_id=read(PROTOCOL)['protocol_id'],protocol_sha256=sha(PROTOCOL),protocol='inductive frozen Source; Target RNA-only; no query fitting',
                existing_checkpoint_seeds=SEEDS if method=='b2' else SEEDS[:3],new_training_seeds=[] if method=='b2' else SEEDS[3:],
                target_fit=False,target_atac_used=False,labels_used_for_fitting=False,total_time_seconds=None,
                processed_input_manifests={f'{s}_to_{t}':sha(INPUTS/s/t/'manifest.json') for s in KEYS for t in KEYS if s!=t},created_at=now))
            (out/'README.md').write_text(f'# {NAMES[method]} PBMC inter\n\n5-seed pre_final; 12 directed Source-to-Target pairs.\n'
                f'Reuse: {old}\n\nFixed scoring/timing: {read(PROTOCOL)["protocol_id"]}. Target RNA only; no Target fitting.\n'
                'Known-K KMeans, harmonized posthoc labels, shared Hungarian mapping. Total time NA for checkpoint reuse.\n'
                'Source training peak and Target inference peak retained separately. scMomer native batch1; others256.\n'
                'Full D1 processed log cache verified against current barcodes and RNA panel; integer count reconstruction audited.\n'
                'Means require all 12 pairs for every one of five seeds; sample SD retained. Final15 not claimed.\n')
            row=dict.fromkeys(fields,'')
            row.update(task_id=str(number),status='QUEUED',experiment_name=name,model_version=NAMES[method],result_path=str(out),
                run_script_path=str(ROOT/'comparison/run/queue_s_mode_inter.py'),stdout_path=str(queue/'queue.log'),stderr_path=str(queue/'queue.log'),
                pid_path=str(queue/'queue.pid'),exitcode_path=str(out/'logs/exitcode'),launcher='direct_detached_queue',scheduler='none',
                launch_method='serial_single_GPU_two_CPU_threads_checkpoint_reuse',created_at=now,updated_at=now,
                datasets=','.join(KEYS),seeds=','.join(map(str,SEEDS)))
            rows.append(row)
        temp=registry.with_suffix('.submit_inter.tmp')
        with temp.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        os.replace(temp,registry)
    # Finish B2 missing timing first; then existing baseline models; finally missing baseline seeds.
    tasks.sort(key=lambda t:t['method']!='b2')
    plan=dict(tasks=tasks,queue_path=str(queue),run_tier='pre_final')
    write(queue/'plan.json',plan)
    env=os.environ.copy(); env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    command=['/home/u202522802045/envs/graph_attention/bin/python','-B',str(ROOT/'comparison/run/queue_s_mode_inter.py'),'--plan',str(queue/'plan.json')]
    with (queue/'queue.log').open('a') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    (queue/'queue.pid').write_text(str(process.pid)+'\n')
    write(queue/'launch.json',dict(argv=command,pid=process.pid,created_at=now))
    print('LAUNCHED',process.pid,[(t['task_id'],t['method']) for t in tasks],flush=True)


if __name__=='__main__': main()
