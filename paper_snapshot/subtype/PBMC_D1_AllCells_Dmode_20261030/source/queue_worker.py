"""Persistent serial worker. Never signals processes or reclaims GPU memory."""
import argparse
import csv
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime,timezone
from common import RUN,SRC,SEEDS,write

PY='/home/u202522802045/envs/graph_attention/bin/python'
FIELDS=['Job_ID','Dataset','Experiment','Seed','Priority','GPU','Status','Scheduler_ID_or_PID','Checkpoint','Log','Attempts','Started','Finished','Error']
def now(): return datetime.now(timezone.utc).isoformat()

def initial():
    rows=[]
    specs=[('PCA','pca',0,'',0)]+[(f'B2_{seed}','full_b2',seed,'0',1) for seed in SEEDS]
    for name,method,seed,gpu,priority in specs:
        rows.append(dict(Job_ID='D1_ALL_'+name,Dataset='D2+D3+D4 -> D1_ALL',Experiment=method,
                    Seed=seed or 'all_three',Priority=priority,GPU=gpu or 'CPU',Status='PENDING',
                    Scheduler_ID_or_PID='',Checkpoint=str(RUN/'FullB2'/f'seed_{seed}'/'checkpoints/student/latest.pt') if gpu else '',
                    Log=str(RUN/'logs'/f'{name}.log'),Attempts=0,Started='',Finished='',Error=''))
    return rows

def save(rows):
    p=RUN/'JOB_QUEUE.csv';tmp=p.with_suffix('.csv.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    tmp.replace(p)
    lines=['# D1 全细胞 D-mode 运行状态','',f'更新：{now()}','',
           'D2/D3/D4 共 16,835 个 Source 细胞训练；D1 共 9,347 个细胞仅推断。',
           '全 Source 重选 HVG、Scaler、B_D 与 F；Teacher/Student 固定 100 轮最终 checkpoint。','',
           '| 任务 | Seed | 状态 | PID |','| --- | --- | --- | --- |']
    lines += [f"| {r['Experiment']} | {r['Seed']} | {r['Status']} | {r['Scheduler_ID_or_PID']} |" for r in rows]
    lines += ['', '实际调度：独立 tmux 会话内的文件锁串行 Worker。GPU 忙时保留 PENDING，每 120 秒检查。',
              '没有执行 kill、pkill、GPU reset 或任何抢占操作。',
              'PCA 包括 Source-fit 正式比较与 Target-fit 描述性敏感性分析；PCA 本身为确定性 SVD。']
    (RUN/'RUN_STATUS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    summaries=[]
    for p in RUN.glob('**/metrics.json'):
        import json
        r=json.loads(p.read_text())
        summaries.append({k:r[k] for k in ['method','seed','cells','subclasses','ari','nmi','ami','cluster_acc','macro_f1','balanced_accuracy','purity_at_15']})
    if summaries:
        import pandas as pd
        pd.DataFrame(summaries).to_csv(RUN/'RESULT_SUMMARY.csv',index=False)

def available():
    try:
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True,timeout=20).strip()
        row=next(r for r in csv.reader(gpu.splitlines()) if r[0].strip()=='0')
        uuid=row[1].strip();mem=int(row[2]);total=int(row[3]);util=int(row[4])
        safe=not apps and mem<500 and util<5 and total-mem>=8000
        return safe,dict(gpu=gpu,processes=apps,uuid=uuid,safe=safe,time=now())
    except Exception as e:
        return False,dict(safe=False,error=str(e),time=now())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--initialize',action='store_true');args=parser.parse_args()
    lock=(RUN/'queue_worker.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=list(csv.DictReader((RUN/'JOB_QUEUE.csv').open())) if (RUN/'JOB_QUEUE.csv').exists() else initial()
    if args.initialize: save(rows);return
    write(RUN/'audit/WORKER_IDENTITY.json',dict(pid=os.getpid(),start=now(),interpreter=sys.executable,serial=True))
    for row in rows:
        if row['Status']=='RUNNING':
            # An interrupted worker may leave a live child. Never relaunch if its PID still exists.
            if (Path('/proc')/str(row['Scheduler_ID_or_PID'])).exists():
                raise RuntimeError('Previous child still exists; do not duplicate or signal it')
            row.update(Status='PENDING',Error='Prior worker exited; safe checkpoint resume requested')
    save(rows)
    while not (RUN/'artifacts/PREPARATION_COMPLETE.json').exists():
        print('WAIT_SOURCE_PREPARATION',now(),flush=True);time.sleep(60)
    for row in rows:
        if row['Status']=='COMPLETED': continue
        gpu=row['GPU']=='0';gpu_lock=None
        if gpu:
            while True:
                safe,snapshot=available()
                write(RUN/'audit/GPU_LAST_CHECK.json',snapshot)
                if safe:
                    gpu_lock=(Path.home()/('.codex_gpu_queue_'+snapshot['uuid']+'.lock')).open('a+')
                    try: fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError: gpu_lock.close();gpu_lock=None
                    if gpu_lock:
                        time.sleep(5);again,second=available()
                        if again: break
                        gpu_lock.close();gpu_lock=None
                row['Status']='PENDING';save(rows);time.sleep(120)
        command=[PY,'-u',str(SRC/'run_model.py'),'--method',row['Experiment']]
        if gpu: command += ['--seed',str(row['Seed'])]
        env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMBA_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        if gpu: env['CUDA_VISIBLE_DEVICES']='0'
        else: env['CUDA_VISIBLE_DEVICES']=''
        for attempt in range(int(row['Attempts']),2):
            row.update(Attempts=attempt+1,Started=now(),Error='')
            with open(row['Log'],'a') as log:
                log.write(f'\nATTEMPT {attempt+1} {now()} COMMAND {command}\n');log.flush()
                child=subprocess.Popen(command,cwd=RUN,env=env,stdout=log,stderr=subprocess.STDOUT)
                row.update(Status='RUNNING',Scheduler_ID_or_PID=child.pid);save(rows)
                code=child.wait()
            if code==0:
                row.update(Status='COMPLETED',Finished=now());save(rows);break
            row.update(Status='FAILED',Error=f'exit {code}; inspect log',Finished=now());save(rows)
            if attempt==0:
                # Retry only our failed process; never retry against an occupied GPU.
                if gpu:
                    safe,_=available()
                    if not safe:
                        row.update(Status='PENDING',Error='Retry deferred: GPU now occupied');save(rows)
                        while not available()[0]: time.sleep(120)
                time.sleep(15)
        if gpu_lock: gpu_lock.close()
    save(rows)
    print('QUEUE_FINISHED',now(),flush=True)

if __name__=='__main__': main()
