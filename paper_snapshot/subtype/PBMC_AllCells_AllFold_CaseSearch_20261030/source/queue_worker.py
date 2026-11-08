"""Persistent serial queue: one GPU child at a time, no process signalling or preemption."""
import csv,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'PBMC_D1_AllCells_Dmode_20261030'
sys.path.insert(0,str(OLD/'source'))
from queue_worker import available
from common import write
PY='/home/u202522802045/envs/graph_attention/bin/python'
FIELDS=['Job_ID','Dataset','Experiment','Seed','Priority','GPU','Status','Scheduler_ID_or_PID','Checkpoint','Log','Attempts','Started','Finished','Error']


def now():return datetime.now(timezone.utc).isoformat()


def initial():
    specs=[]
    for fold in ['D2','D3','D4']:specs += [(fold,'pca',0),(fold,'full_b2',20020130)]
    specs += [(fold,'full_b2',s) for s in [20020131,20020132] for fold in ['D2','D3','D4']]
    rows=[]
    for i,(fold,method,seed) in enumerate(specs):
        base=ROOT/'folds'/fold
        name=f'{fold}_ALL_{method}_{seed or "PCA"}'
        rows.append(dict(Job_ID=name,Dataset=fold,Experiment=method,Seed=seed or 'all_three',Priority=i,
            GPU='0' if seed else 'CPU',Status='PENDING',Scheduler_ID_or_PID='',
            Checkpoint=str(base/'FullB2'/f'seed_{seed}/checkpoints/student/latest.pt') if seed else '',
            Log=str(ROOT/'logs'/f'{name}.log'),Attempts=0,Started='',Finished='',Error=''))
    return rows


def save(rows):
    p=ROOT/'JOB_QUEUE.csv';tmp=p.with_suffix('.csv.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    tmp.replace(p)
    lines=['# PBMC all-cell LODO case search','',now(),'',
      'D1 prior all-cell checkpoints are reused. D2/D3/D4: full Source refits; full held-out Target inference.',
      'Serial file-locked tmux worker. Busy or unverified GPU remains PENDING. No process signalling.', '',
      '|Job|Fold|Seed|Status|PID|','|---|---|---|---|---|']
    lines += [f"|{r['Experiment']}|{r['Dataset']}|{r['Seed']}|{r['Status']}|{r['Scheduler_ID_or_PID']}|" for r in rows]
    (ROOT/'RUN_STATUS.md').write_text('\n'.join(lines)+'\n')


def main():
    lock=(ROOT/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=ROOT/'JOB_QUEUE.csv'
    rows=list(csv.DictReader(p.open())) if p.exists() else initial();save(rows)
    write(ROOT/'audit/WORKER_IDENTITY.json',dict(pid=os.getpid(),started=now(),serial=True,mode='tmux_no_scheduler'))
    for row in rows:
        if row['Status']=='RUNNING':
            if (Path('/proc')/str(row['Scheduler_ID_or_PID'])).exists():raise RuntimeError('Existing child remains alive; no duplicate launch')
            row.update(Status='PENDING',Error='Dead prior child; resume from checkpoint allowed')
        if row['Status']=='COMPLETED':continue
        fold=row['Dataset'];base=ROOT/'folds'/fold
        while not (base/'artifacts/PREPARATION_COMPLETE.json').exists():
            print('WAIT_PREPARATION',fold,now(),flush=True);time.sleep(30)
        gpu=row['GPU']=='0';gpu_lock=None
        if gpu:
            while True:
                safe,snapshot=available();write(ROOT/'audit/GPU_LAST_CHECK.json',snapshot)
                if safe:
                    gpu_lock=(Path.home()/('.codex_gpu_queue_'+snapshot['uuid']+'.lock')).open('a+')
                    try:fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:gpu_lock.close();gpu_lock=None
                    if gpu_lock:
                        time.sleep(5)
                        if available()[0]:break
                        gpu_lock.close();gpu_lock=None
                row['Status']='PENDING';save(rows);time.sleep(120)
        cmd=[PY,'-u',str(base/'source/run_model.py'),'--method',row['Experiment']]
        if gpu:cmd+=['--seed',str(row['Seed'])]
        env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMBA_NUM_THREADS='2',
            PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0' if gpu else '')
        for attempt in range(int(row['Attempts']),2):
            row.update(Attempts=attempt+1,Started=now(),Error='')
            with open(row['Log'],'a') as out:
                out.write('\n'+json.dumps(dict(command=cmd,attempt=attempt+1,time=now()))+'\n');out.flush()
                child=subprocess.Popen(cmd,cwd=base,env=env,stdout=out,stderr=subprocess.STDOUT)
                row.update(Status='RUNNING',Scheduler_ID_or_PID=child.pid);save(rows)
                code=child.wait()
            if code==0:
                row.update(Status='COMPLETED',Finished=now());save(rows);break
            row.update(Status='FAILED',Finished=now(),Error=f'exit {code}; retained log');save(rows)
            if attempt==0:
                if gpu:
                    while not available()[0]:time.sleep(120)
                time.sleep(15)
        if gpu_lock:gpu_lock.close()
    save(rows);write(ROOT/'audit/TRAIN_QUEUE_COMPLETE.json',dict(status='COMPLETE' if all(r['Status']=='COMPLETED' for r in rows) else 'PARTIAL',jobs=len(rows),GPU_jobs=9))
    print('QUEUE_FINISHED',now(),flush=True)


if __name__=='__main__':main()
