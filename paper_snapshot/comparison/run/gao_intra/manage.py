"""Sequential method x seed queue with cooperative unit-boundary pause/stop."""
import argparse
import csv
import fcntl
import os
import platform
import resource
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from common import ROOT, CACHE, SEEDS, TIER, METHODS, NAMES, SCORES, read, write, sha, stamp, memory, score

ENV = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
PYTHON = {'multigrate':'multivi_scvi', 'eclare':'scmomer_eclare_sintra', 'scmomer':'scmomer_eclare_sintra'}
EFF = ['total_seconds','fitting_seconds','preprocessing_seconds','teacher_pretraining_seconds',
       'student_adaptation_seconds','validation_checkpoint_seconds','trainable_parameters',
       'inference_resident_parameters','train_allocated_GiB','train_reserved_GiB',
       'inference_allocated_GiB','inference_reserved_GiB','inference_seconds','cells_per_second','host_peak_RSS_GiB']


def csv_write(path, rows, columns):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def verify(unit):
    m = read(unit/'completion_manifest.json')
    assert m['status'] in ['COMPLETE', 'SMOKE_PASS']
    for name, expected in m['hashes'].items():
        assert sha(unit/name) == expected, 'Changed completed artifact: '+str(unit/name)
    return m


def alive(pid):
    try:
        os.kill(pid, 0)
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except (ProcessLookupError, FileNotFoundError):
        return False


def registry(out, status):
    path = ROOT/'result_contain/task_registry.csv'
    with (ROOT/'result_contain/task_registry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with path.open() as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        row = next(r for r in rows if r['result_path'] == str(out))
        row.update(status=status, updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), pid=str(os.getpid()))
        temp = path.with_name(path.name+'.gao.tmp')
        with temp.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fields)
            writer.writeheader();writer.writerows(rows)
        temp.replace(path)


def initialize(out):
    plan = read(out/'RUN_PLAN.json')
    configs = {}
    for m in METHODS:
        path = ROOT/f'comparison/configs/{m}/s_mode_pbmc_intra.json'
        configs[m] = read(path) if path.exists() else {}
        configs[m].update(seeds=SEEDS, run_tier=TIER, dataset='Gao P56 VIS', inference_batch=256)
    configs['b2'].update(official_runner='main/run/s_mode/train.py', core='main/core/run_b2_strict_lodo.py',
                         frozen_core_sha256='3de1ad3642e7af276658d9b239ccaf9457adf7f8e6078b95a237cf4605f2c7e1')
    configs['scbutterfly'].update(R2R_epochs=100, A2A_epochs=100, translator_epochs=200,
        patience=50, batch_size=64, rna_hidden=[256,128], atac_hidden=[32,128],
        representation='RNA-predicted ATAC, threshold1e-4; same PBMC route', loss_weight=[1,2,1])
    configs['rna_pca'].update(n_components=64, solver='randomized', fit='Source-train zscore and PCA only')
    configs['rna_encoder']['status'] = 'BLOCKED_AWAITING_FROZEN_BASELINE_IDENTITY'
    configs['scmomer']['status'] = 'BLOCKED_METHOD_ADAPTATION'
    write(out/'METHOD_CONFIGS.json', configs)
    downloads = read(ROOT/'comparison/docs/OFFICIAL_CODE_DOWNLOADS.json')['current']
    provenance = []
    for m in METHODS:
        old = downloads.get('scbutterfly_b' if m=='scbutterfly' else m, {})
        provenance.append(dict(method=NAMES[m], repository=old.get('repository'), commit=old.get('commit'),
            source_path=old.get('source_path'), status=configs[m].get('status','PENDING_SMOKE'),
            variant='Source-only adapter' if m in ['eclare','scmomer','palette'] else 'frozen project runner / official module'))
    csv_write(out/'METHOD_PROVENANCE.csv', provenance, ['method','repository','commit','source_path','status','variant'])
    code = list(Path(__file__).parent.glob('*.py'))
    code += [ROOT/'main/run/s_mode/train.py', ROOT/'main/core/run_b2_strict_lodo.py',
        ROOT/'main/core/regulatory/s_mode/single_source.py', ROOT/'comparison/run/s_mode_intra.py',
        ROOT/'comparison/run/neural_s_mode_intra.py', ROOT/'comparison/run/eclare/s_mode_adapter.py',
        ROOT/'comparison/run/palette/fit_source.R']
    for method in ['eclare','multigrate','cobolt','palette','scbutterfly_b']:
        code += list((ROOT/f'comparison/methods/{method}/official').rglob('*.R' if method=='palette' else '*.py'))
    plan.update(execution_source_hashes={str(p):sha(p) for p in code},
        config_sha256=sha(out/'METHOD_CONFIGS.json'), controller_sha256=sha(__file__))
    write(out/'RUN_PLAN.json', plan)
    write(out/'CONTROL.json', dict(request='RUN', pause_granularity='finish current method x seed'))
    (out/'PAUSE_RESUME_AUDIT.md').write_text('# Pause/resume audit\n\n'
        'Pause and stop are cooperative at method x seed boundaries, including evaluation. '
        'The current unit completes and persists its checkpoint before exit. No epoch-resume claim. '
        'Resume verifies all completed artifact hashes and never retrains completed units. '
        'Failed/partial units are preserved and require an explicit reviewed retry; no automatic overwrite.\n')
    (out/'GPU_TIMING_PROTOCOL.md').write_text('# Gao three-seed timing protocol\n\n'
        'This task uses the explicit Task84 request: Total includes scoring and persistence, unlike PBMC V2 inter end-to-end. '
        'Continuous perf_counter starts after the A30 lock and before frozen processed input loading; '
        'ends after inference, StandardScaler, known-K KMeans, all nine scores, full label-based Silhouette and standard result writes. '
        'Final timer/completion-hash bookkeeping is outside its own interval. Shared author QC, split, fixed HVG identity, '
        'lossless paired-only format export, software/download and lock wait are excluded. Method ATAC transforms and B2 fresh B/F are included. '
        'Internal source pretraining is included; historical public pretraining production is excluded and disclosed. '
        'Inference starts with model resident and RNA in CPU RAM, one batch warmup excluded, CUDA sync, batch256, '
        'ends when all representations are in CPU RAM. No disk writes/scoring inside inference. '
        'Allocated and reserved peaks are measured separately for train and inference. Palette/PCA use CPU; GPU inference fields NA. '
        'Original stage timers include validation and checkpoint work. Where a stage cannot be split exactly, the subcomponent is NA. '
        'Three seeds are pre_final_explicit_3seeds; sample SD ddof=1. No partial-seed mean reported as 3-seed.\n')
    (out/'FAILURES_AND_EXCEPTIONS.md').write_text('# Failures and exceptions\n\n'
        '- scMomer-S blocked: current 16,906-gene pretrained reference is human; PBMC Ensembl/symbol mapping does not cover Gao mouse genes. '
        'No validated mouse checkpoint/orthology mapping provided; no uppercase-symbol substitution or de novo core change. '
        'External pretrained training-cell manifest is unavailable: potential Gao overlap remains UNKNOWN.\n'
        '- RNA-only Encoder pending: only Task38 ReLU DAE is located; it differs from B2 LN/GELU Encoder. '
        'No new baseline is invented; user clarification requested.\n'
        '- Author peak universe and Task80 reference assumptions retained; preprocessing is not claimed independent of author annotation-based peak calling.\n')
    for name in ['PREFLIGHT_REPORT.md','FINAL_COMPARISON.md','FINAL_REPORT.md']:
        (out/name).write_text('# '+name[:-3]+'\n\nPENDING: not a completed experiment.\n')
    summarize(out)


def unit_run(out, method, seed, smoke):
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2)
    unit = out/('smoke' if smoke else 'units')/method/f'seed_{seed}'
    if (unit/'completion_manifest.json').exists():
        verify(unit)
        return
    assert not any(p.name != 'run.log' for p in unit.iterdir()), 'Partial unit preserved; review before retry'
    config = read(out/'METHOD_CONFIGS.json')[method]
    assert not config.get('status','').startswith('BLOCKED'), config.get('status')
    device = 'cpu' if method in ['palette','rna_pca'] else 'cuda'
    with (ROOT/'.benchmark_a30.lock').open('a') as lock, threadpool_limits(2):
        fcntl.flock(lock, fcntl.LOCK_EX)
        occupied = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
        assert not occupied, 'Another GPU process is active outside the shared lock: '+occupied
        if device == 'cuda':
            torch.cuda.set_device(0);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        tick = stamp(device)
        import random
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        import methods
        write(unit/'config.json', dict(method=method, seed=seed, run_tier='temporary_smoke' if smoke else TIER, configuration=config))
        write(unit/'environment.json', dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
            numpy=np.__version__, cuda=torch.version.cuda, CPU=platform.processor(), cpu_threads=2,
            GPU=torch.cuda.get_device_name(0) if device=='cuda' else 'CPU',
            gpu_identity=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True).strip()))
        write(unit/'input_manifest.json',dict(shared_manifest_sha256=sha(out/'DATASET_AND_SPLIT_MANIFEST.json'),
            source_code_sha256=read(out/'RUN_PLAN.json')['execution_source_hashes'], test_atac_access=False,
            split='Task80 fixed', HVG='Task80 fixed Top2000', labels_for_training=False))
        if method == 'b2':
            z, resources = methods.b2_run(out, unit, seed, smoke, config)
        else:
            data, cache, preprocessing = methods.data_for(out, unit, method, smoke)
            z, resources = methods.run(method, data, config, unit, seed, smoke, cache)
            resources['preprocessing_seconds'] = preprocessing
        assert z.ndim == 2 and np.isfinite(z).all() and len(z) == (64 if smoke else 5280)
        np.save(unit/'test_embedding.npy', z)
        history = resources.pop('training_history', [])
        write(unit/'training_history.json', history)
        if not smoke:
            scoring_tick = time.perf_counter()
            score(unit)
            resources['scoring_seconds'] = time.perf_counter()-scoring_tick
        resources.update(device='CPU' if device=='cpu' else torch.cuda.get_device_name(0),
            host_peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            total_time_scope='Task84 continuous full intra workflow, including scoring and persistence',
            run_tier='temporary_smoke' if smoke else TIER)
        for stage in ['train','inference']:
            for kind in ['allocated','reserved']:
                resources[f'{stage}_{kind}_GiB'] = resources.get(stage+'_peak',{}).get(kind+'_GiB')
        if device=='cpu':
            resources['CPU_inference_seconds'] = resources['inference_seconds']
            resources['CPU_cells_per_second'] = resources['cells_per_second']
            resources['GPU_inference_seconds'] = None
        write(unit/'resource_usage.json', resources)
        resources['total_seconds'] = stamp(device)-tick
        write(unit/'timing.json', resources)
        files = [p for p in unit.rglob('*') if p.is_file() and p.name != 'run.log']
        assert any(p.suffix in ['.pt','.rds','.npz'] and 'checkpoint' in str(p.relative_to(unit)) for p in files), 'Missing fitted checkpoint'
        write(unit/'completion_manifest.json', dict(status='SMOKE_PASS' if smoke else 'COMPLETE', method=method,
            seed=seed, run_tier='temporary_smoke' if smoke else TIER, fresh=True, RNA_only_test=True,
            labels_loaded_after_embedding_saved=not smoke, config_sha256=sha(out/'METHOD_CONFIGS.json'),
            hashes={str(p.relative_to(unit)):sha(p) for p in files}))


def summarize(out):
    import numpy as np
    configs = read(out/'METHOD_CONFIGS.json')
    rows, failures, states = [], [], []
    for method in METHODS:
        for seed in SEEDS:
            unit = out/'units'/method/f'seed_{seed}'
            status = configs[method].get('status','PENDING')
            if (unit/'completion_manifest.json').exists():
                verify(unit)
                m, e = read(unit/'metrics.json'), read(unit/'timing.json')
                rows.append(dict(Method=NAMES[method], seed=seed, **{k:m[k] for k in SCORES}, **e))
                status = 'COMPLETE'
            elif (unit/'failure.json').exists():
                failure = read(unit/'failure.json')
                failures.append(dict(Method=NAMES[method], seed=seed, **failure))
                status = 'FAILED_PRESERVED'
            elif (unit/'config.json').exists():
                status = 'RUNNING_OR_INTERRUPTED'
            states.append(dict(method=method, seed=seed, status=status))
    columns = ['Method','seed']+SCORES+EFF+['device','inference_batch']
    csv_write(out/'metrics_per_seed.csv',rows,columns)
    csv_write(out/'efficiency_per_seed.csv',rows,['Method','seed']+EFF+['device','inference_batch'])
    summaries = []
    for method in METHODS:
        subset = [r for r in rows if r['Method']==NAMES[method]]
        row = dict(Method=NAMES[method], completed_seeds=len(subset), expected_seeds=3, run_tier=TIER)
        for key in SCORES+EFF:
            values = [r.get(key) for r in subset]
            valid = len(values)==3 and all(isinstance(v,(float,int)) for v in values)
            row[key+'_mean'] = float(np.mean(values)) if valid else None
            row[key+'_SD'] = float(np.std(values,ddof=1)) if valid else None
        summaries.append(row)
    summary_columns=['Method','completed_seeds','expected_seeds','run_tier']+[k+s for k in SCORES+EFF for s in ['_mean','_SD']]
    csv_write(out/'metrics_mean_std.csv',summaries,summary_columns)
    csv_write(out/'efficiency_mean_std.csv',summaries,['Method','completed_seeds']+[k+s for k in EFF for s in ['_mean','_SD']])
    deltas = []
    for seed in SEEDS:
        pair = {r['Method']:r for r in rows if r['seed']==seed}
        if all(k in pair for k in ['Full B2','RNA-only Encoder']):
            deltas.append(dict(seed=seed, **{k:pair['Full B2'][k]-pair['RNA-only Encoder'][k] for k in ['ARI','NMI','AMI','Macro-F1']}))
    csv_write(out/'paired_deltas.csv',deltas,['seed','ARI','NMI','AMI','Macro-F1'])
    csv_write(out/'RESOURCE_USAGE.csv',rows,['Method','seed']+EFF+['device','inference_batch'])
    csv_write(out/'TIMING_BOUNDARY_AUDIT.csv',[dict(Method=r['Method'],seed=r['seed'],
        fresh_continuous_total=True,inference_excludes_IO_scoring=True,split_stage_validation='included, inseparable') for r in rows],
        ['Method','seed','fresh_continuous_total','inference_excludes_IO_scoring','split_stage_validation'])
    write(out/'QUEUE_STATUS.json',dict(units=states, complete=len(rows), expected_main=24, expected_pca=3,
        failures=failures, control=read(out/'CONTROL.json') if (out/'CONTROL.json').exists() else None))
    headers=['Method']+SCORES+['fitting_seconds','total_seconds','trainable_parameters','inference_resident_parameters',
        'train_allocated_GiB','train_reserved_GiB','inference_allocated_GiB','inference_reserved_GiB',
        'inference_seconds','cells_per_second']
    lines=['# Gao S-mode intra — '+TIER,'','Pending rows are NA; no partial 3-seed mean. Total includes scoring. '
           'Palette-S and RNA PCA inference time/throughput are CPU measurements; their GPU fields are NA. '
           'Every neural inference uses batch256 on NVIDIA A30.','',
           '| '+' | '.join(headers+['Seeds'])+' |','|'+'---|'*(len(headers)+1)]
    for row in summaries:
        values=[row['Method']]
        for k in headers[1:]:
            values.append('NA' if row[k+'_mean'] is None else f"{row[k+'_mean']:.4f}±{row[k+'_SD']:.4f}")
        lines.append('| '+' | '.join(values+[str(row['completed_seeds'])+'/3'])+' |')
    (out/'FINAL_COMPARISON.md').write_text('\n'.join(lines)+'\n')
    gates=[]
    for method in METHODS:
        path=out/'smoke'/method/f'seed_{SEEDS[0]}'/'completion_manifest.json'
        gates.append(f"- {NAMES[method]}: "+('SMOKE_PASS' if path.exists() else configs[method].get('status','PENDING_SMOKE')))
    (out/'PREFLIGHT_REPORT.md').write_text('# Preflight\n\n'+'\n'.join(gates)+'\n')
    (out/'FINAL_REPORT.md').write_text('# Experiment status\n\n'
        f'{len(rows)}/27 total units completed; {len(failures)} failed units preserved. '
        'Eight main methods require 24 units, plus three PCA units.\n\n'
        'No final 15-seed or statistical-significance claim. See FINAL_COMPARISON.md, '
        'FAILURES_AND_EXCEPTIONS.md and QUEUE_STATUS.json. B2 versus RNA Encoder conclusion remains pending unless both complete 3/3.\n')


def queue(out):
    with (out/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        (out/'queue.pid').write_text(str(os.getpid()))
        registry(out,'RUNNING_PREFLIGHT_AND_3SEED_QUEUE')
        def boundary():
            request=read(out/'CONTROL.json')['request']
            if request != 'RUN':
                write(out/'CONTROL.json',dict(request=request,status='PAUSED' if request=='PAUSE' else 'STOPPED',current_unit_finished=True))
                registry(out,'PAUSED' if request=='PAUSE' else 'STOPPED')
                summarize(out)
                return False
            return True
        for method in METHODS:
            config=read(out/'METHOD_CONFIGS.json')[method]
            if config.get('status','').startswith('BLOCKED'):
                continue
            for smoke, seeds in [(True,SEEDS[:1]),(False,SEEDS)]:
                if not smoke and not (out/'smoke'/method/f'seed_{SEEDS[0]}'/'completion_manifest.json').exists():
                    break
                for seed in seeds:
                    if not boundary():return
                    unit=out/('smoke' if smoke else 'units')/method/f'seed_{seed}'
                    if (unit/'completion_manifest.json').exists():verify(unit);continue
                    if unit.exists() and any(unit.iterdir()):
                        print('PRESERVE_PARTIAL',unit,flush=True)
                        continue
                    plan=read(out/'RUN_PLAN.json')
                    assert sha(out/'METHOD_CONFIGS.json')==plan['config_sha256'],'Frozen method configs changed'
                    for p,digest in plan['execution_source_hashes'].items():assert sha(p)==digest,'Frozen source changed: '+p
                    unit.mkdir(parents=True,exist_ok=True)
                    executable=f'/home/u202522802045/envs/{PYTHON.get(method,"graph_attention")}/bin/python'
                    command=[executable,'-u',__file__,'unit','--output',str(out),'--method',method,'--seed',str(seed)]
                    if smoke:command.append('--smoke')
                    print('START',method,seed,'SMOKE' if smoke else 'FORMAL',flush=True)
                    with (unit/'run.log').open('x') as log:
                        result=subprocess.run(command,env=ENV,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                    if result.returncode:
                        write(unit/'failure.json',dict(exitcode=result.returncode,log=str(unit/'run.log'),error=(unit/'run.log').read_text()[-6000:]))
                        with (out/'FAILURES_AND_EXCEPTIONS.md').open('a') as handle:
                            handle.write(f'\n- {method} seed{seed} smoke={smoke} FAILED; preserved {unit}/run.log\n')
                        if smoke:break
                    else:verify(unit)
                    summarize(out)
        summarize(out)
        registry(out,'COMPLETED' if read(out/'QUEUE_STATUS.json')['complete']==27 else 'PARTIAL_BLOCKED_OR_FAILED')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['initialize','status','pause','resume','stop','queue','unit','summarize'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0])
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();out=args.output
    if args.action=='initialize':initialize(out)
    elif args.action in ['pause','stop']:
        write(out/'CONTROL.json',dict(request='PAUSE' if args.action=='pause' else 'STOP',requested_at=time.time()))
        with (out/'PAUSE_RESUME_AUDIT.md').open('a') as h:h.write(f'\n{time.ctime()}: {args.action} requested; waiting for current unit boundary.\n')
        print('Requested; current atomic method x seed is allowed to finish safely.')
    elif args.action=='resume':
        assert (out/'DATASET_AND_SPLIT_MANIFEST.json').exists(),'Wait for frozen input audit/export to complete'
        if (out/'queue.pid').exists() and alive(int((out/'queue.pid').read_text())):
            write(out/'CONTROL.json',dict(request='RUN'));print('Existing queue active; no duplicate launched.');return
        summarize(out)
        write(out/'CONTROL.json',dict(request='RUN'))
        with (out/'queue.log').open('a') as log:
            process=subprocess.Popen([sys.executable,'-u',__file__,'queue','--output',str(out)],
                env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
        (out/'queue.pid').write_text(str(process.pid));print('QUEUE_PID',process.pid)
    elif args.action=='queue':
        try:
            queue(out)
        except Exception:
            write(out/f'controller_failure_{time.time_ns()}.json',dict(error=traceback.format_exc()))
            registry(out,'FAILED_CONTROLLER_PRESERVED')
            raise
    elif args.action=='unit':unit_run(out,args.method,args.seed,args.smoke)
    else:
        summarize(out)
        if args.action=='status':print((out/'QUEUE_STATUS.json').read_text())


if __name__=='__main__':main()
