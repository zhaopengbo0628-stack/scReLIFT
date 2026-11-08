"""Fresh, isolated Source fit plus three inter Targets; never overwrite old models."""
import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from pbmc_intra_common import ROOT, CACHE_ROOT, read, write, sha, cache_for, expected_hash
from efficiency_v2 import PROTOCOL

METHODS = ['multigrate','cobolt','palette','eclare','scmomer','b2']


def fit(args):
    if args.method == 'b2':
        sys.path.insert(0,str(ROOT))
        from main.run.s_mode import train as b2
        cache=cache_for(args.source,'pbmc_corrected_s10x20_v1_20261012')
        manifest=read(cache/'manifest.json')
        for name in ['rna_train.npy','rna_val.npy','F_train.npy','F_val.npy','rna_state.npz','gene_ids.npy']:
            assert sha(cache/name)==expected_hash(manifest['hashes'],cache/name),name
        b2.torch.set_num_threads(2)
        with (ROOT/'.benchmark_a30.lock').open('a') as lock, b2.threadpool_limits(2):
            fcntl.flock(lock,fcntl.LOCK_EX)
            b2.train(args.source,args.seed,args.output,cache,manifest,evaluation_targets=[])
    else:
        sys.argv=[str(ROOT/f'comparison/run/{args.method}/s_mode_intra.py'),
            '--stage','unit','--dataset',args.source,'--seed',str(args.seed),
            '--device','cpu' if args.method=='palette' else 'cuda',
            '--output-root',str(args.output/'checkpoints'),
            '--run-seeds',*[str(s) for s in range(20020130,20020135)]]
        if args.method in ['eclare','scmomer']:
            from neural_s_mode_intra import main
        else:
            from s_mode_intra import main
        main(args.method)


def run(args):
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[name]='2'
    metadata=args.output/'efficiency_request.json'
    config=ROOT/('main/configs/smode_pbmc_corrected.json' if args.method=='b2' else f'comparison/configs/{args.method}/s_mode_pbmc_intra.json')
    code=[Path(__file__),ROOT/'comparison/run/efficiency_v2.py',ROOT/'comparison/run/s_mode_inter.py',
          ROOT/'comparison/run/neural_s_mode_intra.py',ROOT/'comparison/run/s_mode_intra.py',
          ROOT/'main/run/s_mode/train.py',ROOT/'main/core/run_b2_strict_lodo.py']
    adapter=ROOT/f'comparison/run/{args.method}/s_mode_adapter.py'
    if adapter.exists():code.append(adapter)
    request=dict(method=args.method,protocol_sha256=sha(PROTOCOL),run_tier='pre_final',
                 source_code_sha256={str(p):sha(p) for p in code},configuration_sha256=sha(config),
                 seeds=list(range(20020130,20020135)),source_scope='4 Sources x 5 seeds; 12 directed pairs per seed')
    if metadata.exists():
        assert read(metadata)==request,'Output directory belongs to a different request'
    elif args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError('Choose a new output directory; historical results preserved')
    else:
        write(metadata,request)
    unit=args.output/('models' if args.method=='b2' else 'checkpoints')/args.source/f'seed_{args.seed}'
    if unit.exists():
        raise RuntimeError(f'Existing Source unit preserved; no automatic retraining: {unit}')
    log=args.output/'logs'/f'{args.source}_{args.seed}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    with log.open('x') as handle:
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--stage','fit',
            '--method',args.method,'--source',args.source,'--seed',str(args.seed),
            '--output',str(args.output)],stdout=handle,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,'-B',str(ROOT/'comparison/run/s_mode_inter.py'),
            '--method',args.method,'--source',args.source,'--seed',str(args.seed),
            '--unit',str(unit),'--output',str(args.output),'--efficiency-v2'],
            stdout=handle,stderr=subprocess.STDOUT,check=True)
    source=read(unit/('training_manifest.json' if args.method=='b2' else 'manifest.json'))
    fitting=source['efficiency_v2']
    workflow=args.output/'workflows'/args.source/f'seed_{args.seed}'
    inter=read(workflow/'inter_efficiency.json')
    assert inter['complete'] and inter['checkpoint_sha256']==fitting['checkpoint_sha256']
    assert fitting['protocol_sha256']==sha(PROTOCOL)
    source_seconds=fitting['source_fitting']['source_workflow_seconds']
    assert source_seconds is not None,'Incomplete or resumed Source timing cannot be reported as fresh'
    elapsed=time.perf_counter()-started
    active_wall=source_seconds+inter['active_wall_seconds']
    assert elapsed>=active_wall
    missing=['B_S selector and F construction (precomputed method-specific cache)'] if args.method=='b2' else []
    end_to_end=active_wall if not missing else None
    import platform
    import torch
    environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        cpu=platform.processor(),cpu_threads=2,inference_batch=1 if args.method=='scmomer' else 256,
        training_batch=256 if args.method=='b2' else read(config)['batch_size'])
    if args.method!='palette':
        gpu=torch.cuda.get_device_properties(0)
        environment.update(gpu=gpu.name,gpu_uuid=str(getattr(gpu,'uuid','unavailable')))
    write(workflow/'efficiency.json',dict(method=args.method,source=args.source,seed=args.seed,
        protocol_sha256=sha(PROTOCOL),run_tier='pre_final',source_record=fitting,
        checkpoint_sha256=fitting['checkpoint_sha256'],
        end_to_end_seconds=end_to_end,source_workflow_seconds=source_seconds,
        launcher_elapsed_seconds=elapsed,excluded_seconds=elapsed-active_wall,
        measured_active_wall_seconds=active_wall,missing_end_to_end_components=missing,environment=environment,
        end_to_end_mode='Sum measured Source and inter active wall intervals; Source fitted once',
        excluded_scope='Queue/lock and process startup; extra Source intra evaluation; posthoc scoring and audit-only replays',
        fresh=True,targets=[t['target'] for t in inter['targets']],
        source_code_sha256={str(p):sha(p) for p in [Path(__file__),PROTOCOL,
            ROOT/'comparison/run/s_mode_inter.py',ROOT/'comparison/run/efficiency_v2.py']},
        source_manifest_path=str(unit/('training_manifest.json' if args.method=='b2' else 'manifest.json'))))
    subprocess.run([sys.executable,'-B',str(ROOT/'comparison/run/summarize_efficiency_v2.py'),
                    '--roots',str(args.output),'--output',str(args.output/'metrics/v2')],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=METHODS,required=True)
    parser.add_argument('--source',choices=['D1','D2','D3','D4'],required=True)
    parser.add_argument('--seed',type=int,choices=range(20020130,20020135),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['run','fit'],default='run')
    args=parser.parse_args()
    try:
        fit(args) if args.stage=='fit' else run(args)
    except Exception as error:
        failure=args.output/'failures'/f'{args.source}_{args.seed}_{time.time_ns()}.json'
        write(failure,dict(status='FAILED',error=repr(error),command=sys.argv,protocol_sha256=sha(PROTOCOL)))
        raise
