"""Read completed ablation artifacts and add accounting to the final report."""
import json
import subprocess
from pathlib import Path

import numpy as np
import run_ablation as run

root, out = run.ROOT, run.OUT
run.summarize()
result = run.b2.read(root/'MULTI_TEACHER_3SEED_REPORT.json')
rows, hashes = [], {}
for fold in run.FOLDS:
    for seed in run.shared.SEEDS:
        unit=out/fold/f'seed_{seed}'
        run.verify(unit)
        selected=run.b2.read(unit/'teacher_selection.json')['selected_student'][-1]
        total=0
        for letter in 'ABC':
            e=run.b2.read(unit/f'S_{letter}'/'efficiency.json')
            total+=e['deployment_path_training_seconds']
            t=run.b2.read(unit/f'T_{letter}'/'training_accounting.json')
            s=run.b2.read(unit/f'S_{letter}'/'training_accounting.json')
            rows.append(dict(fold=fold,seed=seed,candidate=letter,selected=letter==selected,
                             teacher_training_seconds=e['teacher']['training_seconds'],
                             student_training_seconds=e['student']['training_seconds'],
                             inference_seconds=e['target_inference_seconds'],params=e['TOTAL_PARAMS'],
                             peak_GPU_MB=e['peak_GPU_MB'],teacher_cells=t['training_cells'],
                             teacher_epochs=t['epochs'],teacher_steps=t['optimizer_steps'],
                             student_epochs=s['epochs'],student_steps=s['optimizer_steps']))
        run.b2.write(unit/'total_experiment_efficiency.json',dict(total_training_seconds=total,GPU_hours=total/3600,
                     deployment_cost_scope='one candidate Teacher+Student',selection_cost_scope='all three Teachers+Students'))
        for p in unit.rglob('*.pt'):hashes[str(p.relative_to(out))]=run.prep.sha(p)
per_fold={}
keys=list(result['overall']['A'])
for strategy in ['A','B','C','BEST']:
    per_fold[strategy]={fold:{k:dict(mean=float(np.mean([r[k] for r in result['rows'] if r['strategy']==strategy and r['fold']==fold])),
                              std=float(np.std([r[k] for r in result['rows'] if r['strategy']==strategy and r['fold']==fold],ddof=1)))
                              for k in keys} for fold in run.FOLDS}
environment=dict(driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version,name,memory.total','--format=csv,noheader'],text=True).strip(),
                 precision='float32; TF32 disabled',CUDA=run.torch.version.cuda,PyTorch=run.torch.__version__)
run.b2.write(root/'EFFICIENCY_AND_CHECKPOINT_AUDIT.json',dict(rows=rows,environment=environment,checkpoint_hashes=hashes,
             mean_teacher_seconds=float(np.mean([r['teacher_training_seconds'] for r in rows])),
             mean_student_seconds=float(np.mean([r['student_training_seconds'] for r in rows])),
             total_training_seconds=sum(r['teacher_training_seconds']+r['student_training_seconds'] for r in rows),
             GPU_hours=sum(r['teacher_training_seconds']+r['student_training_seconds'] for r in rows)/3600,
             mean_inference_seconds=float(np.mean([r['inference_seconds'] for r in rows])),
             mean_peak_GPU_MB=float(np.mean([r['peak_GPU_MB'] for r in rows])),per_fold=per_fold))
report=root/'MULTI_TEACHER_3SEED_REPORT.md'
result.update(per_fold=per_fold,efficiency=rows,environment=environment,
              source_validation_matrices={f"{v['fold']}/seed_{v['seed']}":run.b2.read(out/v['fold']/f"seed_{v['seed']}"/'teacher_selection.json') for v in result['selection']})
for key in result['source_validation_matrices']:
    fold,seed=key.split('/')
    names=run.b2.read(out/fold/seed/'config.json')['teacher_source_order']
    result['source_validation_matrices'][key]['matrix']={names[c]:run.b2.read(out/fold/seed/f'T_{c}'/'cross_source_validation.json') for c in 'ABC'}
run.b2.write(root/'MULTI_TEACHER_3SEED_REPORT.json',result)
lines=['','## All Metrics','', '| Strategy | ARI | NMI | ACC | Macro-F1 | Balanced ACC | kNN@15 | Silhouette |',
       '| --- | --- | --- | --- | --- | --- | --- | --- |']
for strategy,values in result['overall'].items():
    lines.append('| '+strategy+' | '+' | '.join(f"{values[k]['mean']:.4f} +/- {values[k]['std']:.4f}" for k in keys)+' |')
lines+=['','## Teacher Selection','', '| Fold | Seed | Selected source | Student |', '| --- | --- | --- | --- |']
for v in result['selection']:lines.append(f"| {v['fold']} | {v['seed']} | {v['selected_source']} | {v['selected_student']} |")
lines+=['','## Cross-Source Validation','', 'Candidate-specific frozen initial scales are retained. This is the requested exact normalized validation objective, not target-based selection.']
for key,v in result['source_validation_matrices'].items():
    matrix=v['matrix'];names=list(matrix)
    lines+=['',f'### {key}','', '| Teacher | '+' | '.join(names)+' |','| --- | --- | --- | --- |']
    for name,cells in matrix.items():lines.append('| '+name+' | '+' | '.join(f"{cells[n]['total']:.6f}" for n in names)+' |')
lines+=['','## Gate Scope','',
        'The shared multi-source consolidation gate is retained as a diagnostic for each source-specific candidate, not a candidate exclusion criterion. All candidates are required by this explicit ablation; excluding poorly generalizing candidates would change the intervention. This does not authorize accepting the blocked Shared Full B2 control.',
        '', '## Efficiency','',f"Total training GPU hours: {sum(r['teacher_training_seconds']+r['student_training_seconds'] for r in rows)/3600:.4f}",
       '| Fold | Seed | Teacher | Train cells | Epochs | Steps | Teacher sec | Student sec | Params | Peak MB | Inference sec |',
       '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
for r in rows:lines.append(f"| {r['fold']} | {r['seed']} | {r['candidate']} | {r['teacher_cells']} | {r['teacher_epochs']} | {r['teacher_steps']} | {r['teacher_training_seconds']:.2f} | {r['student_training_seconds']:.2f} | {r['params']} | {r['peak_GPU_MB']:.2f} | {r['inference_seconds']:.4f} |")
report.write_text(report.read_text()+'\n'.join(lines)+'\n')
print('ARCHIVE_AUDIT_PASS',len(hashes),'checkpoint hashes')
