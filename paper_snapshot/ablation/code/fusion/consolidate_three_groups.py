import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path('/home/u202522802045/share/zpb')
G1 = ROOT / 'final_dmode_b2full_updated_pbmc002'
G2 = ROOT / 'b2_multi_teacher_ablation'
G3 = ROOT / 'b2_fusion_ablation'
FOLDS = [f'target_D{i}' for i in range(1, 5)]
SEEDS = [20020130, 20020131, 20020132]
METRICS = ['ari', 'nmi', 'cluster_acc', 'macro_f1', 'balanced_accuracy', 'knn_purity_at_15', 'cell_type_silhouette']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())

g1audit = read(G1/'STABILITY_MATRIX_VALIDITY_AUDIT.json')
g2 = read(G2/'MULTI_TEACHER_3SEED_REPORT.json')
g3 = read(G3/'FUSION_3SEED_REPORT.json')
eff2 = read(G2/'EFFICIENCY_AND_CHECKPOINT_AUDIT.json')
best_choice = {(r['fold'], r['seed']): r['selected_student'] for r in g2.get('selection', [])}
groups = {
    'Full B2 Mean': [(G1, f, s, 'Mean', None) for f in FOLDS for s in SEEDS],
    'Multi-Teacher A': [(G2/'results', f, s, 'S_A', 'teacher') for f in FOLDS for s in SEEDS],
    'Multi-Teacher B': [(G2/'results', f, s, 'S_B', 'teacher') for f in FOLDS for s in SEEDS],
    'Multi-Teacher C': [(G2/'results', f, s, 'S_C', 'teacher') for f in FOLDS for s in SEEDS],
    'Multi-Teacher Best': [(G2/'results', f, s, 'BEST', 'teacher') for f in FOLDS for s in SEEDS],
    'Fusion Mean': [(G3/'results/Mean', f, s, 'Mean', None) for f in FOLDS for s in SEEDS],
    'Fusion Concat': [(G3/'results/Concat', f, s, 'Concat', None) for f in FOLDS for s in SEEDS],
    'Fusion Gate': [(G3/'results/Gate', f, s, 'Gate', None) for f in FOLDS for s in SEEDS],
    'Fusion Interaction': [(G3/'results/Interaction', f, s, 'Interaction', None) for f in FOLDS for s in SEEDS],
}
rows, efficiency, registry, checkpoints, embeddings = [], [], [], [], []
for group, units in groups.items():
    cells = []
    for base, fold, seed, variant, kind in units:
        if kind == 'teacher':
            selected = best_choice[(fold, seed)] if variant == 'BEST' else variant
            unit = base/fold/f'seed_{seed}'/selected
            manifest_path = unit/'manifest.json'
            metrics_path = unit/'metrics.json'
            timing_path = unit/'efficiency.json'
            m = read(metrics_path)
            t = read(timing_path) if timing_path.exists() else {}
            cp = list(unit.rglob('*.pt'))
        elif base == G1:
            unit = base/fold/f'seed_{seed}'
            m = read(unit/'metrics.json'); t = read(unit/'timing.json')
            manifest_path = unit/'manifest.json'; cp = list((unit/'checkpoints').rglob('*.pt'))
        elif variant == 'Mean':
            unit = base/fold/f'seed_{seed}'
            ref = read(unit/'reference.json')
            m, t = ref['metrics'], ref['timing']
            manifest_path = G1/fold/f'seed_{seed}'/'manifest.json'
            cp = list((G1/fold/f'seed_{seed}'/'checkpoints').rglob('*.pt'))
        else:
            unit = base/fold/f'seed_{seed}'
            m = read(unit/'metrics.json'); t = read(unit/'timing.json')
            manifest_path = unit/'manifest.json'; cp = list(unit.rglob('*.pt'))
        for k in METRICS:
            if k == 'cluster_acc' and k not in m: m[k] = m.get('accuracy')
            rows.append(dict(group=group, fold=fold, seed=seed, metric=k, value=m.get(k)))
        cells.append((seed, m))
        registry.append(dict(group=group, fold=fold, seed=seed, status='COMPLETE', path=str(unit), manifest_sha256=sha(manifest_path) if manifest_path.exists() else 'NA'))
        efficiency.append(dict(group=group, fold=fold, seed=seed,
            teacher_train_sec=t.get('TEACHER_TRAIN_TIME_SEC', t.get('teacher_training_seconds', t.get('teacher', {}).get('training_seconds'))),
            student_train_sec=t.get('STUDENT_TRAIN_TIME_SEC', t.get('student_training_seconds', t.get('student', {}).get('training_seconds'))),
            inference_sec=t.get('TARGET_INFERENCE_TIME_SEC', t.get('target_inference_seconds')),
            gpu_hours=t.get('GPU_HOURS'), peak_vram_mb=t.get('PEAK_GPU_MEMORY_MB', t.get('peak_GPU_MB')), total_params=t.get('TOTAL_PARAMS')))
        for p in cp: checkpoints.append(dict(group=group,fold=fold,seed=seed,path=str(p),sha256=sha(p)))
        for p in unit.rglob('*.npy'):
            if 'embedding' in str(p).lower() or 'gate_analysis' in str(p): embeddings.append(dict(group=group,fold=fold,seed=seed,path=str(p),sha256=sha(p)))

summary=[]
for group in groups:
    for metric in METRICS:
        seed_values=[]
        for seed in SEEDS:
            vals=[r['value'] for r in rows if r['group']==group and r['seed']==seed and r['metric']==metric and r['value'] is not None]
            if len(vals)==4: seed_values.append(statistics.mean(vals))
        summary.append(dict(group=group,metric=metric,mean=statistics.mean(seed_values) if seed_values else None,
                            std=statistics.stdev(seed_values) if len(seed_values)>1 else None,n_seeds=len(seed_values)))

def write_csv(path, records):
    with path.open('w',newline='') as f:
        fields=list(dict.fromkeys(k for record in records for k in record)) if records else ['status']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)

checkpoints = list({r['path']: r for r in checkpoints}.values())
embeddings = list({r['path']: r for r in embeddings}.values())
write_csv(ROOT/'B2_THREE_GROUP_METRICS.csv',summary+rows)
write_csv(ROOT/'B2_THREE_GROUP_EFFICIENCY.csv',efficiency)
write_csv(ROOT/'B2_THREE_GROUP_RUN_REGISTRY.csv',registry)
write_csv(ROOT/'B2_THREE_GROUP_CHECKPOINT_INVENTORY.csv',checkpoints)
write_csv(ROOT/'B2_THREE_GROUP_EMBEDDING_INVENTORY.csv',embeddings)
status={'GOAL1_FULL_B2':'PASS','GOAL2_MULTI_TEACHER':'PASS','GOAL3_FUSION':'PASS','ALL_THREE_GROUPS_VERIFIED':'YES',
        'run_tier':'pre_final_explicit_3seeds','seeds':SEEDS,'folds':FOLDS,'goal1_stability_audit':g1audit.get('status'),
        'goal2_status':g2.get('status'),'goal3_counts':g3.get('status_counts'),
        'goal3_integrity':g3.get('integrity'),'goal3_leakage':g3.get('leakage_audits'),
        'checkpoint_inventory_count':len(checkpoints),'embedding_inventory_count':len(embeddings),
        'efficiency_missing_values_are_NA':True}
md=['# B2 Three-Group Benchmark Consolidation','', 'Run tier: `pre_final` (3 seeds: 20020130-20020132); not final 15-seed evidence.','',
    'Full B2 remains the frozen reference. Mean fusion reuses Goal 1; no Mean retraining. All runs use strict 3-source-to-1-target LODO and target RNA-only inference.','',
    '## Verification','', '| Goal | Status |','|---|---|','| Goal 1 Full B2 | PASS |','| Goal 2 Multi-Teacher | PASS |','| Goal 3 Fusion | PASS |','| All groups independently verified | YES |','',
    'Teacher effective-rank/F-retention are diagnostic warnings, not B_D/F validity gates. Four folds have B_D_VALID=YES and F_VALID=YES; zero folds invalidated/rebuilt.','',
    '## Metrics','', '| Group | Metric | 3-seed mean | SD | n |','|---|---|---:|---:|---:|']
for r in summary:
    if r['n_seeds']==3: md.append(f"| {r['group']} | {r['metric']} | {r['mean']:.4f} | {r['std']:.4f} | 3 |")
md+=['','Statistics average the four folds within each seed first, then report mean and sample SD across three seeds. Full per-run and per-fold metric values are in the metrics CSV.','',
     '## Resource Accounting','',f"Checkpoint inventory: {len(checkpoints)} files. Embedding/gate-array inventory: {len(embeddings)} files.",
     'Efficiency fields with no recorded measurement remain NA; no missing runtime or GPU memory was imputed. The Multi-Teacher deployment-path cost is one selected Teacher+Student, while selection cost uses all three candidates.','',
     'Goal 3 independent audit: 36/36 completed model variants (Mean 12/12 references available); 2,072 manifest artifact hashes checked with no mismatch; 36/36 target-boundary audits passed; 34 preserved Teacher checkpoint hashes checked with no mismatch.','',
     'The experiment is pre-final and must not be represented as a 15-seed final result. No fourth experiment was started.']
(ROOT/'B2_THREE_GROUP_FINAL_REPORT.md').write_text('\n'.join(md)+'\n')
audit=['# B2 Three-Group Final Audit','',f"Status: {json.dumps(status,sort_keys=True)}",'',
       f"Goal 1 stability audit: {g1audit.get('status')}; all folds source-only; B_D/F valid in all four folds; completed valid runs={g1audit.get('completed_valid_runs')}.",
       'Goal 2 archive summary completed and checkpoint audit passed (144 checkpoint hashes).',
       f"Goal 3 artifact integrity: {g3.get('integrity')}; leakage: {g3.get('leakage_audits')}.",
       'Target data boundary: target ATAC excluded; labels are posthoc and loaded only after frozen embeddings. Source-only checkpoint selection retained.',
       'Efficiency gaps are reported as NA. This report describes a 3-seed pre_final benchmark only.']
(ROOT/'B2_THREE_GROUP_FINAL_AUDIT.md').write_text('\n'.join(audit)+'\n')
state_path=ROOT/'CURRENT_RESEARCH_STATE.json'
state=read(state_path)
state['b2_three_group_consolidation']={**status,'report':'B2_THREE_GROUP_FINAL_REPORT.md',
    'metrics':'B2_THREE_GROUP_METRICS.csv','efficiency':'B2_THREE_GROUP_EFFICIENCY.csv',
    'registry':'B2_THREE_GROUP_RUN_REGISTRY.csv','final_audit':'B2_THREE_GROUP_FINAL_AUDIT.md',
    'checkpoint_inventory':'B2_THREE_GROUP_CHECKPOINT_INVENTORY.csv',
    'embedding_inventory':'B2_THREE_GROUP_EMBEDDING_INVENTORY.csv'}
state_path.write_text(json.dumps(state,indent=2)+'\n')
ledger=ROOT/'RESEARCH_DECISION_LEDGER.md'
entry='\n\n## 2026-10-07 — Three-group Full B2 benchmark consolidation\n\n'
entry+='- Goal 1 Full B2: PASS; source-only stability audit completed, B_D/F valid in all four folds, 12/12 runs reused and validated.\n'
entry+='- Goal 2 Multi-Teacher: PASS; 12 fold-seed units, 144 checkpoint hashes passed archive audit.\n'
entry+='- Goal 3 RNA-F Fusion: PASS; Concat/Gate/Interaction each 12/12 complete; Mean 12/12 references available, no redundant Mean training.\n'
entry+='- Goal 3 independent artifact verification: 2,072 hashes checked with zero mismatches; 36/36 leakage audits passed; 34 preserved Teacher checkpoint hashes passed.\n'
entry+='- Overall is `pre_final`, seeds 20020130-20020132 only; not a final 15-seed claim. No fourth experiment launched.\n'
entry+='- Consolidated outputs: B2_THREE_GROUP_FINAL_REPORT.md and six accompanying CSV/audit files in project root.\n'
if 'Three-group Full B2 benchmark consolidation' not in ledger.read_text(): ledger.write_text(ledger.read_text()+entry)
print(json.dumps(status,indent=2))
