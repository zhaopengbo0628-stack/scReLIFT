"""Posthoc evidence and checkpoint closure for Task38; no training or tuning."""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/home/u202522802045/share/zpb/result_contain/36_b2_three_causal_ablations_20261003/source/experiments/v5d')
import run_b2_strict_lodo as b


def average(values):
    return float(np.mean(values))


def finish(out):
    report = b.read(out/'FINAL_REPORT.json')
    sd=b.read(out/'B2_SD_ADJUDICATION.json')
    sd['status']='COMPLETED' if sd['METHOD_FREEZE']=='B2_SD' else 'COMPLETED_REJECT'
    sd['seed20020132_run']=False
    sd['rejection_reason']='seed31 D4 regression -0.051506 exceeds fixed -0.05 floor' if sd['METHOD_FREEZE']=='FULL_B2' and sd.get('ran') else None
    b.write(out/'B2_SD_ADJUDICATION.json',sd)
    report['B2_SD_RESULT']=sd
    old35=b.ROOT/'result_contain/35_b2_shared_regulatory_teacher_dense_kd_20261003'
    old36=b.ROOT/'result_contain/36_b2_three_causal_ablations_20261003'
    report['ADDITIONAL_EXISTING_5SEED_MATCHED_REFERENCE']=b.read(old36/'summary.json')['summaries']
    original=[]
    for seed in [20020130,20020131,20020132]:
        per_fold=[b.read(old35/'student'/fold/f'seed_{seed}'/'metrics.json') for fold in b.FOLDS]
        original.append(dict(seed=seed,**{m:average([r[m] for r in per_fold]) for m in ['ari','nmi']}))
    report['ORIGINAL_FIXED_TEACHER_B2_REFERENCE']=dict(teacher_seed=20020130,per_seed=original,
        mean={m:average([r[m] for r in original]) for m in ['ari','nmi']},
        std={m:float(np.std([r[m] for r in original],ddof=1)) for m in ['ari','nmi']},
        scope='not independent Teacher replication; SD uses this matched parent')
    with (out/'CORE_ABLATION_TABLE.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    for group, summary in report['CORE_ABLATION_TABLE'].items():
        for fold in b.FOLDS:
            values = [r for r in rows if r['group']==group and r['fold']==fold]
            summary['per_fold'][fold] = {m:dict(mean=average([float(r[m]) for r in values]),
                std=float(np.std([float(r[m]) for r in values],ddof=1)) if len(values)>1 else None,
                n_seeds=len(values)) for m in ['ari','nmi']}
    mechanisms = {}
    for group in ['FULL','RNA_ONLY','SHUFFLED_F','UNSCREENED']:
        units = [b.read(p) for p in (out/'mechanisms'/group).rglob('*.json')]
        source_rows = [r for u in units for r in u['rows']]
        mechanisms[group] = dict(units=len(units),source_observations=len(source_rows),
            latent_effective_ranks={k:average([r[k]['effective_rank'] for r in source_rows if r[k] is not None])
                if group!='RNA_ONLY' or k!='z_F' else None for k in ['z_R','z_F','z_T']},
            F_effective_rank=average([r['F_rank']['effective_rank'] for r in source_rows]),
            F_mean_variance=average([average(r['F_rank']['variance']) for r in source_rows]),
            RNA_reconstruction=average([r['RNA_reconstruction'] for r in source_rows]),
            F_reconstruction=average([r['F_reconstruction'] for r in source_rows]) if group!='RNA_ONLY' else None,
            paired_to_shuffled_MSE_ratio=average([r['same_cell_mse']/max(r['shuffled_cell_mse'],1e-12) for r in source_rows]) if group!='RNA_ONLY' else None,
            teacher_domain_balanced_accuracy=average([u['domain_balanced_accuracy']['teacher'] for u in units]),
            field_domain_balanced_accuracy=average([u['domain_balanced_accuracy']['field'] for u in units]),
            student_vs_fused={k:average([r['student_vs_fused'][k] for r in source_rows]) for k in source_rows[0]['student_vs_fused']},
            student_vs_RNA={k:average([r['student_vs_RNA'][k] for r in source_rows]) for k in source_rows[0]['student_vs_RNA']})
    report['SOURCE_MECHANISM_SUMMARY'] = mechanisms
    report['CONFIDENCE_ABSOLUTE_ERROR_CHECK'] = b.read(out/'CONFIDENCE_ABSOLUTE_ERROR_CHECK.json')
    report['MECHANISM_LIMITATIONS'] = [
        'F domain classifier is linear and observes separately SOURCE-standardized fields; chance accuracy does not exclude nonlinear domain structure.',
        'A3 retains reliable weights/signs: only hard Top50 selection is removed. No causal claim about removing all reliability estimation.',
        'SOURCE crossfit confidence is conditional on fixed outer Teacher and inherited probes, not fully nested Teacher LODO.',
        'Confidence gate uses preregistered relative correction error. High confidence has larger absolute error and correction energy; energy-dependent predictability is not excluded.',
        'Shuffled-F alignment measures trained broken pairs; it does not become true same-cell supervision.',
        'Existing paired cohort/QC and label-stratified splits inherited; target labels absent from new model fitting/selection.']
    for condition, claim in [('RNA_ONLY','CLAIM_2_REGULATORY_F_CONTRIBUTION'),('SHUFFLED_F','CLAIM_3_TRUE_PAIRING_CONTRIBUTION')]:
        c = report['paired_contrasts'][condition]
        seed_means = [average([r['ari'] for r in c['per_unit_FULL_minus_control'] if r['seed']==seed]) for seed in [20020130,20020131,20020132]]
        evidence = c['mean']['ari']>0 and c['mean']['nmi']>0 and all(v>0 for v in seed_means)
        if condition=='SHUFFLED_F':
            evidence = evidence and mechanisms['FULL']['paired_to_shuffled_MSE_ratio']<mechanisms[condition]['paired_to_shuffled_MSE_ratio']
        report[claim] = 'SUPPORTED' if evidence else 'INCONCLUSIVE'
    report['CLAIM_1_INTERPRETATION'] = 'Small mean advantage is inconsistent across seeds, NMI nearly unchanged, and SOURCE field statistics close; screening benefit not established.'
    c_rows = []
    for fold in b.FOLDS:
        for method, unit in [('C0',b.GOAL1/'parent'/fold/'seed_20020130'),('C1',out/'C1'/fold/'seed_20020130')]:
            diag = b.read(unit/'diagnostics.json')
            history = b.read(unit/'training.json')['history']
            gradients = diag['gradient_records']
            c_rows.append(dict(method=method,fold=fold,near_bound=diag['near_bound_fraction'],bound_hit=diag['bound_hit_fraction'],
                delta_over_R=diag['delta_over_scaffold_norm'],gradient_cosine_mean=average([g['cosine'] for g in gradients]),
                normalized_gradient_norm_ratio_median=float(np.median([g['normalized_global_over_local_norm'] for g in gradients])),
                epoch_first=history[0],epoch_last=history[-1],topology_peak=max(history,key=lambda r:r['val_topology_score_teacher_recall_at_10'])))
    report['C_CAPACITY_GRADIENT_TRAJECTORY'] = c_rows
    geometry_audit = b.read(out/'C_GEOMETRY_SOURCE_AUDIT.json')
    report['C_RAW_UNIT_SCALE_RATIO'] = [dict(fold=r['fold'],source=r['source'],
        student_over_teacher_norm=r['initial_student_norm_mean']/r['teacher_norm_mean']) for r in geometry_audit['records']]
    report['CLAIM_5_C_GEOMETRY'] = 'INCONCLUSIVE'
    report['C_CAUSAL_LIMITATION'] = 'FIX-B corrects raw-vs-normalized definition and preserves Teacher, but frozen RNA scaffold raw-coordinate magnitude remains mismatched. No main-cause adjudication follows from this ineffective correction; no second C variant authorized.'
    report['C_NORMALIZATION_PROVENANCE'] = 'Identical initial SOURCE scale-estimation rule; local/global numeric initial scales recalculated under changed geometry. No fixed numeric coefficients retuned.'
    units = list(out.glob('UNSCREENED/*/seed_*/teacher'))+list(out.glob('UNSCREENED/*/seed_*/student'))+list(out.glob('C1/*/seed_*'))+list(out.glob('B2_SD/*/seed_*/student'))
    checked = []
    for unit in units:
        c = unit.name.startswith('seed_')
        best = unit/('best_by_topology.pt' if c else 'best.pt')
        for path in [best,unit/'latest.pt',unit/'initial.pt']:
            assert path.exists(),path
            saved = torch.load(path,map_location='cpu',weights_only=False)
            assert 'optimizer' in saved and 'rng' in saved and 'epoch' in saved and 'input_hashes' in saved,path
            if path.name!='initial.pt':
                assert saved['config_hash']==b.sha(unit/'config.json'),path
        saved = torch.load(best,map_location='cpu',weights_only=False)
        resource_path = unit/('resources.json' if c else 'manifest.json')
        resources = b.read(resource_path)
        manifest = dict(status='COMPLETE',checkpoint_sha256=b.sha(best),latest_sha256=b.sha(unit/'latest.pt'),
            initial_sha256=b.sha(unit/'initial.pt'),best_epoch=saved['epoch'],input_hashes=saved['input_hashes'],
            config_sha256=b.sha(unit/'config.json'),resources=resources,target_selection=False)
        b.write(unit/'archive_provenance.json',manifest)
        checked.append(dict(unit=str(unit),best_checkpoint_sha256=b.sha(best),best_epoch=saved['epoch']))
    with (out/'ABLATION_REUSE_AUDIT.csv').open() as handle:
        audit = list(csv.DictReader(handle))
    seen={(r['group'],r['fold'],int(r['seed'])) for r in audit}
    extras=[(condition,fold,seed,old36/condition/fold/f'seed_{seed}',True)
        for condition in ['FULL','RNA_ONLY','SHUFFLED_F'] for fold in b.FOLDS for seed in [20020133,20020134]]
    extras += [('FULL_FIXED_TEACHER_REFERENCE',fold,seed,old35/'student'/fold/f'seed_{seed}',False)
        for fold in b.FOLDS for seed in [20020130,20020131,20020132]]
    for group,fold,seed,unit,independent in extras:
        if (group,fold,seed) in seen:continue
        metric=b.read(unit/'metrics.json')
        ep=unit/'z_target_test.npy';cp=unit/('student/best.pt' if independent else 'best.pt')
        assert b.sha(ep)==metric['embedding_sha256'] and metric['evaluator']==b.EVALUATOR
        audit.append(dict(group=group,fold=fold,seed=seed,artifact=str(unit),status='EXACT_REUSE',
            reason='additional existing independent 5seed evidence' if independent else 'original fixed Teacher30 reference, separately reported',
            checkpoint_hash=b.sha(cp),embedding_hash=b.sha(ep),action='reuse reference, no retraining'))
    with (out/'ABLATION_REUSE_AUDIT.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(audit[0]));writer.writeheader();writer.writerows(audit)
    report['REUSE_AUDIT_SUMMARY']={s:sum(r['status']==s for r in audit) for s in sorted({r['status'] for r in audit})}
    b.write(out/'REUSE_SUMMARY.json',report['REUSE_AUDIT_SUMMARY'])
    for row in audit:
        if row['status']!='EXACT_REUSE':continue
        unit=Path(row['artifact'])
        embedding=unit/('predicted_atac.npy' if row['group']=='scButterfly' else 'z_target_test.npy')
        assert b.sha(embedding)==row['embedding_hash'],embedding
        if row['checkpoint_hash']!='NA':
            assert any(b.sha(p)==row['checkpoint_hash'] for p in unit.rglob('*.pt')),unit
    archive=out/'source'
    archive.mkdir(exist_ok=True)
    protocol=b.read(out/'protocol.json')
    for path,digest in protocol['code_hashes'].items():
        assert b.sha(path)==digest,path
        shutil.copy2(path,archive/Path(path).name)
    shutil.copy2(__file__,archive/Path(__file__).name)
    report['ARCHIVE_CLOSURE']=dict(status='COMPLETE',new_training_units=len(checked),checkpoints_verified=checked,
        reused_units_reverified=sum(r['status']=='EXACT_REUSE' for r in audit),optimizer_rng_config_input_checks='PASS')
    b.write(out/'FINAL_REPORT.json',report)
    b.write(out/'metrics.json',report['CORE_ABLATION_TABLE'])
    b.write(out/'MECHANISM_REPORT.json',dict(source=mechanisms,C_capacity_gradient_trajectory=c_rows,limitations=report['MECHANISM_LIMITATIONS']))
    lines=['# Task38 Final Adjudication','','Strict SOURCE-only inference; D1-D4 development benchmark; user-declared three-seed pre-final, not 15-seed validation.','',
           '| Condition | Seeds | ARI mean | ARI std | NMI mean | NMI std |','| --- | --- | --- | --- | --- | --- |']
    for group,s in report['CORE_ABLATION_TABLE'].items():
        lines.append(f"| {group} | {len(s['seeds'])} | {s['mean']['ari']:.6f} | {s['std']['ari']} | {s['mean']['nmi']:.6f} | {s['std']['nmi']} |")
    lines+=['',json.dumps({k:report[k] for k in report if k.startswith('CLAIM_') or k in ['METHOD_FREEZE','NEXT_EXTERNAL_VALIDATION_READY']},indent=2),
            '',report['C_NORMALIZATION_PROVENANCE'],'',*report['MECHANISM_LIMITATIONS'],'','External dataset not launched. STOP.']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    b.write(out/'CHECKPOINT_INDEX.json',{str(path):b.sha(path) for path in out.rglob('*.pt')})
    b.write(out/'ARCHIVE_FILES_SHA256.json',{str(path.relative_to(out)):b.sha(path) for path in out.rglob('*') if path.is_file() and path.name!='ARCHIVE_FILES_SHA256.json'})
    state=b.read(b.ROOT/'CURRENT_RESEARCH_STATE.json')
    state['B2_COMPLETE_CONTROLS_Task38'].update(report)
    state['B2_COMPLETE_CONTROLS_Task38']['archive_closure']={'status':'SERVER_COMPLETE_LOCAL_PENDING'}
    state['current_main_model']=report['METHOD_FREEZE']
    state['current_primary_bottleneck']='UNTOUCHED_EXTERNAL_GENERALIZATION_UNVALIDATED'
    state['current_secondary_bottleneck']='SCREENING_CAUSAL_CONTRIBUTION_INCONCLUSIVE'
    state['next_action']='STOP; frozen method requires untouched external validation, not launched'
    b.write(b.ROOT/'CURRENT_RESEARCH_STATE.json',state)
    with (b.ROOT/'RESEARCH_DECISION_LEDGER.md').open('a') as handle:
        handle.write('\n- Task38 mechanism/archive closure: '+json.dumps({k:report[k] for k in ['CLAIM_1_RELIABILITY_SCREENING',
            'CLAIM_2_REGULATORY_F_CONTRIBUTION','CLAIM_3_TRUE_PAIRING_CONTRIBUTION','CLAIM_5_C_GEOMETRY','METHOD_FREEZE']})+'\n'
            '- SOURCE mechanism quantities and C residual/gradient/trajectory retained. All new best/latest/initial optimizer and RNG verified.\n')
    print('ARCHIVE_COMPLETE',len(checked),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    finish(parser.parse_args().out)
