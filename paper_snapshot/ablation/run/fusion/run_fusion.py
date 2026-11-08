"""Fixed fusion interventions over the unchanged corrected Full B2 stages."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, '/home/u202522802045/share/zpb')
import run_updated_b2_validated as shared
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from fusion import build_fusion

prep, b2 = shared.prep, shared.b2
BASE_TEACHER, BASE_LOAD = b2.Teacher, shared.load_sources
MEAN_ROOT = shared.OUT
ROOT = prep.ROOT / 'b2_fusion_ablation'
OUT = ROOT / 'results'
VARIANTS = ['Concat', 'Gate', 'Interaction']
FOLDS = [f'target_D{i}' for i in range(1, 5)]
VARIANT = None
LEGACY_RUNNER_HASH = 'e7d6127407b1b8f73deb339e6db057dd65aa43cec23b1ac333a5918456dc3164'
LEGACY_SHARED_HASH = '958327b72dcc020317b6183a2883c867a60a716e9955ed57da8ba71f818d1e92'


class FusionTeacher(BASE_TEACHER):
    def __init__(self, rdim, fdim):
        super().__init__(rdim, fdim)
        self.fusion = build_fusion(VARIANT.lower())

    def forward(self, rna, field):
        zr, zf = self.rna_encoder(rna), self.field_encoder(field)
        zt = self.fusion(zr, zf)
        return zr, zf, zt, self.rna_decoder(zt), self.field_decoder(zt)


def load_sources(fold):
    sources, mean, std, manifest = BASE_LOAD(fold)
    manifest['hashes'] = dict(manifest['hashes'])
    for p in [Path(__file__), ROOT/'fusion.py', Path(shared.__file__), Path(prep.__file__)]:
        manifest['hashes'][str(p)] = prep.sha(p)
    return sources, mean, std, manifest


def configure(variant):
    global VARIANT
    VARIANT = variant
    shared.OUT = OUT / variant
    shared.load_sources = load_sources
    b2.Teacher = FusionTeacher


def mean_references():
    rows = []
    benchmark_audit = b2.read(MEAN_ROOT/'STABILITY_MATRIX_VALIDITY_AUDIT.json')
    assert benchmark_audit.get('full_b2_updated_reference_ready') == 'YES'
    assert benchmark_audit.get('fatal_integrity_gates') == 'PASS'
    audit_folds = {item['fold']: item for item in benchmark_audit['folds']}
    for fold in FOLDS:
        for seed in shared.SEEDS:
            unit = MEAN_ROOT/fold/f'seed_{seed}'
            row = dict(fold=fold, seed=seed, reference=str(unit), status='MISSING')
            if (unit/'manifest.json').exists():
                manifest = b2.read(unit/'manifest.json')
                audit = b2.read(unit/'evaluation/source_teacher_audit.json')
                leakage = b2.read(unit/'leakage_audit.json')
                for name, digest in manifest['artifact_hashes'].items():
                    if name != 'logs/run.log':assert prep.sha(unit/name) == digest, str(unit/name)
                fold_audit = audit_folds[fold]
                latent_ok = all(
                    np.isfinite(item[key].get('variance', [])).all() and
                    np.any(np.asarray(item[key].get('variance', []), dtype=float) > 0)
                    for item in audit.get('source_rows', []) for key in ['z_R', 'z_F', 'z_T'])
                fatal_ok = (manifest.get('status') == 'COMPLETE' and
                            fold_audit.get('B_D_valid') == 'YES' and
                            fold_audit.get('F_valid') == 'YES' and
                            fold_audit.get('three_seeds_share_source_hashes') is True and
                            audit.get('target_read') is False and
                            audit.get('biological_labels_used') is False and
                            len(audit.get('source_rows', [])) == 3 and
                            latent_ok and
                            leakage.get('target_atac_loaded') is False and
                            leakage.get('target_hvg_fitted') is False and
                            leakage.get('target_pca_fitted') is False and
                            leakage.get('target_population_fit_before_embedding') is False and
                            leakage.get('target_checkpoint_selection') is False and
                            leakage.get('source_only_training') is True)
                row.update(status='AVAILABLE' if fatal_ok else 'FATAL_AUDIT_UNVERIFIED',
                           teacher_gate=audit.get('TEACHER_SHARED_ALIGNMENT_VALID'),
                           manifest_sha256=prep.sha(unit/'manifest.json'))
                if row['status']=='AVAILABLE':
                    row.update(metrics=b2.read(unit/'metrics.json'),timing=b2.read(unit/'timing.json'),
                               parameters=b2.read(unit/'parameter_count.json'))
            b2.write(OUT/'Mean'/fold/f'seed_{seed}'/'reference.json',row)
            rows.append(row)
    b2.write(ROOT/'MEAN_REFERENCE_STATUS.json',dict(rows=rows,complete=all(r['status']=='AVAILABLE' for r in rows),
                                                  redundant_mean_training=False))
    return rows


def preflight():
    audit=b2.read(prep.ROOT/'UPDATED_FOUR_DATASET_INDEPENDENCE_AUDIT.json')
    assert audit['FOUR_DATASET_INDEPENDENCE']=='PASS'
    for record in audit['datasets'].values():assert prep.sha(record['rna_path'])==record['rna_sha256']
    mean_references();rows=[]
    for variant in VARIANTS:
        configure(variant)
        shared.OUT.mkdir(parents=True,exist_ok=True)
        for name in ['validated_source_artifacts','validated_cache']:
            link=shared.OUT/name
            if not link.exists():link.symlink_to(MEAN_ROOT/name,target_is_directory=True)
            assert link.resolve()==(MEAN_ROOT/name).resolve()
        for fold in FOLDS:
            sources,mean,std,manifest=load_sources(fold)
            shared.target_cache(fold)
            b2.seed_all(20020130);base=BASE_TEACHER(len(mean),len(mean))
            b2.seed_all(20020130);model=FusionTeacher(len(mean),len(mean))
            for name,value in base.state_dict().items():assert torch.equal(value,model.state_dict()[name])
            x=torch.from_numpy(sources[0]['train']['rna'][:8]);f=torch.from_numpy(sources[0]['train']['field'][:8])
            losses=b2.teacher_losses(model,x,f);sum(losses).backward()
            assert all(torch.isfinite(v) for v in losses)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            student=shared.copy.deepcopy(model.rna_encoder).eval()
            assert torch.equal(student(x),model.rna_encoder.eval()(x))
            kd=torch.nn.functional.mse_loss(student(x),model(x,f)[2].detach());kd.backward();assert torch.isfinite(kd)
            rows.append(dict(variant=variant,fold=fold,genes=len(mean),fusion_params=sum(p.numel() for p in model.fusion.parameters()),
                             teacher_params=sum(p.numel() for p in model.parameters()),student_params=sum(p.numel() for p in student.parameters())))
        b2.write(shared.OUT/'_benchmark_meta/VALIDATED_STEP1_GATE.json',dict(status='PASS',
                 frozen_reference_sha256=prep.sha(shared.REFERENCE),wrapper_sha256=prep.sha(shared.__file__)))
    b2.write(ROOT/'PREFLIGHT.json',dict(status='PASS',rows=rows,runner_sha256=prep.sha(__file__),
             fusion_sha256=prep.sha(ROOT/'fusion.py'),wrapper_sha256=prep.sha(shared.__file__),
             unchanged_base_parameters_initialization=True,optimizer_step=False,
             teacher_gate='DIAGNOSTIC_WARNING_ONLY_FATAL_LATENT_CHECKS_UNCHANGED',
             run_tier='pre_final_explicit_3seeds'))
    print('FUSION_PREFLIGHT_PASS',flush=True)


def mechanism(unit, variant, fold):
    sources,mean,std,_=load_sources(fold)
    teacher_dir=unit/'checkpoints/teacher'
    teacher=FusionTeacher(len(mean),len(mean))
    best=torch.load(teacher_dir/'best.pt',map_location='cpu',weights_only=False)
    teacher.load_state_dict(best['model']);teacher.eval()
    history=b2.read(teacher_dir/'history.json')
    selected=next(r for r in history if r['epoch']==best['epoch'])
    value=dict(variant=variant,fold=fold,teacher_best_epoch=best['epoch'],teacher_validation_components=selected['val'],
               teacher_component_order=selected['component_order'],teacher_val_by_source=selected['val_by_source'],
               initial_scales=best['initial_scales'],teacher_epochs=history[-1]['epoch'])
    if (unit/'checkpoints/student/manifest.json').exists():
        value['student_KD_validation']=b2.read(unit/'checkpoints/student/manifest.json')['best_source_val_score']
    if variant=='Gate':
        directory=unit/'gate_analysis';directory.mkdir(exist_ok=True)
        for split in ['train','val']:
            blocks=[];per_source={}
            with torch.no_grad():
                for source in sources:
                    parts=[]
                    for start in range(0,len(source[split]['rna']),512):
                        r=torch.from_numpy(source[split]['rna'][start:start+512]);f=torch.from_numpy(source[split]['field'][start:start+512])
                        zr,zf=teacher.rna_encoder(r),teacher.field_encoder(f)
                        parts.append(torch.sigmoid(teacher.fusion.gate(torch.cat([zr,zf],dim=1))).numpy())
                    g=np.concatenate(parts);blocks.append(g);per_source[source['name']]=dict(mean=float(g.mean()),std=float(g.std()))
            gates=np.concatenate(blocks);np.save(directory/f'gate_values_source_{split}.npy',gates)
            shared.cell_order(sources,split,directory/f'gate_source_{split}_cell_order.tsv')
            value[f'gate_{split}']=dict(mean=float(gates.mean()),std=float(gates.std()),per_source=per_source,
                 per_dimension_mean=gates.mean(0).tolist(),per_dimension_std=gates.std(0).tolist())
        b2.write(directory/'metadata.json',dict(latent_dimension_indices=list(range(64)),target_gate_computed=False))
    b2.write(unit/'mechanism.json',value)


def run(variant,fold,seed):
    gate=b2.read(ROOT/'PREFLIGHT.json')
    assert gate['runner_sha256']==prep.sha(__file__) and gate['fusion_sha256']==prep.sha(ROOT/'fusion.py')
    configure(variant);unit=shared.OUT/fold/f'seed_{seed}'
    if (unit/'manifest.json').exists():
        manifest=b2.read(unit/'manifest.json')
        if manifest.get('status')=='COMPLETE':
            for rel,digest in manifest['artifact_hashes'].items():
                assert prep.sha(unit/rel)==digest, rel
            assert b2.read(unit/'metrics.json')['evaluator']==b2.EVALUATOR
            print('ALREADY_COMPLETE',variant,fold,seed,flush=True)
            return 0
        raise RuntimeError(f'Existing non-complete manifest must be reviewed: {unit}')
    unit.mkdir(parents=True,exist_ok=True)
    b2.write(unit/'fusion_config.json',dict(variant=variant,operator=str(build_fusion(variant.lower())),
             architecture_change='fusion_only',teacher_gate='DIAGNOSTIC_WARNING_ONLY_FATAL_LATENT_CHECKS_UNCHANGED',
             source_hashes=gate,FULL_B2_REFERENCE_UNMODIFIED=True))
    blocker=unit/'SCIENTIFIC_BLOCKER.json'
    legacy_teacher = blocker.exists() and (unit/'checkpoints/teacher/manifest.json').exists()
    original_train_stage=b2.train_stage
    original_validate_teacher=shared.validate_reusable_teacher
    legacy_record=None
    if legacy_teacher:
        sources,mean,_,source_manifest=load_sources(fold)
        current_inputs=dict(source_manifest['hashes'])
        current_inputs[str(shared.REFERENCE)]=prep.sha(shared.REFERENCE)
        teacher_dir=unit/'checkpoints/teacher'
        teacher_manifest=b2.read(teacher_dir/'manifest.json')
        old_inputs=teacher_manifest['input_hashes']
        assert set(old_inputs)==set(current_inputs)
        mismatches=[key for key in current_inputs if current_inputs[key]!=old_inputs[key]]
        expected_wrapper_changes={str(Path(__file__)),str(Path(shared.__file__))}
        assert set(mismatches)==expected_wrapper_changes, mismatches
        assert old_inputs[str(Path(__file__))]==LEGACY_RUNNER_HASH
        assert old_inputs[str(shared.__file__)]==LEGACY_SHARED_HASH
        assert current_inputs[str(shared.REFERENCE)]==prep.sha(shared.REFERENCE)
        cfg=teacher_dir/'config.json'
        assert b2.read(cfg)==dict(**b2.CONFIG,stage='teacher',seed=seed)
        assert teacher_manifest['config_hash']==prep.sha(cfg)
        for name,key in [('best.pt','best_checkpoint_sha256'),('latest.pt','latest_checkpoint_sha256')]:
            cp=teacher_dir/name
            assert prep.sha(cp)==teacher_manifest[key]
            state=torch.load(cp,map_location='cpu',weights_only=False)
            assert state['input_hashes']==old_inputs and state['config_hash']==prep.sha(cfg)
            assert all(torch.isfinite(t).all() for t in state['model'].values() if torch.is_tensor(t))
        assert b2.read(teacher_dir/'data_manifest.json')['input_hashes']==old_inputs
        legacy_record=dict(status='COMPATIBLE_FROZEN_TEACHER_REUSE',
            historical_runner_sha256=LEGACY_RUNNER_HASH,
            current_runner_sha256=prep.sha(__file__),
            historical_shared_wrapper_sha256=LEGACY_SHARED_HASH,
            current_shared_wrapper_sha256=prep.sha(shared.__file__),
            other_input_hashes_exact=True,teacher_config_hash=prep.sha(cfg),
            teacher_best_sha256=prep.sha(teacher_dir/'best.pt'),teacher_latest_sha256=prep.sha(teacher_dir/'latest.pt'),
            optimizer_updates_for_reused_teacher=0,student_retrained=True)
        b2.write(unit/'legacy_teacher_reuse_audit.json',legacy_record)
        def reuse_teacher(model,teacher_sources,stage_dir,stage_seed,device,is_teacher,inputs):
            if is_teacher and stage_dir.resolve()==teacher_dir.resolve():
                assert stage_seed==seed and len(teacher_sources)==3
                state=torch.load(teacher_dir/'best.pt',map_location=device,weights_only=False)
                model.load_state_dict(state['model'],strict=True)
                return model.eval()
            return original_train_stage(model,teacher_sources,stage_dir,stage_seed,device,is_teacher,inputs)
        def validate_reused_teacher(stage_unit,stage_fold,stage_seed,inputs):
            assert stage_unit.resolve()==unit.resolve() and stage_fold==fold and stage_seed==seed
            assert inputs==current_inputs and old_inputs[str(shared.REFERENCE)]==inputs[str(shared.REFERENCE)]
        b2.train_stage=reuse_teacher
        shared.validate_reusable_teacher=validate_reused_teacher
    try:shared.run(fold,seed,unit)
    except AssertionError:
        audit=unit/'evaluation/source_teacher_audit.json'
        if not audit.exists() or b2.read(audit).get('FATAL_GATES_MODE')!='FATAL':raise
        mechanism(unit,variant,fold)
        b2.write(unit/'SCIENTIFIC_BLOCKER.json',dict(status='SCIENTIFIC_BLOCKER',variant=variant,fold=fold,seed=seed,
                 reason='Fatal model/data integrity failure; Student not trained',teacher_audit_sha256=prep.sha(audit),
                 thresholds_changed=False,preserved_teacher_checkpoints={str(p.relative_to(unit)):prep.sha(p)
                                                                       for p in (unit/'checkpoints').rglob('*.pt')}))
        print('SCIENTIFIC_BLOCKER',variant,fold,seed,flush=True);return 20
    finally:
        b2.train_stage=original_train_stage
        shared.validate_reusable_teacher=original_validate_teacher
    mechanism(unit,variant,fold)
    b2.write(unit/'manifest.json',dict(status='COMPLETE',variant=variant,fold=fold,seed=seed,
             artifact_hashes={str(p.relative_to(unit)):prep.sha(p) for p in unit.rglob('*')
                             if p.is_file() and p.name not in ['manifest.json','run.log']},
             legacy_scientific_blocker_preserved=blocker.exists(),legacy_teacher_reuse=legacy_record,
             run_tier='pre_final_explicit_3seeds'))
    print('FUSION_UNIT_COMPLETE',variant,fold,seed,flush=True);return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--variant',choices=VARIANTS);parser.add_argument('--fold',choices=FOLDS)
    parser.add_argument('--seed',type=int,choices=shared.SEEDS);args=parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.dry_run:preflight()
        else:
            assert args.variant and args.fold and args.seed
            sys.exit(run(args.variant,args.fold,args.seed))
