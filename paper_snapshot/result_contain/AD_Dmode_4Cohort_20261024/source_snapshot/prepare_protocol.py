"""Create the new four-cohort registry and enforce the fixed SEA-AD eligibility rule."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('E:/my_research')
OUT = Path(__file__).resolve().parent.parent
OLD = ROOT/'result_contain/AD_Dmode_Goal_20261024/rev02_three_source_seaad_holdout'
TARGET = ROOT/'data/downloads/GSE272082_original_20261024'
SERVER = '/home/u202522802045/share/zpb'


def main():
    if (OUT/'metadata/protocol.json').exists():
        raise FileExistsError('Preserve the protocol; inspect before making a revision')
    registry = pd.read_csv(OLD/'donor_registry.csv', dtype={'author_participant_id': str})
    registry['model_partition'] = registry.cohort.map({
        'GSE214979': 'SOURCE1', 'GSE268609': 'SOURCE2_IDENTITY_HOLD', 'SEAAD_MTG': 'SOURCE3'})
    registry['label_endpoint'] = registry.cohort.map({
        'GSE214979': 'author_clinical_AD_vs_Control',
        'GSE268609': 'confirmed_AD_vs_HA_IDENTITY_UNRESOLVED',
        'SEAAD_MTG': 'ADNC_IntermediateHigh_vs_NotAD'})
    registry['binary_label'] = np.nan
    a = registry.cohort.eq('GSE214979')
    registry.loc[a, 'binary_label'] = registry.loc[a, 'author_clinical_group'].map({'AD': 1, 'Ctrl': 0, 'Control': 0})
    sea = registry.cohort.eq('SEAAD_MTG')
    registry.loc[sea, 'binary_label'] = registry.loc[sea, 'ADNC'].map({'Not AD': 0, 'Intermediate': 1, 'High': 1})
    registry['representation_eligibility'] = 'ELIGIBLE_COUNTS_VERIFIED'
    registry.loc[registry.cohort.eq('GSE268609'), 'representation_eligibility'] = 'HOLD_TRUE_DONOR_VALIDATION_AND_OVERLAP'
    registry.loc[sea, 'representation_eligibility'] = 'ELIGIBLE_PENDING_PAIRED_COUNTS'
    low = sea & registry.ADNC.eq('Low')
    registry.loc[low, 'representation_eligibility'] = 'EXCLUDED_LOW_ADNC_ALL_CELLS'
    registry.loc[low, 'model_partition'] = 'EXCLUDED'
    assert low.sum() == 4
    assert registry.loc[sea & ~low, 'binary_label'].value_counts().to_dict() == {1.0: 21, 0.0: 3}

    samples = pd.read_csv(TARGET/'manifests/server_verified/sample_manifest.csv')
    crosswalk = pd.read_csv(TARGET/'metadata/author_donor_sample_crosswalk.csv')
    check = samples.merge(crosswalk[['library_id', 'author_Individual_ID']], on='library_id', validate='one_to_one')
    assert (check.donor_id == check.author_Individual_ID_y).all()
    assert len(samples) == 21 and samples.donor_id.nunique() == 9
    target_records = []
    for donor, g in samples.groupby('donor_id'):
        for field in ['author_clinical_diagnosis', 'age', 'sex']:
            assert g[field].nunique() == 1
        target_records.append(dict(cohort='GSE272082', registry_id='GSE272082:'+donor,
            author_participant_id=donor, verified_donor_id=donor,
            identity_status='author_Data_S1_explicit_mapping', cells=int(g.barcodes.sum()),
            author_clinical_group=g.author_clinical_diagnosis.iloc[0],
            clinical_diagnosis=g.author_clinical_diagnosis.iloc[0],
            binary_label={'Control': 0, 'EOAD': 1}[g.author_clinical_diagnosis.iloc[0]],
            label_endpoint='sporadic_EOAD_vs_Control', age=g.age.iloc[0], sex=g.sex.iloc[0],
            PMI=g.PMI_hours.iloc[0], repository=g.tissue_source.iloc[0],
            library_ids='|'.join(g.library_id), brain_region='|'.join(sorted(g.tissue.unique())),
            technology='10x_single_nucleus_Multiome',
            disease_evidence='author Data S1; UT04/UT09 GEO title conflicts retained in sample manifest',
            cross_study_identity_status='no_confirmed_match; no_global_crosswalk_available',
            model_partition='EXTERNAL_TARGET4', representation_eligibility='TARGET_AUTHOR_QC_PENDING'))
    registry = pd.concat([registry, pd.DataFrame(target_records)], ignore_index=True)
    registry.to_csv(OUT/'donor_registry.csv', index=False, na_rep='NA')

    eligibility = registry[['registry_id','cohort','model_partition','representation_eligibility',
                            'binary_label','label_endpoint','identity_status']].copy()
    eligibility['probe_eligible'] = eligibility.binary_label.notna()
    eligibility.loc[eligibility.cohort.eq('GSE268609'), 'probe_eligible'] = False
    eligibility['reason'] = eligibility.representation_eligibility
    eligibility.to_csv(OUT/'clinical_eligibility.csv', index=False, na_rep='NA')
    excluded = registry.loc[low.reindex(registry.index, fill_value=False)].copy()
    excluded['reason'] = 'User-fixed rule: exclude all Low ADNC cells from SOURCE3 training'
    excluded['exclusion_scope'] = 'representation_and_probe_all_cells'
    prior_eligibility = pd.read_csv(OLD/'clinical_eligibility.csv')
    outside = prior_eligibility[prior_eligibility.cohort.eq('GSE268609') & prior_eligibility.confirmed_excluded]
    b_excluded = registry[registry.registry_id.isin(outside.registry_id)].merge(
        outside[['registry_id','reason']],on='registry_id',validate='one_to_one')
    b_excluded['exclusion_scope'] = 'disease_probe_only; unsupervised_identity_gate_separate'
    excluded = pd.concat([excluded,b_excluded],ignore_index=True)
    excluded.to_csv(OUT/'excluded_donors.csv', index=False, na_rep='NA')

    cell_path = OLD/'clinical_evidence/SEAAD_paired_clinical_cells.csv.gz'
    cells = pd.read_csv(cell_path, usecols=['sample_id','Donor ID','Overall AD neuropathological Change'])
    # This cell table's row order is not assumed to be the export row order.
    cells['include_source3'] = ~cells['Overall AD neuropathological Change'].eq('Low')
    cells['binary_label'] = cells['Overall AD neuropathological Change'].map({'Not AD':0,'Intermediate':1,'High':1})
    assert set(cells['Donor ID']) == set(registry.loc[sea.reindex(registry.index, fill_value=False), 'verified_donor_id'])
    assert len(cells) == 133084 and cells.sample_id.is_unique
    included = cells[cells.include_source3]
    assert included['Donor ID'].nunique() == 24
    assert not included.binary_label.isna().any()
    cells.to_csv(OUT/'metadata/seaad_cell_eligibility_by_sample_id.csv.gz', index=False)
    cells.groupby(['Donor ID','Overall AD neuropathological Change','include_source3'], dropna=False).size().rename('paired_cells').to_csv(OUT/'qc/seaad_donor_eligibility_counts.csv')
    samples.to_csv(OUT/'metadata/target_libraries.csv', index=False)
    samples.groupby(['donor_id','tissue']).barcodes.sum().to_csv(OUT/'qc/target_prefinal_qc_cells_by_donor_region.csv')
    pfc = samples[samples.tissue.eq('Prefrontal cortex')]
    assert pfc.donor_id.nunique() == 9

    for name in ['GSE268609_donor_crosswalk.csv','GSE268609_DONOR_ID_AUDIT.md']:
        shutil.copyfile(OLD/name, OUT/'data_audit'/name)
    mapping = ROOT/'data/benchmark/AD_Dmode_gene_overlap_audit_20261024/celltype_author_mapping.csv'
    shutil.copyfile(mapping, OUT/'celltype_mapping.csv')
    labels = pd.DataFrame([
        dict(cohort='GSE214979', endpoint='author_clinical_AD_vs_Control', negative='Ctrl/Control', positive='AD', protocol_A=True, protocol_B=True, status='within_study_verified'),
        dict(cohort='GSE268609', endpoint='confirmed_AD_vs_HA', negative='HA', positive='AD', protocol_A=False, protocol_B=False, status='UNRESOLVED_real_donor_and_10_12_clinical_conflicts'),
        dict(cohort='SEAAD_MTG', endpoint='ADNC_IntermediateHigh_vs_NotAD', negative='Not AD', positive='Intermediate/High', protocol_A=True, protocol_B=False, status='Low_excluded_not_dementia_label'),
        dict(cohort='GSE272082', endpoint='sporadic_EOAD_vs_Control', negative='Control', positive='EOAD', protocol_A='external_test', protocol_B='external_test', status='author_Data_S1_verified')])
    labels.to_csv(OUT/'disease_label_protocol.csv', index=False)
    config = dict(status='PREPARATION_IN_PROGRESS_SCIENTIFIC_GATES_PENDING', seeds=[20020130,20020131,20020132],
        sample_standard_deviation_ddof=1, sources=['GSE214979','GSE268609','SEAAD_MTG'], target='GSE272082',
        prior_seaad_holdout_split_used=False, source_train_val_split_frozen=False,
        SEAAD_endpoint='ADNC_IntermediateHigh_vs_NotAD', SEAAD_excluded_ADNC=['Low'],
        SEAAD_eligible_donors=24, SEAAD_eligible_metadata_cells=int(len(included)),
        SEAAD_excluded_metadata_cells=int((~cells.include_source3).sum()),
        hvg='each_SOURCE_TRAIN_top2000_intersection; donor_balanced_training_statistics',
        hvg_fallback='not_invoked; diagnose_SOURCE_only_before_any_revision',
        student_loss='embedding_MSE', student_decoder=False, student_disease_supervision=False,
        source_balanced_training=True, validation='real_donor_disjoint_SOURCE_only',
        p2_types=['Excitatory neurons','Inhibitory neurons','Astrocytes','Oligodendrocytes','OPC'],
        p2_missing_type='SOURCE_training_donor_mean_for_type; unavailable_type_count_reported; no_TARGET_fitting',
        target_primary_region='Prefrontal cortex', target_secondary_regions=['Entorhinal cortex','Hippocampus'],
        multi_region_pooling='equal_region_mean_of_Donor_x_Region_x_Type_means_then_concatenate_types',
        target_used_for_fit=False, external_atac_used=False,
        gpu_training_started=False, target_prediction_started=False,
        blockers=['SEAAD_complete_paired_counts_pending','GSE268609_real_donor_identity_unresolved',
                  'TARGET_author_QC_and_celltype_annotation_pending'],
        user_request_sha256=hashlib.sha256((OUT/'source_snapshot/USER_REQUEST.txt').read_bytes()).hexdigest(),
        goal_tool_status='new_objective_rejected_because_old_goal_unfinished; old_goal_not_falsely_completed',
        result_server_path=SERVER+'/result_contain/AD_Dmode_4Cohort_20261024')
    (OUT/'metadata/protocol.json').write_text(json.dumps(config, indent=2)+'\n')
    print(json.dumps(config, indent=2))


if __name__ == '__main__':
    main()
