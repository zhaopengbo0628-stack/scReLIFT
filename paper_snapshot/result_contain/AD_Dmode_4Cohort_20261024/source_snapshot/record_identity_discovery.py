"""Add the recovered author labels without claiming unverified biological identities."""
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent


if __name__ == '__main__':
    cross = pd.read_csv(OUT/'data_audit/GSE268609_active_ident_crosswalk.csv')
    assert len(cross) == 39 and cross['orig.ident'].is_unique
    labels = cross.set_index('orig.ident').author_active_ident
    cells = pd.read_csv(OUT/'data_audit/GSE268609_active_identity_by_cell.csv.gz')
    assert len(cells) == 153530
    assert cells.groupby('orig.ident').author_active_ident.nunique().eq(1).all()
    # Only the literal author '-repeat' declaration is linked; clinical suffixes stay intact.
    cross['author_repeat_group'] = cross.author_active_ident.str.removesuffix('-repeat')
    cross['identity_link_status'] = 'AUTHOR_ACTIVE_IDENT_TO_TECHNICAL_GROUP_VERIFIED; BIOLOGICAL_DONOR_SEMANTICS_UNRESOLVED'
    cross['split_safety_group'] = cross.author_repeat_group.replace({'7426HA':'POTENTIAL_SHARED_7426','7426':'POTENTIAL_SHARED_7426'})
    cross['split_safety_group_is_verified_donor'] = False
    cross.to_csv(OUT/'data_audit/GSE268609_author_identity_groups.csv',index=False)
    original = pd.read_csv(OUT/'data_audit/GSE268609_donor_crosswalk.csv')
    original['author_active_ident'] = original['orig.ident'].map(labels)
    original['author_active_ident_evidence'] = 'GSE268609_seurat_object.rds active.ident named factor; exact barcode join; 153530/153530 cells'
    original['author_explicit_repeat_group'] = original.author_active_ident.str.removesuffix('-repeat')
    assert original.author_active_ident.notna().all() and original.donor_id.isna().all()
    original.to_csv(OUT/'data_audit/GSE268609_donor_crosswalk.csv',index=False,na_rep='NA')
    registry = pd.read_csv(OUT/'donor_registry.csv')
    b = registry.cohort.eq('GSE268609')
    registry.loc[b,'author_active_ident'] = registry.loc[b,'author_participant_id'].astype(int).map(labels)
    registry.loc[b,'identity_status'] = 'author_active_ident_recovered; original_subject_semantics_and_global_crosswalk_unresolved'
    registry.to_csv(OUT/'donor_registry.csv',index=False,na_rep='NA')
    report = dict(status='NEW_DIRECT_RDS_IDENTITY_EVIDENCE_BIOLOGICAL_MAPPING_STILL_UNRESOLVED',
        technical_groups=39, recovered_author_labels=39, exact_barcode_matches=153530,
        original_RDS='E:/my_research/RAWdata/AD/2/GSE268609_seurat_object.rds',
        original_RDS_prior_verified_sha256='8e16547850a7bfbac2580e5d5a19e9614a9c8b35330b5c21c2e74f973f767ec3',
        explicit_repeat_pairs=[dict(orig_ident=[5,23],labels=['7182','7182-repeat'],cells=9082),
                               dict(orig_ident=[14,25],labels=['7320','7320-repeat'],cells=12133)],
        unresolved_alias_pair=dict(orig_ident=[6,29],labels=['7426','7426HA'],cells=9513,
                                   action='keep_in_same_split_safety_group; do_not_assert_same_donor'),
        distinct_labels_after_literal_repeat_linking=37,
        independent_biological_donors=None,
        paper_38_vs_table_39_explained=False,
        clinical_conflicts_still_held=dict(orig_ident=[10,12],author_active_ident=['6991','7162'],
                                         conflict='GEO_MCI_vs_RDS_and_paper_AD'),
        shared_SEAAD_donor_confirmed=False, shared_SEAAD_donor_excluded=False,
        donor_probe_training_started=False,
        author_fields_still_needed=['orig.ident/SampleNumber/active.ident to deidentified biological subject ID',
            'explicit repeat/alias map including 7426 and 7426HA',
            '38-participant manuscript inclusion list and explanation of 39 technical groups',
            'confirmed diagnosis for 6991 and 7162',
            'biobank/study identifiers cross-referenceable with SEA-AD and other SOURCE/TARGET cohorts'])
    (OUT/'data_audit/GSE268609_identity_discovery.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
