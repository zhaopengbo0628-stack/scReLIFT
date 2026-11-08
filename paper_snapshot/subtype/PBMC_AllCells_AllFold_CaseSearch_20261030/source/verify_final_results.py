"""Numerical and membership verification of completed all-fold analysis."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analyses/D1/source'))
from analysis_common import bh,sha
Q='FDR_BH_all_folds_all_seeds'

def main():
    assert (ROOT/'audit/FUNCTIONAL_BH_REPAIR_COMPLETE.json').exists()
    inv=pd.read_csv(ROOT/'candidate_inventory.csv');ev=pd.read_csv(ROOT/'candidate_seed_evidence_matrix.csv')
    de=pd.read_csv(ROOT/'all_candidate_deg_summary.csv')
    assert len(inv)==82 and inv.candidate_id.is_unique and len(ev)==179
    assert ev[['candidate_id','seed']].duplicated().sum()==0
    assert set(inv.candidate_id)==set(ev.candidate_id)==set(de.candidate_id)
    assert len(de)==537 and (de.tested_genes==36601).all()
    assert set(de.contrast)=={'A_vs_B','A_vs_subclass_rest','B_vs_subclass_rest'}
    assert (de.groupby(['candidate_id','seed']).size()==3).all()
    gp=[];gq=[]
    for r in de.itertuples():
        d=pd.read_csv(r.allfold_global_file)
        assert len(d)==36601 and d.gene_id.is_unique
        assert np.isfinite(d.p_value).all() and d.p_value.between(0,1).all()
        assert sha(Path(r.allfold_global_file))==r.allfold_global_sha256
        assert int(((d[Q]<.05)&(d.log2FC_meanCP10K.abs()>.25)).sum())==r.allfold_strict_effect025
        gp.append(d.p_value.to_numpy());gq.append(d[Q].to_numpy())
    np.testing.assert_allclose(bh(np.concatenate(gp)),np.concatenate(gq),atol=1e-13,rtol=1e-12)
    del gp,gq
    pathway=[]
    for scope in ['GSEA_full','GSEA_expressed','ORA']:
        pcol='nominal_p' if scope=='ORA' else 'nominal_p_floored'
        pp=[];qq=[];invalid=0;comps=set();sig=0
        for d in pd.read_csv(ROOT/f'all_fold_all_seed_{scope}.csv.gz',chunksize=30000):
            assert np.isfinite(d[Q]).all() and d[Q].between(0,1).all()
            raw=d[pcol].to_numpy();use=np.where(np.isfinite(raw)&(raw>=0)&(raw<=1),raw,1.)
            np.testing.assert_array_equal(use,d.P_used_for_global_BH.to_numpy())
            pp.append(use);qq.append(d[Q].to_numpy());invalid+=int((~np.isfinite(raw)).sum())
            if scope!='ORA':
                assert not ((~d.pathway_test_estimable)&d[Q].lt(.05)).any()
                assert (d.permutations==10000).all()
            comps.update(map(tuple,d[['candidate_id','seed','contrast','database']].drop_duplicates().to_numpy()))
            sig+=int(d[Q].lt(.05).sum())
        np.testing.assert_allclose(bh(np.concatenate(pp)),np.concatenate(qq),atol=1e-13,rtol=1e-12)
        assert len(comps)==537*2
        pathway.append(dict(scope=scope,tests=sum(map(len,pp)),candidate_seed_contrast_databases=len(comps),
            invalid_raw_P=invalid,global_significant=sig,global_BH_recomputed=True))
        print('VERIFIED_PATHWAYS',scope,flush=True)
    atac=pd.read_csv(ROOT/'atac_evidence_summary.csv')
    expected=ev[ev.method.eq('FullB2')]
    assert len(atac)==len(expected)==68
    assert set(zip(atac.candidate_id,atac.seed))==set(zip(expected.candidate_id,expected.seed))
    ap=[];aq=[]
    for r in atac.itertuples():
        p=Path(r.path);assert sha(p)==r.result_sha256
        d=pd.read_csv(p);q='FDR_BH_all_B2_candidate_seed_peak_tests'
        assert len(d)==r.tested_peaks and d.peak.is_unique
        assert int((d[q].lt(.05)&d.delta_accessibility.abs().ge(.05)).sum())==r.global_strict_DA
        ap.append(d.p_value.to_numpy());aq.append(d[q].to_numpy())
    np.testing.assert_allclose(bh(np.concatenate(ap)),np.concatenate(aq),atol=1e-13,rtol=1e-12)
    cases=json.loads((ROOT/'SELECTED_CASES.json').read_text());figures=[]
    for c in cases:
        p=ROOT/'case_studies'/c['id'];mem=pd.read_csv(p/'fixed_membership.csv').set_index('barcode').group
        zz=pd.read_csv(p/'umap_all_seeds.csv.gz')
        assert len(zz.groupby(['method','seed']))==9
        for key,d in zz.groupby(['method','seed']):
            fixed=d[d.fixed_group.isin(['A','B'])].set_index('barcode').fixed_group
            assert fixed.to_dict()==mem.to_dict()
            assert d.barcode.is_unique
        for png in (p/'figures').glob('*.png'):
            with Image.open(png) as im:
                assert min(im.size)>1000 and abs(im.info['dpi'][0]-600)<1
            assert png.with_suffix('.pdf').stat().st_size>1000
            assert png.with_name(png.stem+'_caption_zh.md').stat().st_size>20
            figures.append(str(png.relative_to(ROOT)))
    assert len(figures)>=36
    report=ROOT/'final_report.md';assert report.stat().st_size>10000
    (ROOT/'audit/FINAL_NUMERICAL_VERIFICATION.json').write_text(json.dumps(dict(status='PASS',
        candidates=82,candidate_seed_cases=179,RNA_contrasts=537,gene_tests=19654737,
        all_gene_BH_recomputed=True,pathway_analyses=pathway,ATAC_comparisons=68,
        all_ATAC_BH_recomputed=True,identical_case_barcodes_in_every_panel=True,
        PNG600_and_vector_PDF_pairs=len(figures),visual_inspection='separate required review; not inferred from code'),indent=2))
    print('FINAL_NUMERICAL_VERIFICATION_PASS',flush=True)

if __name__=='__main__':main()
