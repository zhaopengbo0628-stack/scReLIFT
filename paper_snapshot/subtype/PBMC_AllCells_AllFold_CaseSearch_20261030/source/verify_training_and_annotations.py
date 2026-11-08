"""Read-only checkpoint verification and barcode-aligned author annotation audit."""
import hashlib,json,os
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'PBMC_D1_AllCells_Dmode_20261030'
SEEDS=[20020130,20020131,20020132]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def write(p,obj):
    p.parent.mkdir(exist_ok=True,parents=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def checkpoints():
    registry=json.loads((ROOT/'input/datasets.json').read_text());rows=[]
    for fold,rec in registry.items():
        base=OLD if fold=='D1' else ROOT/'folds'/fold
        genes=np.load(base/'artifacts/gene_ids.npy').astype(str)
        cells=(base/f'artifacts/{fold}_barcodes.tsv').read_text().splitlines()
        assert len(cells)==len(set(cells))==rec['cells']
        assert cells==Path(rec['barcodes']).read_text().splitlines()
        for seed in SEEDS:
            run=base/'FullB2'/f'seed_{seed}'
            metric=json.loads((run/'metrics.json').read_text())
            assert metric['cells']==rec['cells'] and metric['seed']==seed
            for stage in ['teacher','student']:
                cp=run/'checkpoints'/stage/'latest.pt'
                manifest=json.loads((cp.parent/'manifest.json').read_text())
                cfg=json.loads((cp.parent/'config.json').read_text())
                obj=torch.load(cp,map_location='cpu',weights_only=False)
                assert obj['epoch']==100 and cfg['seed']==seed
                digest=sha(cp);assert digest==manifest['latest_checkpoint_sha256']
                assert cfg['target_atac_used'] is False and cfg['target_adaptation'] is False
                # Check every frozen input actually used by training, including source matrices.
                for p,h in manifest['input_hashes'].items():assert sha(p)==h,(fold,seed,stage,p)
                assert obj['input_hashes']==manifest['input_hashes']
                rows.append(dict(fold=fold,seed=seed,stage=stage,epoch=obj['epoch'],checkpoint=str(cp),
                    sha256=digest,target_cells=rec['cells'],source_cells=sum(r['cells'] for k,r in registry.items() if k!=fold),
                    source_HVG_intersection=len(genes),all_training_input_hashes_verified=True,
                    target_ATAC_in_training=False,target_adaptation=False,
                    note='source all cells; held-out target inference only; final epoch fixed in advance'))
    pd.DataFrame(rows).to_csv(ROOT/'audit/CHECKPOINT_VERIFICATION.csv',index=False)
    write(ROOT/'audit/CHECKPOINT_VERIFICATION_COMPLETE.json',dict(status='COMPLETE',checked_stages=len(rows),
        checked_student_models=12,reused_D1_students=3,new_student_models=9,epoch=100,device='cpu',
        timestamp=datetime.now(timezone.utc).isoformat(),no_training_or_process_signals=True))

def author_metadata():
    run=ROOT/'analyses/D2'
    author_path=OLD/'audit/D2_author_metadata.csv'
    meta=pd.read_csv(author_path,index_col=0)
    rec=json.loads((ROOT/'input/datasets.json').read_text())['D2']
    cells=Path(rec['barcodes']).read_text().splitlines()
    assert len(meta)==len(cells)==6317 and meta.index.is_unique
    if not set(cells).issubset(meta.index):
        assert 'original' in meta and meta.original.is_unique
        meta=meta.set_index('original',drop=False)
    assert set(cells)==set(meta.index)
    meta=meta.loc[cells]
    labels=pd.read_csv(rec['labels'],index_col=0).loc[cells,'cell_type'].astype(str)
    mapping={'B naive':'Naive B','B intermediate':'Intermediate B','B memory':'Memory B',
             'NK_CD56bright':'NK','NK Proliferating':'NK','cDC1':'cDC','cDC2':'cDC'}
    harmonized=meta['predicted.celltype.l2'].astype(str).replace(mapping)
    assert np.array_equal(labels.to_numpy(),harmonized.to_numpy())
    columns=['predicted.celltype.l2','MarkerAnnotations','NewAnnotations','NewAnnotations_2','ClusterAnnotations','ClusterAnnotations_Broad',
             'BridgeAnnotations','sample','Sex','Age','Status','batches','batches_2','batches_3']
    columns=[c for c in columns if c in meta]
    membership=pd.read_csv(run/'tables/candidate_membership.csv.gz')
    rows=[]
    for (cid,seed,group),mem in membership.groupby(['candidate_id','seed','group']):
        sub=meta.loc[mem.barcode]
        for col in columns:
            for val,n in sub[col].fillna('MISSING').astype(str).value_counts().items():
                rows.append(dict(fold='D2',candidate_id=cid,seed=int(seed),group=group,
                    reference_subclass=mem.reference_subclass.iloc[0],author_column=col,author_value=val,
                    cells=int(n),group_cells=len(mem),fraction=float(n/len(mem)),
                    note='same-cohort annotation, not independent validation; missing labels not imputed'))
    pd.DataFrame(rows).to_csv(run/'tables/D2_candidate_author_annotation_composition.csv',index=False)
    cross=[]
    for col in columns:
        for (label,val),n in pd.DataFrame({'reference':labels,'value':meta[col].fillna('MISSING').astype(str)}).value_counts().items():
            cross.append(dict(reference_subclass=label,author_column=col,author_value=val,cells=int(n)))
    pd.DataFrame(cross).to_csv(run/'tables/D2_reference_author_annotation_crosswalk.csv',index=False)
    write(run/'audit/D2_AUTHOR_METADATA_VERIFIED.json',dict(status='COMPLETE',cells=len(cells),metadata=str(author_path),
        metadata_sha256=sha(author_path),barcode_alignment='exact set and explicit reindex; not row-order assumption',
        existing_reference_harmonization=mapping,
        sample='HDa',donor_count=1,age=36,sex='Female',donor_health='Healthy',
        constant_batch_columns={c:str(meta[c].iloc[0]) for c in ['batches','batches_2','batches_3']},
        correction='Earlier generic missing donor/batch statements do not apply to D2; one donor and constant batch are documented.',
        warning='Reference l2 CD8 Naive vs author MarkerAnnotations is strongly discordant; HSPC fine labels available but same-cohort only.',
        remaining_unavailable=['independent donor replication','doublet score','ATAC fragments','TSS enrichment']))
    # Keep all author annotation strings, including No_Annotation, as supplied.
    meta.to_csv(run/'tables/D2_author_metadata_aligned.csv')
    print('D2_AUTHOR_ANNOTATION_AUDITED',len(rows),flush=True)

if __name__=='__main__':
    checkpoints();author_metadata();print('TRAINING_AND_METADATA_VERIFIED',flush=True)
