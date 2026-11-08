"""Read-only input audit and fold wrappers around frozen, already validated Full B2 code."""
import os,sys,json,shutil,hashlib,time,importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'PBMC_D1_AllCells_Dmode_20261030'
sys.path.insert(0,str(OLD/'source'))
import common as oldcommon
import prepare_allcells as prior
from common import sha,write,normalize_log1p,hvg_train_indices,graph_fast
COUNTS={'D1':9347,'D2':6317,'D3':8105,'D4':2413}
SHARED=oldcommon.SHARED


def audit_inputs():
    registry={}
    for key,n in COUNTS.items():
        if key=='D2':
            source=OLD/'input/D2_author'
            meta=json.loads((OLD/'audit/D2_data_audit.json').read_text())
            xp=source/'rna_counts.npz'
            ap=prior.OLD/'source_inputs/D2/atac_binary.npz'
            assert sha(xp)==meta['rna_sha256'] and sha(ap)==meta['atac_sha256']
            x=sparse.load_npz(xp).tocsr();a=sparse.load_npz(ap).tocsr()
            cells=np.asarray((source/'barcodes.txt').read_text().splitlines())
            canonical=SHARED/'data/staging/PBMC/legacy_dmode_data_20261004/derived/common/D3/genes.tsv'
            genes=pd.read_csv(canonical,sep='\t')
            names=genes.iloc[:,1].astype(str).to_numpy();seen=set();unique=[]
            for name in names:
                v=name;suffix=1
                while v in seen:v=f'{name}.{suffix}';suffix+=1
                unique.append(v);seen.add(v)
            assert unique==(source/'gene_names.txt').read_text().splitlines()
            peaks=np.load(prior.OLD/'source_inputs/D2/peak_names.npy').astype(str)
            gp=ROOT/'input/D2_genes.tsv';genes.to_csv(gp,sep='\t',index=False)
            cp=source/'barcodes.txt'
            evidence=dict(author_counts_and_ATAC_hashes_verified=True,prior_audit=str(OLD/'audit/D2_data_audit.json'))
        else:
            x,a,cells,ids,names,peaks,evidence=prior.staged(key)
            p=SHARED/'data/staging/PBMC/legacy_dmode_data_20261004/derived/common'/key
            xp=p/'rna_counts.npz';ap=p/'atac_counts.npz';gp=p/'genes.tsv';cp=p/'barcodes.tsv'
        assert x.shape==(n,36601) and a.shape[0]==n and len(set(cells))==n
        assert np.isfinite(x.data).all() and np.all(x.data>=0) and np.array_equal(x.data,np.round(x.data))
        labels=pd.read_csv(ROOT/'input'/f'{key}_label_harmonized.csv').set_index('barcode')
        assert labels.index.is_unique and set(cells)==set(labels.index),(key,len(labels))
        labels=labels.loc[cells,'cell_type'].astype(str)
        assert labels.notna().all() and not labels.isin(['nan','Unknown','']).any()
        lp=ROOT/'input'/f'{key}_allcell_labels.csv';labels.to_frame().to_csv(lp)
        registry[key]=dict(cells=n,genes=36601,rna=str(xp),atac=str(ap),gene_table=str(gp),barcodes=str(cp),labels=str(lp),
            rna_sha256=sha(xp),atac_sha256=sha(ap),labels_sha256=sha(lp),atac_is_binary=key=='D2',subclasses=labels.nunique(),audit=evidence)
        print('DATA_VERIFIED',key,n,labels.nunique(),flush=True)
        if key=='D1':
            dest=ROOT/'native/D1';dest.mkdir(parents=True,exist_ok=True)
            tss=prior.read_tss(prior.GTF)
            mapped=np.flatnonzero([g.split('.')[0] in tss for g in ids])
            graph=graph_fast(ids[mapped],peaks,tss)
            eligible=mapped[np.flatnonzero(graph['candidate_mask'].any(1))]
            selected=hvg_train_indices(x,np.arange(n),eligible,2000)
            graph=graph_fast(ids[selected],peaks,tss)
            kept=np.unique(graph['candidate_index'][graph['candidate_mask']])
            np.save(dest/'rna.npy',normalize_log1p(x)[:,selected].toarray().astype(np.float32))
            np.save(dest/'gene_ids.npy',ids[selected]);np.save(dest/'gene_names.npy',names[selected])
            np.save(dest/'peak_names.npy',peaks[kept]);sparse.save_npz(dest/'atac_binary.npz',a[:,kept].astype(bool).astype(np.float32))
            np.savez_compressed(dest/'metadata.npz',train_index=np.arange(n),val_index=np.array([],dtype=np.int64),test_index=np.array([],dtype=np.int64))
            (dest/'barcodes.tsv').write_text('\n'.join(cells)+'\n')
        del x,a
    for key in ['D2','D3','D4']:
        dest=ROOT/'native'/key
        if not dest.exists():dest.symlink_to(OLD/'artifacts/native'/key,target_is_directory=True)
        assert (dest/'barcodes.tsv').read_text().splitlines()==Path(registry[key]['barcodes']).read_text().splitlines()
    write(ROOT/'input/datasets.json',registry)
    write(ROOT/'audit/DATA_INPUTS_VERIFIED.json',dict(status='PASS',datasets=registry,
       total_cells=sum(COUNTS.values()),D2_identity='GSE196990 HDa; not staging D2',source_target_roles_vary_by_fold=True))
    return registry


def setup_folds(registry):
    for target in ['D2','D3','D4']:
        run=ROOT/'folds'/target
        for sub in ['source','artifacts/native','input','audit','logs','figures','tables']:(run/sub).mkdir(parents=True,exist_ok=True)
        sources=[k for k in COUNTS if k!=target]
        config=dict(target=target,sources=sources,target_cells=COUNTS[target],source_cells=sum(COUNTS[k] for k in sources),registry=str(ROOT/'input/datasets.json'))
        write(run/'fold_config.json',config)
        frozen=run/'source/frozen'
        if not frozen.exists():frozen.symlink_to(OLD/'source/frozen',target_is_directory=True)
        shutil.copy2(OLD/'source/common.py',run/'source/common.py')
        for key in sources:
            dest=run/'artifacts/native'/key
            if not dest.exists():dest.symlink_to(ROOT/'native'/key,target_is_directory=True)
        shutil.copy2(registry[target]['labels'],run/'audit'/f'{target}_allcell_labels.csv')
        s=(OLD/'source/run_model.py').read_text()
        s=s.replace('from common import *','from common import *\nFOLD_CONFIG=json.loads((RUN/"fold_config.json").read_text())\nTARGET=FOLD_CONFIG["target"]\nSOURCES=FOLD_CONFIG["sources"]\nN_TARGET=FOLD_CONFIG["target_cells"]')
        s=s.replace("['D2','D3','D4']",'SOURCES')
        s=s.replace("'artifacts/D1_'", "f'artifacts/{TARGET}_'")
        s=s.replace("'audit/D1_allcell_labels.csv'", "f'audit/{TARGET}_allcell_labels.csv'")
        s=s.replace("'D1_barcodes.tsv'", "f'{TARGET}_barcodes.tsv'")
        s=s.replace("'D1_rna_all.npy'", "f'{TARGET}_rna_all.npy'")
        s=s.replace("target='D1'",'target=TARGET').replace('9347','N_TARGET')
        assert 'D1_' not in s and "'D1'" not in s
        compile(s,str(run/'source/run_model.py'),'exec')
        (run/'source/run_model.py').write_text(s)
        shutil.copy2(ROOT/'source/prepare_fold.py',run/'source/prepare_fold.py')
    write(ROOT/'audit/FOLD_WRAPPERS_CREATED.json',dict(status='COMPLETE',folds=['D2','D3','D4'],
        original_model_sha256=sha(OLD/'source/frozen/run_b2_strict_lodo.py'),network_and_loss_changed=False,
        D1_reused=str(OLD),source_cells_all=True,target_cells_all_inference_only=True))


if __name__=='__main__':
    for sub in ['native','audit','input','logs','folds']:(ROOT/sub).mkdir(exist_ok=True)
    registry=audit_inputs();setup_folds(registry)
    print('BOOTSTRAP_COMPLETE',flush=True)
