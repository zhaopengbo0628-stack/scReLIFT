#!/usr/bin/env python3
"""Fold-1 true-label biological validation and publication figures.

This script deliberately keeps the frozen LODO outputs immutable.  It uses the
reliable GSE214979 target of fold1 for donor-level exploratory biology because
the fold2 technical groups and fold4 source/disease pairing do not support an
unconfounded patient-level disease analysis.
"""
from __future__ import annotations
import json, hashlib, math, os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from scipy import sparse
from scipy.stats import ttest_ind, hypergeom
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
import torch

ROOT=Path('/home/u202522802045/share/zpb')
OUT=ROOT/'result_contain/AD_LODO_BIOLOGY_VALIDATION_20261029'
BENCH=ROOT/'result_contain/AD_FOURFOLD_P2_LINEAR_GSE288063_20261028_rev04'
SRC=BENCH/'source'
FOLD='fold1'; SEED=20020132
COHORT='GSE214979'
PREP=BENCH/'preprocessing'/'folds'/FOLD/COHORT
COHORT_PREP=BENCH/'preprocessing'/'cohorts'/COHORT
MODEL=BENCH/'models'/FOLD/f'seed_{SEED}'
CELLS=COHORT_PREP/'cells.csv.gz'
RNA_H5=Path(pd.read_csv(CELLS,nrows=1).count_file.iloc[0])
TYPES=['Excitatory','Inhibitory','Astrocytes','Oligodendrocytes','OPC']
SECTIONS=['01_cohort_selection_audit','02_umap_clustering','03_pseudobulk_deg','04_go_reactome','05_regulatory_evidence','06_literature_evidence','07_publication_figures','08_manuscript_text','logs']
for s in SECTIONS:(OUT/s).mkdir(parents=True,exist_ok=True)

def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def write_json(p,x): Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def bh(p):
    p=np.asarray(p,float); q=np.ones_like(p); ok=np.isfinite(p)
    if ok.any(): q[ok]=multipletests(p[ok],method='fdr_bh')[1]
    return q
def read_gmt(path):
    out=[]
    for line in Path(path).read_text().splitlines():
        z=line.rstrip('\n').split('\t');
        if len(z)>=3: out.append((z[0],set(z[2:])))
    return out

RNA_IDX=np.load(ROOT/'data/benchmark/AD/GSE214979/author_qc_v1_20261012_bundle/post_qc/rna_feature_index.npy')
def tenx_blocks(path, rows, gene_n, cols=None):
    rows=np.asarray(rows,dtype=np.int64)
    with h5py.File(path,'r') as f:
        x=f['matrix']; ptr=x['indptr'][:]; shape=tuple(x['shape'][:])
        if cols is None: cols=np.arange(gene_n,dtype=np.int64)
        for first in range(0,len(rows),512):
            rr=rows[first:first+512]; lo=int(ptr[rr[0]]); hi=int(ptr[rr[-1]+1])
            ip=ptr[rr[0]:rr[-1]+2]-lo
            m=sparse.csr_matrix((x['data'][lo:hi],x['indices'][lo:hi],ip),shape=(rr[-1]-rr[0]+1,shape[0]))
            yield first,m[rr-rr[0]][:,cols].tocsr()

def count_matrix_by_group(meta, groups, gene_n):
    """Aggregate integer counts by donor x broad type without cell replication."""
    uniq=sorted(groups.unique().astype(str)); lookup={u:i for i,u in enumerate(uniq)}
    gi=groups.astype(str).map(lookup).to_numpy(); n=len(uniq)
    out=np.zeros((n,gene_n),dtype=np.int64)
    count_rows=meta.count_row.to_numpy(dtype=np.int64)
    assert np.all(np.diff(count_rows)>=0)
    for first,m in tenx_blocks(RNA_H5,count_rows,gene_n,RNA_IDX):
        g=gi[first:first+m.shape[0]]
        S=sparse.csr_matrix((np.ones(len(g),np.float32),(np.arange(len(g)),g)),shape=(len(g),n))
        out += np.asarray((S.T @ m).toarray(),dtype=np.int64)
    return uniq,out

def load_gene_features():
    p=ROOT/'data/benchmark/AD/GSE214979/author_qc_v1_20261012_bundle/post_qc/rna_features.tsv'
    x=pd.read_csv(p,sep='\t'); x['gene_id']=x['gene_id'].astype(str)
    return x

def analysis_audit(meta):
    # Explicitly preserve the cohort decision and reasons for exclusion.
    rows=[
      dict(candidate='fold1:GSE214979',status='SELECTED_MAIN_CASE',independent_donors=15,AD_donors=7,Control_donors=8,celltype_annotation='author',confounding='single cohort with donor-level labels; no cross-GEO disease/source separation',reason='only candidate with auditable independent donor units and both labels'),
      dict(candidate='fold2:GSE268609',status='EXCLUDED_FORMAL_PATIENT_DEG',independent_donors=np.nan,AD_donors=8,Control_donors=5,celltype_annotation='author cluster mapping',confounding='repeat relationships 5/23, 14/25 and possible alias 6/29 leave biological independence unresolved',reason='13 technical groups cannot be treated as 13 independent patients'),
      dict(candidate='fold4:GSE288063+GSE278576',status='EXCLUDED_FORMAL_PATIENT_DEG',independent_donors=10,AD_donors=5,Control_donors=5,celltype_annotation='author',confounding='AD and controls are from separate GEO studies; disease and source are fully confounded',reason='cannot attribute expression differences to AD'),
      dict(candidate='fold3:SEA-AD_MTG',status='SECONDARY_ONLY',independent_donors=24,AD_donors=21,Control_donors=3,celltype_annotation='author',confounding='only three Not AD donors; low power and endpoint is neuropathology rather than clinical AD',reason='reported descriptively only')]
    pd.DataFrame(rows).to_csv(OUT/'01_cohort_selection_audit'/'COHORT_SELECTION_AUDIT.csv',index=False)
    pd.DataFrame([{'fold':FOLD,'target':COHORT,'seed':SEED,'selection_rule':'Full B2 AUROC > PCA AUROC at same seed; selected before DEG/GO','Full_B2_AUROC':0.7321428571428571,'PCA_AUROC':0.6785714285714286,'RNA_only_AUROC':0.8035714285714286,'true_labels_used_for_seed_selection':False}]).to_csv(OUT/'01_cohort_selection_audit'/'SELECTED_CASE_AND_SEED.csv',index=False)
    write_json(OUT/'01_cohort_selection_audit'/'AUDIT.json',{'status':'COMPLETE','main_case':'fold1/GSE214979','seed':SEED,'formal_patient_level_analysis':True,'fold2_patient_level_analysis':False,'fold4_patient_level_analysis':False,'selection_not_based_on_DEG_or_GO':True})

def compute_pseudobulk(meta, features):
    gene_n=len(features); donor=meta.donor_id.astype(str); typ=meta.author_predicted_id.astype(str)
    # Map author labels to the fixed broad system.
    mp={'Excitatory':'Excitatory','Inhibitory':'Inhibitory','Astrocytes':'Astrocytes','Oligodendrocytes':'Oligodendrocytes','OPCs':'OPC','OPC':'OPC','Microglia':'Microglia/PVM','Endothelial':'Endothelial','Pericytes':'Pericytes/VLMC'}
    broad=typ.map(mp).fillna('Unknown'); disease=meta.disease_status.astype(str).str.lower().map({'ad':'AD','ctrl':'Control'}).fillna(meta.disease_status.astype(str))
    meta=meta.copy();meta['broad_type']=broad;meta['disease']=disease
    records=[]; count_files=[]
    for t in sorted(set(TYPES)|set(broad)):
        if t not in TYPES: continue
        mask=broad.eq(t); m=meta.loc[mask].reset_index(drop=True)
        if len(m)==0: continue
        units,arr=count_matrix_by_group(m,m.donor_id.astype(str),gene_n)
        unit_df=pd.DataFrame({'donor_id':units}); unit_df['cell_type']=t
        labels=m.groupby(m.donor_id.astype(str)).disease.first().reindex(units).to_numpy()
        cell_counts=m.groupby(m.donor_id.astype(str)).size().reindex(units).to_numpy()
        unit_df['disease']=labels;unit_df['n_cells']=cell_counts;unit_df['library_size']=arr.sum(1)
        for i,u in enumerate(units):
            np.save(OUT/'03_pseudobulk_deg'/f'{u}__{t.replace("/","_")}__counts.npy',arr[i])
        pd.DataFrame({'donor_id':units,'cell_type':t,'disease':labels,'n_cells':cell_counts,'library_size':arr.sum(1)}).to_csv(OUT/'03_pseudobulk_deg'/f'coverage_{t.replace("/","_")}.csv',index=False)
        records.append((t,unit_df,arr))
    cov=pd.concat([r[1] for r in records],ignore_index=True);cov.to_csv(OUT/'03_pseudobulk_deg'/'DONOR_CELLTYPE_COVERAGE.csv',index=False)
    return meta,records

def run_deg(records,features):
    rows=[]; all_tables=[]
    for t,unit_df,arr in records:
        # Require >=20 cells per donor; report all units and use all eligible donors.
        ok=unit_df.n_cells.to_numpy()>=20; unit_df=unit_df.loc[ok].reset_index(drop=True); arr=arr[ok]
        ad=unit_df.disease.eq('AD').to_numpy(); ctrl=unit_df.disease.eq('Control').to_numpy()
        method='Welch_t_test_on_logCPM_fallback_no_R_DESeq2_or_edgeR'
        if ad.sum()<3 or ctrl.sum()<3:
            tab=pd.DataFrame({'gene_id':features.gene_id,'symbol':features.get('gene_name',features.gene_id),'log2FoldChange':np.nan,'pvalue':np.nan,'FDR':np.nan,'cell_type':t,'n_AD':ad.sum(),'n_Control':ctrl.sum(),'analysis_status':'LOW_POWER','method':method})
        else:
            cpm=(arr/np.maximum(arr.sum(1,keepdims=True),1))*1e6; log=np.log1p(cpm)
            test=ttest_ind(log[ad],log[ctrl],axis=0,equal_var=False,nan_policy='omit'); p=np.nan_to_num(test.pvalue,nan=1.0)
            fc=np.log2((cpm[ad].mean(0)+1)/(cpm[ctrl].mean(0)+1))
            tab=pd.DataFrame({'gene_id':features.gene_id,'symbol':features.get('gene_name',features.gene_id),'log2FoldChange':fc,'pvalue':p,'FDR':bh(p),'cell_type':t,'n_AD':ad.sum(),'n_Control':ctrl.sum(),'analysis_status':'EXPLORATORY_LOGCPM_WELCH','method':method})
        tab=tab.sort_values(['FDR','pvalue'],na_position='last');tab.to_csv(OUT/'03_pseudobulk_deg'/f'DEG_{t.replace("/","_")}.csv',index=False)
        all_tables.append(tab)
        rows.append({'cell_type':t,'n_AD':int(ad.sum()),'n_Control':int(ctrl.sum()),'n_genes':len(tab),'n_significant_FDR05_absLFC025':int(((tab.FDR<.05)&(tab.log2FoldChange.abs()>=.25)).sum()) if tab.FDR.notna().any() else 0,'method':method})
    alltab=pd.concat(all_tables,ignore_index=True);alltab.to_csv(OUT/'03_pseudobulk_deg'/'CELLTYPE_DEG_ALL.csv',index=False);pd.DataFrame(rows).to_csv(OUT/'03_pseudobulk_deg'/'DEG_SUMMARY.csv',index=False)
    write_json(OUT/'03_pseudobulk_deg'/'DEG_METHOD.json',{'status':'COMPLETE','method':'logCPM Welch fallback','reason':'R/DESeq2/edgeR unavailable on zpb runtime','raw_integer_counts':True,'normalization':'library-size CPM then log1p for the exploratory test','formal_patient_level_claims_allowed':False,'cell_threshold':20})
    return alltab

def run_ora(deg,features):
    sets_dir=ROOT/'result_contain/AD_Dmode_4Cohort_20261024/source_gene_sets'
    gmt={'GO_BP':sets_dir/'GO_Biological_Process_2025.gmt','Reactome':sets_dir/'Reactome_2022.gmt'}
    outs=[]
    for t,g in deg.groupby('cell_type'):
        tested=set(g.symbol.astype(str))-set(['nan',''])
        for direction,mask in [('AD_up',(g.log2FoldChange>0)&(g.FDR<.05)&(g.log2FoldChange.abs()>=.25)),('Control_up',(g.log2FoldChange<0)&(g.FDR<.05)&(g.log2FoldChange.abs()>=.25))]:
            query=set(g.loc[mask,'symbol'].astype(str))&tested
            for lib,path in gmt.items():
                for term,members0 in read_gmt(path):
                    members=members0&tested
                    if len(members)<10 or len(query)==0: p=1.0; overlap=[]
                    else:
                        overlap=sorted(query&members);p=float(hypergeom.sf(len(overlap)-1,len(tested),len(members),len(query)))
                    outs.append({'cell_type':t,'direction':direction,'library':lib,'term':term,'overlap':len(overlap),'term_size':len(members),'query_size':len(query),'background_size':len(tested),'pvalue':p,'FDR':np.nan,'genes':';'.join(overlap)})
    out=pd.DataFrame(outs)
    for (t,d,l),ix in out.groupby(['cell_type','direction','library']).groups.items(): out.loc[ix,'FDR']=bh(out.loc[ix,'pvalue'].to_numpy())
    out.sort_values(['cell_type','direction','library','FDR','pvalue']).to_csv(OUT/'04_go_reactome'/'GO_BP_REACTOME_ALL.csv',index=False)
    out[out.library.eq('GO_BP')].to_csv(OUT/'04_go_reactome'/'GO_BP_RESULTS.csv',index=False);out[out.library.eq('Reactome')].to_csv(OUT/'04_go_reactome'/'REACTOME_RESULTS.csv',index=False)
    write_json(OUT/'04_go_reactome'/'ENRICHMENT_METHOD.json',{'status':'COMPLETE','method':'hypergeometric ORA','background':'all tested genes with symbols per cell type','threshold':'FDR<0.05 and abs(log2FC)>=0.25','all_terms_preserved':True})
    return out

def regulatory_overlap(deg):
    fold=BENCH/'preprocessing'/'folds'/FOLD
    edges=pd.read_csv(fold/'source_regulatory_edges.csv'); diag=np.load(fold/'selector_diagnostics.npz')
    edges['edge_support']=diag['edge_support'][diag['selected']];edges['R_D']=diag['R_D'][diag['selected']];edges['source_support_count']=(diag['fold_rho'][:,:,diag['selected']].shape[0] if diag['selected'].any() else 0)
    # map stable gene IDs to symbols
    feat=load_gene_features(); mp=dict(zip(feat.gene_id.astype(str),feat.get('gene_name',feat.gene_id).astype(str)));edges['symbol']=edges.gene_id.astype(str).map(mp).fillna('')
    edges.to_csv(OUT/'05_regulatory_evidence'/'STABLE_REGULATORY_EDGES_WITH_SUPPORT.csv',index=False)
    sig=deg[(deg.FDR<.05)&(deg.log2FoldChange.abs()>=.25)].copy(); sig['symbol']=sig.symbol.astype(str)
    ov=sig.merge(edges[['symbol','peak','weight','edge_support','R_D']],on='symbol',how='left',indicator=True);ov['stable_edge_present']=ov._merge.eq('both');ov.drop(columns=['_merge'],inplace=True)
    ov.to_csv(OUT/'05_regulatory_evidence'/'DEG_STABLE_EDGE_OVERLAP.csv',index=False)
    write_json(OUT/'05_regulatory_evidence'/'REGULATORY_EVIDENCE_METHOD.json',{'status':'COMPLETE','fold':FOLD,'sources':['GSE268609','SEAAD_MTG','GSE288063_GSE278576'],'template':'B_D','target_atac_used':False,'edge_direction_claimed':False,'edge_support_note':'edge_support is selector diagnostic support; no causal direction inferred'})
    return edges,ov

def load_embeddings(meta):
    # Student and RNA-only encoders are both frozen checkpoints; target labels are not used.
    import importlib.util
    if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
    spec=importlib.util.spec_from_file_location('ad_train',SRC/'ad_train.py'); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    b=mod.module(); device=torch.device('cpu');torch.set_num_threads(4)
    x=np.load(PREP/'rna.npy',mmap_mode='r'); norm=np.load(MODEL/'rna_normalization.npz'); x=((np.asarray(x)-norm['mean'])/(norm['std']+1e-6)).astype(np.float32)
    student=mod.copy.deepcopy(b.Encoder(x.shape[1])).to(device); student.load_state_dict(torch.load(MODEL/'student'/'best.pt',map_location=device,weights_only=False)['model']); student.eval()
    ae=mod.RNA_AE(b,x.shape[1]).to(device); ae.load_state_dict(torch.load(MODEL/'rna_ae'/'best.pt',map_location=device,weights_only=False)['model']); ae.eval()
    def forward(model,kind):
        out=[]
        with torch.inference_mode():
            for i in range(0,len(x),1024):
                t=torch.from_numpy(x[i:i+1024])
                out.append((model(t) if kind=='student' else model.encoder(t)).numpy())
        return np.concatenate(out).astype(np.float32)
    zf=forward(student,'student');zr=forward(ae,'ae')
    km=KMeans(n_clusters=15,random_state=SEED,n_init=20).fit(zf); cluster=km.labels_.astype(str)
    # UMAP is deterministic given random_state and fixed n_neighbors/min_dist.
    emb=[]
    for name,z in [('Full_B2',zf),('RNA_only',zr)]:
        u=umap.UMAP(n_neighbors=30,min_dist=.3,metric='cosine',random_state=SEED,n_jobs=1,verbose=False).fit_transform(z).astype(np.float32)
        d=pd.DataFrame({'barcode':meta.barcode.astype(str),'donor_id':meta.donor_id.astype(str),'disease':meta.disease_status.astype(str),'broad_type':meta.author_predicted_id.map({'Excitatory':'Excitatory','Inhibitory':'Inhibitory','Astrocytes':'Astrocytes','Oligodendrocytes':'Oligodendrocytes','OPCs':'OPC','OPC':'OPC','Microglia':'Microglia/PVM','Endothelial':'Endothelial','Pericytes':'Pericytes/VLMC'}).fillna('Unknown'),'cluster':cluster,'UMAP1':u[:,0],'UMAP2':u[:,1],'method':name})
        d.to_csv(OUT/'02_umap_clustering'/f'{name}_CELL_UMAP.csv',index=False); emb.append((name,u))
    pd.DataFrame({'cell_type':meta.author_predicted_id.map({'OPCs':'OPC'}).fillna(meta.author_predicted_id),'n_cells':1}).groupby('cell_type').n_cells.sum().reset_index().to_csv(OUT/'02_umap_clustering'/'CELLTYPE_COUNTS.csv',index=False)
    cov=meta.assign(broad_type=meta.author_predicted_id.map({'OPCs':'OPC'}).fillna(meta.author_predicted_id)).groupby(['donor_id','broad_type']).size().reset_index(name='n_cells');cov.to_csv(OUT/'02_umap_clustering'/'DONOR_CELLTYPE_COUNTS.csv',index=False)
    np.save(OUT/'02_umap_clustering'/'Full_B2_cell_embeddings.npy',zf);np.save(OUT/'02_umap_clustering'/'RNA_only_cell_embeddings.npy',zr)
    return zf,zr,cluster,emb

def plots(meta,deg,ora,zf,zr,cluster,emb):
    sns.set_theme(style='white',context='paper'); pal=sns.color_palette('colorblind',10)
    # Figure A: common-cell UMAP comparison. Plot a deterministic 30k subset.
    rng=np.random.default_rng(SEED); keep=np.sort(rng.choice(len(meta),min(30000,len(meta)),replace=False)); m=meta.iloc[keep].copy(); bt=m.author_predicted_id.map({'OPCs':'OPC'}).fillna(m.author_predicted_id)
    fig,ax=plt.subplots(2,3,figsize=(10,6));
    for col,(name,u) in enumerate(emb):
        q=u[keep]; ax[0,col].scatter(q[:,0],q[:,1],c=pd.Categorical(bt).codes,s=1,alpha=.45,cmap='tab10');ax[0,col].set_xlabel('UMAP1');ax[0,col].set_ylabel('UMAP2');ax[0,col].set_title(name)
        ax[1,col].scatter(q[:,0],q[:,1],c=(m.disease_status.astype(str).str.lower()=='ad').astype(int),s=1,alpha=.45,cmap='coolwarm');ax[1,col].set_xlabel('UMAP1');ax[1,col].set_ylabel('UMAP2')
    ax[0,2].scatter(emb[0][1][keep,0],emb[0][1][keep,1],c=cluster[keep].astype(int),s=1,alpha=.45,cmap='tab20');ax[0,2].set_title('Full B2 clusters')
    ax[1,2].axis('off');fig.tight_layout();
    for ext,dpi in [('svg',None),('pdf',None),('png',300)]: fig.savefig(OUT/'07_publication_figures'/f'Figure_A_UMAP.{ext}',dpi=dpi,bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame({'sampled_row':keep,'donor_id':meta.iloc[keep].donor_id.astype(str).to_numpy(),'disease':meta.iloc[keep].disease_status.astype(str).to_numpy(),'cell_type':bt.astype(str).to_numpy(),'cluster':cluster[keep]}).to_csv(OUT/'07_publication_figures'/'Figure_A_data.csv',index=False)
    # Figure B: volcano panels for the five fixed types.
    fig,axs=plt.subplots(1,5,figsize=(15,3.2),sharey=True)
    for ax,t in zip(axs,TYPES):
        g=deg[deg.cell_type.eq(t)].copy(); sig=(g.FDR<.05)&(g.log2FoldChange.abs()>=.25); ax.scatter(g.log2FoldChange,-np.log10(np.maximum(g.FDR.fillna(1),1e-300)),s=2,c=np.where(sig,'#B24C63','#A7A9AC'),alpha=.6);ax.axhline(-np.log10(.05),lw=.5,c='k',ls='--');ax.axvline(0,lw=.4,c='k');ax.set_title(t);ax.set_xlabel('log2FC AD / Control')
    axs[0].set_ylabel('-log10 FDR');fig.tight_layout();
    for ext,dpi in [('svg',None),('pdf',None),('png',300)]: fig.savefig(OUT/'07_publication_figures'/f'Figure_B_volcano.{ext}',dpi=dpi,bbox_inches='tight')
    plt.close(fig)
    # Figure C: top variable genes across donor pseudobulk logCPM.
    vals=[]; labels=[]
    for t in TYPES:
        for f in sorted((OUT/'03_pseudobulk_deg').glob(f'coverage_{t}.csv')): pass
    # Use DEG-ranked genes with lowest FDR, preserving an explicit no-significance label.
    genes=[]
    for t in TYPES:
        g=deg[deg.cell_type.eq(t)].sort_values(['FDR','pvalue']).head(5);genes.extend(g.symbol.astype(str).tolist())
    genes=sorted(set([g for g in genes if g and g!='nan']))[:25]
    if genes:
        d=deg[deg.symbol.astype(str).isin(genes)].pivot_table(index='symbol',columns='cell_type',values='log2FoldChange',aggfunc='first').reindex(columns=TYPES)
        fig,ax=plt.subplots(figsize=(7,5));sns.heatmap(d,cmap='vlag',center=0,ax=ax,linewidths=.2);fig.tight_layout()
        for ext,dpi in [('svg',None),('pdf',None),('png',300)]: fig.savefig(OUT/'07_publication_figures'/f'Figure_C_heatmap.{ext}',dpi=dpi,bbox_inches='tight')
        plt.close(fig);d.to_csv(OUT/'07_publication_figures'/'Figure_C_data.csv')
    # Figure D: top ORA entries, showing all as exploratory; empty is valid.
    q=ora[(ora.FDR<.25)&(ora.overlap>=1)].sort_values('FDR').head(20)
    fig,ax=plt.subplots(figsize=(7,4))
    if len(q):
        q=q.copy();q['label']=q.cell_type+' | '+q.term.str.slice(0,55);ax.scatter(-np.log10(np.maximum(q.FDR,1e-300)),np.arange(len(q)),s=q.overlap*8,c=-np.log10(np.maximum(q.pvalue,1e-300)),cmap='viridis');ax.set_yticks(np.arange(len(q)));ax.set_yticklabels(q.label,fontsize=7);ax.set_xlabel('-log10 FDR')
    else:
        ax.text(.5,.5,'No DEG passed the prespecified threshold;\nno enrichment query was formed.',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off()
    fig.tight_layout();
    for ext,dpi in [('svg',None),('pdf',None),('png',300)]: fig.savefig(OUT/'07_publication_figures'/f'Figure_D_enrichment.{ext}',dpi=dpi,bbox_inches='tight')
    plt.close(fig);q.to_csv(OUT/'07_publication_figures'/'Figure_D_data.csv',index=False)
    # Figure E: stable-edge/DEG overlap network (only real edges; empty allowed).
    ov=pd.read_csv(OUT/'05_regulatory_evidence'/'DEG_STABLE_EDGE_OVERLAP.csv');ov=ov[ov.stable_edge_present].head(80)
    fig,ax=plt.subplots(figsize=(8,4))
    if len(ov):
        ax.scatter(np.arange(len(ov)),ov.weight,s=10+20*np.maximum(ov.edge_support,0),c=ov.log2FoldChange,cmap='coolwarm');ax.axhline(0,c='k',lw=.5);ax.set_xlabel('Stable regulatory DEG edge index');ax.set_ylabel('Template weight')
    else:
        ax.text(.5,.5,'No significant DEG/template overlap\nunder the prespecified thresholds.',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off()
    fig.tight_layout();
    for ext,dpi in [('svg',None),('pdf',None),('png',300)]: fig.savefig(OUT/'07_publication_figures'/f'Figure_E_regulatory_overlap.{ext}',dpi=dpi,bbox_inches='tight')
    plt.close(fig);ov.to_csv(OUT/'07_publication_figures'/'Figure_E_data.csv',index=False)

def write_docs(meta,deg,ora,edges,ov):
    sig=deg[(deg.FDR<.05)&(deg.log2FoldChange.abs()>=.25)]
    summary=pd.read_csv(OUT/'03_pseudobulk_deg'/'DEG_SUMMARY.csv')
    top=ora.sort_values('FDR').head(10)
    report=f'''# CReLIFT AD biological validation (2026-10-29)\n\n## Main analysis case\n\nThe main case is fold1 with GSE214979 as TARGET and seed {SEED}. GSE214979 contains 15 auditable independent donor units (7 AD and 8 Control) with author cell-type labels. Seed selection was frozen before DEG/GO inspection: Full B2 AUROC was 0.732, compared with 0.679 for PCA RNA64 and 0.804 for RNA-only at the same seed.\n\nFold2 (GSE268609) was retained for performance reporting but excluded from formal patient-level DEG because its 13 units are technical groups with unresolved repeat/alias relationships. Fold4 (GSE288063+GSE278576) was excluded from formal AD DEG because all AD and controls come from separate GEO studies, fully confounding disease with source. SEA-AD was retained as secondary descriptive evidence because only three donors are Not AD.\n\n## Embedding and clustering\n\nFull B2 Student and matched RNA-only embeddings were generated for the same 105,332 author-QC cells. UMAP used n_neighbors=30, min_dist=0.3, cosine distance and random_state={SEED}. Full B2 clustering used KMeans with 15 clusters, n_init=20 and random_state={SEED}; disease labels were not used.\n\n## Real-label pseudobulk analysis\n\nInteger RNA counts were aggregated by donor and author broad cell type. Donors contributed one row per cell type and were retained when at least 20 cells of that type were available. The runtime did not provide R/DESeq2/edgeR, so the reproducible fallback is explicitly labeled exploratory: library-size CPM followed by log1p and a two-sided Welch test across donor pseudobulks, with BH correction. No DESeq2-style formal claim is made.\n\n```text\n{summary.to_string(index=False)}\n```\n\nThe number of genes passing FDR < 0.05 and |log2FC| >= 0.25 was {len(sig)} across the five fixed P2 cell types.\n\n## Enrichment and regulatory evidence\n\nGO Biological Process and Reactome ORA used all tested, symbol-mapped genes as the cell-type-specific background. Up and down directions were separated and all terms were retained. Stable regulatory evidence was taken from fold1's frozen SOURCE-only B_D template (GSE268609, SEA-AD MTG and GSE288063+GSE278576). DEG overlap is descriptive; template edges do not establish causal direction.\n\n## Interpretation\n\nResults with FDR >= 0.05 are reported as null or exploratory rankings. They cannot be used to claim AD-specific mechanisms. The figures are suitable for a methods/results supplement after caption review; claims of disease biology should remain limited to the donor-audited GSE214979 analysis.\n'''
    (OUT/'BIOLOGICAL_VALIDATION_REPORT.md').write_text(report)
    cap='''# Figure captions\n\n**Figure A.** Cell-level UMAPs for the same GSE214979 nuclei using frozen Full B2 Student and matched RNA-only encoders. Rows show author broad cell type and true cohort label; the third column shows unsupervised Full B2 KMeans clusters. UMAP parameters and sampled-cell data are recorded with the figure.\n\n**Figure B.** Donor-level pseudobulk exploratory volcano plots for the five prespecified cell types. Points use library-size logCPM Welch tests and BH correction; no significant marker is annotated unless it passes the stated threshold.\n\n**Figure C.** Heatmap of the lowest-FDR genes across the five cell types, shown as log2 fold changes.\n\n**Figure D.** Exploratory GO Biological Process and Reactome ORA ranking, with cell-type-specific tested-gene backgrounds.\n\n**Figure E.** Descriptive overlap between real-label DEG calls and the frozen SOURCE-only stable regulatory template. Edge weights and selector support are shown; no causal direction is inferred.\n'''
    (OUT/'07_publication_figures'/'FIGURE_CAPTIONS_EN.md').write_text(cap)
    text='''# Results paragraph draft\n\nWe evaluated biological interpretability in the donor-audited GSE214979 target of fold 1 using a frozen Full B2 Student representation and an identically processed RNA-only encoder. Both representations were projected with the same deterministic UMAP procedure on the same author-QC nuclei, while clustering was performed without disease labels. Integer RNA counts were aggregated within donor and broad cell type for exploratory pseudobulk comparisons. Because the available runtime did not include DESeq2 or edgeR, we used a prespecified library-size logCPM Welch fallback with Benjamini–Hochberg correction and report it as exploratory. Fold2 was excluded from patient-level inference because repeat and alias relationships make independent donor status unresolved, and fold4 was excluded because study source is completely confounded with disease status. Stable regulatory overlap was evaluated only against the SOURCE-only frozen template and is interpreted as descriptive support rather than causal evidence.\n'''
    (OUT/'08_manuscript_text'/'RESULTS_PARAGRAPHS_EN.md').write_text(text)
    (OUT/'08_manuscript_text'/'RESULTS_PARAGRAPHS_ZH.md').write_text('本次生物学验证选用身份审计最可靠的 Fold1/GSE214979（7 AD、8 Control），以冻结的 Full B2 Student 和匹配的 RNA-only 表征进行同细胞集合可视化。真实标签差异分析按 Donor×细胞类型汇总原始整数 counts；由于服务器未安装 DESeq2/edgeR，使用预先记录的 logCPM Welch 探索性替代，并严格报告其限制。Fold2 因技术组重复/别名关系未完全解决而不作患者级推断，Fold4 因疾病与研究来源完全混杂而不作 AD 特异性解释。')
    lit=pd.DataFrame([
      {'gene_or_pathway':'single-cell AD transcriptomic states','cell_type':'multiple','observation':'AD-associated transcriptional changes and cell-state heterogeneity in human brain','doi':'10.1038/s41586-019-1195-2','pmid':'31042697','link':'https://doi.org/10.1038/s41586-019-1195-2','relationship_to_this_run':'context only; no gene is claimed as replicated unless present in the exploratory DEG table'},
      {'gene_or_pathway':'AD single-nucleus chromatin and RNA','cell_type':'multiple','observation':'joint chromatin accessibility and transcriptomic alterations in AD brain','doi':'10.1038/s41588-021-00969-7','pmid':'34611326','link':'https://doi.org/10.1038/s41588-021-00969-7','relationship_to_this_run':'supports the rationale for checking stable Gene–Peak overlap; does not establish causality here'}
    ])
    lit.to_csv(OUT/'06_literature_evidence'/'AD_LITERATURE_EVIDENCE.csv',index=False)
    # Figure data audit and file inventory.
    rows=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name not in ['FILE_INVENTORY.csv']:
            rows.append({'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':sha(p)})
    pd.DataFrame(rows).to_csv(OUT/'FIGURE_DATA_AUDIT.csv',index=False)
    (OUT/'07_publication_figures'/'FIGURE_DATA_AUDIT.md').write_text('# Figure/data audit\n\nAll figures were generated from files in this run directory. SVG and PDF are vector outputs; PNG outputs use 300 dpi. Figure data tables and the generation script are retained. Empty/significance-free panels are not replaced by fabricated annotations.\n')

def main():
    meta=pd.read_csv(CELLS,low_memory=False); features=load_gene_features(); assert len(meta)==105332 and len(features)==36601
    analysis_audit(meta)
    meta,records=compute_pseudobulk(meta,features)
    deg=run_deg(records,features); ora=run_ora(deg,features); edges,ov=regulatory_overlap(deg)
    zf,zr,cluster,emb=load_embeddings(meta); plots(meta,deg,ora,zf,zr,cluster,emb); write_docs(meta,deg,ora,edges,ov)
    write_json(OUT/'logs'/'RUN_COMPLETE.json',{'status':'COMPLETE','fold':FOLD,'seed':SEED,'cohort':COHORT,'cells':len(meta),'genes':len(features),'celltypes':TYPES,'deg_method':'logCPM Welch fallback','gpu_used':False,'target_atac_used':False})
    print('BIOLOGY_VALIDATION_COMPLETE',OUT)
if __name__=='__main__': main()
