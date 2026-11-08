"""Exploratory TARGET predicted-group donor-level DEG and GO/Reactome ORA."""
import argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, hypergeom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ad_runtime as a
import ad_rna as r

SETS=a.RESULTS/'AD_Dmode_4Cohort_20261024/source_gene_sets'
LIBRARIES={'GO_BP':SETS/'GO_Biological_Process_2025.gmt','Reactome':SETS/'Reactome_2022.gmt'}

def bh(p):
    p=np.asarray(p,float);order=np.argsort(p,kind='stable')
    ranked=p[order]*len(p)/np.arange(1,len(p)+1)
    out=np.empty_like(p);out[order]=np.minimum(1,np.minimum.accumulate(ranked[::-1])[::-1]);return out

def donor_means(name):
    dest=a.OUT/'biology/donor_RNA'/name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():
        cm=a.read(dest/'COMPLETE.json');assert a.sha(dest/'donor_means.npz')==cm['sha256']
        # Own cache, verified above; pandas previously serialized unit IDs as objects.
        return np.load(dest/'donor_means.npz',allow_pickle=True)
    ids=a.identities(name);units=np.sort(ids.unit_id.unique());lookup={u:i for i,u in enumerate(units)}
    group=ids.unit_id.map(lookup).to_numpy();sizes=np.bincount(group,minlength=len(units))
    logged=np.zeros((len(units),len(r.GENES)),float);cp=np.zeros_like(logged)
    for first,z in r.normalized_blocks(name):
        for j in np.unique(group[first:first+z.shape[0]]):
            rows=np.flatnonzero(group[first:first+z.shape[0]]==j);block=z[rows]
            logged[j]+=np.asarray(block.sum(0)).ravel()
            block=block.copy();block.data=np.expm1(block.data);cp[j]+=np.asarray(block.sum(0)).ravel()
    logged/=sizes[:,None];cp/=sizes[:,None]
    np.savez_compressed(dest/'donor_means.npz',units=units.astype(str),genes=r.GENES.astype(str),
                        log_mean=logged,cp10k_mean=cp,cells=sizes)
    a.write(dest/'COMPLETE.json',dict(status='COMPLETE',units=len(units),cells=int(sizes.sum()),
            aggregation='mean across all QC cells per donor/technical group',normalization='full36601CP10K,log1p',
            sha256=a.sha(dest/'donor_means.npz')))
    return np.load(dest/'donor_means.npz')

def ora(query,background,path):
    query=set(query)&background;rows=[]
    with path.open() as f:
        for line in f:
            fields=line.rstrip('\n').split('\t');term=fields[0]
            members=set(fields[2:])&background
            if not 10<=len(members)<=2000:continue
            overlap=sorted(query&members)
            p=float(hypergeom.sf(len(overlap)-1,len(background),len(members),len(query))) if query else 1.
            rows.append(dict(term=term,overlap=len(overlap),term_size=len(members),query_size=len(query),
                             background_size=len(background),pvalue=p,genes=';'.join(overlap)))
    out=pd.DataFrame(rows,columns=['term','overlap','term_size','query_size','background_size','pvalue','genes'])
    out['FDR']=bh(out.pvalue) if len(out) else pd.Series(dtype=float)
    return out.sort_values(['FDR','pvalue','term'])

def run(fold,seed,method):
    dest=a.OUT/'biology'/fold/f'seed_{seed}'/method;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():return
    ep=a.OUT/'evaluation'/fold/f'seed_{seed}'/method
    assert a.read(ep/'COMPLETE.json')['status']=='COMPLETE'
    # Only prediction and unit identifiers are imported. True diagnoses are never read here.
    pred=pd.read_csv(ep/'predictions_blind.csv',usecols=['cohort','unit_id','prediction','probability_AD'],dtype={'unit_id':str})
    pos=int(pred.prediction.sum());neg=len(pred)-pos
    base=dict(fold=fold,seed=seed,method=method,n_predicted_AD=pos,n_predicted_Control=neg,
              groups='predicted AD vs predicted Control',true_diagnosis_used=False,exploratory=True,
              prediction_sha256=a.sha(ep/'predictions_blind.csv'))
    if min(pos,neg)<2:
        a.write(dest/'COMPLETE.json',dict(status='NOT_EVALUABLE',reason='at least2 units required in each predicted group',**base))
        return
    name=a.protocol()['folds'][fold]['target'];arrays=donor_means(name)
    lookup={u:i for i,u in enumerate(arrays['units'])};ix=[lookup[u] for u in pred.unit_id]
    log=arrays['log_mean'][ix];cp=arrays['cp10k_mean'][ix];ad=pred.prediction.to_numpy()==1
    keep=(cp>0).sum(0)>=2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');test=ttest_ind(log[ad][:,keep],log[~ad][:,keep],axis=0,equal_var=False)
    p=np.nan_to_num(test.pvalue,nan=1.,posinf=1.,neginf=1.)
    genes=arrays['genes'][keep];symbol_key=next(k for k in ['gene_name','gene_symbol','name','symbol'] if k in r.FEATURES)
    sym=dict(zip(r.GENES,r.FEATURES[symbol_key].fillna('').astype(str)))
    table=pd.DataFrame(dict(gene_id=genes,symbol=[sym[g] for g in genes],pvalue=p,FDR=bh(p),
        log2FC=np.log2((cp[ad][:,keep].mean(0)+1)/(cp[~ad][:,keep].mean(0)+1)),
        predicted_AD_mean_log=log[ad][:,keep].mean(0),predicted_Control_mean_log=log[~ad][:,keep].mean(0),
        detected_units=(cp[:,keep]>0).sum(0)))
    table=table.sort_values(['FDR','pvalue','gene_id']);table.to_csv(dest/'DEG_all_tested.csv.gz',index=False)
    sig=table[(table.FDR<.05)&(table.log2FC.abs()>.25)]
    sig.to_csv(dest/'DEG_significant.csv',index=False)
    table.head(200).to_csv(dest/'DEG_top200_by_FDR.csv',index=False)
    all_library_genes=set()
    for path in LIBRARIES.values():
        for line in path.read_text().splitlines():all_library_genes.update(line.split('\t')[2:])
    background=set(table.symbol)-{''};mapped_background=background&all_library_genes
    pd.DataFrame({'symbol':sorted(mapped_background)}).to_csv(dest/'ORA_background.csv',index=False)
    ora_counts={}
    for direction,mask in [('AD_up',sig.log2FC>0),('Control_up',sig.log2FC<0)]:
        query=sig[mask].head(200);query.to_csv(dest/(direction+'_query_genes.csv'),index=False)
        for library,path in LIBRARIES.items():
            library_genes=set(g for line in path.read_text().splitlines() for g in line.split('\t')[2:])
            bg=background&library_genes
            out=ora(query.symbol,bg,path)
            out.to_csv(dest/(direction+'_'+library+'_all.csv.gz'),index=False)
            enriched=out[(out.FDR<.05)&(out.overlap>=3)]
            enriched.to_csv(dest/(direction+'_'+library+'_significant.csv'),index=False)
            out.head(20).to_csv(dest/(direction+'_'+library+'_top20.csv'),index=False)
            ora_counts[direction+'_'+library]=len(enriched)
    fig,ax=plt.subplots(figsize=(6,4.5));significant=(table.FDR<.05)&(table.log2FC.abs()>.25)
    ax.scatter(table.log2FC,-np.log10(np.maximum(table.FDR,1e-300)),c=np.where(significant,'#c8403a','#999999'),s=5,alpha=.5)
    ax.axhline(-np.log10(.05),color='black',lw=.7,ls='--');ax.axvline(-.25,color='black',lw=.5,ls=':');ax.axvline(.25,color='black',lw=.5,ls=':')
    ax.set(xlabel='log2FC: predicted AD / predicted Control',ylabel='-log10 BH FDR',
           title=f'{fold} | {method} | seed {seed}\nExploratory donor-level Welch test ({pos} vs {neg})')
    ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(dest/'DEG_volcano.png',dpi=200);plt.close(fig)
    a.write(dest/'COMPLETE.json',dict(status='COMPLETE',**base,tested_genes=len(table),significant_genes=len(sig),
        enrichment_counts=ora_counts,background='tested genes mapped to each gene-set library',
        gene_set_sha256={str(p):a.sha(p) for p in LIBRARIES.values()},
        limitations='expression-derived predicted groups, technical-group independence and cohort endpoint heterogeneity'))
    print('BIOLOGY_COMPLETE',fold,seed,method,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',required=True);p.add_argument('--seed',required=True,type=int)
    p.add_argument('--method',required=True);v=p.parse_args();run(v.fold,v.seed,v.method)
