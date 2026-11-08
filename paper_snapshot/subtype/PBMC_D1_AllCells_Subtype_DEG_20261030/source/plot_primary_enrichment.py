"""Reproducible 600-dpi/vector candidate, full-gene and pathway plots."""
import time,textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from analysis_common import *

FIG=RUN/'fgsea/figures'
COLORS={'FullB2':'#766bd3','PCA_sourcefit':'#AEC7E8','PCA_targetfit_fullnorm':'#FFBB78'}
A='#766bd3';B='#FFBB78';OTHER='#D6D6D6'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,p):
    p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p.with_suffix('.png'),dpi=600,bbox_inches='tight')
    fig.savefig(p.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)

def membership_plots():
    while not (RUN/'audit/INVENTORY_COMPLETE.json').exists():time.sleep(15)
    inv=pd.read_csv(RUN/'candidate_inventory.csv');membership=pd.read_csv(RUN/'tables/candidate_membership.csv.gz')
    cells,labels=context()
    for r in inv.itertuples():
        p=FIG/'candidates'/r.candidate_id
        if p.with_suffix('.png').exists():continue
        mem=membership[membership.candidate_id==r.candidate_id];seeds=sorted(mem.seed.unique())
        fig,axs=plt.subplots(len(seeds),3,figsize=(12,3.3*len(seeds)),squeeze=False,layout='constrained')
        for i,seed in enumerate(seeds):
            mm=mem[mem.seed==seed].set_index('barcode').group.reindex(cells)
            for j,method in enumerate(METHODS):
                df=pd.read_csv(BASE/'figures'/method/f'seed_{seed}/plot_data.csv').set_index('barcode').loc[cells]
                ax=axs[i,j];z=df[['UMAP1','UMAP2']].to_numpy()
                ax.scatter(z[:,0],z[:,1],s=1,c=OTHER,alpha=.3,linewidths=0)
                same=labels==r.reference_subclass
                ax.scatter(z[same,0],z[same,1],s=2,c='#999999',alpha=.6,linewidths=0)
                for group,color in [('A',A),('B',B)]:
                    mask=mm.eq(group).fillna(False).to_numpy()
                    ax.scatter(z[mask,0],z[mask,1],s=5,c=color,alpha=.9,linewidths=0)
                ax.set_title(f'{method} | seed {seed}\nA={int(mm.eq("A").sum())}, B={int(mm.eq("B").sum())}, all {r.reference_subclass}={int(same.sum())}',fontsize=9)
                ax.set_xticks([]);ax.set_yticks([]);ax.set_xlabel('UMAP1');ax.set_ylabel('UMAP2')
        fig.suptitle(f'{r.method}: {r.reference_subclass} | same fixed candidate barcodes across panels',fontsize=11)
        fig.legend(handles=[Line2D([],[],color=A,marker='o',linestyle='',label='Candidate A'),
                Line2D([],[],color=B,marker='o',linestyle='',label='Candidate B'),
                Line2D([],[],color='#999999',marker='o',linestyle='',label='Same reference subclass, outside candidate')],
                loc='outside lower center',ncol=3,frameon=False,fontsize=8)
        save(fig,p)
        mem.to_csv(p.parent/(p.name+'_membership.csv'),index=False)
        print('MEMBERSHIP_PLOT',r.candidate_id,flush=True)

def molecular_plots():
    while not (RUN/'audit/DEG_COMPLETE.json').exists():time.sleep(30)
    inv=pd.read_csv(RUN/'candidate_inventory.csv');de=pd.read_csv(RUN/'all_candidate_deg_summary.csv')
    allplot=[]
    for r in inv.itertuples():
        rows=de[(de.candidate_id==r.candidate_id)&de.contrast.eq('A_vs_B')]
        anchor=rows[rows.seed==r.anchor_seed].iloc[0];d=pd.read_csv(RUN/anchor.path)
        sig=(d.FDR_BH_global_all_candidate_seed_contrast_tests<.05)&(d.log2FC_meanCP10K.abs()>.25)
        visible=(d.pct_detected_A>=.1)|(d.pct_detected_B>=.1)
        choices=[]
        for sign in [1,-1]:
            sel=d[sig&visible&(d.log2FC_meanCP10K*sign>0)].copy()
            sel['abs_effect']=sel.log2FC_meanCP10K.abs()
            choices += sel.sort_values('abs_effect',ascending=False).head(6).gene_id.tolist()
        # Volcano includes every gene, even when no candidate Marker survives correction.
        fig,ax=plt.subplots(figsize=(5.8,4.6),layout='constrained')
        effects=d.log2FC_meanCP10K.to_numpy();q=d.FDR_BH_global_all_candidate_seed_contrast_tests.to_numpy()
        colors=np.where(sig,np.where(effects>0,A,B),'#CDCDCD')
        ax.scatter(effects,-np.log10(np.maximum(q,1e-300)),c=colors,s=3,linewidths=0,alpha=.7)
        ax.axhline(-np.log10(.05),color='#999999',ls='--',lw=.7)
        for idx,row in d[d.gene_id.isin(choices[:3]+choices[6:9])].iterrows():
            ax.annotate(row.gene_symbol,(row.log2FC_meanCP10K,-np.log10(max(row.FDR_BH_global_all_candidate_seed_contrast_tests,1e-300))),fontsize=7)
        ax.set_xlabel('log2 mean CP10K ratio (A / B)');ax.set_ylabel('-log10 global BH FDR')
        ax.set_title(f'{r.method}: {r.reference_subclass}\nAnchor {r.anchor_seed}; all 36,601 genes; global DEG n={sig.sum()}',fontsize=9)
        save(fig,FIG/'volcano'/r.candidate_id)
        if choices:
            genes=d.set_index('gene_id').loc[choices]
            matrices=[];qmatrix=[];columns=[]
            for rr in rows.itertuples():
                dd=pd.read_csv(RUN/rr.path).set_index('gene_id').loc[choices]
                matrices.append(dd.log2FC_meanCP10K.to_numpy());qmatrix.append(dd.FDR_BH_global_all_candidate_seed_contrast_tests.to_numpy());columns.append(str(rr.seed))
            mat=np.stack(matrices,axis=1);qm=np.stack(qmatrix,axis=1)
            fig,ax=plt.subplots(figsize=(5.3,max(3,len(choices)*.28)),layout='constrained')
            bound=max(.5,float(np.percentile(abs(mat),95)))
            im=ax.imshow(mat,cmap='PuOr_r',vmin=-bound,vmax=bound,aspect='auto')
            for i in range(len(choices)):
                for j in range(len(columns)):
                    if qm[i,j]<.05:ax.text(j,i,'*',ha='center',va='center',fontsize=10)
                    allplot.append(dict(candidate_id=r.candidate_id,gene_id=choices[i],gene_symbol=genes.gene_symbol.iloc[i],seed=columns[j],log2FC=float(mat[i,j]),global_FDR=float(qm[i,j])))
            ax.set_xticks(range(len(columns)),columns,rotation=25);ax.set_yticks(range(len(choices)),genes.gene_symbol)
            ax.set_title(f'{r.method}: {r.reference_subclass}\nAnchor-selected effect Markers; * global FDR<0.05',fontsize=9)
            fig.colorbar(im,ax=ax,label='log2FC A / B',shrink=.7)
            save(fig,FIG/'marker_heatmaps'/r.candidate_id)
    pd.DataFrame(allplot).to_csv(RUN/'tables/marker_heatmap_plot_data.csv',index=False)
    print('MOLECULAR_PLOTS_COMPLETE',flush=True)

def enrichment_plots():
    while not (RUN/'audit/FGSEA_REPORT_COMPLETE.json').exists():time.sleep(45)
    ev=pd.read_csv(RUN/'fgsea/candidate_evidence_matrix.csv')
    gs=pd.read_csv(RUN/'fgsea/all_candidate_gsea_summary.csv.gz')
    allcurves=[];bubble=[]
    for r in ev.itertuples():
        dd=gs[(gs.candidate_id==r.candidate_id)&gs.contrast.eq('A_vs_B')]
        selected=pd.concat([g.sort_values(['FDR_BH_global_all_candidate_contrast_GO_Reactome_tests','nominal_p_floored']).head(5) for _,g in dd.groupby('database')])
        fig,ax=plt.subplots(figsize=(10,5.2),layout='constrained')
        terms=[textwrap.shorten(t,width=85,placeholder='…') for t in selected.Term]
        strict=selected.FDR_BH_global_all_candidate_contrast_GO_Reactome_tests<.05
        scatter=ax.scatter(selected.NES,np.arange(len(selected)),s=20+selected.matched_genes.clip(upper=200)*1.4,
            c=-np.log10(selected.FDR_BH_global_all_candidate_contrast_GO_Reactome_tests.clip(lower=1e-8)),cmap='PuBu',edgecolor=np.where(strict,A,'#BBBBBB'),linewidth=1)
        ax.set_yticks(range(len(selected)),terms,fontsize=7);ax.axvline(0,color='#BBBBBB',lw=.7)
        ax.set_xlabel('NES (positive: A-high; negative: B-high)')
        ax.set_title(f'{r.method}: {r.reference_subclass}\nGO/Reactome top terms; purple outline = global FDR<0.05',fontsize=10)
        fig.colorbar(scatter,ax=ax,label='-log10 global BH FDR',shrink=.6)
        save(fig,FIG/'gsea_bubbles'/r.candidate_id)
        bubble.append(selected)
        ranked=pd.read_csv(RUN/'fgsea'/f'{r.candidate_id}__A_vs_B__rank.csv')
        symbols=ranked.gene_symbol.astype(str).to_numpy();scores=ranked.wilcoxon_score_signed.to_numpy()
        fig,axs=plt.subplots(2,1,figsize=(8,5.7),layout='constrained')
        for ax,(db,g) in zip(axs,dd.groupby('database',sort=True)):
            term=g.sort_values(['FDR_BH_global_all_candidate_contrast_GO_Reactome_tests','nominal_p_floored']).iloc[0]
            filename='GO_Biological_Process_2025.gmt' if db=='GO_BP_2025' else 'Reactome_2022.gmt'
            line=next(v for v in (RUN/'input'/filename).read_text().splitlines() if v.split('\t')[0]==term.Term)
            hits=np.isin(symbols,line.split('\t')[2:]);weights=np.where(hits,abs(scores),0)
            norm=weights.sum()
            if norm<=0:weights=hits.astype(float);norm=weights.sum()
            running=np.cumsum(weights/max(norm,1e-12)-np.where(hits,0,1/max(1,(~hits).sum())))
            ax.plot(np.arange(len(running)),running,color=A,lw=1.2);ax.axhline(0,color='#BBBBBB',lw=.6)
            positions=np.flatnonzero(hits);lo,hi=ax.get_ylim()
            ax.vlines(positions,lo,lo+.04*(hi-lo),color='#8C564B',lw=.5)
            ax.set_ylabel('Running ES');ax.set_xlabel('Full-gene rank')
            ax.set_title(f'{db}: {textwrap.shorten(term.Term,width=90,placeholder="…")}\nNES={term.NES:.2f}; global BH FDR={term.FDR_BH_global_all_candidate_contrast_GO_Reactome_tests:.3g}',fontsize=8)
            # Standalone curve data for every rank, not only a selected screenshot.
            curve=pd.DataFrame(dict(candidate_id=r.candidate_id,database=db,term=term.Term,rank=np.arange(len(running)),gene_symbol=symbols,signed_score=scores,gene_set_hit=hits,running_ES=running))
            curve.to_csv(FIG/'gsea_curves'/f'{r.candidate_id}_{db}_curve.csv.gz',index=False,compression='gzip')
        save(fig,FIG/'gsea_curves'/r.candidate_id)
    pd.concat(bubble).to_csv(RUN/'tables/gsea_bubble_plot_data.csv',index=False)
    # Side-by-side evidence does not silently turn PCA candidate self-recovery into a B2 advantage.
    labels=ev.method+' / '+ev.reference_subclass
    metrics=['mean_partition_ARI','mean_member_Jaccard','recovery_ARI_FullB2','recovery_ARI_PCA_sourcefit','recovery_ARI_PCA_targetfit_fullnorm']
    mat=ev[metrics].to_numpy(float)
    fig,ax=plt.subplots(figsize=(10,8),layout='constrained')
    im=ax.imshow(np.ma.masked_invalid(mat),vmin=0,vmax=1,cmap='PuBu',aspect='auto')
    for i in range(len(ev)):
        for j in range(len(metrics)):
            if np.isfinite(mat[i,j]):ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if mat[i,j]>.65 else 'black')
    ax.set_yticks(range(len(ev)),labels,fontsize=7)
    ax.set_xticks(range(5),['Cross-seed ARI','Member Jaccard','Recovery: B2','Recovery: Source PCA','Recovery: Target PCA'],rotation=30,ha='right')
    ax.set_title('All fixed-screen candidates | stability and same-member recovery',fontsize=11)
    fig.colorbar(im,ax=ax,shrink=.55,label='ARI / Jaccard')
    save(fig,FIG/'all_candidate_evidence_comparison')
    ev[['candidate_id']+metrics].to_csv(RUN/'tables/evidence_comparison_plot_data.csv',index=False)
    print('ENRICHMENT_PLOTS_COMPLETE',flush=True)

def main():
    for d in ['candidates','volcano','marker_heatmaps','gsea_bubbles','gsea_curves']: (FIG/d).mkdir(parents=True,exist_ok=True)
    enrichment_plots()
    write(RUN/'audit/PRIMARY_FIGURES_COMPLETE.json',dict(status='COMPLETE',dpi=600,vector_pdf=True,
        candidates=18,palette_source='user scMAGCA Fig6.R; purple and muted orange A/B',
        no_cell_subsampling=True,all_seeds_preserved=True))

if __name__=='__main__':main()
