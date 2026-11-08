"""Reproducible plots read exported CSVs only; shared scMAGCA Fig6 palette."""
import json,sys,textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT=Path(__file__).resolve().parents[1]
Q='FDR_BH_all_folds_all_seeds'
COLORS={'A':'#766bd3','B':'#FFBB78','Other':'#D8D8D8'}
METHODS=['PCA_sourcefit','PCA_targetfit_fullnorm','FullB2']
NAMES={'PCA_sourcefit':'Source-fit PCA','PCA_targetfit_fullnorm':'Target-fit PCA','FullB2':'Full B2'}
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,'savefig.pad_inches':.06})

def context():
    p=ROOT/'case_studies'/sys.argv[1]
    return p,json.loads((p/'case.json').read_text())

def save(fig,p,name,caption):
    out=p/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(out/(name+'.png'),dpi=600,bbox_inches='tight');plt.close(fig)
    (out/(name+'_caption_zh.md')).write_text(caption+'\n',encoding='utf-8')

def scatter(ax,d,group='fixed_group',focus=False):
    for g in ['Other','A','B']:
        dd=d[d[group].eq(g)]
        ax.scatter(dd.UMAP1,dd.UMAP2,s=2 if g=='Other' else 6,c=COLORS[g],alpha=.45 if g=='Other' else .85,linewidths=0)
    ax.set_xticks([]);ax.set_yticks([])
    ax.set_ylabel('UMAP 2')

def umap():
    p,c=context();d=pd.read_csv(p/'umap_all_seeds.csv.gz');d=d[d.seed.eq(c['seed'])]
    fig,axes=plt.subplots(2,3,figsize=(10,6.4),layout='constrained')
    for j,m in enumerate(METHODS):
        dd=d[d.method.eq(m)]
        scatter(axes[0,j],dd)
        scatter(axes[1,j],dd[dd.reference_label.eq(c['reference_subclass'])])
        for i in range(2):axes[i,j].set_xlabel(NAMES[m]+' | UMAP 1')
    handles=[Line2D([],[],marker='o',ls='',color=COLORS[g],label=f'{g} (n={c["n_"+g]})') for g in ['A','B']]
    fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False)
    save(fig,p,'01_same_cells_umap',f"{c['fold']} {c['reference_subclass']}，seed {c['seed']}。上排显示完整 Target {c['target_cells']} 个细胞，下排显示全部同参考类 {c['subclass_cells']} 个细胞；下排直接截取原 UMAP 坐标，没有重算局部 UMAP。三种方法以完全相同的 {c['n_A']+c['n_B']} 个固定 barcode 着色，A={c['n_A']}、B={c['n_B']}，灰色为其他细胞。候选来自 {c.get('origin_method','FullB2')}，因此存在发现方法选择偏倚；二维分离不能代替独立亚型验证。代表 fold/seed 为事后探索性选择。")
    fig,axes=plt.subplots(3,3,figsize=(10,8.8),layout='constrained')
    d=pd.read_csv(p/'umap_all_seeds.csv.gz')
    for i,s in enumerate(sorted(d.seed.unique())):
        for j,m in enumerate(METHODS):
            dd=d[d.seed.eq(s)&d.method.eq(m)&d.reference_label.eq(c['reference_subclass'])]
            scatter(axes[i,j],dd)
            axes[i,j].set_xlabel(NAMES[m]+' | UMAP 1')
            axes[i,j].set_ylabel(f'seed {s}\nUMAP 2')
    fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False)
    save(fig,p,'02_fixed_members_all_seeds',f"相同的代表 seed {c['seed']} 候选成员在三个模型/可视化 seeds 下的投影。未在某个 seed 独立发现分裂时仍显示固定成员；该图不能证明该 seed 自行恢复了候选，独立分裂状态见 all_seed_evidence.csv 和 cross_seed_stability.csv。")

def markers():
    p,c=context();d=pd.read_csv(p/'marker_dotplot.csv');genes=list(dict.fromkeys(d.gene))
    fig,ax=plt.subplots(figsize=(max(7,len(genes)*.42),3),layout='constrained')
    cm=LinearSegmentedColormap.from_list('scmagca_expression',['#F4F1F7','#C5B0D5','#766bd3'])
    dots=ax.scatter(d.gene.map({g:i for i,g in enumerate(genes)}),d.group.map({'A':0,'B':1}),
        s=15+220*d.fraction_detected,c=d.mean_log1pCP10K,cmap=cm,vmin=0,
        edgecolors='#888888',linewidths=.35)
    ax.set_xticks(range(len(genes)),genes,rotation=55,ha='right',fontstyle='italic')
    ax.set_yticks([0,1],[f'A (n={c["n_A"]})',f'B (n={c["n_B"]})']);ax.set_ylim(-.5,1.5)
    fig.colorbar(dots,ax=ax,label='Mean log1p(CP10K)',shrink=.7)
    handles=[plt.scatter([],[],s=15+220*v,c='#C5B0D5',label=f'{int(v*100)}%') for v in [.1,.5,1]]
    ax.legend(handles=handles,title='Detected',loc='upper left',bbox_to_anchor=(1.25,1),frameon=False)
    save(fig,p,'03_marker_dotplot',f"全基因原始 RNA counts 按完整文库 CP10K/log1p 计算。颜色为组内平均 log1p(CP10K)，面积为检测比例。基因包括预先定义的功能标记和两方向各最多4个全局 FDR<0.05、|log2FC|>0.5、任一组检测率≥10%的大效应基因；预定义基因可能不显著，逐基因统计见 marker_dotplot.csv。没有仅限于模型输入基因。")
    d=pd.read_csv(p/'all_gene_DEG.csv.gz');sig=(d[Q]<.05)&d.log2FC_meanCP10K.abs().gt(.25)
    fig,ax=plt.subplots(figsize=(6,4.8),layout='constrained')
    for label,color,mask in [('Other','#D8D8D8',~sig),('A higher',COLORS['A'],sig&d.log2FC_meanCP10K.gt(0)),('B higher',COLORS['B'],sig&d.log2FC_meanCP10K.lt(0))]:
        dd=d[mask];ax.scatter(dd.log2FC_meanCP10K,-np.log10(dd[Q].clip(lower=1e-300)),s=5,c=color,alpha=.6,linewidths=0,label=f'{label} ({len(dd)})')
    ax.axhline(-np.log10(.05),c='#999999',ls='--',lw=.6)
    for v in [-.25,.25]:ax.axvline(v,c='#AAAAAA',ls=':',lw=.6)
    for g in genes[:8]:
        hit=d[d.gene_symbol.eq(g)].sort_values(Q).iloc[0]
        if hit[Q]<.05:ax.annotate(g,(hit.log2FC_meanCP10K,-np.log10(max(hit[Q],1e-300))),fontsize=7,xytext=(2,2),textcoords='offset points')
    ax.set_xlabel('log2 fold change (A / B; mean CP10K)');ax.set_ylabel('-log10 BH FDR (all folds / seeds / contrasts)')
    ax.legend(frameon=False,fontsize=8)
    save(fig,p,'04_all_gene_volcano',f"所有36,601个已检验基因均保留；横轴为平均 CP10K 比值的 log2FC（伪计数1e-4），纵轴为覆盖四折全部候选、seed和三种比较的 BH-FDR。彩点要求 FDR<0.05 且 |log2FC|>0.25。细胞级检验为同队列聚类后的关联分析，未把细胞或模型 seed 视作独立供者重复。")

def pathways():
    p,c=context()
    for scope in ['full','expressed']:
        d=pd.read_csv(p/f'GSEA_{scope}.csv');tops=[]
        for db,g in d.groupby('database'):
            tops.append(g.sort_values([Q,'nominal_p_floored']).head(5))
        top=pd.concat(tops,ignore_index=True);top.to_csv(p/f'GSEA_{scope}_bubble_plot_data.csv',index=False)
        labels=[textwrap.fill(('GO: ' if r.database.startswith('GO') else 'Reactome: ')+r.Term,58) for r in top.itertuples()]
        fig,ax=plt.subplots(figsize=(10,7),layout='constrained')
        norm=Normalize(vmin=0,vmax=max(2.,float(-np.log10(top[Q].clip(lower=1e-12)).min())+.15))
        cm=plt.get_cmap('Purples');color=cm(norm(-np.log10(top[Q].clip(lower=1e-12))))
        sig=top[Q].lt(.05).to_numpy();yy=np.arange(len(top));sizes=20+np.sqrt(top.matched_genes)*10
        ax.scatter(top.NES[sig],yy[sig],s=sizes[sig],facecolors=color[sig],edgecolors='#555555',linewidths=.8)
        ax.scatter(top.NES[~sig],yy[~sig],s=sizes[~sig],facecolors='none',edgecolors=color[~sig],linewidths=1.3)
        ax.set_yticks(range(len(top)),labels,fontsize=8);ax.axvline(0,color='#AAAAAA',ls=':',lw=.6)
        ax.set_xlabel('Normalized enrichment score (positive: A; negative: B)')
        fig.colorbar(ScalarMappable(norm=norm,cmap=cm),ax=ax,label='-log10 global BH FDR',shrink=.6)
        ax.legend(handles=[Line2D([],[],marker='o',ls='',markerfacecolor='#766bd3',markeredgecolor='#555555',label='FDR < 0.05'),
            Line2D([],[],marker='o',ls='',markerfacecolor='none',markeredgecolor='#766bd3',label='FDR >= 0.05')],
            frameon=False,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2)
        save(fig,p,f'05_GSEA_{scope}_bubble',f"{scope} 排序背景下 GO BP2025 与 Reactome2022 各按四折全局 BH-FDR/nominal P 取前5条。大小表示匹配基因数，实心为全局 FDR<0.05，空心为未达显著。完整排序保留所有被检验的有符号统计量；expressed 为只保留任一比较组检测到基因的敏感性分析，两者不能混写。nperm=10,000，逐项 NES、FDR、leading-edge 见同名 CSV。")
        curve=pd.read_csv(p/f'GSEA_{scope}_curves.csv.gz');meta=pd.read_csv(p/f'GSEA_{scope}_curves_metadata.csv')
        fig,axes=plt.subplots(1,2,figsize=(10,3.2),layout='constrained')
        for ax,r in zip(axes,meta.itertuples()):
            z=curve[curve.Term.eq(r.Term)&curve.database.eq(r.database)]
            ax.plot(z['rank'],z.running_ES,c=COLORS['A'] if r.NES>0 else COLORS['B'],lw=1.1)
            hits=z[z.hit];ax.plot(hits['rank'],np.full(len(hits),min(z.running_ES.min(),0)-.04),'|',c='#777777',ms=2)
            ax.set_xlabel('Gene rank');ax.set_ylabel('Running enrichment score')
            ax.text(.40 if r.NES>0 else .02,.98,textwrap.fill(r.Term,38)+f'\nNES={r.NES:.2f}; global FDR={r.global_FDR:.3g}',
                    transform=ax.transAxes,va='top',fontsize=7)
        save(fig,p,f'06_GSEA_{scope}_curves',f"{scope} 背景下每个数据库在免疫/造血相关条目中按全局 FDR 最低选择一条作事后说明，未用于筛选候选成员。曲线从原有完整排序精确重建并与 fgsea 的 ES 数值核对。若标示 FDR≥0.05，仅为探索性趋势；不能因名称包含特定组织或感染而诊断对应身份或疾病。")
    d=pd.read_csv(p/'ORA.csv');top=pd.concat([g.sort_values([Q,'nominal_p']).head(4) for _,g in d.groupby('database')],ignore_index=True)
    top.to_csv(p/'ORA_plot_data.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    sig=top[Q].lt(.05).to_numpy();yy=np.arange(len(top));cc=top.direction.map({'A_high':COLORS['A'],'B_high':COLORS['B']}).to_numpy()
    ax.scatter(-np.log10(top.loc[sig,Q].clip(lower=1e-300)),yy[sig],s=20+top.loc[sig,'overlap']*10,
       c=cc[sig],edgecolors='#555555',lw=.8)
    ax.scatter(-np.log10(top.loc[~sig,Q].clip(lower=1e-300)),yy[~sig],s=20+top.loc[~sig,'overlap']*10,
       facecolors='none',edgecolors=cc[~sig],lw=1.2)
    ax.set_yticks(range(len(top)),[textwrap.fill(r.direction+' / '+r.Term,65) for r in top.itertuples()],fontsize=8)
    ax.axvline(-np.log10(.05),c='#AAAAAA',ls='--',lw=.6);ax.set_xlabel('-log10 all-fold BH FDR (ORA)')
    ax.legend(handles=[Line2D([],[],marker='o',ls='',markerfacecolor=COLORS[g],markeredgecolor='#555555',label=g+' higher') for g in ['A','B']]+
        [Line2D([],[],marker='o',ls='',markerfacecolor='none',markeredgecolor='#777777',label='Open: FDR >= 0.05')],frameon=False,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3)
    save(fig,p,'07_GO_Reactome_ORA',"ORA 的基因输入沿用各折全部候选/seed/比较校正后的 DEG（FDR<0.05、|log2FC|>0.25），背景为本比较检测到的基因；随后对四折全部通路/比较/方向做 BH。它与全基因排序 GSEA 是不同检验，不能合并显著条目数。大小表示命中基因数；空心为不显著条目，实心为FDR<0.05，紫/橙分别为 A/B 上调。")

def qc_stability():
    p,c=context();d=pd.read_csv(p/'cell_qc_modules.csv');stats=pd.read_csv(p/'QC_module_statistics.csv')
    wanted=['RNA_UMI','detected_genes','percent_mitochondrial','ATAC_peak_read_counts','ATAC_detected_peaks','module_Cell_cycle','module_Cytotoxic','module_Naive_memory']
    cols=[x for x in wanted if x in d];fig,axes=plt.subplots(2,4,figsize=(11,5.8),layout='constrained')
    for ax,col in zip(axes.flat,cols):
        vals=[d.loc[d.group.eq(g),col].to_numpy() for g in ['A','B']]
        bp=ax.boxplot(vals,tick_labels=['A','B'],patch_artist=True,showfliers=False,widths=.55)
        for box,g in zip(bp['boxes'],['A','B']):box.set_facecolor(COLORS[g]);box.set_alpha(.75)
        lower,upper=ax.get_ylim();ax.set_ylim(lower,upper+(upper-lower)*.19)
        ax.set_ylabel(col.replace('_',' '),fontsize=8)
        row=stats[stats.variable.eq(col)].iloc[0]
        ax.text(.03,.97,f"r={row.rank_biserial:.2f}\nq={row.FDR_BH_all_qc_and_module_tests:.2g}",transform=ax.transAxes,va='top',fontsize=7)
    for ax in list(axes.flat)[len(cols):]:ax.axis('off')
    save(fig,p,'08_QC_and_modules',f"固定 A/B 成员的可获得 QC 与功能模块分布，A={c['n_A']}、B={c['n_B']}；r 为 rank-biserial 关联效应，q 覆盖本折全部候选的 QC/模块检验。模块为每基因 z(log1p CP10K) 的简单均值，非 Seurat AddModuleScore；Cell_cycle 是简化增殖模块，非完整 S/G2M 分数。Peak read-count/binary 非 fragments。缺少 doublet/TSS 或多供者控制时不能声称完全排除混杂。")
    d=pd.read_csv(p/'cross_seed_stability.csv');fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained')
    for i,r in enumerate(d.itertuples()):
        for offset,col,color in [(-.22,'member_Jaccard','#AEC7E8'),(0,'group_Jaccard','#C5B0D5'),(.22,'partition_ARI','#766bd3')]:
            val=getattr(r,col)
            if np.isfinite(val):ax.bar(i+offset,val,width=.2,color=color,label=col if i==0 else None)
            else:ax.text(i+offset,.02,'NA',rotation=90,ha='center',fontsize=7)
    ax.set_xticks(range(len(d)),[f'{str(r.seed_a)[-2:]} vs {str(r.seed_b)[-2:]}\nshared n={r.shared_cells}' for r in d.itertuples()])
    ax.set_ylabel('Barcode stability / partition agreement');ax.set_ylim(min(-.05,float(d.partition_ARI.min())-.05) if d.partition_ARI.notna().any() else -.05,1.05)
    ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.25),fontsize=8)
    save(fig,p,'09_cross_seed_stability',"成员 Jaccard 以两次独立候选的 barcode 集合计算；组 Jaccard 允许 A/B 标签交换；分区 ARI 仅在共有成员中双方均保留两组时计算，无效组合标 NA。共有成员 ARI 很高不代表全部候选成员一致，应同时查看 Jaccard 与 shared n。未通过固定筛选的 seed 不补造独立分群；三个 seed 不构成生物学重复。")
    d=pd.read_csv(p/'same_members_embedding_recovery.csv').set_index('evaluated_method').loc[METHODS].reset_index()
    d.to_csv(p/'embedding_comparison_plot_data.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(8,3.5),layout='constrained')
    for ax,col in zip(axes,['ARI','candidate_label_silhouette']):
        ax.bar(range(3),d[col],color=['#AEC7E8','#C5B0D5','#766bd3'])
        ax.set_xticks(range(3),[NAMES[m] for m in METHODS],rotation=20,ha='right');ax.set_ylabel(col.replace('_',' '))
        for i,v in enumerate(d[col]):ax.text(i,v,f'{v:.3f}',ha='center',va='bottom',fontsize=8)
    save(fig,p,'10_same_members_embedding_comparison',"固定相同候选成员在原始标准化 embedding 中的比较：ARI 来自在该固定细胞集合内 K=2、n_init=100 的 KMeans，silhouette 来自既定 A/B 标签。未重新选择细胞，也不等于独立发现能力或准确的生物学真值；候选来源方法享有选择偏倚。二维 UMAP 指标另见 all_seed_evidence.csv。")
