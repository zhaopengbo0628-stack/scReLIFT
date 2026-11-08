"""All-candidate evidence and favorable PCA controls, from unfiltered CSVs."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def main():
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'pdf.fonttype':42})
    d=pd.read_csv(ROOT/'candidate_evidence_matrix.csv')
    d=d.sort_values(['fold','method','reference_subclass'])
    cols=['max_measured_markers','GSEA_full_GO_significant_terms_max','GSEA_full_Reactome_significant_terms_max',
          'GSEA_expressed_GO_significant_terms_max','GSEA_expressed_Reactome_significant_terms_max',
          'mean_partition_ARI','mean_member_Jaccard','mean_UMAP_gain_vs_sourcePCA','mean_UMAP_gain_vs_targetPCA']
    names=['Markers (max)','Full GO (max)','Full Reactome (max)','Expr GO (max)','Expr Reactome (max)',
           'Seed ARI','Member Jaccard','UMAP gain/source','UMAP gain/target']
    x=d[cols].copy()
    # Counts log transformed; no invented common scale or combined score.
    for c in cols[:5]:x[c]=np.log10(1+x[c])
    gain_limit=max(.6,float(np.ceil(np.nanmax(np.abs(x[cols[7:]].to_numpy()))*10)/10))
    fig,axes=plt.subplots(1,len(cols),figsize=(12,24),sharey=True,layout='constrained')
    for j,(ax,col,name) in enumerate(zip(axes,cols,names)):
        values=x[col].to_numpy()[:,None]
        cm=plt.get_cmap('coolwarm' if j==5 or j>=7 else 'Purples').copy();cm.set_bad('#D9D9D9')
        maximum=gain_limit if j>=7 else 1 if j>=5 else max(1.,float(np.nanmax(values)))
        minimum=-gain_limit if j>=7 else -1 if j==5 else 0
        image=ax.imshow(np.ma.masked_invalid(values),aspect='auto',cmap=cm,
            vmin=minimum,vmax=maximum)
        ax.set_xticks([0],[name],rotation=80,ha='right');ax.xaxis.tick_top()
        ax.set_yticks(range(len(d)))
        if j==0:ax.set_yticklabels([f'{r.fold} {r.method.replace("PCA_targetfit_fullnorm","PCA target").replace("PCA_sourcefit","PCA source")} | {r.reference_subclass}' for r in d.itertuples()],fontsize=7)
        else:ax.tick_params(axis='y',left=False,labelleft=False)
        fig.colorbar(image,ax=ax,location='bottom',shrink=.7,aspect=10)
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'all_candidates_evidence.pdf',bbox_inches='tight')
    fig.savefig(out/'all_candidates_evidence.png',dpi=600,bbox_inches='tight');plt.close(fig)
    d.to_csv(out/'all_candidates_evidence_plot_data.csv',index=False)
    (out/'all_candidates_evidence_caption_zh.md').write_text('全部82组方法×参考类候选均保留，包括PCA有利结果。前5列为各候选所有seed中的最大计数，颜色采用log10(1+count)，不是共同证据评分；计数列白色=0，灰色=缺失。后续为共有成员ARI、成员Jaccard与相同成员UMAP差值（Full B2减PCA）；灰色格为缺失或无有效跨seed比较。ARI色轴覆盖[-1,1]，UMAP差值两列共用对称色轴且覆盖全部实际值，负值完整保留。max不代表所有seed均达到该结果。候选由各方法自身聚类定义，跨方法计数不能直接当作发现能力排名；同队列DEG/GSEA/ATAC不是独立验证。\n',encoding='utf-8')

if __name__=='__main__':main()
