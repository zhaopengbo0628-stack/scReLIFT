from case_plot_common import *

def main():
    p,c=context();path=p/'all_peaks_DA.csv.gz'
    if not path.exists():return
    q='FDR_BH_all_B2_candidate_seed_peak_tests';d=pd.read_csv(path)
    sig=d[q].lt(.05)&d.delta_accessibility.abs().ge(.05)
    fig,ax=plt.subplots(figsize=(6,4.6),layout='constrained')
    for label,color,mask in [('Other','#D8D8D8',~sig),('A more accessible',COLORS['A'],sig&d.delta_accessibility.gt(0)),('B more accessible',COLORS['B'],sig&d.delta_accessibility.lt(0))]:
        dd=d[mask];ax.scatter(dd.delta_accessibility,-np.log10(dd[q].clip(lower=1e-300)),s=4,c=color,alpha=.6,linewidths=0,label=f'{label} ({len(dd)})')
    ax.axhline(-np.log10(.05),color='#AAAAAA',ls='--',lw=.6)
    ax.set_xlabel('Accessibility fraction difference (A - B)');ax.set_ylabel('-log10 BH FDR (all B2 candidate peaks)')
    ax.legend(frameon=False,fontsize=8)
    save(fig,p,'11_ATAC_differential_peaks',f"固定同细胞配对ATAC的事后差异Peak。显著要求覆盖全部29组Full B2候选、所有可用seed的Peak检验BH-FDR<0.05，且检测比例差≥0.05；检验基于原始Peak值/已核验二值缓存，未控制深度等协变量，属于同队列关联。Target ATAC未参与训练。全部{len(d):,}个Peak保留，彩点{int(sig.sum())}个；不能把Peak reads/binary值称为fragments。")
    pp=p/'exploratory_source_template_overlap.csv.gz'
    if not pp.exists():return
    d=pd.read_csv(pp);hit=d[d.overlapping_target_peak.notna()].copy()
    hit=hit.sort_values(['gene_and_peak_associated',q,'gene_global_FDR'],ascending=[False,True,True]).head(12)
    hit.to_csv(p/'gene_peak_network_plot_data.csv',index=False)
    if hit.empty:return
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    genes=list(dict.fromkeys(hit.gene_symbol));peaks=list(dict.fromkeys(hit.overlapping_target_peak))
    gy={g:y for g,y in zip(genes,np.linspace(0,1,len(genes)))};py={g:y for g,y in zip(peaks,np.linspace(0,1,len(peaks)))}
    for r in hit.itertuples():
        strong=r.gene_and_peak_associated
        ax.plot([0,1],[gy[r.gene_symbol],py[r.overlapping_target_peak]],color='#766bd3' if strong else '#CCCCCC',lw=1.4 if strong else .6,alpha=.7)
    for g,y in gy.items():ax.scatter([0],[y],s=30,c='#C5B0D5');ax.text(-.03,y,g,ha='right',va='center',fontsize=8,fontstyle='italic')
    for peak,y in py.items():ax.scatter([1],[y],s=25,c='#FFBB78');ax.text(1.03,y,peak,ha='left',va='center',fontsize=6)
    ax.set_xlim(-.3,1.8);ax.set_ylim(-.08,1.08);ax.axis('off')
    save(fig,p,'12_exploratory_gene_peak_links',"探索性Source模板连接图：Target Peak与Source稳定模板Peak要求双向区间重叠≥50%，不意味着坐标相同或功能元件相同。紫色连接要求对应基因和Peak分别达到其全局FDR及效应门槛，灰色未达；最多展示12条并保留绘图CSV。Source B_D只包含Source交集基因，面板外Marker没有模板连接不能解释为不存在调控。不是Target重建的因果Gene–Peak边，也不构成独立验证；完整精确坐标匹配结果单独保留。")

if __name__=='__main__':main()
