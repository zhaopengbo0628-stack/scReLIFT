"""A dated Chinese readout; never label pending pathway stages complete."""
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def main():
    inv=pd.concat([pd.read_csv(ROOT/'analyses'/f/'candidate_inventory.csv').assign(fold=f) for f in ['D1','D2','D3','D4']],ignore_index=True)
    inv.to_csv(ROOT/'candidate_inventory.csv',index=False)
    stages=pd.read_csv(ROOT/'ANALYSIS_STAGE_STATUS.csv')
    deg=pd.read_csv(ROOT/'all_fold_all_seed_DEG_summary.csv')
    lines=['# PCA 与 Full B2 亚型发现：当前结果简报','',f'核验时间：{datetime.now().isoformat()}','',
        '四折 Source 全部细胞训练，完整 held-out Target 仅推断；D1=9,347，D2=6,317，D3=8,105，D4=2,413。D1复用三个模型，D2/D3/D4新增九个模型全部完成。24个Teacher/Student最终checkpoint和全部训练输入哈希已核验。','',
        f'共 {len(inv)} 组方法×参考类候选、{len(deg)//3} 个候选×seed分裂；每个分裂做 A/B、A/同类其余、B/同类其余三种比较，合计{len(deg)}次、{len(deg)*36601:,}次基因检验。候选不是已经确认的新细胞亚型。','',
        '|折|Full B2候选|Source-fit PCA候选|Target-fit PCA候选|全基因DEG|全seed完整排序GSEA|表达背景GSEA|',
        '|---|---:|---:|---:|---|---|---|']
    for r in stages.itertuples():
        n=inv[inv.fold.eq(r.fold)].groupby('method').size()
        lines.append(f"|{r.fold}|{n.get('FullB2',0)}|{n.get('PCA_sourcefit',0)}|{n.get('PCA_targetfit_fullnorm',0)}|{'完成' if r.DEG else '运行中'}|{'完成' if r.GSEA_full else '运行中'}|{'完成' if r.GSEA_expressed else '运行中'}|")
    lines+=['','## 重点候选（全局基因校正已完成；通路全折校正尚在汇总）','',
        '|候选|Seed|A/B细胞|全折BH-FDR<0.05且\|log2FC\|>0.25的基因数|共有成员分区ARI|成员Jaccard|UMAP silhouette：Full B2 / Source PCA / Target PCA|',
        '|---|---|---:|---:|---:|---:|---|']
    for f,sub,seed in [('D1','CD8 TEM',20020132),('D3','CD8 TEM',20020132),('D4','CD8 TEM',20020130),('D2','CD4 TCM',20020131),('D2','HSPC',20020130)]:
        ev=pd.read_csv(ROOT/'analyses'/f/'candidate_seed_RNA_UMAP_evidence.csv')
        r=ev[ev.method.eq('FullB2')&ev.reference_subclass.eq(sub)&ev.seed.eq(seed)].iloc[0]
        dd=deg[deg.candidate_id.eq(r.candidate_id)&deg.seed.eq(seed)&deg.contrast.eq('A_vs_B')].iloc[0]
        lines.append(f'|{f} {sub}|{seed}|{r.n_A}/{r.n_B}|{dd.allfold_strict_effect025}|{r.mean_partition_ARI:.3f}|{r.mean_member_Jaccard:.3f}|{r.UMAP_silhouette_FullB2:.3f} / {r.UMAP_silhouette_PCA_sourcefit:.3f} / {r.UMAP_silhouette_PCA_targetfit_fullnorm:.3f}|')
    lines+=['',
        'D1、D3、D4 的 CD8 TEM 均有细胞毒相关（FGFBP2/GZMB/GZMH等）与记忆相关（IFNG-AS1/IL7R/CD28等）程序差异。D1目前分子信号更强，D3/D4跨seed更稳定；正文案例需等GO/Reactome统一比较后再决定。',
        'D2 HSPC效应很强，且有作者HSC/MPP、LMPP、MEP等细注释可解释；但PCA也能恢复，存在深度相关混杂、成员Jaccard较低。其平均分区ARI=1仅来自seed30/32一个有效共有双组比较，seed31的两个比较无效，不能说三个seed完全一致。',
        'D2 CD8 Naive在predicted.celltype.l2与MarkerAnnotations之间明显冲突；不能直接作为可靠CD8新亚型。D1 CD8 Naive跨seed分区ARI仅约0.031，不适合作强发现。','',
        '## 统计口径与限制','',
        'DEG使用完整36,601基因原始counts，完整文库CP10K/log1p，冻结Task99的双侧Mann–Whitney统计与效应量；全折BH覆盖所有候选、seed和三种比较。显著基因仍是聚类后同队列关联，不能当作独立验证。',
        'GO BP2025、Reactome2022均做ORA和10,000置换的全基因排序GSEA；表达基因背景GSEA独立标为敏感性分析。正在运行或只有nominal P的结果不会写成正式显著通路。',
        'PCA_targetfit_fullnorm不是“所有基因PCA”：与Source-fit PCA使用相应折Source Top2000交集面板（355/488/365/398基因），区别是Target拟合且不使用Source RNA Scaler。全基因DEG与PCA输入面板不是同一概念。',
        '相同成员的UMAP对比有候选来源方法选择偏倚，二维更分离不等于新亚型成立。完整候选表保留PCA有利案例。当前没有同一全细胞协议的RNA-only Encoder，不混入旧划分结果。',
        'D2已核实为健康36岁女性HDa单一样本，三个batch列在内部恒定；不是缺失供者信息，但仍没有多供者生物学重复。其他缺失的doublet/TSS等未补造。','',
        '本简报是阶段快照；最终排序、通路全局校正、ATAC事后结果和图包仍在完成，详见RUN_STATUS_CN.md。']
    (ROOT/'当前结果简报.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
