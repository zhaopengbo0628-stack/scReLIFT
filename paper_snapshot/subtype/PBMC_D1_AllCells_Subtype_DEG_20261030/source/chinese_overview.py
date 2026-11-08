"""Create a Chinese overview with exact primary and sensitivity pathway evidence."""
import time
from analysis_common import *


def main():
    while not (RUN/'audit/DELIVERY_VERIFIED.json').exists():
        time.sleep(15)
    ev = pd.read_csv(RUN/'fgsea/candidate_evidence_matrix_with_sensitivity.csv')
    de = pd.read_csv(RUN/'all_candidate_deg_summary.csv')
    ora = pd.read_csv(RUN/'fgsea/all_candidate_ora_summary.csv.gz')
    gs = pd.read_csv(RUN/'fgsea/all_candidate_gsea_summary.csv.gz')
    ss = pd.read_csv(RUN/'fgsea_expressed/all_candidate_gsea_summary.csv.gz')
    oq = 'FDR_BH_global_all_candidate_contrast_database_direction_tests'
    gq = 'FDR_BH_global_all_candidate_contrast_GO_Reactome_tests'
    lines = ['# D1 全细胞：PCA 与 Full B2 亚型发现、差异基因和 GO/Reactome', '',
        '## 已完成的范围', '',
        'D2/D3/D4 共 16,835 个 Source 细胞用于训练，D1 全部 9,347 个 Target 细胞仅用于冻结模型后的推断与分析。D1 的 CD8 TEM 共 622 个细胞。',
        '本轮分析新全细胞结果；旧 Task99 的 test-only 数值没有混入。固定三个 seeds 均保留，未重训网络、未按显著性重选成员。',
        '对 Full B2、Source-fit PCA 和 Target-fit PCA 统一执行原 Task99 K=34 候选筛查，获得 18 组方法特定候选、36 个候选/seed 分裂。每个 seed 候选做 A/B、A/同类其余、B/同类其余三项比较，共 108 项 DEG、3,952,908 个基因检验。',
        'Source-fit PCA 与 Full B2 使用相同的 355 个 Source-only 交集基因；Target-fit PCA 在 Target 完整归一化 RNA 上拟合，作为描述性补充对照，特征数和拟合范围均不同。', '',
        '## 候选总表', '',
        '分区 ARI 按实际 barcode 在共有成员上计算。DEG 范围覆盖该候选所有可用 seeds，阈值为全局 BH<0.05 且 |log2FC|>0.25。模型 seeds 不是生物学重复。', '',
        '|方法|参考类别|通过 seed 数|跨 seed 分区 ARI|各 seed 全局 DEG 数|结论|', '|---|---|---:|---:|---|---|']
    for r in ev.itertuples():
        d = de[de.candidate_id.eq(r.candidate_id) & de.contrast.eq('A_vs_B')].sort_values('seed')
        counts = '；'.join(f'{int(x.seed)}: {int(x.global_strict_effect025)}' for x in d.itertuples())
        ari = '不可计算' if pd.isna(r.mean_partition_ARI) else f'{r.mean_partition_ARI:.3f}'
        lines.append(f'|{r.method}|{r.reference_subclass}|{r.n_seeds}|{ari}|{counts}|{r.evidence_grade}|')
    lines += ['', '## 建议正文与备选案例', '',
        '**正文可优先考虑 Full B2 的 CD8 TEM，但证据级别应写为探索性的细胞状态分裂。**它的功能解释比仅凭簇分开更具体：A 的 FGFBP2、GZMB、GZMH 与细胞毒模块较高，B 的 IFNG-AS1、DPP4、CD28 和记忆相关模块相对较高。不要将其直接改名为已确认的新细胞亚型。',
        '该候选在 seed30 和 seed32 通过筛查，seed31 未通过。A/B 分别为 296/131、330/182；其余同类细胞完整保留但不强行分入 A/B。跨 seed 成员 Jaccard=0.792、共有成员分区 ARI=0.627。',
        '在完全相同的候选成员上二分恢复，平均 ARI 为 Full B2 0.963、Source-fit PCA 0.315、Target-fit PCA 0.526。此为事后成员恢复，有起源表征偏向；两个 PCA 在统一的全体细胞 K34 筛查中均未独立提出 TEM 分裂，这也不等于证明 PCA 在所有合理参数下都不能分开。',
        '备选为 Full B2 NK：分区更稳定，A 相对高 FGFBP2/GZMH，B 相对高 XCL1/KLRC1；尚不足以确认 CD56 bright/dim。PCA 的 CD4 Naive 应作为有利于 PCA 的对照保留。',
        'Full B2 的 CD8 Naive、Intermediate B 和 Memory B 分区不稳定，建议放补充或暂不作生物学发现结论。Target-fit PCA CD14 Mono 分裂稳定，但 RNA UMI 与检测基因数有关联，须先排除深度等技术因素。', '',
        '## GO/Reactome：严格支持与趋势分开', '',
        '每候选在 DEG 前固定为最小可用 seed，三个比较分别做两库 ORA 与全基因排序 GSEA。主 GSEA 和表达背景敏感性各 108 项数据库分析，每项 10,000 次置换。敏感性仅排除该比较双方均未表达基因，并未筛选显著基因。',
        '**TEM 的主分析 A/B 全排序 GSEA 没有通过全局 FDR<0.05 的 GO 或 Reactome 条目。**不能把其 nominal P 或敏感性分析的显著条目写成主 GSEA 已获显著支持。',
        '**TEM 的 DEG-ORA 有严格统计支持：GO 17 个、Reactome 8 个条目通过全体候选/比较/方向/数据库的全局 BH<0.05。**下面列出具有 PBMC 功能解释的代表条目，完整表保留所有通路。', '',
        '|分析|方向/数据库|通路|效应或重叠|全局 BH-FDR|相关基因|', '|---|---|---|---|---|---|']
    cid = 'D1_ALL_FullB2_CD8_TEM_C1'
    curated = ['Positive Regulation of Leukocyte Mediated Cytotoxicity', 'T Cell Activation',
               'Immunoregulatory Interactions Between A Lymphoid And A non-Lymphoid Cell', 'DAP12 Interactions']
    o = ora[ora.candidate_id.eq(cid) & ora.contrast.eq('A_vs_B')]
    for term in curated:
        g = o[o.Term.str.startswith(term)].sort_values(oq).head(1)
        for r in g.itertuples():
            q = getattr(r, oq)
            lines.append(f'|DEG-ORA|{r.direction} / {r.database}|{r.Term}|{r.overlap} 个重叠基因|{q:.3g}|{r.overlap_genes}|')
    candidates = ['Immunoregulatory Interactions Between A Lymphoid And A non-Lymphoid Cell',
                  'Antigen processing-Cross Presentation', 'Natural Killer Cell Mediated Immunity']
    for source, label in [(gs, '主全排序 GSEA'), (ss, '表达背景敏感性 GSEA')]:
        d = source[source.candidate_id.eq(cid) & source.contrast.eq('A_vs_B')]
        for term in candidates:
            g = d[d.Term.str.startswith(term)].sort_values(gq).head(1)
            for r in g.itertuples():
                lines.append(f'|{label}|{r.database}|{r.Term}|NES={r.NES:.3f}|{getattr(r,gq):.3g}|{r.leading_edge_genes}|')
    lines += ['', '注：GO 的“NK 介导免疫/细胞毒性”基因集在 CD8 T 细胞中富集，表明共享的细胞毒程序，不能据此把参考 CD8 TEM 细胞重新注释为 NK。',
        '表达背景敏感性包含额外显著条目，结论限定为背景设置敏感的功能支持；所有候选两背景交叉表见 fgsea/pathway_background_sensitivity.csv.gz。嗅觉受体等不相符条目不用于生物学命名。', '',
        '## 混杂、统计和独立验证限制', '',
        'TEM 两个 seeds 的 RNA UMI 和检测基因数无明显差异；seed32 线粒体比例和 ATAC Peak read-count 有关联，效应约 -0.20 和 -0.25。NK 也有线粒体与 ATAC 深度关联。Donor、batch、doublet、fragments、TSS 信息缺失，不能声称混杂已完全排除。',
        'RNA DEG、ORA、GSEA、功能模块均来自同一个 D1 Target；细胞级 P 值未解决聚类后选择偏差，没有独立 donor 的 pseudobulk 验证。严格 FDR 指本轮事后检验中的多重检验控制，不等于新亚型独立确证。',
        '本轮未补训匹配的 RNA-only Encoder，也没有完成新候选差异 Peak/motif 验证。需要独立供者/队列、RNA-only 对照、配对 ATAC 调控证据与蛋白/流式验证才能提高证据等级。', '',
        '## 文件与图表', '',
        '- 完整中文技术报告：fgsea/final_report.md。',
        '- 全候选清单：candidate_inventory.csv；证据矩阵：fgsea/candidate_evidence_matrix_with_sensitivity.csv。',
        '- 108 个完整 DEG 表：deg/；比较汇总：all_candidate_deg_summary.csv。',
        '- 主富集完整表：fgsea/all_candidate_gsea_summary.csv.gz、fgsea/all_candidate_ora_summary.csv.gz。',
        '- 表达背景敏感性：fgsea_expressed/；通路背景比较：fgsea/pathway_background_sensitivity.csv.gz。',
        '- 三方法相同成员 UMAP、火山图、跨 seed Marker 热图：figures/。',
        '- GO/Reactome 气泡图、富集曲线、全候选比较：fgsea/figures/。所有正式图均有 600 dpi PNG、矢量 PDF 和绘图数据。',
        '- 可复现脚本：source/；数据库及哈希：input/、audit/；统计与文件验收：audit/DELIVERY_VERIFIED.json。',
        '较早启动的 GSEApy 额外计算继续保留在远程，未中断；本次已完成交付的主结果为 fgseaSimple，不依赖该额外任务完成。', '']
    (RUN/'中文结果总览.md').write_text('\n'.join(lines), encoding='utf-8')
    write(RUN/'audit/CHINESE_OVERVIEW_COMPLETE.json', dict(status='COMPLETE', report='中文结果总览.md'))
    print('CHINESE_OVERVIEW_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
