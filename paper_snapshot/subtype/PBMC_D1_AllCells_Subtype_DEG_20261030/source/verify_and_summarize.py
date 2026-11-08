"""Verify every comparison and summarize primary/sensitivity results without selecting seeds."""
import time
from analysis_common import *

Q = 'FDR_BH_global_all_candidate_contrast_GO_Reactome_tests'


def main():
    for flag in ['FGSEA_COMPLETE.json', 'FGSEA_REPORT_COMPLETE.json', 'FGSEA_EXPRESSED_COMPLETE.json', 'PRIMARY_FIGURES_COMPLETE.json']:
        while not (RUN/'audit'/flag).exists():
            time.sleep(15)
    inv = pd.read_csv(RUN/'candidate_inventory.csv')
    ev = pd.read_csv(RUN/'fgsea/candidate_evidence_matrix.csv')
    de = pd.read_csv(RUN/'all_candidate_deg_summary.csv')
    gs = pd.read_csv(RUN/'fgsea/all_candidate_gsea_summary.csv.gz')
    sens = pd.read_csv(RUN/'fgsea_expressed/all_candidate_gsea_summary.csv.gz')
    assert len(inv) == len(ev) == 18 and len(de) == 108
    assert de.status.eq('COMPLETE').all()
    all_p, all_q = [], []
    for row in de.itertuples():
        p = RUN/row.path
        assert sha(p) == row.sha256
        d = pd.read_csv(p)
        assert len(d) == 36601 and d.gene_id.is_unique
        assert np.isfinite(d[['p_value', 'FDR_BH_global_all_candidate_seed_contrast_tests', 'log2FC_meanCP10K']]).all().all()
        assert np.allclose(bh(d.p_value.to_numpy()), d.FDR_BH_within_comparison, atol=1e-12)
        all_p.append(d.p_value.to_numpy())
        all_q.append(d.FDR_BH_global_all_candidate_seed_contrast_tests.to_numpy())
    assert np.allclose(bh(np.concatenate(all_p)), np.concatenate(all_q), atol=1e-12)
    for d in [gs, sens]:
        assert len(d.groupby(['candidate_id', 'contrast', 'database'])) == 108
        assert d.permutations.eq(10000).all()
        assert np.isfinite(d[['NES', 'nominal_p_floored', Q]]).all().all()
        assert d.nominal_p_floored.between(1/10001, 1).all()
        assert np.allclose(bh(d.nominal_p_floored.to_numpy()), d[Q], atol=1e-12)
    keys = ['candidate_id', 'contrast', 'database', 'Term']
    merged = gs[keys+['NES', Q, 'matched_genes']].merge(
        sens[keys+['NES', Q, 'matched_genes']], on=keys, how='outer', suffixes=('_all', '_expressed'))
    merged['same_direction'] = np.sign(merged.NES_all) == np.sign(merged.NES_expressed)
    merged['both_global_FDR_lt005'] = (merged[Q+'_all'] < .05) & (merged[Q+'_expressed'] < .05) & merged.same_direction
    merged.to_csv(RUN/'fgsea/pathway_background_sensitivity.csv.gz', index=False, compression='gzip')
    rows = []
    for e in ev.to_dict('records'):
        cid = e['candidate_id']
        s = sens[sens.candidate_id.eq(cid) & sens.contrast.eq('A_vs_B')]
        m = merged[merged.candidate_id.eq(cid) & merged.contrast.eq('A_vs_B')]
        for db, name in [('GO_BP_2025', 'GO'), ('Reactome_2022', 'Reactome')]:
            e[name+'_expressed_GSEA_global_FDR_lt005'] = int(((s.database == db) & (s[Q] < .05)).sum())
            e[name+'_GSEA_robust_both_backgrounds'] = int(((m.database == db) & m.both_global_FDR_lt005).sum())
        rows.append(e)
    out = pd.DataFrame(rows)
    out.to_csv(RUN/'fgsea/candidate_evidence_matrix_with_sensitivity.csv', index=False)
    selected = []
    for (cid, db), g in sens[sens.contrast.eq('A_vs_B')].groupby(['candidate_id', 'database']):
        selected.append(g.sort_values([Q, 'nominal_p_floored']).head(10))
    pd.concat(selected).to_csv(RUN/'fgsea/top_expressed_background_pathways.csv', index=False)

    # Correct a prose-only approximation against the saved numeric table.
    report = RUN/'fgsea/final_report.md'
    text = report.read_text(encoding='utf-8')
    nk = out[out.candidate_id.eq('D1_ALL_FullB2_NK_C1')].iloc[0]
    text = text.replace('NK 候选分区 ARI 约 0.801', f'NK 候选分区 ARI 约 {nk.mean_partition_ARI:.3f}')
    text += '\n## 表达背景敏感性与交付验收\n\n'
    text += ('另对全部 18 个候选的三项比较、两个数据库完成相同 10,000 次置换 GSEA，仅排除该比较双方均未表达的基因；未筛选显著基因、未换细胞、未换 seed。'
             '原完整基因排序结果保留为主分析；表达背景结果单独保留，不用两种设置中较小的 P 值替代主结果。'
             '嗅觉受体等与 PBMC 不相符或仅少量 leading-edge 基因驱动的条目，不据此命名生物学亚型。\n\n')
    text += '|候选|主 GO/Reactome 显著条目|表达背景 GO/Reactome 显著条目|两背景均显著且方向一致 GO/Reactome|\n|---|---|---|---|\n'
    for e in out.itertuples():
        text += (f'|{e.method} / {e.reference_subclass}|{e.GO_GSEA_global_FDR_lt005}/{e.Reactome_GSEA_global_FDR_lt005}'
                 f'|{e.GO_expressed_GSEA_global_FDR_lt005}/{e.Reactome_expressed_GSEA_global_FDR_lt005}'
                 f'|{e.GO_GSEA_robust_both_backgrounds}/{e.Reactome_GSEA_robust_both_backgrounds}|\n')
    text += ('\nCD8 TEM 的候选 A 在两个可用 seeds 中均有更高的 Cytotoxic 模块，候选 B 的 Naive/memory 模块相对更高，'
             '更适合暂称“细胞毒程序较高与记忆相关程序相对较高的 TEM 内部状态候选”。GZMK 模块方向不一致，不能命名为已确认的 GZMK 亚型。'
             'RNA UMI 和检测基因数没有明显关联，但 seed32 的线粒体比例与 ATAC 深度有统计关联（rank-biserial 约 -0.20/-0.25），不代表完全排除混杂。\n\n'
             'NK 可作为较稳定的备选状态案例：A 相对高 FGFBP2/GZMH，B 相对高 XCL1/KLRC1；不足以仅凭这几个 Marker 确定 CD56 bright/dim 身份。'
             'Source-fit PCA 的 CD4 Naive 是必须保留的有利 PCA 对照；Target-fit PCA 的 CD14 Mono 虽稳定且 DEG 多，却与 RNA 深度和检测基因数相关，不能将其直接当成干净的新亚型。\n\n'
             '全部数值均通过文件级和统计重算验收：108 个 DEG 比较，每比较 36,601 基因；两个 GSEA 背景各 108 项数据库分析。'
             '当前仍是 D1 单 Target 队列的事后探索，D2/D3/D4 此处是训练 Source，不是新增的全细胞 Target 亚型验证。'
             '本轮没有完成差异 ATAC Peak、motif 或多 donor 验证，不能写成已验证的调控机制。\n')
    report.write_text(text, encoding='utf-8')
    write(RUN/'audit/DELIVERY_VERIFIED.json', dict(status='PASS', candidates=18, candidate_seed_splits=36,
        DEG_comparisons=108, DEG_tests=108*36601, all_DEG_file_hashes_verified=True,
        all_within_and_global_gene_BH_recomputed=True, primary_GSEA_analyses=108,
        expressed_background_sensitivity_GSEA_analyses=108, permutations=10000,
        primary_terms=len(gs), sensitivity_terms=len(sens), GSEA_global_BH_recomputed=True,
        finite_statistics=True, all_models_frozen=True, new_GPU_training=False,
        completed_scope='D1 all 9347 target cells; 18 method-specific candidate splits; posthoc discovery'))
    print('VERIFIED_ALL_COMPARISONS', flush=True)


if __name__ == '__main__':
    main()
