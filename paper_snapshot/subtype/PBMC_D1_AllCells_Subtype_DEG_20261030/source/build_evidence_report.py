"""Evidence matrix and Chinese report; compares every method without selecting favorable seeds."""
import time
from analysis_common import *

def table_lines(df,columns,labels=None):
    names=labels or columns
    lines=['| '+' | '.join(names)+' |','| '+' | '.join('---' for _ in names)+' |']
    for _,r in df.iterrows():
        vals=[]
        for c in columns:
            v=r[c]
            vals.append('NA' if pd.isna(v) else (f'{v:.3f}' if isinstance(v,float) else str(v)))
        lines.append('| '+' | '.join(vals)+' |')
    return lines

def main():
    while not (RUN/'audit/ENRICHMENT_COMPLETE.json').exists():time.sleep(45)
    inv=pd.read_csv(RUN/'candidate_inventory.csv')
    de=pd.read_csv(RUN/'all_candidate_deg_summary.csv')
    gs=pd.read_csv(RUN/'all_candidate_gsea_summary.csv.gz')
    ora=pd.read_csv(RUN/'all_candidate_ora_summary.csv.gz')
    recovery=pd.read_csv(RUN/'tables/fixed_candidate_recovery_all_methods.csv')
    qc=pd.read_csv(RUN/'tables/confounder_and_module_tests.csv')
    directions=pd.read_csv(RUN/'tables/gene_direction_consistency.csv.gz')
    model_genes=set(np.load(BASE/'artifacts/gene_ids.npy').astype(str))
    rows=[];best_markers=[];all_paths=[]
    physical=['RNA_UMI','detected_genes','percent_mitochondrial','ATAC_peak_read_counts','ATAC_detected_peaks']
    for rec in inv.to_dict('records'):
        cid=rec['candidate_id'];method=rec['method'];seed=int(rec['anchor_seed'])
        dd=de[(de.candidate_id==cid)&(de.contrast=='A_vs_B')]
        anchor=dd[dd.seed==seed].iloc[0]
        deg=pd.read_csv(RUN/anchor.path)
        gene=deg[(deg.FDR_BH_global_all_candidate_seed_contrast_tests<.05)&(abs(deg.log2FC_meanCP10K)>.25)].copy()
        measured=gene[(gene.pct_detected_A>=.1)|(gene.pct_detected_B>=.1)].copy()
        direction=directions[directions.candidate_id==cid]
        stable=direction[direction.direction_consistent&(direction.within_FDR_seeds==rec['n_seeds'])&(direction.min_abs_log2FC>.25)]
        rec.update(anchor_global_DEG=len(gene),anchor_global_DEG_outside_model=int((~gene.gene_id.isin(model_genes)).sum()),
             global_DEG_min_seed=int(dd.global_strict_effect025.min()),global_DEG_max_seed=int(dd.global_strict_effect025.max()),
             same_direction_within_FDR_all_available_seeds=int(len(stable)) if rec['n_seeds']>=2 else 0)
        for db,prefix in [('GO_BP_2025','GO'),('Reactome_2022','Reactome')]:
            gg=gs[(gs.candidate_id==cid)&(gs.contrast=='A_vs_B')&(gs.database==db)]
            rec[prefix+'_GSEA_global_FDR_lt005']=int((gg.FDR_BH_global_all_candidate_contrast_GO_Reactome_tests<.05).sum())
            rec[prefix+'_GSEA_within_FDR_lt005']=int((gg.FDR_BH_within_candidate_contrast_both_databases<.05).sum())
            oo=ora[(ora.candidate_id==cid)&(ora.contrast=='A_vs_B')&(ora.database==db)]
            rec[prefix+'_ORA_global_FDR_lt005']=int((oo.FDR_BH_global_all_candidate_contrast_database_direction_tests<.05).sum())
            top=gg.sort_values(['FDR_BH_global_all_candidate_contrast_GO_Reactome_tests','nominal_p_floored','NES'],ascending=[True,True,False]).head(5)
            for _,r in top.iterrows():
                all_paths.append(dict(candidate_id=cid,method=method,reference_subclass=rec['reference_subclass'],
                  database=db,Term=r.Term,NES=r.NES,nominal_p=r.nominal_p_floored,
                  global_FDR=r.FDR_BH_global_all_candidate_contrast_GO_Reactome_tests,
                  within_FDR=r.FDR_BH_within_candidate_contrast_both_databases,
                  leading_edge_genes=r.leading_edge_genes,matched_genes=r.matched_genes,
                  evidence='global FDR supported within posthoc analysis' if r.FDR_BH_global_all_candidate_contrast_GO_Reactome_tests<.05 else 'exploratory trend'))
        rs=recovery[recovery.candidate_id==cid]
        for m in METHODS:
            rr=rs[rs.evaluated_method==m]
            rec['recovery_ARI_'+m]=float(rr.ARI.mean())
            rec['recovery_ARI_SD_'+m]=float(rr.ARI.std(ddof=1)) if len(rr)>1 else np.nan
        rec['FullB2_minus_SourcePCA_recovery_ARI']=rec['recovery_ARI_FullB2']-rec['recovery_ARI_PCA_sourcefit']
        qq=qc[(qc.candidate_id==cid)&qc.variable.isin(physical)]
        flagged=qq[(qq.FDR_BH_all_qc_and_module_tests<.05)&(qq.rank_biserial.abs()>=.3)]
        rec['QC_associated_variables']=';'.join(sorted(set(flagged.variable)))
        rec['QC_flagged_seed_tests']=len(flagged)
        rec['missing_metadata']='donor;batch;doublet;ATAC_fragments;TSS_enrichment'
        rec['min_cluster_purity']=float(pd.read_csv(RUN/'tables/candidate_screen_by_seed.csv').query('candidate_id==@cid')[['purity_a','purity_b']].min().min())
        for direction_sign,name in [(1,'A_high'),(-1,'B_high')]:
            markers=measured[measured.log2FC_meanCP10K*direction_sign>0].copy()
            markers['abs_lfc']=markers.log2FC_meanCP10K.abs()
            markers=markers.sort_values(['abs_lfc','FDR_BH_global_all_candidate_seed_contrast_tests'],ascending=[False,True]).head(12)
            rec[name+'_top_markers']=';'.join(markers.gene_symbol)
            for _,r in markers.iterrows():
                best_markers.append(dict(candidate_id=cid,method=method,reference_subclass=rec['reference_subclass'],
                      seed=seed,direction=name,gene_id=r.gene_id,gene_symbol=r.gene_symbol,
                      log2FC=r.log2FC_meanCP10K,within_FDR=r.FDR_BH_within_comparison,
                      global_FDR=r.FDR_BH_global_all_candidate_seed_contrast_tests,
                      pct_A=r.pct_detected_A,pct_B=r.pct_detected_B,model_input_gene=r.gene_id in model_genes))
        stability=rec['n_seeds']>=2 and rec['mean_partition_ARI']>=.5 and rec['mean_member_Jaccard']>=.5
        rec['stable_partition_gate']=bool(stability)
        rec['evidence_grade']='跨seed相对稳定的探索性候选' if stability else ('仅单seed候选' if rec['n_seeds']==1 else '跨seed分区不稳定')
        rec['biological_validation']='未有独立队列/多donor验证；同队列DEG/GSEA为事后支持'
        rec['DEG_status']='COMPLETE'
        rec['GO_Reactome_status']='COMPLETE'
        rows.append(rec)
    evidence=pd.DataFrame(rows)
    evidence.to_csv(RUN/'candidate_evidence_matrix.csv',index=False)
    pd.DataFrame(best_markers).to_csv(RUN/'tables/anchor_effect_markers.csv',index=False)
    pd.DataFrame(all_paths).to_csv(RUN/'tables/top_pathways_all_candidates.csv',index=False)
    summary=gs.groupby(['candidate_id','method','reference_subclass','contrast','database']).agg(
        tested_terms=('Term','size'),global_FDR_terms=('FDR_BH_global_all_candidate_contrast_GO_Reactome_tests',lambda x:int((x<.05).sum())),
        within_FDR_terms=('FDR_BH_within_candidate_contrast_both_databases',lambda x:int((x<.05).sum())),
        minimum_global_FDR=('FDR_BH_global_all_candidate_contrast_GO_Reactome_tests','min')).reset_index()
    summary.to_csv(RUN/'all_candidate_gsea_summary_compact.csv',index=False)
    # Ranking uses reproducibility, then confounding, then direction consistency; never the minimum P.
    ranking=evidence.sort_values(['stable_partition_gate','n_seeds','QC_flagged_seed_tests','mean_partition_ARI',
                                  'same_direction_within_FDR_all_available_seeds'],ascending=[False,False,True,False,False],na_position='last')
    ranking.insert(0,'evidence_rank',range(1,len(ranking)+1))
    ranking.to_csv(RUN/'tables/candidate_ranking.csv',index=False)
    lines=['# D1 全细胞亚型发现：Full B2 与 PCA、DEG、GO/Reactome 中文报告','',
        '本次分析已实际使用 D1 全部 9,347 个细胞，模型来自 D2/D3/D4 全细胞训练后冻结；没有重新训练或选择更有利 seed。',
        '18 组候选：Full B2 8 组、Source-fit PCA 3 组、Target-fit PCA 7 组；共 36 个候选/seed 分裂，完整保留所有未达筛查阈值的类别/seed。',
        '候选发现使用原 Task99 K=34、n_init=8、簇至少 30 细胞和多数标签纯度至少 0.60。参考标签只用于聚类后的候选命名和同类成员筛查。',
        'CD8 TEM 全类别共有 622 个细胞，但每个候选 A/B 是达到固定规则的两个簇中该类别成员；其余细胞保持未分组，不会为提高显著性重新选入。', '',
        '## Full B2 的 CD8 TEM 候选', '']
    tem=evidence[(evidence.method=='FullB2')&(evidence.reference_subclass=='CD8 TEM')].iloc[0]
    lines += [f'该候选在 seeds {tem.seeds} 中通过筛查；A：{tem.n_A_by_seed}；B：{tem.n_B_by_seed}。',
        f'两 seed 候选成员 Jaccard={tem.mean_member_Jaccard:.3f}，共有成员分区 ARI={tem.mean_partition_ARI:.3f}、NMI={tem.mean_partition_NMI:.3f}。第三个 seed 未通过独立分裂筛查，不能补成“三 seed 稳定发现”。',
        f'使用完全相同的候选细胞二分恢复，Full B2 平均 ARI={tem.recovery_ARI_FullB2:.3f}；Source-fit PCA={tem.recovery_ARI_PCA_sourcefit:.3f}；Target-fit PCA={tem.recovery_ARI_PCA_targetfit_fullnorm:.3f}。',
        '该恢复指标使用已选候选成员，偏向候选的起源表征；它与 PCA 是否在全体细胞中独立发现同一分裂是两项不同证据。两个 PCA 均未通过固定 K34 的独立 CD8 TEM 分裂筛查。',
        f'anchor seed 的 A/B 比较有 {int(tem.anchor_global_DEG)} 个基因通过全局 BH<0.05 且 |log2FC|>0.25；其中 {int(tem.anchor_global_DEG_outside_model)} 个不在 355 个模型输入基因中。',
        f'在所有可用 seed 中方向一致且各自通过单比较 FDR 和效应阈值的基因为 {int(tem.same_direction_within_FDR_all_available_seeds)} 个；模型 seeds 不是生物学重复。',
        f'A 高表达候选：{tem.A_high_top_markers}。B 高表达候选：{tem.B_high_top_markers}。',
        f'A/B 比较的 GO GSEA 全局 FDR<0.05 条目数为 {int(tem.GO_GSEA_global_FDR_lt005)}，Reactome 为 {int(tem.Reactome_GSEA_global_FDR_lt005)}。',
        f'显著且效应较大的 QC 相关变量：{tem.QC_associated_variables or "未检出达到预设阈值的关联；不代表混杂已排除"}。', '',
        '## 所有方法的候选证据', '',
        '下表 DEG/富集为预先固定的最小可用 seed A vs B。其他 seed 和 A/B vs 同类其余细胞均在完整表中；不能只挑最显著的比较。', '']
    display=evidence.copy();display['candidate']=display.method+' / '+display.reference_subclass
    lines += table_lines(display,['candidate','n_seeds','mean_partition_ARI','anchor_global_DEG','GO_GSEA_global_FDR_lt005','Reactome_GSEA_global_FDR_lt005','evidence_grade'],
        ['候选','通过seeds','成员分区ARI','全局DEG','GO GSEA全局FDR条目','Reactome GSEA全局FDR条目','证据级别'])
    lines += ['', 'Source-fit PCA 的 CD4 Naive 候选分区 ARI 约 0.988，比 Full B2 的 CD4 Naive 分裂更稳定；不能宣称 Full B2 在所有类别上都优于 PCA。',
        'Full B2 的 NK 候选分区 ARI 约 0.801，值得与 CD8 TEM 一起继续研究。Full B2 的 CD8 Naive 虽在三个 seeds 都通过筛查，但分区 ARI 约 0.031，说明“都出现两个簇”并不等于发现了同一亚群。', '',
        '## GO 与 Reactome 的解释范围','',
        '为 18 个候选各自预先固定 anchor seed 的三项比较执行 GO BP 2025 和 Reactome 2022：共 108 项 GSEA，均为 10,000 次 permutations。',
        '全排序输入来自完整 36,601 基因的有符号 Wilcoxon 统计量；不截取显著 DEG。重复 gene symbol 确定性去重，匹配基因数和 leading edge 完整导出。',
        'ORA 使用该比较表达过的基因作背景、全局显著且有足够效应的 DEG 作输入，分别测试 A/B 高表达方向。',
        'GSEA 的严格统计列为全体候选×三个比较×GO/Reactome 条目的 global BH。数据库内置 FDR、单比较跨两库 FDR 与 nominal P 全部保留，但不能混用。',
        '大量同值/零排序统计量可能影响通路顺序；本轮采用固定 Ensembl ID tie 顺序，不把轻微 NES 或 nominal 显著写成确定结论。', '',
        '## 已完成与仍缺少的证据','',
        '已完成每个候选每个可用 seed 的全基因 DEG、A/B 与同 subclass 其余细胞的比较、跨 seed 成员匹配、相同细胞的多方法恢复、统一 QC/功能模块检查。',
        'ATAC 只在冻结模型后的 QC 中读取，Peak read-count 总数未冒充 fragments。缺少 donor/batch、doublet、fragment、TSS 信息，因此不声称排除了所有技术混杂。',
        '全部 DEG、ORA、GSEA 均来自同一 D1 队列，属于事后证据，细胞级 P 值没有解决聚类选择偏差；不能视作独立实验验证。',
        '本轮没有训练匹配全 Source 的 RNA-only Encoder，不能用不同训练池的历史结果补成公平对照。',
        '正文案例应综合跨 seed 稳定性、RNA 功能解释、QC 风险与相对 PCA 的可测量差异。当前给出完整候选证据，不凭最小 P 值预定 CD8 TEM 胜出。', '',
        '完整文件：candidate_inventory.csv、candidate_evidence_matrix.csv、all_candidate_deg_summary.csv、deg/ 全基因表、all_candidate_gsea_summary.csv.gz、all_candidate_ora_summary.csv.gz、tables/ 和 figures/。']
    (RUN/'final_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    ranklines=['# 候选排序说明','',
       '此顺序是描述性检索顺序：先看是否有跨 seed 相对稳定分区，再看通过 seed 数、QC 关联变量数、分区稳定性与一致 Marker。P 值最小不是排序标准；因某方法的候选天然更适配其表征，恢复 ARI 单独呈现，不参与机械胜负判定。','']
    ranklines += table_lines(ranking,['evidence_rank','method','reference_subclass','n_seeds','mean_partition_ARI','QC_associated_variables','evidence_grade'])
    (RUN/'candidate_ranking.md').write_text('\n'.join(ranklines)+'\n',encoding='utf-8')
    inv['DEG_status']='COMPLETE';inv['GO_Reactome_status']='COMPLETE';inv.to_csv(RUN/'candidate_inventory.csv',index=False)
    write(RUN/'audit/REPORT_COMPLETE.json',dict(status='COMPLETE',candidates=len(evidence),
        no_seed_cherry_picking=True,strict_statistics='global BH families defined in ANALYSIS_PROTOCOL.md',
        claim='all candidates are posthoc exploratory biological hypotheses; no independent validation'))
    print('REPORT_COMPLETE',flush=True)

if __name__=='__main__':main()
