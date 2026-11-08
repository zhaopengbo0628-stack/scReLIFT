"""Chinese scientific report, numbers read from verified CSVs, all cases retained."""
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
Q='FDR_BH_all_folds_all_seeds'

def main():
    cases=json.loads((ROOT/'SELECTED_CASES.json').read_text())
    seed=pd.read_csv(ROOT/'candidate_seed_evidence_matrix.csv');inv=pd.read_csv(ROOT/'candidate_evidence_matrix.csv')
    deg=pd.read_csv(ROOT/'all_fold_all_seed_DEG_summary.csv');atac=pd.read_csv(ROOT/'atac_evidence_summary.csv')
    verification=json.loads((ROOT/'audit/CHECKPOINT_VERIFICATION_COMPLETE.json').read_text())
    b2=seed[seed.method.eq('FullB2')]
    lines=['# PBMC 全细胞 D-mode：PCA 与 Full B2 候选亚群、差异基因及 GO/Reactome 中文总报告','',
        '## 结论与论文案例','',
        f"本次完成四折完整 Target 的亚群系统探索：{len(inv)}组方法×参考类候选、{len(seed)}个候选×seed分裂、{len(deg)}次全基因比较，共{int(deg.tested_genes.sum()):,}次基因检验。全部候选和全部固定seed保留，未按显著性重新挑选细胞。",'',
        '最有希望的生物学解释是 CD8 TEM 内部细胞毒效应与记忆相关程序的状态分化。现有数据支持“具有分子和染色质关联的候选状态”，不足以证明全新的细胞类型；DEG、GO/Reactome和ATAC均来自同队列，不能加总成三次独立验证。','',
        '**关键阴性结果：选入的D1/D3/D4 CD8 TEM案例，其完整基因排序GSEA在四折统一校正后，GO和Reactome均无FDR<0.05条目。** DEG-ORA和表达背景GSEA可提供各自口径下的统计关联，不能替代这个主分析阴性结果。因此本次选择的是最完整的探索性案例，并非已达到强发现标准的新亚型。','',
        '|用途|案例|Seed|A/B细胞|全局DEG|较大效应Marker|全局差异Peak|共有成员ARI|成员Jaccard|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for c in cases:
        r=seed[seed.candidate_id.eq(c['candidate_id'])&seed.seed.eq(c['seed'])].iloc[0]
        a=atac[atac.candidate_id.eq(c['candidate_id'])&atac.seed.eq(c['seed'])]
        peaks=int(a.global_strict_DA.iloc[0]) if len(a) else '不可用'
        lines.append(f"|{c['role']}|{c['fold']} {r.reference_subclass}|{c['seed']}|{r.n_A}/{r.n_B}|{int(r.allfold_strict_effect025)}|{int(r.allfold_measured_markers)}|{peaks}|{r.mean_partition_ARI:.3f}|{r.mean_member_Jaccard:.3f}|")
    for c in cases:
        p=ROOT/'case_studies'/c['id'];meta=json.loads((p/'case.json').read_text())
        r=seed[seed.candidate_id.eq(c['candidate_id'])&seed.seed.eq(c['seed'])].iloc[0]
        lines += ['',f"### {c['role']}：{c['fold']} {r.reference_subclass} / seed {c['seed']}",'',c['rationale'],'',
            f"完整 Target {meta['target_cells']:,}个细胞，参考{subtype_name(r.reference_subclass)}共{meta['subclass_cells']}个；固定候选A={r.n_A}、B={r.n_B}。未被固定筛查选入A/B的同类细胞没有消失：它们保留在完整Target图和同类rest比较中。",'',
            '|相同候选成员的二维指标|Full B2|Source-fit PCA|Target-fit PCA|','|---|---:|---:|---:|',
            f'|UMAP silhouette|{r.UMAP_silhouette_FullB2:.3f}|{r.UMAP_silhouette_PCA_sourcefit:.3f}|{r.UMAP_silhouette_PCA_targetfit_fullnorm:.3f}|',
            f'|UMAP balanced Purity@15|{r.UMAP_balanced_purity15_FullB2:.3f}|{r.UMAP_balanced_purity15_PCA_sourcefit:.3f}|{r.UMAP_balanced_purity15_PCA_targetfit_fullnorm:.3f}|',
            '',f"Full B2相对Source-fit PCA的silhouette差值为{r.UMAP_silhouette_gain_vs_PCA_sourcefit:+.3f}，相对Target-fit PCA为{r.UMAP_silhouette_gain_vs_PCA_targetfit_fullnorm:+.3f}。这些是固定候选标签下二维几何分离，候选来自Full B2，存在来源选择偏倚；不能解释为已知真亚型预测准确率。",'',
            'GO/Reactome 对 A/B 的全局统计：','',
            '|分析|GO显著条目|Reactome显著条目|解释口径|','|---|---:|---:|---|']
        recovery=pd.read_csv(p/'same_members_embedding_recovery.csv').set_index('evaluated_method')
        position=lines.index('GO/Reactome 对 A/B 的全局统计：',len(lines)-5)
        extra=['原始标准化 embedding 中也做了固定成员对比，以避免仅凭二维图判断：','',
            '|原始 embedding 指标|Full B2|Source-fit PCA|Target-fit PCA|','|---|---:|---:|---:|']
        for column,label in [('ARI','固定成员内K=2分区与候选A/B的ARI'),
                             ('candidate_label_silhouette','候选标签silhouette')]:
            extra.append('|'+label+'|'+'|'.join(f'{recovery.loc[m,column]:.3f}' for m in ['FullB2','PCA_sourcefit','PCA_targetfit_fullnorm'])+'|')
        extra+=['','这些指标沿用相同成员，KMeans为n_init=100。A/B本身由Full B2产生，因此该对照是候选结构的事后恢复分析，不能将较高ARI称为独立发现准确率；PCA也恢复了部分结构。','']
        lines[position:position]=extra
        for prefix,desc in [('GSEA_full','完整排序主分析'),('GSEA_expressed','表达基因背景敏感性分析'),('ORA','DEG过度富集分析')]:
            lines.append(f"|{desc}|{int(r[prefix+'_GO_significant_terms'])}|{int(r[prefix+'_Reactome_significant_terms'])}|FDR覆盖四折全部候选、seed和比较；三个分析家族分别校正|")
        lines+=['','代表性免疫程序（每个数据库/排序背景按全局FDR取免疫相关前两项，属于事后说明，完整结果均保留）：','',
            '|背景|数据库|条目|NES|全折FDR|','|---|---|---|---:|---:|']
        for scope in ['full','expressed']:
            t=pd.read_csv(p/f'GSEA_{scope}.csv')
            t=t[t.Term.str.contains('cytotox|lymph|immun|t cell|hematopo|myelo|stem cell|eryth',case=False,regex=True)]
            for db,g in t.groupby('database'):
                for v in g.sort_values([Q,'nominal_p_floored']).head(2).itertuples():
                    lines.append(f'|{scope}|{db}|{v.Term}|{v.NES:.2f}|{getattr(v,Q):.4g}|')
        genes=pd.read_csv(p/'all_gene_DEG.csv.gz');markers=genes[genes.gene_symbol.isin(['FGFBP2','GZMB','GZMH','GNLY','IFNG-AS1','IL7R','CD28','GZMK','DPP4'])]
        lines+=['','关键功能标记（正log2FC为A高，负为B高；不显著仍列出）：','',
            '|基因|log2FC|全折FDR|A检测率|B检测率|','|---|---:|---:|---:|---:|']
        for v in markers.itertuples():lines.append(f'|{v.gene_symbol}|{v.log2FC_meanCP10K:+.2f}|{getattr(v,Q):.3g}|{v.pct_detected_A:.2f}|{v.pct_detected_B:.2f}|')
        stats=pd.read_csv(p/'QC_module_statistics.csv');stats=stats[~stats.variable.str.startswith('module_')]
        lines+=['','混杂检查不是“全部排除”：','']
        for v in stats.itertuples():
            lines.append(f'- {v.variable}：rank-biserial={v.rank_biserial:+.3f}，本折QC/模块BH-FDR={v.FDR_BH_all_qc_and_module_tests:.3g}。')
        ar=atac[atac.candidate_id.eq(c['candidate_id'])&atac.seed.eq(c['seed'])].iloc[0]
        lines+=['',f"ATAC差异Peak={int(ar.global_strict_DA)}，精确坐标Source模板匹配边={int(ar.exact_target_template_edges)}；单列的双向≥50%区间重叠连接中，基因与Peak均过各自门槛的连接={int(ar.overlap_strict_gene_and_peak_links)}。区间重叠并不保证功能元件相同，这些连接仅为探索性支持，不能称为因果调控验证。Source模板受模型基因面板限制，面板外Marker没有连接不等于缺乏调控。",'',
            f"跨seed有效双组比较{int(r.valid_two_group_pairs)}/{int(r.seed_pairs_tested)}对，最少共有成员{int(r.min_shared_cells)}个。所有seed图均保留；固定成员跨seed投影图与每个seed独立候选的稳定性是两件事。"]
    lines+=['','## PCA 对照与其余候选','',
        '两个PCA均保留，包括PCA有利的候选。Source-fit PCA用Source标准化RNA拟合后转换Target；Target-fit PCA在Target上拟合，属于描述性敏感性对照。两者使用本折相同的Source交集基因面板，均不是完整36,601基因PCA。PCA采用确定性full SVD，三个seed改变下游聚类/UMAP，而不是三次独立PCA拟合。','',
        'D2 HSPC有很强的全基因差异和作者HSC/MPP、LMPP、MEP等同队列细注释，但PCA同样可恢复，且存在深度混杂与成员不一致。因此更适合作为“共同恢复已知谱系结构”的补充对照，不能宣称Full B2独有发现。其ARI=1仅来自一个有效seed对，其他seed对不满足共有双组条件。',
        'D2 CD4 TCM可列补充探索；D1 NK在已有seed间较稳定，可列补充。D1/D2 CD8 Naive分区稳定性低，D2还存在不同作者注释列的身份冲突，不宜写作强CD8新亚型。其他B细胞/髓系候选按统一矩阵保留，RNA效应很少、seed不稳或QC关联明显者不作为正文发现。完整82组逐项结果见candidate_evidence_matrix.csv，所有179个候选×seed详见candidate_seed_evidence_matrix.csv。','',
        'D3 CD14 Mono seed32虽然有350个全局DEG、95条ORA显著项，但跨seed分区ARI约0.021，且Marker中出现TRBC2/TCF7等T细胞相关程序；在缺少doublet和正交身份核验时，不能据此优先认定髓系新亚型。D3 Intermediate B也有IGHG1/FCRL5等状态差异，但稳定性和混杂控制不足；D1 NK的PTGDS/FGFBP2与XCL1等程序差异值得补充核验。上述候选未因“不如CD8 TEM有利”被删除。','',
        '## 实验协议及统计边界','',
        f"- 训练核验：{verification['checked_student_models']}个Student、共{verification['checked_stages']}个Teacher/Student阶段，最终epoch100和全部训练输入哈希已核验。D1三套复用，D2/D3/D4新增九套。未重新训练RNA-only Encoder；不混用旧train/test划分的结果。",
        '- 每折三个Source原train/val/test全部用于训练及train-only HVG选择；Target全细胞仅在模型冻结后推断。Source交集基因数D1/D2/D3/D4分别355/488/365/398。Source训练监控集与训练集相同，不称为独立validation；固定100轮，不按Target表现选checkpoint。',
        '- RNA输入Scaler和B_D/F仅Source拟合。但沿用Task99评估时Target-fitted embedding StandardScaler，另在完整Target构图/UMAP。因此目前没有证明完全归纳式的零样本分离优势；论文中须保留这个评估适配限制，严格归纳式敏感性仍待补。',
        '- 候选筛查保留冻结Task99规则：K=min(max(2×参考类数,30),50)，每cluster≥30细胞、majority purity≥0.60，每参考类取两个最大合格簇；仅将该参考类成员纳入A/B。K依赖参考类数且事后筛查使用参考标签，不能称为完全无标签发现。',
        '- DEG：全36,601基因counts，完整文库CP10K/log1p，双侧渐近Mann–Whitney；log2FC基于平均CP10K与1e-4伪计数。BH覆盖四折全部基因×候选×seed×三比较；门槛FDR<0.05且|log2FC|>0.25。大效应Marker另要求|log2FC|>0.5与任一组检测率≥0.1。',
        '- 四折统一BH与早先单折BH的假设家族不同，校正后的计数可增可减；例如D1 seed32从原折内305变为全折309，不是更换细胞、counts或效应门槛。报告统一采用全折列，旧列作为追溯保留。',
        '- GO BP2025、Reactome2022版本GMT哈希保留。全排序GSEA用所有可映射去重gene symbols的有符号统计量，包括零表达/零统计量，不截取显著DEG。表达背景GSEA为独立敏感性分析，不能替代不利的完整排序结果。10,000置换，P下限1/10001。',
        '- GO/Reactome条目高度重叠，显著条目数不是独立生物学程序数。CD8候选的NK相关条目反映共享细胞毒程序，不证明细胞身份变成NK；同一基因出现在正/负调控基因集中不能单靠NES推断因果调控方向。D1的GZMK方向跨seed不一致，不将候选命名为已确认的GZMK亚型。',
        '- ORA以各折全候选/seed/比较基因BH通过的DEG为输入、该比较检测到基因为背景；再对四折全部通路/方向做BH。ORA输入门槛与报告中跨折DEG门槛分别保留，未偷换。',
        '- 最终核验发现完整排序4项、表达背景2项通路无法估计P/NES（早先首先发现D2表达背景两项）；原始NA全部保留，全局BH中按P=1保守计入，未删除假设或伪造NES。修正前输出也在服务器audit/pre_nonfinite_pathway_BH_repair保留，主表使用修正后的全折FDR列，逐项数量见audit/*BH_AUDIT.json。',
        '- ATAC：所有Full B2候选×seed的全部Peak检验统一BH，显著另要求检测比例差≥0.05；复用冻结Task99统计，未把Peak read-count/binary缓存当作fragments。未控制depth等协变量，motif扫描背景尚不可用。',
        '- 同队列聚类后DEG、GO及ATAC均是事后关联；统计FDR不消除聚类选择偏倚、细胞相关性或事后选择最佳Fold/seed偏倚。三个seed不是三个生物学重复。代表案例与seed经全量检索后选取，所有其他结果保留。',
        '- D2作者元数据已按barcode核验：HDa单一健康36岁女性，sample与三个batch列内部恒定。其他供者信息缺失与D2单供者限制区分记录；缺少doublet、完整S/G2M、TSS、真实fragments的限制不补造。','',
        '## 仍缺什么独立验证','',
        '优先在有独立供者且标签一致的新配对PBMC队列冻结候选定义和Marker程序后复现；用供者级pseudobulk/混合模型检验RNA和ATAC，控制depth、doublet、细胞周期等；在Source-only latent scaling下做严格归纳式敏感性对照，并补同一全细胞协议的RNA-only Encoder。若论文要主张新亚型或特定调控机制，还需正交注释、蛋白/流式或功能实验与可信motif/Gene–Peak验证。跨LODO Fold训练来源重叠且供者独立性未完整确认，不能将四折重复趋势称为四个独立验证队列。','',
        '## 本地交付索引','',
        '- candidate_inventory.csv：82组候选与有效稳定性比较数。',
        '- candidate_seed_evidence_matrix.csv / candidate_evidence_matrix.csv：逐seed及汇总证据，完整保留PCA有利结果。',
        '- all_candidate_deg_summary.csv / deg_allfold_global/：537个比较及完整基因表。',
        '- all_fold_all_seed_GSEA_full.csv.gz / GSEA_expressed.csv.gz / ORA.csv.gz：全通路结果，NES、FDR、leading-edge、版本与原始P保留。',
        '- atac_evidence_summary.csv / atac_posthoc/：差异Peak与Source模板精确/区间重叠连接。',
        '- case_studies/：正文探索案例及两个备选的固定成员、12类图、600dpi PNG、PDF与CSV。PDF保留可提取文本和矢量元素，部分散点/热图图元为栅格，并非每个图元全矢量。',
        '- figures/all_candidates_evidence.*：全部候选证据比较。',
        '- source/、analyses/*/source/、audit/、JOB_QUEUE.csv：复现代码、输入哈希、核验记录和真实队列。','']
    (ROOT/'final_report.md').write_text('\n'.join(lines),encoding='utf-8')
    (ROOT/'中文总报告.md').write_text('\n'.join(lines),encoding='utf-8')
    ranking=['# 全部候选排序与去留说明','','排序用于检索和审阅，不是新亚型显著性评分。正文代表Fold/seed为事后探索性选择；未重新选择细胞。','']
    for c in cases:ranking += [f"- {c['role']}：{c['candidate_id']}，seed {c['seed']}。{c['rationale']}"]
    ranking+=['','|Fold|方法|候选|全局DEG范围|最大大效应Marker|共有ARI|成员Jaccard|有效seed对|建议|','|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in inv.sort_values(['fold','method','reference_subclass']).itertuples():
        chosen=[c for c in cases if c['candidate_id']==r.candidate_id]
        if chosen:use=chosen[0]['role']
        elif r.method!='FullB2':use='完整PCA对照；可有相同生物结构，不能排除'
        elif r.reference_subclass=='HSPC':use='补充：已知谱系；PCA可恢复且QC/稳定性受限'
        elif r.reference_subclass=='CD8 Naive':use='不作强发现：分区不稳/注释风险'
        elif r.max_measured_markers<10:use='弱RNA证据，保留探索记录'
        elif r.any_large_QC_association:use='补充探索，需控制QC'
        else:use='补充探索，需独立生物学复现'
        ranking.append(f'|{r.fold}|{r.method}|{r.reference_subclass}|{r.min_global_DEG}–{r.max_global_DEG}|{r.max_measured_markers}|{r.mean_partition_ARI:.3f}|{r.mean_member_Jaccard:.3f}|{r.valid_two_group_pairs}/{r.seed_pairs_tested}|{use}|')
    (ROOT/'candidate_ranking.md').write_text('\n'.join(ranking)+'\n',encoding='utf-8')
    print('CHINESE_FINAL_REPORT_WRITTEN',flush=True)

def subtype_name(x):return str(x)

if __name__=='__main__':main()
