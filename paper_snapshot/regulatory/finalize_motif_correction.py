"""Build a reviewable local/remote report from completed corrected results."""
from pathlib import Path
import json
import hashlib
import shutil
import pandas as pd
import numpy as np

ROOT=Path('/home/u202522802045/share/zpb/regulatory/motif_null_corrected_20261101')
WORK=ROOT/'strict_length_null'
done=json.loads((WORK/'completion.json').read_text())
assert done['status']=='completed'
df=pd.read_csv(WORK/'results/motif_enrichment_fimo.csv')
seeds=pd.read_csv(WORK/'results/motif_per_fold_per_seed_summary.csv')
coverage=pd.read_csv(WORK/'diagnostics/matching_coverage.csv')
diag=pd.read_csv(WORK/'diagnostics/null_sampling_diagnostics.csv')
assert len(df)==4076 and not df.duplicated(['fold','motif_id']).any()
assert len(seeds)==12 and (seeds.motifs_tested==1019).all()
assert len(diag)==2400 and (diag.sampled_peaks==diag.requested_peaks).all()
assert (diag.actual_chrom_match_fraction==1).all()
assert (diag.actual_chrom_length_caliper_match_fraction==1).all()
assert (diag.missing_peaks==0).all()
assert (df.fdr_bh_max.between(0,1)).all()
reg=json.loads((WORK/'diagnostics/duplicate_regression_check.json').read_text())
assert reg['passed']
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
audit=pd.read_csv(ROOT/'diagnostics/frozen_input_audit.csv')
art=Path('/eaas/default/groups/public_cluster/home/share/zpb/result_contain/99_pbmc_lodo_dmode_20261024/full_b2/validated_source_artifacts')
for r in audit.itertuples():assert sha(art/r.fold/'B_D.npz')==r.b_sha256
for name in ['frozen_input_audit.csv','preparation.json','bedtools.stderr.log']:
    shutil.copy2(ROOT/'diagnostics'/name,WORK/'diagnostics'/name)

old=pd.read_csv(ROOT/'results/motif_enrichment_by_seed.csv')
old_sum=old.groupby(['fold','seed'],as_index=False).agg(corrected_counts_original_sampler=('fdr_bh',lambda x:int((x<.05).sum())),median_corrected_effect=('fold_change','median'))
old_sum.to_csv(WORK/'diagnostics/counting_only_correction_original_sampler.csv',index=False)

tables=[]
for r in coverage.itertuples():
    ss=seeds[seeds.fold==r.fold]
    sig=int(df[(df.fold==r.fold)&df.robust_significant].shape[0])
    tables.append(f'| {r.fold.replace("target_", "")} | {r.all_selected_peaks} | {r.evaluable_selected_peaks} | {r.evaluable_fraction*100:.2f}% | {r.unmatched_selected_peaks} | {ss.significant_fdr05.mean():.2f} ± {ss.significant_fdr05.std(ddof=1):.2f} | {sig} |')
quality=diag.groupby('fold',as_index=False).agg(mean_length_ratio=('mean_length_ratio','mean'),mean_duplicate_fraction=('duplicate_fraction','mean'),mean_unique_peaks=('unique_peaks','mean'))
quality.to_csv(WORK/'diagnostics/matching_quality_summary.csv',index=False)
recurrent=pd.read_csv(WORK/'results/motif_recurrent_summary.csv')
report='''# TF motif null 修正结果（2026-11-01）

本目录为修正后的主要结果。旧 2026-10-28 表保留为历史，不再作为最终 motif 富集结论。

## 修正内容

旧命中数去重、分母保留重复抽样次数，导致背景命中率低估。新代码按每次抽样计数，频数矩阵与直接重复索引结果交叉核对。还发现旧背景长度不足，因此主分析使用同染色体、长度比 0.9–1.1 的固定匹配池，不放宽；无匹配背景的 selected Peak 单列。以旧 sampler 仅修复计数的结果置于 diagnostics，仅供追踪，不能作为严格匹配主结论。

## 四折结果

每种子 200 轮，种子 20020130、20020131、20020132。数量报告三种子均值与样本标准差；稳健数量要求三个种子都 FDR < 0.05。

| 折 | 全部 selected | 可评价 selected | 匹配覆盖 | 无匹配 | 显著数均值 ± SD | 三种子均显著 |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(tables) + '\n\n'
report+='## 背景匹配质量\n\n'
for r in quality.itertuples():
    report+=f'- {r.fold}：每次均满足染色体和长度限制，平均长度比 {r.mean_length_ratio:.4f}，背景抽样重复比例 {r.mean_duplicate_fraction*100:.2f}%。\n'
report+='\n## 重复出现的代表性 motif\n\n'
if len(recurrent):
    for r in recurrent.head(10).itertuples():
        report+=f'- {r.tf_name}（{r.motif_id}）：{r.folds} 折通过三种子一致显著标准，平均倍数 {r.mean_fold_change:.3f}。\n'
else:report+='没有 motif 满足三种子一致显著标准，应如实报告阴性结果。\n'
report+='''
## 解释边界

结论只适用于有匹配背景的 selected 子集。未匹配 Peak 不等于 motif 阴性。高重复率说明独立背景稀缺，200 次抽样不是 200 个生物学重复；没有额外控制 GC、可及性、TSS 距离或 motif 家族冗余。经验 P 最小为 1/201，BH 逐折逐种子覆盖 1,019 motifs。不得平均三个种子的 P/FDR。FIMO 位点 P、富集经验 P 和富集 BH-FDR 为不同统计量。

原始命中表已在 zpb 新主机补扫，因为旧主机临时目录/命中表未迁移。复用已有 JASPAR 和完整 hg38，没有重新下载参考或原始测序数据。冻结 B_D 及四折 canonical Peak 均未修改。

## 文件

- results/motif_enrichment_fimo.csv：4,076 条折–motif 汇总；含全体 selected 和可评价子集分母。
- results/motif_enrichment_by_seed.csv：12,228 条逐种子记录，提供 enrichment_p 和 fdr_bh。
- results/motif_enrichment_fimo_fdr05.csv：三个种子均显著的折–motif。
- results/motif_per_fold_summary.csv 与 motif_per_fold_per_seed_summary.csv：数量及种子波动。
- diagnostics/unmatched_selected_peaks.csv：不可评价 Peak。
- diagnostics/null_sampling_diagnostics.csv：全部 2,400 轮匹配核验。
- figures/motif_enrichment_dotplot.png/svg：修正后的图。
'''
(WORK/'MOTIF_NULL_CORRECTED_REPORT_ZH.md').write_text(report,encoding='utf-8')
(WORK/'motif_protocol.md').write_text((WORK/'MOTIF_NULL_CORRECTION_PROTOCOL_20261101.md').read_text(),encoding='utf-8')
(WORK/'PAPER_READY_MOTIF_METHODS.md').write_text('''# Corrected motif enrichment methods

We used frozen Task99 nonzero B_D rows as selected Peaks and unique candidate edge Peak indices as the eligible universe in each fold. Unique exact-coordinate intervals across folds were scanned once with FIMO 5.5.9 against 1,019 JASPAR 2026 CORE vertebrate motifs on the existing hg38 reference, using a motif-site P-value threshold of 1e-4. Overlapping intervals were not merged; motif presence was binary within each Peak. All interval bounds, sequence lengths and extraction coverage were checked.

For each selected Peak, the matched pool comprised nonselected candidate Peaks on the same chromosome with a background-to-selected length ratio between 0.9 and 1.1. Selected Peaks without a matching background were excluded from enrichment inference and listed separately. In each of 200 null draws, one background Peak was independently sampled with replacement per evaluable selected Peak. Repeated draws contributed equally to both motif-positive counts and the denominator. The observed proportion was computed on the same evaluable selected subset.

One-sided empirical enrichment P-values used (1 + number of null proportions at least as large as observed)/201. Benjamini–Hochberg correction included all 1,019 motifs within each fold and random seed. We used seeds 20020130, 20020131 and 20020132, retained per-seed statistics, and reported count/effect means and sample standard deviations. Robust motif enrichment required FDR < 0.05 in all three seeds; P-values and FDRs were not averaged. Matching coverage and repeated-background fractions were reported. These analyses assess sequence enrichment only, without additional GC/accessibility matching or evidence of TF binding.
''',encoding='utf-8')
shutil.copy2('/home/u202522802045/share/zpb/regulatory/run_motif_null_strict.py',WORK/'run_motif_null_corrected.py')
checks={str(p.relative_to(WORK)):sha(p) for p in WORK.rglob('*') if p.is_file() and p.name!='checksums.json' and 'inputs' not in p.relative_to(WORK).parts and 'scan' not in p.relative_to(WORK).parts}
(WORK/'checksums.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False)+'\n')
print(report,flush=True)
