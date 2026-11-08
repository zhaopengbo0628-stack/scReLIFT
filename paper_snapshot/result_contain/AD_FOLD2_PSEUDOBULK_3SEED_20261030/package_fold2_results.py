from pathlib import Path
import json, hashlib, tarfile
import pandas as pd
ROOT=Path('/home/u202522802045/share/zpb')
OUT=ROOT/'result_contain/AD_FOLD2_PSEUDOBULK_3SEED_20261030'
summary=pd.read_csv(OUT/'ALL_ANALYSES_SUMMARY.csv')
terms=pd.read_csv(OUT/'ALL_SIGNIFICANT_GO_REACTOME.csv')
report=OUT/'FOLD2_PSEUDOBULK_REPORT_ZH.md'
text=report.read_text()
addon='''

## 本次主要发现与旧结果的关系

本次合计17条GO BP、34条Reactome显著条目（按Seed和细胞类型分别计数，可能包含重复术语）。均为预测Control-up：

| Seed | 细胞类型 | DEG | GO BP | Reactome |
| --- | --- | --- | --- | --- |
| 20020131 | OPC | 382 | 3 | 3 |
| 20020132 | Excitatory | 179 | 2 | 0 |
| 20020132 | Astrocytes | 542 | 11 | 31 |
| 20020132 | OPC | 322 | 1 | 0 |

20020131/OPC主要涉及包涵体组装调节、分子伴侣介导蛋白重折叠、HSF1热休克反应；20020132/Excitatory涉及Hippo信号调节；20020132/Astrocytes主要涉及翻译、细胞呼吸、氧化磷酸化。术语重叠、基因重复以及两组预测对照带来的不稳定性需要保留。

真实AD/HA共同参考：Astrocytes的CPAMD8（FDR约0.00993，log2FC约2.941）和OPC的VCAN（FDR约0.0308，log2FC约0.542）各1个显著基因，均为AD-up，但没有显著GO/Reactome。它们仍是技术组层面的探索性发现。

此前rev04 Fold2无显著GO的报告基于“全细胞混合的每细胞CP10K/log1p均值”；本次使用“原始counts求和→每技术组每类型CPM/log1p”的pseudobulk。数据聚合和检验对象不同，两个结果可以同时成立；不得覆盖或混合两个版本，也不能把显著术语增加解释成模型改进。

## 完整性核验

独立直接读取HDF5的28组Oligodendrocytes共316个细胞，重新求和后36,601维向量与本次pseudobulk完全一致。另对Seed 20020131/OPC用独立Welch公式和statsmodels BH重算，p值/FDR最大绝对误差分别小于7e-16/1.4e-15。24项比较的DEG及GO/Reactome计数已与明细核对。

旧rev04运行器最初因协议哈希失配停止。核实差异仅为清理时supersedes历史说明中的废弃队列名字被替换，全部科学字段与原冻结快照一致；本次只在独立分析目录记录兼容性核验，不修改正式协议或运行器。
'''
if '## 本次主要发现与旧结果的关系' not in text:
    report.write_text(text+addon)
assert len(terms[terms.library.eq('GO_BP')])==17
assert len(terms[terms.library.eq('Reactome')])==34
assert json.loads((OUT/'INDEPENDENT_VERIFICATION.json').read_text())['status']=='PASS'

root_names={'FOLD2_PSEUDOBULK_REPORT_ZH.md','ALL_ANALYSES_SUMMARY.csv','ALL_SIGNIFICANT_GO_REACTOME.csv',
 'DONOR_CELLTYPE_COVERAGE.csv','UNIT_ROSTER.csv','FROZEN_ANALYSIS_CONFIG.json','AGGREGATION_CHECK.json',
 'SOFTWARE_VERSIONS.json','SOURCE_MANIFEST.json','PROTOCOL_COMPATIBILITY_AUDIT.json',
 'PROTECTED_INPUT_HASHES_BEFORE.json','PROTECTED_INPUT_HASHES_AFTER.json','RUN_COMPLETE.json','INDEPENDENT_VERIFICATION.json'}
detail_names={'SUMMARY.csv','GROUP_ASSIGNMENTS.csv','TRUE_LABEL_REFERENCE.json','ALL_SIGNIFICANT_ENRICHMENT.csv',
 'ALL_SIGNIFICANT_DEG.csv','UNIT_INCLUSION.csv','DEG_significant.csv','ENRICHMENT_significant.csv',
 'COMPLETE.json','ENRICHMENT_STATUS.json','PSEUDOBULK_VOLCANO.png'}
files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and
       ((p.parent==OUT and p.name in root_names) or (p.parent!=OUT and p.name in detail_names))]
manifest=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
(OUT/'LOCAL_EXPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
tarpath=OUT/'small_results.tar.gz'
with tarfile.open(tarpath,'w:gz') as tar:
    for p in files+[OUT/'LOCAL_EXPORT_MANIFEST.json']:
        tar.add(p,arcname=str(p.relative_to(OUT)),recursive=False)
print(json.dumps(dict(files=len(files),uncompressed_bytes=sum(r['bytes'] for r in manifest),
                     archive_bytes=tarpath.stat().st_size,archive_sha256=hashlib.sha256(tarpath.read_bytes()).hexdigest()),indent=2))
