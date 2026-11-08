"""Write a Chinese progress snapshot from actual files, not planned work."""
import json,subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
old=ROOT.parent/'PBMC_D1_AllCells_Dmode_20261030'
registry=json.loads((ROOT/'input/datasets.json').read_text())
lines=['# PBMC 全细胞 LODO 数据就绪状态','','Source 全部合格细胞训练，完整 held-out Target 仅推断。','','|数据|全部细胞|标签类别|RNA原始counts|真实配对ATAC|','|---|---:|---:|---|---|']
for k,d in registry.items():lines.append(f"|{k}|{d['cells']}|{d['subclasses']}|已核验 36,601 基因|{'已核验二值缓存' if d['atac_is_binary'] else '已核验 counts'}|")
lines += ['', 'D2 是 HDa 6,317 个细胞，不使用同名 staging D2。D1/D3/D4 已与对应原始配对 10x H5 按实际 barcode、基因及 Peak 完整矩阵核对；D2 复用作者计数矩阵与元数据、原有严格核验的哈希。',
          'D2 的二值 ATAC 行和不是 fragments 或 Peak read-count。缺少的 donor/batch/doublet/TSS 信息不会补造。']
(ROOT/'DATA_READINESS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
rows=[];stages=[]
for fold in ['D1','D2','D3','D4']:
    base=old if fold=='D1' else ROOT/'folds'/fold
    for method in ['FullB2','PCA_sourcefit','PCA_targetfit_fullnorm']:
        for seed in [20020130,20020131,20020132]:
            p=base/method/f'seed_{seed}/metrics.json'
            if p.exists():rows.append(dict(fold=fold,**json.loads(p.read_text())))
    run=ROOT/'analyses'/fold
    stages.append(dict(fold=fold,inventory=(run/'audit/INVENTORY_COMPLETE.json').exists(),
        DEG=(run/'audit/DEG_COMPLETE.json').exists(),RNA_UMAP=(run/'audit/RNA_UMAP_EVIDENCE_COMPLETE.json').exists(),
        GSEA_expressed=(run/'audit/GSEA_ALL_SEEDS_EXPRESSED_COMPLETE.json').exists(),
        GSEA_full=(run/'audit/GSEA_ALL_SEEDS_COMPLETE.json').exists(),complete=(run/'audit/ALL_SEED_ANALYSIS_COMPLETE.json').exists()))
pd.DataFrame(stages).to_csv(ROOT/'ANALYSIS_STAGE_STATUS.csv',index=False)
if rows:
    df=pd.DataFrame(rows);df.to_csv(ROOT/'RESULT_SUMMARY.csv',index=False)
    df.groupby(['fold','method'])[['ari','nmi','ami','cluster_acc','macro_f1','balanced_accuracy','purity_at_15']].agg(['mean','std','count']).to_csv(ROOT/'TRAINING_METRIC_MEAN_SD.csv')
queue=pd.read_csv(ROOT/'JOB_QUEUE.csv')
lines=['# PBMC 全细胞、全 Fold 案例搜索进展','',f'快照时间：{datetime.now().isoformat()}','',
    '今晚优先级：DEG 与 GO/Reactome → 相同细胞的 Full B2/PCA UMAP → Jaccard 与跨 seed 稳定性。',
    'D1 三个模型复用；新增 D2/D3/D4 九个 GPU 训练任务，正规调度器不可用，采用 tmux 文件锁串行 Worker。','',
    '|折|候选清单|DEG|RNA/UMAP比较|全seed表达背景GSEA|全seed完整排序GSEA|','|---|---|---|---|---|---|']
for s in stages:lines.append('|'+s['fold']+'|'+'|'.join('完成' if s[k] else '等待/运行中' for k in ['inventory','DEG','RNA_UMAP','GSEA_expressed','GSEA_full'])+'|')
lines += ['', '|训练任务|Fold|Seed|状态|','|---|---|---|---|']
for r in queue.itertuples():lines.append(f'|{r.Experiment}|{r.Dataset}|{r.Seed}|{r.Status}|')
lines += ['', '代表 Fold/seed 可以在完整比较后探索性选择，所有结果保留；尚未完成的阶段不会写成完成。PCA_targetfit_fullnorm 仍为相应 Source 交集面板，不是全部基因 PCA。']
(ROOT/'RUN_STATUS_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('STATUS_SNAPSHOT_WRITTEN')
