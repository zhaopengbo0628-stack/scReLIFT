"""Read-only scientific reporting and small-artifact export manifest."""
from pathlib import Path
import json, os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ad_runtime as a

def main():
    state={}
    for key,p in [('training',a.OUT/'queue/status.json'),('analysis',a.OUT/'queue/analysis_status.json'),
                  ('extra_preprocessing',a.EXTRA/'queue/cpu_status.json'),
                  ('finalization',a.OUT/'queue/finish_status.json')]:
        state[key]=a.read(p) if p.exists() else {'stage':'NOT_STARTED'}
    stages=list((a.OUT/'models').glob('fold*/seed_*/TRAINING_COMPLETE.json'))
    evals=list((a.OUT/'evaluation').glob('fold*/seed_*/*/COMPLETE.json'))
    biology=list((a.OUT/'biology').glob('fold*/seed_*/*/COMPLETE.json'))
    state.update(utc=datetime.now(timezone.utc).isoformat(),training_units_complete=len(stages),
                 expected_training_units=12,evaluations_complete=len(evals),expected_evaluations=36,
                 biology_runs_resolved=len(biology),protocol_sha256=a.EXPECTED_PROTOCOL)
    reports=a.OUT/'reports';reports.mkdir(exist_ok=True)
    a.write(reports/'run_status.json',state)
    lines=['# AD 四折正式运行状态','',f"更新时间（UTC）：{state['utc']}",'',
           f'训练完成：{len(stages)}/12（每单位包含 Teacher、Student、RNA AE）；评测完成：{len(evals)}/36。','',
           'retired_legacy_cohort已移除；每折用其余三个有作者注释的SOURCE训练Probe。',
           '固定 P2、固定 L2 LogisticRegression C=0.1；三 seeds，sample SD(ddof=1)。','',
           '当前训练：'+str(state['training']), '','当前分析：'+str(state['analysis']),'',
           '完整四折三 seeds 未全部完成前，部分结果仅供进度核对，不能据此选择参数或判断方法最终优劣。']
    if biology:
        records=[]
        for p in biology:
            d=a.read(p);records.append({k:d.get(k) for k in ['fold','seed','method','status','n_predicted_AD',
                     'n_predicted_Control','tested_genes','significant_genes','reason']})
        pd.DataFrame(records).to_csv(reports/'biology_run_summary.csv',index=False)
        lines+=['','生物学分析使用 TARGET 预测组，按 donor/技术组检验；任何一组少于2单位时记录 NOT_EVALUABLE。',
                'GO BP2025 与 Reactome2022 均已在服务器找到。表达驱动的预测组差异属于探索性关联。']
    if (a.OUT/'EVALUATION_COMPLETE.json').exists():
        f=pd.read_csv(a.OUT/'metrics_per_fold_seed.csv')
        fig,axes=plt.subplots(1,2,figsize=(11,4.2));methods=['Full_B2','RNA_only_Encoder','PCA_RNA64']
        colors=['#227c9d','#e5a02b','#8b64ad']
        for ax,metric in zip(axes,['AUROC','Balanced_Accuracy']):
            for j,m in enumerate(methods):
                g=f[f.method.eq(m)].groupby('fold')[metric].agg(['mean','std']).reindex(['fold1','fold2','fold3','fold4'])
                ax.bar(np.arange(4)+(j-1)*.24,g['mean'],.24,yerr=g['std'],label=m,color=colors[j],capsize=2)
            ax.set_xticks(np.arange(4),['GSE214979','GSE268609','SEAAD_MTG','GSE288063+278576'],rotation=20)
            ax.set(ylabel=metric,title=metric+' (mean ± seed sample SD)',ylim=(0,1.15))
            ax.axhline(.5,color='grey',ls=':',lw=.8);ax.spines[['top','right']].set_visible(False)
        axes[1].legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(reports/'method_comparison.png',dpi=200);plt.close(fig)
        lines+=['','全部36次评测已完成，比较表见 metrics_summary.csv 和 macro_summary.csv。',
                '宏平均先按每 seed 等权平均四折，再对三个 seed 报 mean/sample SD；误差条不代表 donor 置信区间。']
    lines+=['','限制：GSE268609为13技术组，独立生物donor未核验；SEAAD为ADNC终点，其他队列临床诊断存在差异；',
             'GSE288063/GSE278576限于作者QC及固定10名donor，队列规模较小。',
             'PCA图仅展示SOURCE拟合投影，不是疾病分类指标。模型、矩阵、完整逐细胞预测留服务器。']
    (reports/'READABLE_STATUS.md').write_text('\n'.join(lines)+'\n')
    selected=[]
    root_names=['frozen_protocol.json','software_versions.json','FREEZE_COMPLETE.json','IMPLEMENTATION_FREEZE.json',
                'PROBE_TESTS.json','ANALYSIS_TESTS.json','author_P2_unit_coverage.csv','metrics_per_fold_seed.csv',
                'metrics_summary.csv','macro_per_seed.csv','macro_summary.csv','predictions_all.csv','EVALUATION_COMPLETE.json']
    selected.extend(a.OUT/n for n in root_names if (a.OUT/n).exists())
    selected.extend(p for p in reports.iterdir() if p.is_file())
    ann=a.OUT/'annotation/fold4'
    selected.extend(ann/n for n in ['type_summary.csv','unit_type_coverage.csv','library_type_coverage.csv',
                    'P2_unit_coverage.csv','fit_diagnostics.json','settings.json','COMPLETE.json'] if (ann/n).exists())
    for p in (a.OUT/'evaluation').glob('fold*/seed_*/*/PCA2_donors.png'):
        if 'seed_20020130' in p.parts:selected.append(p)
    for p in (a.OUT/'evaluation').glob('fold*/seed_*/*/p2_coverage.csv'):
        if 'seed_20020130' in p.parts:selected.append(p)
    for folder in (a.OUT/'biology').glob('fold*/seed_*/*'):
        if not folder.is_dir():continue
        selected.extend(folder/n for n in ['COMPLETE.json','DEG_significant.csv'] if (folder/n).exists())
        selected.extend(folder.glob('*_significant.csv'))
        if 'seed_20020130' in folder.parts:
            selected.extend(folder/n for n in ['DEG_top200_by_FDR.csv','DEG_volcano.png'] if (folder/n).exists())
            selected.extend(folder.glob('*_top20.csv'))
    files=[]
    for p in sorted(set(selected)):
        if p.stat().st_size>10<<20:continue
        files.append(dict(path=str(p.relative_to(a.OUT)),bytes=p.stat().st_size,sha256=a.sha(p)))
    manifest=dict(remote_root=str(a.OUT),files=files,finished=(reports/'FINAL_COMPLETE.json').exists() and
                  a.read(reports/'FINAL_COMPLETE.json')['status']=='COMPLETE',state=state)
    a.write(a.OUT/'EXPORT_MANIFEST.json',manifest)
    print(json.dumps(manifest,ensure_ascii=True))

if __name__=='__main__':
    import fcntl
    with (a.OUT/'.export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
