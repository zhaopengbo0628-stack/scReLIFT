"""Finalize only after all36 fixed evaluations and biology runs resolve."""
import os,sys,time,subprocess,traceback
from collections import Counter
import numpy as np
import pandas as pd
import ad_runtime as a

def final_report():
    reports=a.OUT/'reports';f=pd.read_csv(a.OUT/'metrics_per_fold_seed.csv')
    assert len(f)==36
    macro=pd.read_csv(a.OUT/'macro_per_seed.csv');methods=['Full_B2','RNA_only_Encoder','PCA_RNA64']
    def summary(g,k):return f'{g[k].mean():.3f} ± {g[k].std(ddof=1):.3f}'
    lines=['# AD 四折 LODO 正式结果','',
      '四折 × 三个固定 seeds × 三种表征，共36次评测，已全部完成。SOURCE-only checkpoint、填充和标准化；',
      '每 donor/技术组一条监督样本。冻结阈值0.5，不依据TARGET性能调参。','',
      '## 表征与相同分类头','',
      'Full B2 Student、RNA-only AE Encoder、PCA RNA64都输出每细胞64维embedding。',
      '按Excitatory、Inhibitory、Astrocytes、Oligodendrocytes、OPC分别取donor内均值，拼为320维。',
      '每种表征分别拟合相同配置的L2 LogisticRegression(C=0.1,lbfgs,max_iter=1000,tol=1e-4,balanced,fit_intercept=True)。',
      'PCA仅由SOURCE train拟合；二维PCA图只作展示。新队列使用作者注释；每折三个SOURCE都参与Encoder与Probe。','',
      '## 等折宏平均','',
      '先对每个seed等权平均四折，再报告三个seed的mean±sample SD(ddof=1)。不是合并所有donor计算的指标。','',
      '| Method | AUROC | Balanced Accuracy | Macro F1 |','|---|---:|---:|---:|']
    for method in methods:
        g=macro[macro.method.eq(method)];assert sorted(g.seed.tolist())==a.SEEDS
        lines.append('| '+method+' | '+' | '.join(summary(g,k) for k in ['AUROC','Balanced_Accuracy','Macro_F1'])+' |')
    full=macro[macro.method.eq('Full_B2')].set_index('seed').loc[a.SEEDS]
    for method in ['RNA_only_Encoder','PCA_RNA64']:
        base=macro[macro.method.eq(method)].set_index('seed').loc[a.SEEDS]
        difference=full.AUROC-base.AUROC
        lines+=['',f'Full B2相对{method}的等折AUROC差值：{difference.mean():+.3f}；配对seed差值的sample SD={difference.std(ddof=1):.3f}。']
    lines+=['','这些SD仅反映seed变化，不能作为donor泛化性能的置信区间，也不能仅凭三个seed断言统计显著。',
            '','## 逐折结果','','| TARGET | Method | AUROC | Balanced Accuracy | Macro F1 |','|---|---|---:|---:|---:|']
    for fold in ['fold1','fold2','fold3','fold4']:
        for method in methods:
            g=f[f.fold.eq(fold)&f.method.eq(method)]
            lines.append('| '+g.target.iloc[0]+' | '+method+' | '+' | '.join(summary(g,k) for k in ['AUROC','Balanced_Accuracy','Macro_F1'])+' |')
    biology=[]
    for path in (a.OUT/'biology').glob('fold*/seed_*/*/COMPLETE.json'):
        m=a.read(path);biology.append(m)
    assert len(biology)==36
    lines+=['','## 预测组差异分析与富集','',
       '按TARGET预测AD/Control分组，以donor/技术组的全基因CP10K log1p均值做Welch检验和BH；',
       '每组至少2单位，否则NOT_EVALUABLE。GO BP2025/Reactome2022用实际检验且映射到各库的基因为背景。',
       '以下为Full B2的12次下游结果；全部方法的状态见biology_run_summary.csv。','',
       '| Fold | Seed | Predicted AD / Control | Status | Significant genes | GO significant terms (AD/Control) | Reactome significant terms (AD/Control) |',
       '|---|---:|---:|---|---:|---:|---:|']
    for m in sorted([x for x in biology if x['method']=='Full_B2'],key=lambda x:(x['fold'],x['seed'])):
        counts=m.get('enrichment_counts',{})
        sig=m.get('significant_genes','NA')
        go=f"{counts.get('AD_up_GO_BP','NA')}/{counts.get('Control_up_GO_BP','NA')}"
        react=f"{counts.get('AD_up_Reactome','NA')}/{counts.get('Control_up_Reactome','NA')}"
        lines.append(f"| {m['fold']} | {m['seed']} | {m['n_predicted_AD']}/{m['n_predicted_Control']} | {m['status']} | {sig} | {go} | {react} |")
    lines+=['','预测组由RNA表达派生，差异和富集是探索性关联，不能作为独立疾病机制验证。',
      '无显著富集时如实保留空结果；未为获得显著性修改阈值、基因列表或术语过滤规则。','',
      '## 数据与解释范围','',
      'GSE268609沿用13技术组(8AD/5HA)，不等同已核验13名独立生物donor；',
      'SEAAD为21 Intermediate/High与3 Not AD，类别不平衡且疾病终点与临床AD不同；',
      'GSE288063/GSE278576为预先固定5AD/5Control，作者QC条形码和作者细胞类型；ad22两个库合并为一个donor。',
      '五类缺失只按Probe-eligible SOURCE donor均值填充，个别单位细胞数少，覆盖表保留。','',
      '## 验证与文件','',
      'FINAL_INTEGRITY_CHECK.json记录36次保存概率重放、指标重算、SOURCE拟合统计量、',
      '固定Probe参数、共同单位/P2覆盖和SOURCE最佳checkpoint核查结果。',
      '本轮完成确定性重放和数据验证；独立语义审查不由这些检查替代。',
      '原始矩阵、cell embedding、模型checkpoint保留服务器；本地仅同步必要图像、概率、指标和差异/富集表。']
    (reports/'FINAL_RESULTS.md').write_text('\n'.join(lines)+'\n')
    a.write(reports/'FINAL_COMPLETE.json',dict(status='COMPLETE',folds=4,seeds=a.SEEDS,evaluations=36,
            representation_training_units=12,biology_runs=36,biology_counts=dict(Counter(m['status'] for m in biology)),
            deterministic_check=a.read(reports/'FINAL_INTEGRITY_CHECK.json')['status'],
            protocol_sha256=a.EXPECTED_PROTOCOL,semantic_audit_scope='inflight snapshot; see independent report'))

def main():
    q=a.OUT/'queue';env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
    last_count=-1
    while True:
        st=a.read(q/'status.json');analysis=a.read(q/'analysis_status.json')
        count=len(list((a.OUT/'evaluation').glob('fold*/seed_*/*/COMPLETE.json')))
        if count!=last_count:
            subprocess.run([sys.executable,str(a.OUT/'source/ad_comparison_report.py')],check=True,env=env)
            last_count=count
        if st['stage']=='FAILED' or analysis['stage']=='FAILED':raise RuntimeError(dict(training=st,analysis=analysis))
        if analysis['stage']=='ALL_ANALYSES_COMPLETE':break
        a.write(q/'finish_status.json',dict(stage='WAIT_COMPLETE_EVIDENCE',evaluations=count))
        time.sleep(30)
    a.write(q/'finish_status.json',dict(stage='VERIFY_FINAL'))
    subprocess.run([sys.executable,str(a.OUT/'source/ad_final_check.py')],check=True,env=env)
    final_report()
    subprocess.run([sys.executable,str(a.OUT/'source/ad_export.py')],check=True,env=env,stdout=subprocess.DEVNULL)
    a.write(q/'finish_status.json',dict(stage='COMPLETE'))
    print('FINAL_REPORT_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        a.write(a.OUT/'queue/finish_status.json',dict(stage='FAILED',traceback=traceback.format_exc()))
        raise
