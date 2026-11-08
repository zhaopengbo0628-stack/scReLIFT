"""Descriptive comparisons of complete seed triplets; never select or tune a method."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ad_runtime as a

METHODS=['Full_B2','RNA_only_Encoder','PCA_RNA64']
METRICS=['AUROC','Balanced_Accuracy','Macro_F1','Sensitivity','Specificity','Accuracy','AUPRC']

def main():
    records=[a.read(p) for p in (a.OUT/'evaluation').glob('fold*/seed_*/*/metrics.json')
             if (p.parent/'COMPLETE.json').exists()]
    if not records:return
    data=pd.DataFrame(records);reports=a.OUT/'reports';reports.mkdir(exist_ok=True)
    rows=[];delta=[];completed=[]
    for fold,g in data.groupby('fold',sort=True):
        if len(g)!=9 or set(g.method)!=set(METHODS):continue
        assert all(sorted(v.seed.tolist())==a.SEEDS for _,v in g.groupby('method'))
        completed.append(fold)
        for method in METHODS:
            subset=g[g.method.eq(method)]
            for metric in METRICS:
                rows.append(dict(fold=fold,target=subset.target.iloc[0],method=method,metric=metric,
                    mean=float(subset[metric].mean()),sample_SD=float(subset[metric].std(ddof=1)),n_seeds=3,
                    n_probe_units=int(subset.n_probe_units.iloc[0]),n_target_units=int(subset.n_target_units.iloc[0])))
        full=g[g.method.eq('Full_B2')].set_index('seed').loc[a.SEEDS]
        for baseline in ['RNA_only_Encoder','PCA_RNA64']:
            base=g[g.method.eq(baseline)].set_index('seed').loc[a.SEEDS]
            for metric in METRICS:
                difference=full[metric]-base[metric]
                baseline_mean=float(base[metric].mean());full_mean=float(full[metric].mean())
                delta.append(dict(fold=fold,target=full.target.iloc[0],method='Full_B2',baseline=baseline,
                    metric=metric,Full_B2_mean=full_mean,baseline_mean=baseline_mean,
                    absolute_delta=full_mean-baseline_mean,
                    relative_delta_percent=(100*(full_mean-baseline_mean)/baseline_mean if baseline_mean else None),
                    paired_seed_difference_SD=float(difference.std(ddof=1)),n_seeds=3,
                    uncertainty='seed variability only; no donor-level confidence interval'))
        fig,ax=plt.subplots(figsize=(8.4,4.8));x=np.arange(3)
        for j,method in enumerate(METHODS):
            v=g[g.method.eq(method)]
            ax.bar(x+(j-1)*.24,[v[k].mean() for k in METRICS[:3]],.24,
                   yerr=[v[k].std(ddof=1) for k in METRICS[:3]],capsize=3,
                   label=method,color=['#227c9d','#e5a02b','#8b64ad'][j])
        ax.set_xticks(x,['AUROC','Balanced accuracy','Macro F1']);ax.set_ylim(0,1.12)
        ax.set_ylabel('Mean ± sample SD across three fixed seeds')
        ax.set_title(f'{fold}: TARGET {g.target.iloc[0]}\nComplete single-fold comparison; four-fold study ongoing'
                     if len(data)<36 else f'{fold}: TARGET {g.target.iloc[0]}')
        ax.legend(frameon=False,fontsize=9);ax.spines[['top','right']].set_visible(False)
        fig.tight_layout();fig.savefig(reports/(fold+'_comparison.png'),dpi=200);plt.close(fig)
    if rows:
        table=pd.DataFrame(rows);table.to_csv(reports/'complete_fold_comparison.csv',index=False)
        pd.DataFrame(delta).to_csv(reports/'complete_fold_deltas.csv',index=False)
        lines=['# 已完成折的三方法比较','',
               '固定三个 seeds：20020130、20020131、20020132；数值为 mean ± sample SD(ddof=1)。',
               '每折必须三个方法各完成全部三个 seeds 才进入此表。未完成折不填补、不据部分结果调参。','',
               '| Fold / TARGET | Method | AUROC | Balanced Accuracy | Macro F1 |',
               '|---|---|---:|---:|---:|']
        for (fold,target,method),g in table.groupby(['fold','target','method'],sort=False):
            vals=g.set_index('metric')
            cells=[f"{vals.loc[k,'mean']:.3f} ± {vals.loc[k,'sample_SD']:.3f}" for k in METRICS[:3]]
            lines.append(f'| {fold} / {target} | {method} | '+' | '.join(cells)+' |')
        lines+=['','当前完整折：'+', '.join(completed)+'。',
                '四折均完成后才能给出正式等折宏平均及总体优劣结论。误差条只反映训练 seed 变异。',
                'GSE268609 单位为13技术组；SEAAD为ADNC终点；GSE288063/GSE278576为固定10名donor及作者QC/注释。']
        (reports/'COMPLETE_FOLD_COMPARISON.md').write_text('\n'.join(lines)+'\n')
    a.write(reports/'comparison_scope.json',dict(complete_folds=completed,total_planned_folds=4,
        evaluations_present=len(data),planned_evaluations=36,seed_selection=False,hyperparameters_changed=False))
    print('COMPLETE_FOLD_REPORT',completed,flush=True)

if __name__=='__main__':main()
