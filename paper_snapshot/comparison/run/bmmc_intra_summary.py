import json
from pathlib import Path
import pandas as pd
ROOT=Path('/home/u202522802045/share/zpb')
OUT=ROOT/'comparison/result/BMMC_intra'
METHODS={'multigrate':'Multigrate','cobolt':'Cobolt','palette':'Palette-U','eclare':'ECLARE-S'}
SITES=['BMMC_Site1','BMMC_Site2','BMMC_Site3','BMMC_Site4']; SEEDS=[0,1,2]
rows=[]
for key,label in METHODS.items():
    for site in SITES:
        for seed in SEEDS:
            p=OUT/key/'pre_final'/site/f'seed_{seed}'/'metrics.json'
            if not p.exists(): continue
            m=json.loads(p.read_text())
            rows.append(dict(Method=label,MethodKey=key,Site=site,Seed=seed,
                ARI=m['ari'],NMI=m['nmi'],AMI=m['ami'],ACC=m['cluster_acc'],
                Macro_F1=m['macro_f1'],Balanced_ACC=m['balanced_accuracy'],
                Weighted_F1=m['weighted_f1'],Purity_at_15=m['knn_purity_at_15'],Silhouette=m['silhouette']))
df=pd.DataFrame(rows); df.to_csv(OUT/'results_raw.csv',index=False)
metrics=['ARI','NMI','AMI','ACC','Macro_F1','Balanced_ACC','Weighted_F1','Purity_at_15','Silhouette']
g=df.groupby(['Method','MethodKey','Site'])[metrics].agg(['mean','std']).reset_index()
g.columns=['Method','MethodKey','Site']+[f'{m}_{s}' for m in metrics for s in ('mean','std')]
g.to_csv(OUT/'results_summary.csv',index=False)
eq=df.groupby(['Method','MethodKey'])[metrics].agg(['mean','std']).reset_index()
eq.columns=['Method','MethodKey']+[f'{m}_{s}' for m in metrics for s in ('mean','std')]
eq.to_csv(OUT/'results_equal_weight.csv',index=False)
lines=['# BMMC Intra comparison — 4 methods × 4 Sites × 3 seeds','','scMomer-S removed by user; not included.','','| Method | ARI | NMI | ACC | Macro-F1 |','|---|---:|---:|---:|---:|']
for row in eq.itertuples(index=False):
    lines.append(f'| {row.Method} | {row.ARI_mean:.4f} ± {row.ARI_std:.4f} | {row.NMI_mean:.4f} ± {row.NMI_std:.4f} | {row.ACC_mean:.4f} ± {row.ACC_std:.4f} | {row.Macro_F1_mean:.4f} ± {row.Macro_F1_std:.4f} |')
lines += ['','Per-Site results: `results_summary.csv`; raw units: `results_raw.csv`.','']
(OUT/'FINAL_REPORT.md').write_text('\n'.join(lines))
print('rows',len(df)); print(eq.to_string(index=False)); print(g.to_string(index=False))
