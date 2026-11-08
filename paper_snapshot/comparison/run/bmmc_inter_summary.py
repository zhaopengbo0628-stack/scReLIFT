import json
from pathlib import Path
import pandas as pd
ROOT=Path('/home/u202522802045/share/zpb')
OUT=ROOT/'comparison/result/BMMC_inter/b2full'
rows=[]
for d in sorted(OUT.glob('D*_to_D*')):
    for p in sorted(d.glob('seed_*/metrics.json')):
        m=json.loads(p.read_text()); rows.append(dict(Direction=d.name,Seed=int(p.parent.name.split('_')[1]),**m))
df=pd.DataFrame(rows); df.to_csv(OUT/'results_raw.csv',index=False)
metrics=['ARI','NMI','AMI','ACC','Macro_F1','Balanced_ACC','Weighted_F1','Purity_at_15','Silhouette']
g=df.groupby('Direction')[metrics].agg(['mean','std']).reset_index(); g.columns=['Direction']+[f'{m}_{s}' for m in metrics for s in ('mean','std')]
g.to_csv(OUT/'results_summary.csv',index=False)
lines=['# BMMC Full B2 inter — 3 seeds','','| Direction | ARI | NMI | ACC | Macro-F1 |','|---|---:|---:|---:|---:|']
for row in g.itertuples(index=False):
    lines.append(f'| {row.Direction} | {row.ARI_mean:.4f} ± {row.ARI_std:.4f} | {row.NMI_mean:.4f} ± {row.NMI_std:.4f} | {row.ACC_mean:.4f} ± {row.ACC_std:.4f} | {row.Macro_F1_mean:.4f} ± {row.Macro_F1_std:.4f} |')
lines += ['','Source sites: D1=BMMC_Site1, D2=Site2, D3=Site3, D4=Site4.','Target RNA-only; target ATAC not used.','']
(OUT/'FINAL_REPORT.md').write_text('\n'.join(lines))
print(g.to_string(index=False))
