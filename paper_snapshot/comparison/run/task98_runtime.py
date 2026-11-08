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
            p=OUT/key/'pre_final'/site/f'seed_{seed}'/'manifest.json'
            if not p.exists(): continue
            d=json.loads(p.read_text()); e=d.get('efficiency_v2',{})
            train=float(d.get('training_seconds',e.get('training',{}).get('seconds',0)))
            infer=float(d.get('inference_seconds',e.get('inference',{}).get('seconds',0)))
            load=float(d.get('input_load_seconds',e.get('input_load_seconds',0)))
            total=d.get('total_wall_seconds')
            if total is None: total=train+infer
            peak=d.get('peak_gpu_allocated_bytes')
            rows.append(dict(Method=label,MethodKey=key,Site=site,Seed=seed,
                Training_seconds=train,Inference_seconds=infer,Input_load_seconds=load,
                Total_seconds=float(total),Trainable_parameters=d.get('trainable_parameters') or e.get('trainable_parameters'),
                Peak_GPU_GiB=(float(peak)/2**30 if peak else None),Device=('cuda' if d.get('gpu_model') else ('cpu' if key=='palette' else 'CPU'))))
df=pd.DataFrame(rows); df.to_csv(OUT/'TASK98_RUNTIME.csv',index=False)
metrics=['Training_seconds','Inference_seconds','Input_load_seconds','Total_seconds']
g=df.groupby(['Method','Site'])[metrics].agg(['mean','std']).reset_index(); g.columns=['Method','Site']+[f'{m}_{s}' for m in metrics for s in ('mean','std')]
g.to_csv(OUT/'TASK98_RUNTIME_SUMMARY.csv',index=False)
eq=df.groupby(['Method'])[metrics].agg(['mean','std']).reset_index(); eq.columns=['Method']+[f'{m}_{s}' for m in metrics for s in ('mean','std')]
eq.to_csv(OUT/'TASK98_RUNTIME_EQUAL.csv',index=False)
lines=['# Task 98 — recorded runtime','','Runtime is the per-unit value stored in each completed manifest. `Total_seconds` is the recorded total-wall field when present; otherwise training + inference. Cache build is not included in these per-unit timers.','','## Equal-weight runtime (all 12 units per method)','','| Method | Training (s) | Inference (s) | Input load (s) | Total (s) |','|---|---:|---:|---:|---:|']
for _,r in eq.iterrows():
    lines.append(f"| {r['Method']} | {r['Training_seconds_mean']:.1f} ± {r['Training_seconds_std']:.1f} | {r['Inference_seconds_mean']:.3f} ± {r['Inference_seconds_std']:.3f} | {r['Input_load_seconds_mean']:.2f} ± {r['Input_load_seconds_std']:.2f} | {r['Total_seconds_mean']:.1f} ± {r['Total_seconds_std']:.1f} |")
lines += ['','## Per-Site runtime','','| Method | Site | Training (s) | Inference (s) | Total (s) |','|---|---|---:|---:|---:|']
for _,r in g.iterrows():
    lines.append(f"| {r['Method']} | {r['Site']} | {r['Training_seconds_mean']:.1f} ± {r['Training_seconds_std']:.1f} | {r['Inference_seconds_mean']:.3f} ± {r['Inference_seconds_std']:.3f} | {r['Total_seconds_mean']:.1f} ± {r['Total_seconds_std']:.1f} |")
lines += ['','## Artifacts','','- `TASK98_RUNTIME.csv` — 48 per-unit rows','- `TASK98_RUNTIME_SUMMARY.csv` — per method/Site mean±SD','- `TASK98_RUNTIME_EQUAL.csv` — per-method mean±SD','']
(OUT/'TASK98_RUNTIME_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(eq.to_string(index=False)); print(g.to_string(index=False))
