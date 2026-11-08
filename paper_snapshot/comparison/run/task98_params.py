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
            d=json.loads(p.read_text())
            rows.append(dict(Method=label,Site=site,Seed=seed,
                trainable=d.get('trainable_parameters'),inference=d.get('inference_parameters'),
                peak=d.get('peak_gpu_allocated_bytes'),gpu=d.get('gpu_model'),
                training=d.get('training_seconds'),inference_seconds=d.get('inference_seconds')))
df=pd.DataFrame(rows)
for m,g in df.groupby('Method'):
    print(m,'trainable',g.trainable.dropna().unique().tolist(),'inference',g.inference.dropna().unique().tolist(),
          'peak_max',g.peak.max(),'gpu',g.gpu.dropna().unique().tolist(),
          'training_mean',g.training.mean(),'inference_mean',g.inference_seconds.mean())
