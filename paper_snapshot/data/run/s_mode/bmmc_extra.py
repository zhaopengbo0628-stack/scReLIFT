import json
from pathlib import Path
import pandas as pd, torch
ROOT=Path('/home/u202522802045/share/zpb')
OUT=ROOT/'result_contain/97_bmmc_b2full_rnaonly_3seed_intra_20261024'
CACHE=ROOT/'data/cache/s_mode/BMMC/bmmc_site_intra_v1_20261024'
SITES=['BMMC_Site1','BMMC_Site2','BMMC_Site3','BMMC_Site4']
SEEDS=[20020130,20020131,20020132]
rows=[]
for site in SITES:
    for seed in SEEDS:
        p=OUT/'models'/site/f'seed_{seed}'/'training_manifest.json'
        if p.exists():
            d=json.loads(p.read_text()); e=d['efficiency_v2']
            rows.append(dict(Method='Full B2-S',Site=site,Seed=seed,Training_seconds=e['training']['seconds'],
                Trainable_params=e['trainable_parameters'],Inference_params=e['inference_resident_parameters'],
                Peak_allocated_GiB=e['training']['peak_allocated_bytes']/2**30,
                GPU=d['environment']['gpu']))
pd.DataFrame(rows).to_csv(OUT/'metrics/b2_efficiency.csv',index=False)
rrows=[]
for site in SITES:
    for seed in SEEDS:
        p=OUT/'rna_only/models'/site/f'seed_{seed}'/'best.pt'
        if p.exists():
            s=torch.load(p,map_location='cpu',weights_only=False)
            params=sum(v.numel() for v in s['model'].values() if torch.is_tensor(v))
            enc=sum(v.numel() for k,v in s['model'].items() if k.startswith('encoder.') and torch.is_tensor(v))
            rrows.append(dict(Method='RNA-only AE',Site=site,Seed=seed,Trainable_params=params,
                Inference_params=enc,Training_seconds=None))
pd.DataFrame(rrows).to_csv(OUT/'metrics/rna_efficiency.csv',index=False)
g=[]
for site in SITES:
    m=json.loads((CACHE/site/'manifest.json').read_text())
    g.append(dict(Dataset=site,Candidate_genes=m['genes'],Candidate_peaks=m['peaks'],
        Candidate_edges=m['candidate_edges'],Retained_genes=m['retained_genes'],Retained_edges=m['retained_edges'],
        Selector_seconds=m['selector_seconds'],Teacher_geometry_seconds=m['field_seconds']))
pd.DataFrame(g).to_csv(OUT/'metrics/regulatory_graph.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(rrows).to_string(index=False)); print(pd.DataFrame(g).to_string(index=False))
