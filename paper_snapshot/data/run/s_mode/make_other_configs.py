import json
from pathlib import Path
ROOT=Path('/home/u202522802045/share/zpb')
sites=['BMMC_Site1','BMMC_Site2','BMMC_Site3','BMMC_Site4']
for method in ['multigrate','cobolt','eclare','scmomer']:
    src=ROOT/f'comparison/configs/{method}/s_mode_pbmc_intra.json'
    d=json.loads(src.read_text())
    d.update(datasets=sites,seeds=[0,1,2])
    d['method']={'eclare':'ECLARE-S','scmomer':'scMomer-S'}.get(method,method)
    out=ROOT/f'comparison/configs/{method}/s_mode_bmmc_intra.json'
    out.write_text(json.dumps(d,indent=2)+'\n')
    print(out)
