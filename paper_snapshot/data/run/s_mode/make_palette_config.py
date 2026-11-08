import json
from pathlib import Path
ROOT=Path('/home/u202522802045/share/zpb')
p=ROOT/'comparison/configs/palette/s_mode_bmmc_intra.json'
d=json.loads((ROOT/'comparison/configs/palette/s_mode_pbmc_intra.json').read_text())
d.update(method='Palette-U',datasets=['BMMC_Site1','BMMC_Site2','BMMC_Site3','BMMC_Site4'],seeds=[0,1,2])
p.write_text(json.dumps(d,indent=2)+'\n')
print(p)
