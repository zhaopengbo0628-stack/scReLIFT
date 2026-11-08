import json
from pathlib import Path
p=Path('/home/u202522802045/share/zpb/comparison/configs/cobolt/s_mode_bmmc_intra.json')
d=json.loads(p.read_text()); d['learning_rate']=0.001
p.write_text(json.dumps(d,indent=2)+'\n')
print('cobolt lr',d['learning_rate'])
