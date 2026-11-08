from pathlib import Path
p=Path('/home/u202522802045/share/zpb/comparison/run/bmmc_neural_intra.py')
s=p.read_text().replace("choices=['D1','D2','D3','D4']","choices=['BMMC_Site1','BMMC_Site2','BMMC_Site3','BMMC_Site4']")
p.write_text(s)
for method in ('eclare','scmomer'):
    q=Path(f'/home/u202522802045/share/zpb/comparison/run/bmmc_{method}_intra.py')
    q.write_text(f'''import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from bmmc_neural_intra import main
if __name__=='__main__':
    main('{method}')
''')
print('patched')
