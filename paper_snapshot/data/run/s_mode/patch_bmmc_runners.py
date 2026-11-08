from pathlib import Path
p=Path('/home/u202522802045/share/zpb/comparison/run/bmmc_neural_intra.py')
s=p.read_text()
old='import sys as _sys; _sys.path.insert(0,str(ROOT/" comparison/run/bmmc_shim))'
new="import sys as _sys; _sys.path.insert(0,str(ROOT/'comparison/run/bmmc_shim'))"
assert old in s, 'old line missing'
p.write_text(s.replace(old,new))
q=Path('/home/u202522802045/share/zpb/comparison/run/bmmc_s_mode_intra.py')
t=q.read_text()
t=t.replace("choices=['D1','D2','D3','D4']","choices=['BMMC_Site1','BMMC_Site2','BMMC_Site3','BMMC_Site4']")
q.write_text(t)
print('patched')
