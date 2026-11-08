from pathlib import Path
p=Path('/home/u202522802045/share/zpb/comparison/run/bmmc_s_mode_intra.py')
s=p.read_text()
old="                value.backward()\n                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):"
new="                value.backward()\n                if method=='cobolt':\n                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)\n                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):"
assert old in s, 'target block missing'
p.write_text(s.replace(old,new))
print('patched cobolt clipping')
