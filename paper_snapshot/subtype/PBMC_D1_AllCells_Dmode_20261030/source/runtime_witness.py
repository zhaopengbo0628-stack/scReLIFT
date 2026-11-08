import importlib
import importlib.util
import json
import subprocess
import sys
from common import RUN, SRC, FROZEN, write
import torch

def main():
    spec=json.loads((RUN/'env-spec.json').read_text())
    versions={}
    for name,expected in spec['packages'].items():
        actual=importlib.import_module(name.replace('scikit-learn','sklearn')).__version__
        assert actual==expected,(name,actual,expected)
        versions[name]=actual
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    assert not apps, 'GPU has existing process; do not run GPU witness'
    for row in gpu.strip().splitlines():
        index,mem,util=[int(x.strip()) for x in row.split(',')]
        assert mem<500 and util<5,(mem,util)
    ref=importlib.util.spec_from_file_location('frozen_b2',FROZEN/'run_b2_strict_lodo.py')
    b2=importlib.util.module_from_spec(ref);ref.loader.exec_module(b2)
    b2.seed_all(20020130)
    model=b2.Teacher(32,32).to('cuda')
    x=torch.randn(8,32,device='cuda');field=torch.randn_like(x)
    loss=sum(b2.teacher_losses(model,x,field));loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    write(RUN/'audit/ENV_KERNEL_WITNESS.json',dict(status='PASS',python=sys.executable,packages=versions,
          seed=20020130,loss=float(loss.detach()),gpu=torch.cuda.get_device_name(),finite_backward=True,shape=[8,64]))
    print('WITNESS_PASS',torch.cuda.get_device_name(),float(loss.detach()),flush=True)

if __name__=='__main__': main()
