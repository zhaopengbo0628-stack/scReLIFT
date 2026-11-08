"""One-shot background sync for the extra CPU RNA view; exits after success."""
import json
import subprocess
import sys
import time
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
REMOTE='/home/u202522802045/codex_jobs/PBMC_D1_AllCells_Dmode_20261030'

def main():
    for attempt in range(240):
        probe=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','zpb',
             f'test -f {REMOTE}/audit/FINAL_COMPLETION.json'],capture_output=True,text=True,timeout=30)
        if probe.returncode==0:
            for name in ['figures/RNA_QC_fullgene_log1p_no_PCA','audit','logs']:
                parent=RUN/Path(name).parent
                parent.mkdir(parents=True,exist_ok=True)
                subprocess.run(['scp','-q','-r',f'zpb:{REMOTE}/{name}',str(parent)],check=True,timeout=300)
            subprocess.run(['scp','-q',f'zpb:{REMOTE}/*.md',f'zpb:{REMOTE}/*.csv',str(RUN)],check=True,timeout=120)
            subprocess.run([sys.executable,str(RUN/'source/verify_download.py')],check=True)
            (RUN/'audit/LOCAL_SYNC_STATUS.json').write_text(json.dumps(dict(status='COMPLETE',
                reports_and_11_plots_local=True,remote=REMOTE),indent=2)+'\n')
            print('LOCAL_FINAL_SYNC_COMPLETE',flush=True);return
        print('WAIT_EXTRA_CPU_RNA_VIEW',attempt+1,flush=True)
        time.sleep(90)
    raise RuntimeError('Extra CPU view sync timed out after six hours; primary training/PCA results were already downloaded')

if __name__=='__main__': main()
