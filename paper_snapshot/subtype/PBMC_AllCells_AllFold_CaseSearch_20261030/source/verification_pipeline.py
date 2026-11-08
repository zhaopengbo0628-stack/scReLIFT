import time,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
while not (ROOT/'audit/FINAL_CASE_ARTIFACTS_COMPLETE.json').exists():time.sleep(15)
subprocess.run([sys.executable,'-u',str(ROOT/'source/verify_final_results.py')],cwd=ROOT,check=True)
