from pathlib import Path
import subprocess,time,os
b=Path("/home/u202522802045/share/zpb/regulatory/motif_null_corrected_20261101")
print("Waiting for 8 successful scan completion markers",flush=True)
while len(list((b/"scan").glob("chunk_*.done.json")))<8:
    if not Path("/proc/650").exists():
        raise RuntimeError("Scan launcher exited before all 8 chunks finished; inspect run.log")
    time.sleep(10)
print("All scan markers present; calculating strict null",flush=True)
p=subprocess.run(["/opt/conda/bin/python","-u","/home/u202522802045/share/zpb/regulatory/run_motif_null_strict.py","--stage","statistics"])
if p.returncode:raise SystemExit(p.returncode)
