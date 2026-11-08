#!/bin/bash
set -euo pipefail
root=/home/u202522802045/share/zpb
out="$root/result_contain/74_scmomer_ad_donor_benchmark_20261015"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
py=/home/u202522802045/envs/scmomer_eclare_sintra/bin/python
echo "$$" > "$root/slum/74_resume_20261022.pid"
echo "$$" > "$root/slum/74_scmomer_ad_donor_benchmark_20261015.pid"
trap 'code=$?; echo "$code" > "$root/slum/74_resume_20261022.exitcode"' EXIT
cd "$root"
"$py" -u comparison/run/scmomer/ad_donor/run.py --out "$out"
