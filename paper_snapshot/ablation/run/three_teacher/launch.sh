#!/bin/bash
set -euo pipefail
ROOT=/home/u202522802045/share/zpb
EXP="$ROOT/b2_multi_teacher_ablation"
PY=/home/u202522802045/envs/glue_156478/bin/python
exec 9>"$ROOT/.benchmark_a30.lock"
echo "$$" > "$EXP/launcher.pid"
trap 'echo "$?" > "$EXP/launcher.exitcode"' EXIT
echo WAITING_GLOBAL_GPU_LOCK
flock 9
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for seed in 20020130 20020131 20020132; do
    for fold in target_D1 target_D2 target_D3 target_D4; do
        mkdir -p "$EXP/results/$fold/seed_$seed"
        "$PY" -u "$EXP/run_ablation.py" --fold "$fold" --seed "$seed" >> "$EXP/results/$fold/seed_$seed/run.log" 2>&1
    done
done
"$PY" -u "$EXP/run_ablation.py" --summarize
