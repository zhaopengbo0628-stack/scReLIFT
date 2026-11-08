#!/bin/bash
set -euo pipefail
ROOT=/home/u202522802045/share/zpb
OUT="$ROOT/main/result/d_mode/corrected_pbmc_legacy"
PY=/home/u202522802045/envs/glue_156478/bin/python
META="$OUT/_benchmark_meta"
mkdir -p "$META"
exec 9>"$ROOT/.benchmark_a30.lock"
printf '%s\n' "$$" > "$META/validated_launcher.pid"
trap 'code=$?; printf "%s\n" "$code" > "$META/validated_launcher.exitcode"' EXIT
echo WAITING_GLOBAL_GPU_LOCK
flock 9
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
    echo WAITING_EXISTING_GPU_WORKLOAD
    sleep 30
done
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo GLOBAL_GPU_LOCK_ACQUIRED
for seed in 20020130 20020131 20020132; do
    for fold in target_D1 target_D2 target_D3 target_D4; do
        unit="$OUT/$fold/seed_$seed"
        mkdir -p "$unit/logs"
        echo "RUNNING $fold $seed"
        "$PY" -u "$ROOT/main/run/d_mode/run_updated_b2_validated.py" --fold "$fold" --seed "$seed" --output-dir "$unit" --resume >> "$unit/logs/run.log" 2>&1
        echo "FINISHED $fold $seed"
    done
done
"$PY" -u "$ROOT/main/run/d_mode/run_updated_b2_validated.py" --summarize
echo BENCHMARK_FINISHED
