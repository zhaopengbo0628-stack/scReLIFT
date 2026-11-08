#!/bin/bash
set -euo pipefail
ROOT=/home/u202522802045/share/zpb
EXP="$ROOT/b2_fusion_ablation"
PY=/home/u202522802045/envs/glue_156478/bin/python
exec 9>"$ROOT/.benchmark_a30.lock"
echo "$$" > "$EXP/launcher.pid"
status() { printf '%s %s\n' "$(date -Is)" "$*" > "$EXP/launcher.status"; }
trap 'code=$?; echo "$code" > "$EXP/launcher.exitcode"; status "EXITED code=$code"' EXIT
echo WAITING_GLOBAL_GPU_LOCK
status WAITING_GLOBAL_GPU_LOCK
flock 9
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
status A30_LOCK_ACQUIRED
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for variant in Concat Gate Interaction; do
    for seed in 20020130 20020131 20020132; do
        for fold in target_D1 target_D2 target_D3 target_D4; do
            unit="$EXP/results/$variant/$fold/seed_$seed"
            mkdir -p "$unit"
            status "RUNNING $variant $fold $seed"
            "$PY" -u "$EXP/run_fusion.py" --variant "$variant" --fold "$fold" --seed "$seed" >> "$unit/run.log" 2>&1 &
            run_pid=$!
            while kill -0 "$run_pid" 2>/dev/null; do
                status "RUNNING $variant $fold $seed pid=$run_pid"
                sleep 60
            done
            if wait "$run_pid"; then
                status "COMPLETE $variant $fold $seed"
                echo "COMPLETE $variant $fold $seed"
            else
                code=$?
                if [ "$code" -eq 20 ]; then status "SCIENTIFIC_BLOCKER $variant $fold $seed"; echo "SCIENTIFIC_BLOCKER $variant $fold $seed"; else exit "$code"; fi
            fi
        done
    done
done
echo FUSION_RUNS_FINISHED
status FUSION_RUNS_FINISHED
