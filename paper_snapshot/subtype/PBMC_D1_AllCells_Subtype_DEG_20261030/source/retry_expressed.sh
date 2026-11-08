#!/usr/bin/env bash
set -euo pipefail
while tmux has-session -t pbmc_d1_expressed_sensitivity_20261030 2>/dev/null; do sleep 15; done
cd /home/u202522802045/codex_jobs/PBMC_D1_AllCells_Subtype_DEG_20261030
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
exec /home/u202522802045/envs/graph_attention/bin/python -u source/fgsea_expressed.py > logs/fgsea_expressed_retry.log 2>&1
