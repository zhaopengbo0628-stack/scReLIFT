#!/usr/bin/env bash
set -euo pipefail
cd /home/u202522802045/codex_jobs/PBMC_D1_AllCells_Dmode_20261030
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
exec /home/u202522802045/envs/graph_attention/bin/python -u source/prepare_allcells.py > logs/prepare.log 2>&1
