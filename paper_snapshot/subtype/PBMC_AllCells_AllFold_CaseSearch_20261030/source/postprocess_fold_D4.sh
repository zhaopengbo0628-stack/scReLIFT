#!/usr/bin/env bash
set -euo pipefail
cd /home/u202522802045/codex_jobs/PBMC_AllCells_AllFold_CaseSearch_20261030
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
exec /home/u202522802045/envs/graph_attention/bin/python -u source/postprocess_fold.py D4 > logs/postprocess_fold_D4.log 2>&1
