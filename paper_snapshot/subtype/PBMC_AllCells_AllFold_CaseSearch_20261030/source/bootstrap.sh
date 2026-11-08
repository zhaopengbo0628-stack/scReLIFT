#!/usr/bin/env bash
set -euo pipefail
cd /home/u202522802045/codex_jobs/PBMC_AllCells_AllFold_CaseSearch_20261030
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
/home/u202522802045/envs/graph_attention/bin/python -u source/bootstrap.py > logs/bootstrap.log 2>&1
for fold in D2 D3 D4; do /home/u202522802045/envs/graph_attention/bin/python -u folds/$fold/source/prepare_fold.py > logs/prepare_$fold.log 2>&1; done
