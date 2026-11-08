#!/usr/bin/env bash
set -euo pipefail
cd /home/u202522802045/codex_jobs/PBMC_AllCells_AllFold_CaseSearch_20261030
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
exec /home/u202522802045/envs/graph_attention/bin/python -u source/score_candidates.py D3 > logs/score_D3.log 2>&1
