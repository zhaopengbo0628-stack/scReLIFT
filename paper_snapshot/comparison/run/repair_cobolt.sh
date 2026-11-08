#!/bin/bash
set -u
cd /home/u202522802045/share/zpb
for item in BMMC_Site1:0 BMMC_Site1:1 BMMC_Site1:2 BMMC_Site2:0 BMMC_Site3:2 BMMC_Site4:0 BMMC_Site4:2; do
  site=${item%%:*}; seed=${item##*:}
  unit="comparison/result/BMMC_intra/cobolt/pre_final/$site/seed_$seed"
  if [ -f "$unit/failure.json" ]; then
    rm -rf "$unit"
    /home/u202522802045/envs/graph_attention/bin/python -u comparison/run/bmmc_s_mode_intra.py --method cobolt --stage unit --dataset "$site" --seed "$seed" --device cuda
  fi
done
