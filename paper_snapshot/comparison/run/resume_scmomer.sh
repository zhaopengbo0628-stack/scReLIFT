#!/bin/bash
set -u
cd /home/u202522802045/share/zpb
# incomplete unit from the killed queue
if [ -d comparison/result/BMMC_intra/scmomer/pre_final/BMMC_Site2/seed_1 ] && [ ! -f comparison/result/BMMC_intra/scmomer/pre_final/BMMC_Site2/seed_1/manifest.json ]; then
  rm -rf comparison/result/BMMC_intra/scmomer/pre_final/BMMC_Site2/seed_1
fi
for item in BMMC_Site2:1 BMMC_Site2:2 BMMC_Site3:0 BMMC_Site3:1 BMMC_Site3:2 BMMC_Site4:0 BMMC_Site4:1 BMMC_Site4:2; do
  site=${item%%:*}; seed=${item##*:}
  unit="comparison/result/BMMC_intra/scmomer/pre_final/$site/seed_$seed"
  if [ -f "$unit/manifest.json" ] && grep -q '"status": "COMPLETE"' "$unit/manifest.json"; then
    continue
  fi
  /home/u202522802045/envs/scmomer_eclare_sintra/bin/python -u comparison/run/bmmc_scmomer_intra.py --stage unit --dataset "$site" --seed "$seed" --device cuda
done
