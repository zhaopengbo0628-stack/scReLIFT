#!/bin/bash
set -e
cd /home/u202522802045/share/zpb
for s in BMMC_Site2 BMMC_Site3 BMMC_Site4; do
  /home/u202522802045/envs/scmomer_eclare_sintra/bin/python -u comparison/run/bmmc_scmomer_intra.py --stage smoke --dataset "$s" --seed 0 --device cuda
done
