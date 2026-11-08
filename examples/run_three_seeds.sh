#!/usr/bin/env bash
set -euo pipefail
# Run from repository root after preparing one fold's source cache.
CACHE=${1:?Usage: bash examples/run_three_seeds.sh CACHE OUTPUT_ROOT}
OUTPUT_ROOT=${2:?Specify an output directory}
for seed in 20020130 20020131 20020132; do
  screlift train --cache "$CACHE" --seed "$seed" --output "$OUTPUT_ROOT/seed_$seed"
done
