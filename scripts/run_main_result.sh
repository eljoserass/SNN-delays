#!/usr/bin/env bash
set -euo pipefail

SEEDS=(0 1 2)
SIGMAS=(0.5 1.0 2.0 5.0)
RUN_NAME_PREFIX=${RUN_NAME_PREFIX:-"MainResultDelayDrop"}

for sigma in "${SIGMAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== Running sigma_drop=${sigma}, seed=${seed} ==="
    python main.py --sigma_drop "${sigma}" --seed "${seed}" --run_name "${RUN_NAME_PREFIX}"
  done
done
