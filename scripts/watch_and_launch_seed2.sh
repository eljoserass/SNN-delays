#!/usr/bin/env bash
set -euo pipefail
cd /workspace/SNN-delays
WATCH_DIR=".run_status/followup_seed2"
mkdir -p "$WATCH_DIR"
LAUNCH_MARKER="$WATCH_DIR/launched.ok"
RUNTIME_LOG="$WATCH_DIR/runtime.log"
LAUNCH_LOG="$WATCH_DIR/launch.log"
CURRENT_PATTERN="python main.py --dataset ssc --sigma_drop 0.25 --sparsity_p 0.96 --seed 0"

{
  echo "[$(date -u +"%F %T")] watcher start (seed=2)"
  if [[ -f "$LAUNCH_MARKER" ]]; then
    echo "[$(date -u +"%F %T")] launch marker exists, exiting"
    exit 0
  fi

  while pgrep -f "$CURRENT_PATTERN" >/dev/null; do
    echo "[$(date -u +"%F %T")] waiting for current seed0 sigma0.25 run to finish"
    sleep 120
  done

  echo "[$(date -u +"%F %T")] current run finished; launching seed=2 sweep"
  bash scripts/run_main_result.sh \
    --dataset ssc \
    --sigmas 0.05,0.1,0.25,0.5 \
    --seeds 2 \
    --sparsity-p 0.96 \
    --run-name-prefix DelayDropSSC \
    --status-dir .run_status/delaydrop_ssc_small_sigma_seed2 \
    >> "$LAUNCH_LOG" 2>&1

  touch "$LAUNCH_MARKER"
  echo "[$(date -u +"%F %T")] seed=2 sweep launcher completed"
} >> "$RUNTIME_LOG" 2>&1
