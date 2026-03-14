#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs

show_usage() {
  cat <<'EOF'
Usage:
  bash scripts/server_baseline_helpers.sh <command>

Commands:
  inspect-live
    Show active parent training process, its workers, and the latest SHD log tail.

  inspect-ssc
    Print the installed SpikingJelly SSC class file and resource metadata.

  inspect-datasets
    Show the first levels of Datasets/SHD and Datasets/SSC.

  prepare-shd
    Remove the broken Datasets/SHD tree and rebuild SHD preprocessing only.

  launch-shd-fc
    Launch SHD fully-connected baselines for seeds 1 and 2, one at a time.

  launch-shd-sparse
    Launch SHD sparse baselines for seeds 1 and 2, one at a time.

  launch-ssc-sparse
    Launch SSC sparse baselines for seeds 0,1,2, at most one parent run at a time.

  launch-ssc-fc
    Launch SSC fully-connected baselines for seeds 0,1,2, at most one parent run at a time.

  launch-all-safe
    Run SHD FC, SHD sparse, SSC sparse, SSC FC sequentially, one parent run at a time.

Notes:
  - This script tracks only parent jobs that it launches; DataLoader workers do not affect scheduling.
  - All runs use sigma_drop=0 and explicit dataset roots.
EOF
}

parent_training_pids() {
  ps -eo pid=,ppid=,args= | awk '
    $3 ~ /python$/ && $4 == "main.py" {
      pid=$1
      ppid=$2
      cmd[pid]=$0
      parent[pid]=ppid
    }
    END {
      for (pid in cmd) {
        if (!(parent[pid] in cmd)) {
          print pid
        }
      }
    }
  ' | sort -n
}

print_parent_processes() {
  local pid
  while read -r pid; do
    [[ -n "${pid}" ]] || continue
    ps -p "${pid}" -o pid,ppid,args --no-headers
  done < <(parent_training_pids)
}

inspect_live() {
  echo "=== Parent training processes"
  print_parent_processes || true
  echo

  while read -r parent_pid; do
    [[ -n "${parent_pid}" ]] || continue
    echo "=== Workers for parent PID ${parent_pid}"
    ps --ppid "${parent_pid}" -o pid,ppid,args || true
    echo
  done < <(parent_training_pids)
  echo

  if [[ -f logs/shd_baseline_seed0.log ]]; then
    echo "=== Tail of logs/shd_baseline_seed0.log"
    tail -n 30 logs/shd_baseline_seed0.log
  fi
}

inspect_ssc() {
  python -c "import spikingjelly.datasets.shd as m; print('module_file:', m.__file__); print('symbols:', [x for x in dir(m) if 'Speech' in x or 'SHD' in x or 'Spiking' in x])"
  python -c "from spikingjelly.datasets.shd import SpikingSpeechCommands as S; print('dataset_name:', getattr(S, 'dataset_name', None)); print('resources:', S.resource_url_md5())"
}

inspect_datasets() {
  echo "=== Datasets/SHD"
  find Datasets/SHD -maxdepth 3 -print 2>/dev/null | sort || true
  echo
  echo "=== Datasets/SSC"
  find Datasets/SSC -maxdepth 3 -print 2>/dev/null | sort || true
}

prepare_shd() {
  rm -rf Datasets/SHD
  python -c "from config import Config; from datasets import SHD_dataloaders; cfg = Config(); cfg.dataset='shd'; cfg.datasets_path='Datasets/SHD'; cfg.seed=0; SHD_dataloaders(cfg); print('SHD ready')"
}

launch_job() {
  local dataset="$1"
  local data_root="$2"
  local sparsity="$3"
  local seed="$4"
  local tag="$5"

  nohup python main.py \
    --dataset "${dataset}" \
    --datasets_path "${data_root}" \
    --sigma_drop 0 \
    --sparsity_p "${sparsity}" \
    --seed "${seed}" \
    --run_name BaselineNoDelayDrop \
    > "logs/${tag}.log" 2>&1 &

  echo "Launched ${tag} pid=$!"
  ACTIVE_PIDS+=("$!")
}

wait_for_slot() {
  local max_parallel="$1"

  while true; do
    local total_parents
    total_parents="$(parent_training_pids | wc -l)"
    if [[ "${total_parents}" -ge "${max_parallel}" ]]; then
      sleep 20
      continue
    fi

    local still_running=()
    local pid
    for pid in "${ACTIVE_PIDS[@]:-}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        still_running+=("${pid}")
      fi
    done
    ACTIVE_PIDS=("${still_running[@]:-}")

    if [[ "${#ACTIVE_PIDS[@]}" -lt "${max_parallel}" ]]; then
      break
    fi

    sleep 20
  done
}

wait_for_all() {
  local pid
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    wait "${pid}"
  done
  ACTIVE_PIDS=()
}

launch_many() {
  local dataset="$1"
  local data_root="$2"
  local sparsity="$3"
  local tag_prefix="$4"
  shift 4
  local max_parallel="${1}"
  shift 1

  ACTIVE_PIDS=()
  local seed
  for seed in "$@"; do
    wait_for_slot "${max_parallel}"
    launch_job "${dataset}" "${data_root}" "${sparsity}" "${seed}" "${tag_prefix}_seed${seed}"
  done

  wait_for_all
}

launch_shd_fc() {
  launch_many shd Datasets/SHD 0.0 shd_fc 1 1 2
}

launch_shd_sparse() {
  launch_many shd Datasets/SHD 0.96 shd_sparse 1 1 2
}

launch_ssc_sparse() {
  launch_many ssc Datasets/SSC 0.98 ssc_sparse 1 0 1 2
}

launch_ssc_fc() {
  launch_many ssc Datasets/SSC 0.0 ssc_fc 1 0 1 2
}

launch_all_safe() {
  launch_shd_fc
  launch_shd_sparse
  launch_ssc_sparse
  launch_ssc_fc
}

case "${1:-}" in
  inspect-live)
    inspect_live
    ;;
  inspect-ssc)
    inspect_ssc
    ;;
  inspect-datasets)
    inspect_datasets
    ;;
  prepare-shd)
    prepare_shd
    ;;
  launch-shd-fc)
    launch_shd_fc
    ;;
  launch-shd-sparse)
    launch_shd_sparse
    ;;
  launch-ssc-sparse)
    launch_ssc_sparse
    ;;
  launch-ssc-fc)
    launch_ssc_fc
    ;;
  launch-all-safe)
    launch_all_safe
    ;;
  *)
    show_usage
    exit 1
    ;;
esac
