#!/usr/bin/env bash
set -euo pipefail

SEEDS=(0 1 2)
SIGMAS=(0.5 1.0 2.0 5.0)
DATASETS=(shd ssc)
RUN_NAME_PREFIX=${RUN_NAME_PREFIX:-"MainResultDelayDrop"}
STATUS_DIR=${STATUS_DIR:-".run_status/main_result"}
SPARSITY_P=${SPARSITY_P:-""}
FORCE=0
RETRY_FAILED=1

declare -A SPARSITY_BY_DATASET=(
  [shd]="0.96"
  [ssc]="0.98"
  [gsc]="0.0"
)

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --datasets shd,ssc      Comma-separated datasets to run (default: shd,ssc)
  --dataset shd           Single dataset shortcut
  --sigmas 0.5,1.0,2.0    Comma-separated sigma_drop values
  --seeds 0,1,2           Comma-separated seeds
  --run-name-prefix NAME  Run name prefix (default: ${RUN_NAME_PREFIX})
  --status-dir DIR        Status/log directory (default: ${STATUS_DIR})
  --sparsity-p VALUE      Global sparsity mask probability p in [0,1] (default: dataset-specific)
  --force                 Re-run completed runs
  --no-retry-failed       Skip runs previously marked as failed
  --help                  Show this help
EOF
}

csv_to_array() {
  local csv="$1"
  local -n out="$2"
  IFS=',' read -r -a out <<< "${csv}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets)
      csv_to_array "$2" DATASETS
      shift 2
      ;;
    --dataset)
      DATASETS=("$2")
      shift 2
      ;;
    --sigmas)
      csv_to_array "$2" SIGMAS
      shift 2
      ;;
    --seeds)
      csv_to_array "$2" SEEDS
      shift 2
      ;;
    --run-name-prefix)
      RUN_NAME_PREFIX="$2"
      shift 2
      ;;
    --status-dir)
      STATUS_DIR="$2"
      shift 2
      ;;
    --sparsity-p)
      SPARSITY_P="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --no-retry-failed)
      RETRY_FAILED=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

mkdir -p "${STATUS_DIR}"

failed_count=0
skipped_count=0
completed_count=0
launched_count=0

for dataset in "${DATASETS[@]}"; do
  dataset_sparsity="${SPARSITY_P}"
  if [[ -z "${dataset_sparsity}" ]]; then
    dataset_sparsity="${SPARSITY_BY_DATASET[${dataset}]:-0.0}"
  fi
  for sigma in "${SIGMAS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      sigma_tag="${sigma//./p}"
      sparsity_tag="${dataset_sparsity//./p}"
      run_id="${dataset}__sigma_${sigma_tag}__sparsity_${sparsity_tag}__seed_${seed}"
      done_file="${STATUS_DIR}/${run_id}.done"
      fail_file="${STATUS_DIR}/${run_id}.failed"
      lock_file="${STATUS_DIR}/${run_id}.lock"
      log_file="${STATUS_DIR}/${run_id}.log"

      if [[ "${FORCE}" -eq 0 && -f "${done_file}" ]]; then
        echo "=== Skipping completed ${run_id}"
        skipped_count=$((skipped_count + 1))
        continue
      fi

      if [[ -f "${lock_file}" ]]; then
        lock_pid="$(cat "${lock_file}" 2>/dev/null || true)"
        if [[ -n "${lock_pid}" ]] && kill -0 "${lock_pid}" 2>/dev/null; then
          echo "=== Skipping in-progress ${run_id} (pid=${lock_pid})"
          skipped_count=$((skipped_count + 1))
          continue
        fi
        rm -f "${lock_file}"
      fi

      if [[ "${FORCE}" -eq 0 && "${RETRY_FAILED}" -eq 0 && -f "${fail_file}" ]]; then
        echo "=== Skipping previously failed ${run_id} (--no-retry-failed)"
        skipped_count=$((skipped_count + 1))
        continue
      fi

      echo "$$" > "${lock_file}"
      echo "=== Running ${run_id}"
      launched_count=$((launched_count + 1))

      set +e
      python main.py \
        --dataset "${dataset}" \
        --sigma_drop "${sigma}" \
        --sparsity_p "${dataset_sparsity}" \
        --seed "${seed}" \
        --run_name "${RUN_NAME_PREFIX}" \
        2>&1 | tee "${log_file}"
      cmd_rc=${PIPESTATUS[0]}
      set -e

      rm -f "${lock_file}"
      if [[ "${cmd_rc}" -eq 0 ]]; then
        touch "${done_file}"
        rm -f "${fail_file}"
        completed_count=$((completed_count + 1))
      else
        echo "exit_code=${cmd_rc}" > "${fail_file}"
        failed_count=$((failed_count + 1))
      fi
    done
  done
done

echo "=== Sweep summary: launched=${launched_count} completed=${completed_count} failed=${failed_count} skipped=${skipped_count}"

if [[ "${failed_count}" -gt 0 ]]; then
  exit 1
fi
