#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is required" >&2
  exit 1
fi

mkdir -p logs

# Ensure SHD resolves to an existing dataset root with current resolver logic.
mkdir -p Datasets/SHD
for p in download extract events_h5 duration_10; do
  if [[ -e "Datasets/${p}" && ! -e "Datasets/SHD/${p}" ]]; then
    ln -s "../${p}" "Datasets/SHD/${p}"
  fi
done

controller_log="logs/seed0_matrix_controller.log"
: > "${controller_log}"

echo "[$(date -Is)] starting seed-0 matrix" | tee -a "${controller_log}"

auto_run() {
  local dataset="$1"
  local data_root="$2"
  local sparsity="$3"
  local sigma="$4"

  local tag="seed0_${dataset}_sp${sparsity}_sig${sigma}"
  local log_file="logs/${tag}.log"

  echo "[$(date -Is)] START ${tag} root=${data_root}" | tee -a "${controller_log}"

  python3 main.py \
    --dataset "${dataset}" \
    --datasets_path "${data_root}" \
    --sparsity_p "${sparsity}" \
    --sigma_drop "${sigma}" \
    --seed 0 \
    --run_name Seed0Matrix \
    > "${log_file}" 2>&1

  echo "[$(date -Is)] DONE  ${tag}" | tee -a "${controller_log}"
}

for sigma in 0 1 2; do
  for sparsity in 0 0.96; do
    auto_run shd Datasets "${sparsity}" "${sigma}"
  done
done

for sigma in 0 1 2; do
  for sparsity in 0 0.96; do
    auto_run ssc /tmp/SSC "${sparsity}" "${sigma}"
  done
done

echo "[$(date -Is)] matrix complete" | tee -a "${controller_log}"
