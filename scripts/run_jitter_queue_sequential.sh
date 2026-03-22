#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-"/workspace/SNN-delays"}
DENSE_STATUS=${DENSE_STATUS:-".run_status/delaydrop_ssc_small_sigma_retryfix"}
SPARSE_STATUS=${SPARSE_STATUS:-".run_status/delaydrop_ssc_small_sigma_sparse"}
STATE_DIR=${STATE_DIR:-".run_status/jitter_eval_seq"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"analysis/jitter_seq"}
RUN_PREFIX=${RUN_PREFIX:-"DelayDropSSC"}
POLL_SEC=${POLL_SEC:-120}
SIGMAS=${SIGMAS:-"0,0.05,0.1,0.25,0.5"}
REPEATS=${REPEATS:-5}
WANDB_PROJECT=${WANDB_PROJECT:-"Wandb Project Name"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
WANDB_GROUP=${WANDB_GROUP:-"DelayDropSSC_jitter_seq"}

BASELINE_DENSE=${BASELINE_DENSE:-"Seed0Matrix||seed=0||snn_delays||ssc||10ms||bins=5||sigma_drop=0||sparsity_p=0_Best_ACC.pt"}
BASELINE_SPARSE=${BASELINE_SPARSE:-"Seed0Matrix||seed=0||snn_delays||ssc||10ms||bins=5||sigma_drop=0||sparsity_p=0.96_Best_ACC.pt"}

cd "$ROOT"
mkdir -p "$STATE_DIR" "$OUTPUT_ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "[jitter-seq] missing .venv/bin/python" >&2
  exit 1
fi
if [[ ! -f eval_jitter.py ]]; then
  echo "[jitter-seq] missing eval_jitter.py" >&2
  exit 1
fi

_tag_to_float() {
  python3 - "$1" <<PY
import sys
print(f"{float(sys.argv[1].replace('p','.')):g}")
PY
}

process_done() {
  local done_path="$1"
  local done_file run_id dataset sigma_tag sparsity_tag seed_tag
  done_file="$(basename "$done_path")"

  if [[ ! "$done_file" =~ ^([^_]+)__sigma_([^_]+)__sparsity_([^_]+)__seed_([^\.]+)\.done$ ]]; then
    return 0
  fi

  dataset="${BASH_REMATCH[1]}"
  sigma_tag="${BASH_REMATCH[2]}"
  sparsity_tag="${BASH_REMATCH[3]}"
  seed_tag="${BASH_REMATCH[4]}"
  run_id="${done_file%.done}"

  local marker_done="$STATE_DIR/${run_id}.jitter.done"
  local marker_failed="$STATE_DIR/${run_id}.jitter.failed"
  local jitter_log="$STATE_DIR/${run_id}.jitter.log"

  [[ -f "$marker_done" ]] && return 0

  local sigma_val sparsity_val baseline_ckpt candidate_ckpt
  sigma_val="$(_tag_to_float "$sigma_tag")"
  sparsity_val="$(_tag_to_float "$sparsity_tag")"

  if [[ "$sparsity_tag" == "0p96" ]]; then
    baseline_ckpt="$BASELINE_SPARSE"
  else
    baseline_ckpt="$BASELINE_DENSE"
  fi

  candidate_ckpt="${RUN_PREFIX}||seed=${seed_tag}||snn_delays||${dataset}||10ms||bins=5||sigma_drop=${sigma_val}||sparsity_p=${sparsity_val}_Best_ACC.pt"

  if [[ ! -f "$baseline_ckpt" ]]; then
    echo "[jitter-seq] baseline missing for ${run_id}: ${baseline_ckpt}" | tee -a "$jitter_log"
    echo "missing_baseline" > "$marker_failed"
    return 0
  fi
  if [[ ! -f "$candidate_ckpt" ]]; then
    echo "[jitter-seq] candidate missing for ${run_id}: ${candidate_ckpt}" | tee -a "$jitter_log"
    echo "missing_candidate" > "$marker_failed"
    return 0
  fi

  local out_dir="$OUTPUT_ROOT/$run_id"
  mkdir -p "$out_dir"

  echo "[jitter-seq] start ${run_id}" | tee -a "$jitter_log"

  local -a cmd=(
    .venv/bin/python eval_jitter.py
    --baseline_ckpt "$baseline_ckpt"
    --candidate_ckpt "$candidate_ckpt"
    --dataset "$dataset"
    --datasets_path "Datasets"
    --sigmas "$SIGMAS"
    --repeats "$REPEATS"
    --output_dir "$out_dir"
    --run_name "jitter_${run_id}"
    --use_wandb
    --wandb_project "$WANDB_PROJECT"
    --wandb_group "$WANDB_GROUP"
    --wandb_job_type "eval_jitter"
  )
  if [[ -n "$WANDB_ENTITY" ]]; then
    cmd+=(--wandb_entity "$WANDB_ENTITY")
  fi

  set +e
  (
    export CUDA_VISIBLE_DEVICES=""
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    "${cmd[@]}"
  ) >> "$jitter_log" 2>&1
  local rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    touch "$marker_done"
    rm -f "$marker_failed"
    echo "[jitter-seq] done ${run_id}" | tee -a "$jitter_log"
  else
    echo "$rc" > "$marker_failed"
    echo "[jitter-seq] failed ${run_id} rc=${rc}" | tee -a "$jitter_log"
  fi
}

echo "[jitter-seq] watching ${DENSE_STATUS} and ${SPARSE_STATUS}; poll=${POLL_SEC}s"
while true; do
  shopt -s nullglob
  done_files=("$DENSE_STATUS"/*.done "$SPARSE_STATUS"/*.done)
  shopt -u nullglob
  for done_path in "${done_files[@]:-}"; do
    [[ -n "${done_path:-}" ]] || continue
    process_done "$done_path"
  done
  sleep "$POLL_SEC"
done
