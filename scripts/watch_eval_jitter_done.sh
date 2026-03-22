#!/usr/bin/env bash
set -euo pipefail

STATUS_DIR=${STATUS_DIR:-".run_status/delaydrop_ssc_small_sigma_retryfix"}
EVAL_STATUS_DIR=${EVAL_STATUS_DIR:-".run_status/jitter_eval_dense"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"analysis/jitter_dense"}
BASELINE_CKPT=${BASELINE_CKPT:-"Seed0Matrix||seed=0||snn_delays||ssc||10ms||bins=5||sigma_drop=0||sparsity_p=0_Best_ACC.pt"}
RUN_PREFIX=${RUN_PREFIX:-"DelayDropSSC"}
POLL_SEC=${POLL_SEC:-60}
SIGMAS=${SIGMAS:-"0,0.05,0.1,0.25,0.5"}
REPEATS=${REPEATS:-5}
WANDB_PROJECT=${WANDB_PROJECT:-"Wandb Project Name"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
WANDB_GROUP=${WANDB_GROUP:-"DelayDropSSC_jitter_dense"}

mkdir -p "$EVAL_STATUS_DIR" "$OUTPUT_ROOT"

if [[ ! -f "$BASELINE_CKPT" ]]; then
  echo "[watch_eval_jitter_done] baseline checkpoint not found: $BASELINE_CKPT" >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "[watch_eval_jitter_done] .venv/bin/python not found" >&2
  exit 1
fi
if [[ ! -f eval_jitter.py ]]; then
  echo "[watch_eval_jitter_done] eval_jitter.py not found" >&2
  exit 1
fi

tag_to_float() {
  local tag="$1"
  python3 - <<PY
v = "${tag}".replace("p", ".")
print(f"{float(v):g}")
PY
}

eval_one() {
  local done_path="$1"
  local done_file
  done_file="$(basename "$done_path")"

  if [[ ! "$done_file" =~ ^([^_]+)__sigma_([^_]+)__sparsity_([^_]+)__seed_([^\.]+)\.done$ ]]; then
    echo "[watch_eval_jitter_done] skip unknown marker format: $done_file"
    return 0
  fi

  local dataset_tag="${BASH_REMATCH[1]}"
  local sigma_tag="${BASH_REMATCH[2]}"
  local seed_tag="${BASH_REMATCH[4]}"
  local run_id="${done_file%.done}"

  local marker_done="$EVAL_STATUS_DIR/${run_id}.jitter.done"
  local marker_failed="$EVAL_STATUS_DIR/${run_id}.jitter.failed"
  local jitter_log="$EVAL_STATUS_DIR/${run_id}.jitter.log"

  if [[ -f "$marker_done" ]]; then
    return 0
  fi

  local sigma_val
  sigma_val="$(tag_to_float "$sigma_tag")"

  shopt -s nullglob
  local pattern="${RUN_PREFIX}||seed=${seed_tag}||snn_delays||${dataset_tag}||10ms||bins=5||sigma_drop=${sigma_val}||sparsity_p=*_Best_ACC.pt"
  local candidates=( $pattern )
  shopt -u nullglob

  if (( ${#candidates[@]} == 0 )); then
    echo "[watch_eval_jitter_done] no candidate ckpt for $run_id (sigma=${sigma_val})" | tee -a "$jitter_log"
    return 0
  fi

  local candidate_ckpt="${candidates[0]}"
  local out_dir="$OUTPUT_ROOT/$run_id"
  mkdir -p "$out_dir"

  echo "[watch_eval_jitter_done] evaluating $run_id vs baseline" | tee -a "$jitter_log"

  local -a wandb_args=(--use_wandb --wandb_project "$WANDB_PROJECT" --wandb_group "$WANDB_GROUP" --wandb_job_type "eval_jitter")
  if [[ -n "$WANDB_ENTITY" ]]; then
    wandb_args+=(--wandb_entity "$WANDB_ENTITY")
  fi

  set +e
  CUDA_VISIBLE_DEVICES="" .venv/bin/python eval_jitter.py \
    --baseline_ckpt "$BASELINE_CKPT" \
    --candidate_ckpt "$candidate_ckpt" \
    --dataset "$dataset_tag" \
    --datasets_path "Datasets" \
    --sigmas "$SIGMAS" \
    --repeats "$REPEATS" \
    --output_dir "$out_dir" \
    --run_name "jitter_${run_id}" \
    "${wandb_args[@]}" \
    >> "$jitter_log" 2>&1
  rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    touch "$marker_done"
    rm -f "$marker_failed"
    echo "[watch_eval_jitter_done] done $run_id" | tee -a "$jitter_log"
  else
    echo "$rc" > "$marker_failed"
    echo "[watch_eval_jitter_done] failed $run_id rc=$rc" | tee -a "$jitter_log"
  fi
}

echo "[watch_eval_jitter_done] watching $STATUS_DIR (poll=${POLL_SEC}s)"
while true; do
  shopt -s nullglob
  done_files=("$STATUS_DIR"/*.done)
  shopt -u nullglob
  for done_path in "${done_files[@]:-}"; do
    [[ -n "${done_path:-}" ]] || continue
    eval_one "$done_path"
  done
  sleep "$POLL_SEC"
done
