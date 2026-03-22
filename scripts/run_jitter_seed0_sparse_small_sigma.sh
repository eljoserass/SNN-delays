#!/usr/bin/env bash
set -euo pipefail
cd /workspace/SNN-delays

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

source .venv/bin/activate

BASELINE="Seed0Matrix||seed=0||snn_delays||ssc||10ms||bins=5||sigma_drop=0||sparsity_p=0.96_Best_ACC.pt"
CANDIDATES=(
  "DelayDropSSC||seed=0||snn_delays||ssc||10ms||bins=5||sigma_drop=0.05||sparsity_p=0.96_Best_ACC.pt"
  "DelayDropSSC||seed=0||snn_delays||ssc||10ms||bins=5||sigma_drop=0.1||sparsity_p=0.96_Best_ACC.pt"
  "DelayDropSSC||seed=0||snn_delays||ssc||10ms||bins=5||sigma_drop=0.25||sparsity_p=0.96_Best_ACC.pt"
  "DelayDropSSC||seed=0||snn_delays||ssc||10ms||bins=5||sigma_drop=0.5||sparsity_p=0.96_Best_ACC.pt"
)
TAGS=("sigma_0p05" "sigma_0p1" "sigma_0p25" "sigma_0p5")

for i in "${!CANDIDATES[@]}"; do
  cand="${CANDIDATES[$i]}"
  tag="${TAGS[$i]}"
  outdir="analysis/jitter_seed0_sparse_small_sigma/${tag}"
  mkdir -p "$outdir"

  echo "=== [$(date -u +"%F %T")] Running jitter ${tag}"
  python eval_jitter.py \
    --baseline_ckpt "$BASELINE" \
    --candidate_ckpt "$cand" \
    --dataset ssc \
    --datasets_path Datasets \
    --sigmas 0,0.05,0.1,0.25,0.5 \
    --repeats 5 \
    --seed 0 \
    --output_dir "$outdir" \
    --run_name "jitter_seed0_sparse_${tag}" \
    --use_wandb \
    --wandb_project "Wandb Project Name" \
    --wandb_group "jitter_seed0_sparse_small_sigma"

  echo "=== [$(date -u +"%F %T")] Done ${tag}"
done

echo "=== [$(date -u +"%F %T")] All jitter comparisons done"
