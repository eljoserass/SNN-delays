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

python scripts/eval_jitter_many.py \
  --baseline_ckpt "Seed0Matrix||seed=0||snn_delays||ssc||10ms||bins=5||sigma_drop=0||sparsity_p=0.96_Best_ACC.pt" \
  --candidate_ckpts "DelayDropSSC||seed=0||snn_delays||ssc||10ms||bins=5||sigma_drop=0.05||sparsity_p=0.96_Best_ACC.pt,DelayDropSSC||seed=0||snn_delays||ssc||10ms||bins=5||sigma_drop=0.1||sparsity_p=0.96_Best_ACC.pt,DelayDropSSC||seed=0||snn_delays||ssc||10ms||bins=5||sigma_drop=0.25||sparsity_p=0.96_Best_ACC.pt,DelayDropSSC||seed=0||snn_delays||ssc||10ms||bins=5||sigma_drop=0.5||sparsity_p=0.96_Best_ACC.pt" \
  --candidate_tags "sigma_0p05,sigma_0p1,sigma_0p25,sigma_0p5" \
  --dataset ssc \
  --datasets_path Datasets \
  --sigmas 0,0.05,0.1,0.25,0.5 \
  --repeats 5 \
  --seed 0 \
  --output_root analysis/jitter_seed0_sparse_small_sigma_cached \
  --use_wandb \
  --wandb_project "Wandb Project Name" \
  --wandb_group "jitter_seed0_sparse_small_sigma_cached" \
  --run_name_prefix "jitter_seed0_sparse_cached"
