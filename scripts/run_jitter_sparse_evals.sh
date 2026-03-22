#!/usr/bin/env bash
set -euo pipefail
cd /workspace/SNN-delays
source .venv/bin/activate
set -a
source .env
set +a

BASE="Seed0Matrix||seed=0||snn_delays||ssc||10ms||bins=5||sigma_drop=0||sparsity_p=0.96_Best_ACC.pt"

python eval_jitter.py \
  --baseline_ckpt "$BASE" \
  --candidate_ckpt "DelayDropSSC||seed=0||snn_delays||ssc||10ms||bins=5||sigma_drop=0.05||sparsity_p=0.96_Best_ACC.pt" \
  --dataset ssc --datasets_path Datasets \
  --sigmas "0,0.05,0.1,0.25,0.5" --repeats 5 \
  --output_dir analysis/jitter_sparse/ssc_sparse_sigma0p05_seed0 \
  --run_name jitter_sparse_sigma0p05_seed0 \
  --use_wandb --wandb_project "Wandb Project Name" \
  --wandb_group "DelayDropSSC_jitter_sparse_cpu" --wandb_job_type "eval_jitter"

python eval_jitter.py \
  --baseline_ckpt "$BASE" \
  --candidate_ckpt "DelayDropSSC||seed=0||snn_delays||ssc||10ms||bins=5||sigma_drop=0.1||sparsity_p=0.96_Best_ACC.pt" \
  --dataset ssc --datasets_path Datasets \
  --sigmas "0,0.05,0.1,0.25,0.5" --repeats 5 \
  --output_dir analysis/jitter_sparse/ssc_sparse_sigma0p1_seed0 \
  --run_name jitter_sparse_sigma0p1_seed0 \
  --use_wandb --wandb_project "Wandb Project Name" \
  --wandb_group "DelayDropSSC_jitter_sparse_cpu" --wandb_job_type "eval_jitter"
