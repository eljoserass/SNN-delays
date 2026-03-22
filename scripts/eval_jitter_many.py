#!/usr/bin/env python3
import argparse
import os
import sys
from argparse import Namespace

import torch

# Allow importing repo-root modules when executed from scripts/.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from eval_jitter import (
    auc_over_sigma,
    build_config,
    load_model,
    load_test_loader,
    maybe_wandb_log,
    run_jitter_curve,
    set_seed,
    write_csv,
)


def parse_csv_list(text: str):
    items = [x.strip() for x in text.split(",") if x.strip()]
    if not items:
        raise ValueError("Expected at least one item")
    return items


def write_summary(path: str, baseline_ckpt: str, candidate_ckpt: str, sigmas, baseline_rows, candidate_rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"baseline_ckpt={baseline_ckpt}\n")
        f.write(f"candidate_ckpt={candidate_ckpt}\n")
        f.write(f"baseline_auc={auc_over_sigma(baseline_rows):.8f}\n")
        f.write(f"candidate_auc={auc_over_sigma(candidate_rows):.8f}\n")
        f.write(f"delta_auc_candidate_minus_baseline={auc_over_sigma(candidate_rows)-auc_over_sigma(baseline_rows):.8f}\n")
        for sigma, b_row, c_row in zip(sigmas, baseline_rows, candidate_rows):
            f.write(
                f"sigma={sigma:.4f} baseline={b_row[1]:.6f}+-{b_row[2]:.6f} "
                f"candidate={c_row[1]:.6f}+-{c_row[2]:.6f}\n"
            )


def main():
    p = argparse.ArgumentParser(description="Evaluate baseline once, compare many candidates.")
    p.add_argument("--baseline_ckpt", required=True)
    p.add_argument("--candidate_ckpts", required=True, help="Comma-separated list of candidate checkpoints.")
    p.add_argument("--candidate_tags", default="", help="Comma-separated tags; defaults to candidate_0..N")
    p.add_argument("--dataset", default="ssc", choices=["ssc", "shd"])
    p.add_argument("--datasets_path", default="Datasets")
    p.add_argument("--sigmas", default="0,0.05,0.1,0.25,0.5")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--n_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_root", default="analysis/jitter_many")

    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="Wandb Project Name")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_group", default="jitter_eval")
    p.add_argument("--wandb_job_type", default="eval")
    p.add_argument("--run_name_prefix", default="jitter_many")

    args = p.parse_args()

    candidates = parse_csv_list(args.candidate_ckpts)
    tags = parse_csv_list(args.candidate_tags) if args.candidate_tags else [f"candidate_{i}" for i in range(len(candidates))]
    if len(tags) != len(candidates):
        raise ValueError("candidate_tags count must match candidate_ckpts count")

    sigmas = [float(x.strip()) for x in args.sigmas.split(",") if x.strip()]
    if not sigmas:
        raise ValueError("No sigma values parsed")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_ns = Namespace(dataset=args.dataset, datasets_path=args.datasets_path)
    cfg = build_config(cfg_ns)
    test_loader = load_test_loader(cfg)

    # Baseline computed ONCE.
    baseline_model = load_model(args.baseline_ckpt, cfg, device)
    baseline_rows = run_jitter_curve(
        baseline_model, test_loader, cfg, device, sigmas, args.repeats, args.n_batches
    )

    os.makedirs(args.output_root, exist_ok=True)

    for tag, candidate_ckpt in zip(tags, candidates):
        outdir = os.path.join(args.output_root, tag)
        os.makedirs(outdir, exist_ok=True)

        candidate_model = load_model(candidate_ckpt, cfg, device)
        candidate_rows = run_jitter_curve(
            candidate_model, test_loader, cfg, device, sigmas, args.repeats, args.n_batches
        )

        csv_path = os.path.join(outdir, "jitter_results.csv")
        write_csv(csv_path, baseline_rows, candidate_rows)

        summary_path = os.path.join(outdir, "summary.txt")
        write_summary(summary_path, args.baseline_ckpt, candidate_ckpt, sigmas, baseline_rows, candidate_rows)

        if args.use_wandb:
            wb_args = Namespace(
                use_wandb=True,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                wandb_group=args.wandb_group,
                wandb_job_type=args.wandb_job_type,
                run_name=f"{args.run_name_prefix}_{tag}",
                dataset=args.dataset,
                repeats=args.repeats,
                n_batches=args.n_batches,
            )
            maybe_wandb_log(wb_args, baseline_rows, candidate_rows, args.baseline_ckpt, candidate_ckpt)

        print(f"Saved {csv_path}")
        print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
