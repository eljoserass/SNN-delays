#!/usr/bin/env python3
import argparse
import csv
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from datasets import SHD_dataloaders, SSC_dataloaders
from snn_delays import SnnDelays
from DCLS.construct.modules import Dcls1d
from spikingjelly.activation_based import functional


def parse_sigmas(text: str) -> List[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("No sigma values parsed")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate robustness to delay jitter and optionally log to W&B.")
    p.add_argument("--baseline_ckpt", required=True)
    p.add_argument("--candidate_ckpt", required=True)
    p.add_argument("--dataset", default="ssc", choices=["ssc", "shd"])
    p.add_argument("--datasets_path", default="Datasets")
    p.add_argument("--sigmas", default="0,0.05,0.1,0.25,0.5")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--n_batches", type=int, default=0, help="0 means full test loader")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="analysis/jitter")
    p.add_argument("--run_name", default="jitter_eval")

    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="Wandb Project Name")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_group", default="jitter_eval")
    p.add_argument("--wandb_job_type", default="eval")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.dataset = args.dataset
    cfg.datasets_path = args.datasets_path
    cfg.use_wandb = False
    cfg.sigma_drop = 0.0
    if args.dataset == "ssc":
        cfg.n_outputs = 35
    elif args.dataset == "shd":
        cfg.n_outputs = 20
    return cfg


def load_test_loader(cfg: Config):
    if cfg.dataset == "ssc":
        _, _, test_loader = SSC_dataloaders(cfg)
        return test_loader
    if cfg.dataset == "shd":
        _, test_loader = SHD_dataloaders(cfg)
        return test_loader
    raise ValueError(f"Unsupported dataset {cfg.dataset}")


def iter_dcls_layers(model: SnnDelays):
    for block in model.blocks:
        if isinstance(block[0][0], Dcls1d):
            yield block[0][0]


def load_model(ckpt_path: str, cfg: Config, device: torch.device) -> SnnDelays:
    model = SnnDelays(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    with torch.no_grad():
        for layer in iter_dcls_layers(model):
            layer.SIG *= 0
            layer.version = "max"
            layer.DCK.version = "max"
    if hasattr(model, "round_pos"):
        model.round_pos()
    return model


def save_positions(model: SnnDelays) -> List[torch.Tensor]:
    return [layer.P.detach().clone() for layer in iter_dcls_layers(model)]


def restore_positions(model: SnnDelays, saved: List[torch.Tensor]) -> None:
    for layer, pos in zip(iter_dcls_layers(model), saved):
        layer.P.data.copy_(pos)


def add_jitter(model: SnnDelays, sigma: float) -> None:
    for layer in iter_dcls_layers(model):
        layer.P.data.add_(torch.randn_like(layer.P) * sigma)
        layer.clamp_parameters()


@torch.no_grad()
def eval_accuracy(model: SnnDelays, loader, cfg: Config, device: torch.device, n_batches: int) -> float:
    softmax_fn = torch.nn.Softmax(dim=2)
    correct = 0
    total = 0

    for i, batch in enumerate(loader):
        if n_batches > 0 and i >= n_batches:
            break

        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        y_oh = F.one_hot(y, cfg.n_outputs).float().to(device)
        x = x.permute(1, 0, 2).float().to(device)

        out = model(x)
        summed = torch.sum(softmax_fn(out), 0)
        preds = torch.argmax(summed, 1)
        labels = torch.argmax(y_oh, 1)

        correct += (preds == labels).sum().item()
        total += x.shape[1]
        functional.reset_net(model)

    return correct / max(total, 1)


def run_jitter_curve(model: SnnDelays, loader, cfg: Config, device: torch.device,
                     sigmas: List[float], repeats: int, n_batches: int) -> List[Tuple[float, float, float]]:
    saved = save_positions(model)
    results = []

    for sigma in sigmas:
        accs = []
        reps = repeats if sigma > 0 else 1
        for _ in range(reps):
            restore_positions(model, saved)
            if sigma > 0:
                add_jitter(model, sigma)
            accs.append(eval_accuracy(model, loader, cfg, device, n_batches))

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs)) if len(accs) > 1 else 0.0
        results.append((sigma, mean_acc, std_acc))

    restore_positions(model, saved)
    return results


def auc_over_sigma(rows: List[Tuple[float, float, float]]) -> float:
    xs = np.array([r[0] for r in rows], dtype=np.float64)
    ys = np.array([r[1] for r in rows], dtype=np.float64)
    if len(xs) < 2:
        return float(ys[0])
    span = xs[-1] - xs[0]
    if abs(span) < 1e-12:
        return float(np.mean(ys))
    return float((np.trapz(ys, xs) if hasattr(np, "trapz") else np.trapezoid(ys, xs)) / span)


def write_csv(path: str, baseline_rows, candidate_rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "sigma_test", "acc_mean", "acc_std"])
        for sigma, mean_acc, std_acc in baseline_rows:
            w.writerow(["baseline", f"{sigma:.6f}", f"{mean_acc:.8f}", f"{std_acc:.8f}"])
        for sigma, mean_acc, std_acc in candidate_rows:
            w.writerow(["candidate", f"{sigma:.6f}", f"{mean_acc:.8f}", f"{std_acc:.8f}"])


def maybe_wandb_log(args: argparse.Namespace, baseline_rows, candidate_rows,
                    baseline_ckpt: str, candidate_ckpt: str) -> None:
    if not args.use_wandb:
        return

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        name=args.run_name,
        config={
            "dataset": args.dataset,
            "baseline_ckpt": baseline_ckpt,
            "candidate_ckpt": candidate_ckpt,
            "sigmas": [r[0] for r in baseline_rows],
            "repeats": args.repeats,
            "n_batches": args.n_batches,
        },
    )

    for idx in range(len(baseline_rows)):
        sigma = baseline_rows[idx][0]
        wandb.log(
            {
                "sigma_test": sigma,
                "jitter_acc_baseline": baseline_rows[idx][1],
                "jitter_std_baseline": baseline_rows[idx][2],
                "jitter_acc_candidate": candidate_rows[idx][1],
                "jitter_std_candidate": candidate_rows[idx][2],
            },
            step=idx,
        )

    b_auc = auc_over_sigma(baseline_rows)
    c_auc = auc_over_sigma(candidate_rows)
    run.summary["baseline_acc_sigma0"] = baseline_rows[0][1]
    run.summary["candidate_acc_sigma0"] = candidate_rows[0][1]
    run.summary["baseline_auc"] = b_auc
    run.summary["candidate_auc"] = c_auc
    run.summary["delta_auc_candidate_minus_baseline"] = c_auc - b_auc
    run.finish()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    sigmas = parse_sigmas(args.sigmas)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = build_config(args)
    test_loader = load_test_loader(cfg)

    baseline = load_model(args.baseline_ckpt, cfg, device)
    candidate = load_model(args.candidate_ckpt, cfg, device)

    baseline_rows = run_jitter_curve(baseline, test_loader, cfg, device, sigmas, args.repeats, args.n_batches)
    candidate_rows = run_jitter_curve(candidate, test_loader, cfg, device, sigmas, args.repeats, args.n_batches)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "jitter_results.csv")
    write_csv(csv_path, baseline_rows, candidate_rows)

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"baseline_ckpt={args.baseline_ckpt}\n")
        f.write(f"candidate_ckpt={args.candidate_ckpt}\n")
        f.write(f"baseline_auc={auc_over_sigma(baseline_rows):.8f}\n")
        f.write(f"candidate_auc={auc_over_sigma(candidate_rows):.8f}\n")
        for sigma, b_row, c_row in zip(sigmas, baseline_rows, candidate_rows):
            f.write(
                f"sigma={sigma:.4f} baseline={b_row[1]:.6f}+-{b_row[2]:.6f} "
                f"candidate={c_row[1]:.6f}+-{c_row[2]:.6f}\n"
            )

    maybe_wandb_log(args, baseline_rows, candidate_rows, args.baseline_ckpt, args.candidate_ckpt)

    print(f"Saved {csv_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
