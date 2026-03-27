#!/usr/bin/env python3
import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

SRC = "analysis/full_sweep_summary"
DST = "analysis/latest"
RERUN_SUMMARY = "analysis/jitter_seed12_sparse_small_sigma/seed1_sigma0p25_rerun/summary.txt"
RERUN_JITTER = "analysis/jitter_seed12_sparse_small_sigma/seed1_sigma0p25_rerun/jitter_results.csv"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def parse_rerun_summary(path):
    out = {"baseline_auc": None, "candidate_auc": None}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("baseline_auc="):
                out["baseline_auc"] = float(line.split("=", 1)[1])
            elif line.startswith("candidate_auc="):
                out["candidate_auc"] = float(line.split("=", 1)[1])
    return out


def parse_rerun_candidate(path):
    points = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["model"] != "candidate":
                continue
            sigma = round(float(row["sigma_test"]), 2)
            points[sigma] = (float(row["acc_mean"]), float(row["acc_std"]))
    return points


def build_latest_csvs():
    os.makedirs(DST, exist_ok=True)

    rerun_summary = parse_rerun_summary(RERUN_SUMMARY)
    rerun_cand = parse_rerun_candidate(RERUN_JITTER)

    models = read_csv(os.path.join(SRC, "jitter_models_summary.csv"))
    for row in models:
        if row["seed"] == "1" and row["sigma_train"] == "0.25":
            clean = rerun_cand[0.00][0]
            row["clean_acc"] = f"{clean:.6f}"
            row["candidate_auc"] = f"{rerun_summary['candidate_auc']:.8f}"
            row["baseline_auc"] = f"{rerun_summary['baseline_auc']:.8f}"
            row["delta_auc"] = f"{(rerun_summary['candidate_auc'] - rerun_summary['baseline_auc']):.8f}"
            row["source"] = RERUN_SUMMARY
    write_csv(
        os.path.join(DST, "jitter_models_summary.csv"),
        models,
        ["seed", "sigma_train", "clean_acc", "candidate_auc", "baseline_auc", "delta_auc", "source"],
    )

    curves = read_csv(os.path.join(SRC, "jitter_curves_long.csv"))
    for row in curves:
        if row["seed"] == "1" and row["sigma_train"] == "0.25":
            s = round(float(row["sigma_test"]), 2)
            if s in rerun_cand:
                row["acc_mean"] = f"{rerun_cand[s][0]:.6f}"
                row["acc_std"] = f"{rerun_cand[s][1]:.6f}"
    write_csv(
        os.path.join(DST, "jitter_curves_long.csv"),
        curves,
        ["seed", "sigma_train", "sigma_test", "acc_mean", "acc_std"],
    )

    # readable legend CSV (same schema as previous)
    readable_rows = []
    for row in sorted(
        curves,
        key=lambda r: (int(float(r["seed"])), float(r["sigma_train"]), float(r["sigma_test"])),
    ):
        sigma_train = float(row["sigma_train"])
        acc_mean = float(row["acc_mean"])
        acc_std = float(row["acc_std"])
        readable_rows.append(
            {
                "seed": int(float(row["seed"])),
                "sigma_train": sigma_train,
                "train_label": f"train_sigma_{sigma_train:g}",
                "sigma_test": float(row["sigma_test"]),
                "acc_mean": acc_mean,
                "acc_std": acc_std,
                "acc_mean_pct": acc_mean * 100.0,
                "acc_std_pct": acc_std * 100.0,
            }
        )
    write_csv(
        os.path.join(DST, "jitter_scatter_by_seed_readable_legend.csv"),
        readable_rows,
        [
            "seed",
            "sigma_train",
            "train_label",
            "sigma_test",
            "acc_mean",
            "acc_std",
            "acc_mean_pct",
            "acc_std_pct",
        ],
    )

    # relative-to-own-seed CSV
    parsed = []
    for row in models:
        parsed.append(
            {
                "seed": int(row["seed"]),
                "sigma_train": float(row["sigma_train"]),
                "clean_acc": float(row["clean_acc"]),
                "auc": float(row["candidate_auc"]),
            }
        )
    seed_base = {}
    for row in parsed:
        if abs(row["sigma_train"]) < 1e-12:
            seed_base[row["seed"]] = (row["clean_acc"], row["auc"])

    rel = []
    for row in sorted(parsed, key=lambda r: (r["seed"], r["sigma_train"])):
        bclean, bauc = seed_base[row["seed"]]
        rel.append(
            {
                "seed": row["seed"],
                "sigma_train": f"{row['sigma_train']:.2f}",
                "clean_acc": f"{row['clean_acc']:.6f}",
                "auc": f"{row['auc']:.8f}",
                "delta_clean_pp_vs_seed_sigma0": f"{(row['clean_acc'] - bclean) * 100.0:.6f}",
                "delta_auc_vs_seed_sigma0": f"{(row['auc'] - bauc):.8f}",
            }
        )
    write_csv(
        os.path.join(DST, "jitter_models_relative_to_own_seed.csv"),
        rel,
        [
            "seed",
            "sigma_train",
            "clean_acc",
            "auc",
            "delta_clean_pp_vs_seed_sigma0",
            "delta_auc_vs_seed_sigma0",
        ],
    )

    # training CSV unchanged (copied for latest package completeness)
    training = read_csv(os.path.join(SRC, "training_best_sparse_ssc.csv"))
    write_csv(
        os.path.join(DST, "training_best_sparse_ssc.csv"),
        training,
        ["seed", "sigma_train", "best_acc_test_pct", "last_acc_test_pct", "source"],
    )


def load_curve_map(curves_rows):
    data = defaultdict(lambda: defaultdict(list))
    for row in curves_rows:
        seed = int(float(row["seed"]))
        sigma_train = float(row["sigma_train"])
        sigma_test = float(row["sigma_test"])
        acc_mean = float(row["acc_mean"]) * 100.0
        acc_std = float(row["acc_std"]) * 100.0
        data[seed][sigma_train].append((sigma_test, acc_mean, acc_std))
    return data


def plot_jitter_scatter(curves_rows, readable=False):
    data = load_curve_map(curves_rows)
    seeds = sorted(data.keys())
    sigmas_train = sorted({s for d in data.values() for s in d.keys()})
    colors = plt.cm.viridis(
        [i / (len(sigmas_train) - 1) if len(sigmas_train) > 1 else 0.5 for i in range(len(sigmas_train))]
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, seed in zip(axes, seeds):
        for color, s_train in zip(colors, sigmas_train):
            rows = sorted(data[seed][s_train], key=lambda x: x[0])
            xs = [x for x, _, _ in rows]
            ys = [y for _, y, _ in rows]
            yerr = [e for _, _, e in rows]
            ax.plot(xs, ys, marker="o", markersize=5, linewidth=2.0, color=color, label=f"train σ={s_train:g}")
            ax.scatter(xs, ys, s=30, color=color)
            ax.fill_between(xs, [a - b for a, b in zip(ys, yerr)], [a + b for a, b in zip(ys, yerr)], color=color, alpha=0.10)
        ax.set_title(f"Seed {seed}")
        ax.set_xlabel("Jitter sigma_test")
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0.0, 0.05, 0.1, 0.25, 0.5])

    axes[0].set_ylabel("Accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    if readable:
        legend = fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, fontsize=11, bbox_to_anchor=(0.5, -0.02))
        legend.get_frame().set_alpha(0.95)
        fig.tight_layout(rect=[0, 0.12, 1, 0.92])
        base = os.path.join(DST, "jitter_scatter_by_seed_readable_legend")
    else:
        fig.legend(handles, labels, loc="upper center", ncol=len(sigmas_train), frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        base = os.path.join(DST, "jitter_scatter_by_seed")

    fig.suptitle("SSC sparse (sparsity=0.96): Jitter robustness by seed and train sigma", y=0.98)
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)

    # compatibility with previous naming
    if not readable:
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        for ax, seed in zip(axes2, seeds):
            for color, s_train in zip(colors, sigmas_train):
                rows = sorted(data[seed][s_train], key=lambda x: x[0])
                xs = [x for x, _, _ in rows]
                ys = [y for _, y, _ in rows]
                yerr = [e for _, _, e in rows]
                ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.5, color=color, label=f"train σ={s_train:g}")
                ax.fill_between(xs, [a - b for a, b in zip(ys, yerr)], [a + b for a, b in zip(ys, yerr)], color=color, alpha=0.15)
            ax.set_title(f"Seed {seed}")
            ax.set_xlabel("sigma_test")
            ax.grid(True, alpha=0.3)
        axes2[0].set_ylabel("Accuracy (%)")
        handles2, labels2 = axes2[0].get_legend_handles_labels()
        fig2.legend(handles2, labels2, loc="upper center", ncol=len(sigmas_train), frameon=False)
        fig2.tight_layout(rect=[0, 0, 1, 0.92])
        fig2.suptitle("Jitter Curves By Seed", y=0.98)
        fig2.savefig(os.path.join(DST, "jitter_curves_by_seed.png"), dpi=180, bbox_inches="tight")
        fig2.savefig(os.path.join(DST, "jitter_curves_by_seed.pdf"), bbox_inches="tight")
        plt.close(fig2)


def plot_seed_relative(rel_rows):
    by_seed = defaultdict(list)
    for row in rel_rows:
        seed = int(row["seed"])
        by_seed[seed].append(
            (
                float(row["sigma_train"]),
                float(row["delta_clean_pp_vs_seed_sigma0"]),
                float(row["delta_auc_vs_seed_sigma0"]),
            )
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for seed in sorted(by_seed):
        rows = sorted(by_seed[seed], key=lambda x: x[0])
        xs = [r[0] for r in rows]
        ys1 = [r[1] for r in rows]
        ys2 = [r[2] for r in rows]
        axes[0].plot(xs, ys1, marker="o", label=f"seed {seed}")
        axes[1].plot(xs, ys2, marker="o", label=f"seed {seed}")

    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[0].set_title("Δ Clean Acc (pp) vs seed sigma=0")
    axes[1].set_title("Δ AUC vs seed sigma=0")
    axes[0].set_xlabel("train sigma")
    axes[1].set_xlabel("train sigma")
    axes[0].set_ylabel("percentage points")
    axes[1].set_ylabel("AUC delta")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(DST, "relative_to_seed_baseline.png"), dpi=180, bbox_inches="tight")
    fig.savefig(os.path.join(DST, "relative_to_seed_baseline.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_training_auc(models_rows, training_rows):
    # map training best acc pct to decimal
    train_map = {}
    for row in training_rows:
        train_map[(int(float(row["seed"])), float(row["sigma_train"]))] = float(row["best_acc_test_pct"]) / 100.0

    seeds = sorted({int(float(r["seed"])) for r in models_rows})
    sigmas = sorted({float(r["sigma_train"]) for r in models_rows})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for seed in seeds:
        xs = sigmas
        ys_train = [train_map.get((seed, s), math.nan) * 100.0 for s in xs]
        ys_auc = [next(float(r["candidate_auc"]) * 100.0 for r in models_rows if int(float(r["seed"])) == seed and float(r["sigma_train"]) == s) for s in xs]
        axes[0].plot(xs, ys_train, marker="o", label=f"seed {seed}")
        axes[1].plot(xs, ys_auc, marker="o", label=f"seed {seed}")

    axes[0].set_title("Best Test Accuracy by Train Sigma")
    axes[1].set_title("Jitter AUC by Train Sigma")
    for ax in axes:
        ax.set_xlabel("train sigma")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Accuracy (%)")
    axes[1].set_ylabel("AUC (%)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(DST, "training_and_auc_by_seed.png"), dpi=180, bbox_inches="tight")
    fig.savefig(os.path.join(DST, "training_and_auc_by_seed.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(models_rows):
    seeds = sorted({int(float(r["seed"])) for r in models_rows})
    sigmas = sorted({float(r["sigma_train"]) for r in models_rows})

    clean = np.zeros((len(seeds), len(sigmas)))
    auc = np.zeros((len(seeds), len(sigmas)))
    for i, seed in enumerate(seeds):
        for j, sigma in enumerate(sigmas):
            row = next(r for r in models_rows if int(float(r["seed"])) == seed and float(r["sigma_train"]) == sigma)
            clean[i, j] = float(row["clean_acc"]) * 100.0
            auc[i, j] = float(row["candidate_auc"]) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    im0 = axes[0].imshow(clean, aspect="auto", cmap="viridis")
    im1 = axes[1].imshow(auc, aspect="auto", cmap="viridis")
    axes[0].set_title("Clean Accuracy (%)")
    axes[1].set_title("Jitter AUC (%)")
    for ax in axes:
        ax.set_xticks(range(len(sigmas)))
        ax.set_xticklabels([f"{s:g}" for s in sigmas])
        ax.set_yticks(range(len(seeds)))
        ax.set_yticklabels([str(s) for s in seeds])
        ax.set_xlabel("train sigma")
    axes[0].set_ylabel("seed")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(DST, "heatmaps_clean_auc.png"), dpi=180, bbox_inches="tight")
    fig.savefig(os.path.join(DST, "heatmaps_clean_auc.pdf"), bbox_inches="tight")
    plt.close(fig)


def write_readme(models_rows):
    # highlight corrected point
    row = next(r for r in models_rows if r["seed"] == "1" and r["sigma_train"] == "0.25")
    with open(os.path.join(DST, "README.md"), "w", encoding="utf-8") as f:
        f.write("# Latest Results Snapshot\n\n")
        f.write("This folder is the latest consolidated export with the corrected jitter rerun for `seed=1`, `sigma_train=0.25`.\n\n")
        f.write("## Key correction\n")
        f.write("- Source used: `analysis/jitter_seed12_sparse_small_sigma/seed1_sigma0p25_rerun/summary.txt`\n")
        f.write(f"- Corrected clean acc: `{row['clean_acc']}`\n")
        f.write(f"- Corrected candidate AUC: `{row['candidate_auc']}`\n")
        f.write(f"- Baseline AUC (matched repeats=5): `{row['baseline_auc']}`\n\n")
        f.write("## Included outputs\n")
        f.write("- `jitter_models_summary.csv`\n")
        f.write("- `jitter_curves_long.csv`\n")
        f.write("- `jitter_models_relative_to_own_seed.csv`\n")
        f.write("- `training_best_sparse_ssc.csv`\n")
        f.write("- `jitter_scatter_by_seed*` and `jitter_curves_by_seed*` plots\n")
        f.write("- `training_and_auc_by_seed*`, `heatmaps_clean_auc*`, `relative_to_seed_baseline*` plots\n")


def main():
    build_latest_csvs()

    models_rows = read_csv(os.path.join(DST, "jitter_models_summary.csv"))
    curves_rows = read_csv(os.path.join(DST, "jitter_curves_long.csv"))
    rel_rows = read_csv(os.path.join(DST, "jitter_models_relative_to_own_seed.csv"))
    training_rows = read_csv(os.path.join(DST, "training_best_sparse_ssc.csv"))

    plot_jitter_scatter(curves_rows, readable=False)
    plot_jitter_scatter(curves_rows, readable=True)
    plot_seed_relative(rel_rows)
    plot_training_auc(models_rows, training_rows)
    plot_heatmaps(models_rows)
    write_readme(models_rows)


if __name__ == "__main__":
    main()
