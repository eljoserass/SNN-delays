# W&B Audit (all runs)

- Project: `joserass/Wandb Project Name`
- Snapshot date: 2026-03-17
- Runs analyzed: 27 (26 finished, 1 running)

## Core findings

- No hard-broken finished runs (no missing `acc_valid`, no ultra-short finished runs).
- SHD `acc_test` is flat zero on all SHD runs by code design (`test_loader=None` for SHD), so W&B `acc_test` chart is not meaningful for SHD.
- No run logs a dedicated sparsity metric (`has_sparsity_metric=False` for all 27 runs), which explains why sparsity is not visible as a plot.
- W&B config metadata is unreliable for several keys (`dataset_name`, `sigma_drop`, `sparsity_p`) because config is logged from class attributes rather than runtime instance values.

## Performance snapshot (SHD)

- Best overall run: `cyohxi6e` with max `acc_valid = 52.85%`.
- Best sparse (name indicates `sparsity_p=0.96`): `yo2fbnze` with max `acc_valid = 31.09%`.
- Baseline no-delay run: `6i5v6k0f` with max `acc_valid = 27.49%`.
- Delay-drop sparse main runs (`sparsity_p=0.96`) are below sparse baseline in current results.

## Cleanup recommendation

- Hard delete candidates: none.
- Optional cleanup (naming inconsistency only): `rpdtvkiq`, `6i5v6k0f`.
- Exclude from aggregate dashboards until completion: `cpvo56hh` (currently running SSC).
