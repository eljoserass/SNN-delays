# Delay Dropout Experiment Status (SSC/SHD)

Last updated: 2026-03-22 19:50 UTC

## 1) What we are testing
We are testing **delay dropout** in `snn_delays` by adding Gaussian noise to learned delay positions `P` during training:

- Implementation: `delay_dropout.py::forward_with_delay_dropout(...)`
- Wired in model forward: `snn_delays.py` (hidden + output DCLS layers)
- Hyperparameter: `sigma_drop` (`config.py`, `main.py --sigma_drop`)
- Extra logging: per-layer delay distribution stats are logged to W&B in `model.py` (`all_layer_delay_stats`)

Interpretation: `sigma_drop=0` is baseline (no delay noise), `sigma_drop>0` adds training-time delay jitter.

## 2) Why we focused on SSC (not SHD)
We observed SHD behavior consistent with a preprocessing/version issue (SpikingJelly + cached extracted frames):

- SHD runs were often near chance/flat and delays appeared ineffective.
- We previously saw evidence of SHD collapsing to very small temporal depth in some environments (version-dependent framing behavior).
- Because this blocks meaningful delay-learning conclusions, we used **SSC** for replication and comparisons.

Practical note: SHD preprocessing can be sensitive to SpikingJelly version and cached `extract`/`duration_*` data.

## 3) Experiment progression

### Phase A: initial larger sigma tests
We initially tested larger values (`sigma_drop` around 1–2; earlier drafts also considered larger values). Outcome: generally degraded performance, especially in sparse setups.

### Phase B: small-sigma sweep (current main line)
We moved to smaller values aligned with effective delay-kernel scales:

- `sigma_drop = {0.05, 0.1, 0.25, 0.5}`
- Main benchmark: **SSC sparse (`sparsity_p=0.96`), seed=0**

Key seed-0 sparse SSC best test accuracies:

- baseline (`sigma=0`): ~**37.03%** (jitter eval @ `sigma_test=0` gives 37.08%)
- `sigma=0.05`: **39.06%**
- `sigma=0.1`: **39.57%** (best)
- `sigma=0.25`: **39.38%**
- `sigma=0.5`: **38.23%**

So the best improvement over sparse baseline is about **+2.54 pp** (`39.57 - 37.03`).

Dense SSC (earlier runs) showed much smaller gains; sparse is where improvements are clearer.

## 4) Jitter evaluation: what it does
`eval_jitter.py` evaluates robustness by perturbing learned delays at test time:

- For each `sigma_test` in a grid, it jitters `P` and measures accuracy.
- `sigma_test=0` is no perturbation.
- For `sigma_test>0`, it repeats N times and reports mean/std.
- Reports per-run AUC over sigma grid (higher = more robust curve).

Current grid used in these runs:

- `sigma_test = [0, 0.05, 0.1, 0.25, 0.5]`
- repeats = 5

### Determinism question
- At `sigma_test=0`, result is deterministic for a fixed checkpoint/eval path (no injected jitter).
- At `sigma_test>0`, it is stochastic; repeat count controls variance estimate.

## 5) Seed-0 sparse jitter results (AUC)
Baseline checkpoint: `Seed0Matrix ... sigma_drop=0 ... sparsity_p=0.96_Best_ACC.pt`

- candidate `sigma=0.05`: baseline AUC `0.36931754`, candidate AUC `0.38274114`, delta `+0.01342360`
- candidate `sigma=0.1`: baseline AUC `0.36931754`, candidate AUC `0.39555392`, delta `+0.02623638`
- candidate `sigma=0.25`: baseline AUC `0.36931754`, candidate AUC `0.39456972`, delta `+0.02525218`
- candidate `sigma=0.5`: baseline AUC `0.36931754`, candidate AUC `0.39079531`, delta `+0.02147777`

This is consistent with training metrics: small sigma helps, with `0.1`/`0.25` strongest.

## 6) Pairwise jitter vs cached jitter
Originally we ran pairwise jobs (`baseline vs candidate`) for each candidate. This is valid but recomputes baseline each time.

Now we added cached scripts to avoid wasted baseline compute:

- `scripts/eval_jitter_many.py`: computes baseline jitter curve once, reuses for many candidates.
- `scripts/run_jitter_seed0_sparse_small_sigma_cached.sh`: ready-made seed-0 sparse batch using cached baseline.

W&B logging remains comparison-style per candidate run (`jitter_acc_baseline`, `jitter_acc_candidate`, `baseline_auc`, `candidate_auc`, `delta_auc...`).

## 7) Ops issues encountered and fixes
1. **Watcher launch used system Python** (`/usr/local/bin/python`) on some machines, causing:
   - `ModuleNotFoundError: spikingjelly`
2. Fix:
   - always run through project venv: `source /workspace/SNN-delays/.venv/bin/activate`
3. SHD instability/flat behavior likely tied to SpikingJelly version + cached preprocessing outputs.

## 8) Current status (live snapshot)
- Seed-0 sparse small-sigma runs are complete and analyzed.
- Seed-1 and seed-2 sparse small-sigma sweeps are currently running on two GPU machines.
- CPU machine is set up for jitter evaluations, with cached baseline workflow available.

Live check at `2026-03-22 19:48 UTC`:

- `213.173.110.196` (`seed=1`, sparse `sparsity_p=0.96`, current `sigma_drop=0.05`)
  - last logged epoch: `130/149`
  - last test acc: `37.20%`
  - best test acc so far in this run: `37.20%`
  - run log mtime: `2026-03-22 19:48:38 UTC`
- `213.173.109.6` (`seed=2`, sparse `sparsity_p=0.96`, current `sigma_drop=0.05`)
  - last logged epoch: `120/149`
  - last test acc: `35.62%`
  - best test acc so far in this run: `35.62%`
  - run log mtime: `2026-03-22 19:48:38 UTC`

Queue state:

- Both queues are still on `sigma_drop=0.05` for their respective seeds.
- After completion, each queue should continue automatically with `sigma_drop=0.1`, then `0.25`, then `0.5`.

## 9) Important artifact paths
- Training status dirs: `.run_status/delaydrop_ssc_small_sigma_*`
- Seed-0 sparse jitter outputs:
  - `analysis/jitter_seed0_sparse_small_sigma/sigma_0p05/summary.txt`
  - `analysis/jitter_seed0_sparse_small_sigma/sigma_0p1/summary.txt`
  - `analysis/jitter_seed0_sparse_small_sigma/sigma_0p25/summary.txt`
  - `analysis/jitter_seed0_sparse_small_sigma/sigma_0p5/summary.txt`
- Cached jitter scripts:
  - `scripts/eval_jitter_many.py`
  - `scripts/run_jitter_seed0_sparse_small_sigma_cached.sh`
