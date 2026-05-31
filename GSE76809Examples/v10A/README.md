# v10A — Noisy Simulation of the v08 Sample-Efficiency Sweep

## Goal

Every v01–v09 result is on a **noiseless** statevector simulator. The
open question is whether any of it survives on real NISQ hardware. v10A
gives a defensible first-order answer for the **small-data regime**
without requiring real-device access.

v10A re-runs **v08's data-reuploading VQC** on PennyLane's
`default.mixed` density-matrix simulator with a parameterised
depolarising channel after every gate, crossing v08's sample-size axis
with a noise axis.

## Why v08? (vs v06 / v06B / v07 / v09)

A noise study is only informative when it degrades a quantity that is
genuinely *quantum* and genuinely *interesting*. Of the five candidates:

| Example | Core variable | Fit for a noise study |
|---------|---------------|------------------------|
| v06   | VQC vs tuned classical on full data | Weak: the v06 headline is "tuned XGBoost wins", so noise just makes a losing model lose harder. |
| v06B  | Classical feature extractor (ANOVA+PCA vs learned) | Wrong axis: the variable is the *classical* front-end; noise degrades all extractors roughly equally. |
| v07   | Cross-dataset transfer | Incomplete grid (only GSE76809 ran); noise adds an axis to an unfinished one. |
| **v08** | **Small-data sample efficiency** | **Chosen here.** Small-data efficiency is the most-cited reason to expect a quantum edge, so "does noise erode small-N faster than large-N?" is a sharp, decision-relevant question. |
| v09   | Encoding ablation | Also strong — covered separately in **v10B**. |

v08 is the right target for a *sample-size* noise study because it is the
one experiment whose whole point is behaviour as a function of training
set size. v08's own result was a **negative** one (no small-data quantum
advantage on this dataset); v10A asks the natural follow-up: under
realistic noise, does the small-N AUC collapse faster than the large-N
AUC, or is the VQC's small-data behaviour noise-robust?

## Experimental design

For each cell in the grid

```
(N_per_class)  x  (subsample_seed)  x  (noise_p)
```

we draw a 1:1 stratified subsample of the v06 training pool, refit
ANOVA / scaler / PCA **inside** the subsample (leakage-safe, identical to
v08's `build_fold_features`), train the noisy VQC, and score AUC on the
**fixed v06 holdout** (54 samples). `noise_p = 0` reproduces v08's
noiseless small-data curve exactly (modulo density-matrix vs statevector
representation).

## Noise model

| Setting              | Value                                                   |
|----------------------|---------------------------------------------------------|
| Device               | `default.mixed` (density-matrix simulator)              |
| Channel              | Depolarising (`DepolarizingChannel`)                    |
| Single-qubit error p | swept: 0, 1e-4, 1e-3, 5e-3, 1e-2                        |
| Two-qubit error      | `min(10 × p, 0.75)` (per NISQ folklore, capped)         |

This is **a model, not the model**. Real superconducting / trapped-ion
devices have correlated, time-dependent, qubit-specific errors. The
depolarising model is the simplest credible stand-in and is the standard
"first noisy experiment" in QML papers.

## Reduced design (compute-aware)

`default.mixed` is ~5–10× slower than `default.qubit` (it tracks the full
density matrix, 2^(2n) entries instead of 2^n), so v10A is leaner than
v08:

- Reduced N grid: `[10, 20, 50]` (vs v08's eight values up to 100).
- 5 subsample seeds (vs 20), single VQC init per subsample (vs 3).
- 40 training epochs (vs 80) — under noise, training plateaus sooner.
- Only the VQC is run (no classical baselines, no quantum kernel).

## How to run

```powershell
cd GSE76809Examples\v10A
python noise_sweep.py                                # full reduced sweep
python noise_sweep.py --noise 0 1e-3                 # 2-level subset
python noise_sweep.py --n-per-class 10 20 --subsample-seeds 3   # quick
```

The run is **resumable**: a checkpoint
(`results/v10A_noise_sample_efficiency_partial.json`) is written after
every cell and reloaded on startup, so a crashed run continues where it
stopped.

## Output

`results/v10A_noise_sample_efficiency.json` — per `(noise_p, N_per_class)`
cell: mean / std / min / max AUC and a bootstrap 95 % CI of the mean over
all subsample seeds.

The headline number to report: **at what single-qubit error rate does the
small-N (N=10) holdout AUC fall to chance (~0.5), and does it degrade
faster than the larger-N (N=50) AUC?**

## Files

| File                          | Purpose                                                            |
|-------------------------------|-------------------------------------------------------------------|
| `model_quantum_vqc_noisy.py`  | v08/v06 data-reuploading VQC ported to `default.mixed` with depolarising channels |
| `noise_sweep.py`              | Driver — sweeps `(N_per_class × subsample_seed × noise_p)`, resumable |

## Caveats

- We do **not** include measurement (shot) noise here; that costs another
  order of magnitude in wall time and is deferred to a follow-up.
- `noise_p = 0` on `default.mixed` is numerically equivalent to v08's
  `default.qubit` run but not bit-identical (different backend).
- Backprop on `default.mixed` is supported but slow; consider
  parameter-shift if you only want a single noise level.
