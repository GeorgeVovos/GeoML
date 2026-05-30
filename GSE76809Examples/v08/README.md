# v08 — Pure Sample-Efficiency Study

## Goal

This is the single experiment that directly addresses the central
question: **does quantum ML beat strong classical baselines when training
data is scarce?**

v06's learning curve hinted at it (VQC AUC 0.683 vs XGBoost 0.500 at 21
samples, 3 repeats). v08 turns that hint into a defensible measurement by:

1. **Decoupling sample size from class imbalance** — every training set is
   built by stratified subsampling with a *forced 1:1 ratio*. The v06
   learning curve mixed "less data" with "almost no minority samples";
   v08 doesn't.
2. **Sweeping N over a wider, finer grid** —
   `N_per_class ∈ {5, 10, 15, 20, 30, 50, 75, 100}` (so total N up to 200).
3. **Many repeats** — 20 subsample seeds at every N, plus 3 VQC init
   seeds per subsample => 60 VQC trainings per (N, model) cell. This is
   the variance budget v06 was missing.
4. **Including LR-L1** — the historical small-data classical champion.
   If QML can't beat it, the small-data quantum-advantage story is dead.
5. **Frozen pipeline** — same v06 ANOVA / scaler / PCA / model
   hyperparameters, refit per subsample. Only N changes.

The fixed test set is **the holdout set from v06's GSE76809 split** (54
samples, 8 healthy), so every (N, seed) cell evaluates on the same
patients — a clean apples-to-apples comparison.

## Models compared

| Model              | Why include                                              |
|--------------------|----------------------------------------------------------|
| Data-reuploading VQC | The quantum candidate                                   |
| Quantum kernel SVM | Second quantum candidate (skipped at N<30, see v06)      |
| Parameter-matched MLP | Apples-to-apples classical NN                         |
| Tuned XGBoost      | Strong gradient-boosted baseline                          |
| RBF SVM            | Classical kernel baseline                                 |
| **LR-L1** (new)    | The small-data linear champion                            |

## How to run

```powershell
# Need GSE76809 downloaded
cd C:\dev\GeoML
python download_geo.py --gse GSE76809

cd GSE76809Examples\v08
python sample_efficiency.py                       # full sweep (LONG)
python sample_efficiency.py --quick               # 5 seeds, fewer N values
python sample_efficiency.py --skip-quantum-kernel # skip QKernel cells
python sample_efficiency.py --models classical_logreg classical_xgb  # subset
```

## Compute cost (warning)

| Cell                                  | Approx time |
|---------------------------------------|-------------|
| 1 VQC train (small N, fast)           | ~30 s       |
| 1 Quantum-Kernel train (full N)       | ~3-5 min    |
| 1 MLP / XGB / SVM / LR-L1 train       | < 5 s       |

Full sweep with all defaults: **8 N values x 20 subsample seeds x 3 init
seeds x 1 VQC + 1 QKernel + 4 fast classicals ~ 30-60 hours on CPU.**

Realistic plan:
- First pass: `--quick --skip-quantum-kernel` (~1 hour) to confirm code
  works end-to-end.
- Second pass: full sweep with `--skip-quantum-kernel` (~10 hours
  overnight) — you usually don't need the quantum kernel for this story.
- Final pass: full sweep including quantum kernel only if the second pass
  motivates it.

## Output

`results/v08_sample_efficiency.json` — for every (N_per_class, model)
cell: list of AUCs across all subsample x init seeds, plus mean/std/95% CI.

## Plotting

`plot_results.py` (provided) generates `results/v08_learning_curves.png`
with one line per model, AUC vs N, shaded 95% confidence intervals from
the seed distribution. That plot is the headline figure of the study.

## Why this beats v06's learning curve

| Concern                    | v06 LC                  | v08                     |
|----------------------------|-------------------------|-------------------------|
| Class ratio at small N     | Uncontrolled (random)   | Fixed 1:1               |
| Repeats per N              | 3                       | 20 x 3 (VQC) / 20 others|
| Smallest N tested          | ~21 samples             | 10 samples (5 per class)|
| LR-L1 baseline             | Absent                  | Included                |
| Init-seed variance for VQC | Not measured            | Measured per N          |
| Test set                   | Same (good)             | Same (good)             |

If the v06 small-data signal survives v08's tighter design, you have a
publishable claim. If it doesn't, you have an honest negative result that
is *itself* worth a chapter ("On the fragility of reported small-data
quantum advantage").

## Preliminary Results (quick mode: 5 subsample seeds, 2 VQC init seeds)

**Dataset:** GSE76809, 266 usable samples (4521 genes after variance filter).
The training pool has only 30 samples per class, so N_per_class ∈ {50, 100}
were skipped (insufficient data). Results below are for N=10 and N=20.

### N_per_class = 10 (total training = 20 samples, 1:1 balanced)

| Model            | Mean AUC | Std   | 95% CI          | n |
|------------------|----------|-------|-----------------|---|
| Classical MLP    | 0.780    | 0.065 | [0.730, 0.830]  | 5 |
| XGBoost          | 0.762    | 0.114 | [0.683, 0.843]  | 5 |
| LR-L1            | 0.753    | 0.114 | [0.670, 0.847]  | 5 |
| **Quantum VQC**  | 0.669    | 0.125 | —               | 9*|
| RBF SVM          | 0.429    | 0.231 | [0.256, 0.601]  | 5 |

*VQC: 5 subsamples × 2 init seeds = 10 cells; 9 completed before timeout.

### N_per_class = 20 (total training = 40 samples, 1:1 balanced)

| Model            | Mean AUC | Std   | 95% CI          | n |
|------------------|----------|-------|-----------------|---|
| LR-L1            | 0.797    | 0.079 | [0.737, 0.858]  | 5 |
| Classical MLP    | 0.790    | 0.089 | [0.721, 0.855]  | 5 |
| XGBoost          | 0.788    | 0.070 | [0.733, 0.840]  | 5 |
| RBF SVM          | 0.496    | 0.295 | [0.279, 0.746]  | 5 |
| Quantum VQC      | —        | —     | —               | — |

VQC results for N=20 were not collected due to runtime constraints.

### Key observations

1. **No small-data quantum advantage detected.** At N=10 (the smallest
   cell), the VQC's mean AUC (0.669) is *below* all three strong classical
   baselines (MLP 0.780, XGBoost 0.762, LR-L1 0.753).
2. **High VQC variance.** The VQC shows extreme seed sensitivity — AUC
   ranges from 0.538 to 0.929 across init seeds and subsamples, suggesting
   the training landscape is poorly conditioned at 6 qubits / 80 epochs.
3. **LR-L1 is competitive.** Even at N=10, logistic regression with L1
   regularisation achieves 0.753 mean AUC, confirming its role as the
   small-data classical champion.
4. **RBF SVM struggles.** The kernel SVM with default grid search performs
   poorly at both N=10 and N=20, likely due to inadequate hyperparameter
   tuning at small sample sizes.
5. **Runtime.** Classical models complete in <30 seconds total; the VQC
   requires ~2.5 minutes per training run (~40 minutes for N=10 alone at
   5 seeds × 2 inits).

### Limitations of this quick run

- Only 5 subsample seeds (design calls for 20).
- Only 2 VQC init seeds (design calls for 3).
- N=50 and N=100 could not be tested because the training pool has only
  30 samples per class.
- VQC was not evaluated at N=20 due to runtime constraints.
- Quantum kernel was skipped entirely.

A full overnight run with the complete seed budget is needed to produce
publication-grade confidence intervals.
