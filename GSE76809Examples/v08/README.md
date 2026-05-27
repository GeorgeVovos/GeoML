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
