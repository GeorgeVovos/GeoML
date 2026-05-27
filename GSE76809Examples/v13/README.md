# v13 — Calibration & Decision-Curve Analysis

## Goal

Every example so far reports AUC, accuracy, and F1. None reports
**calibration** (do the predicted probabilities mean what they say?) or
**clinical net benefit** (would a clinician acting on the model be
better off than treating everyone / nobody?).

For biomedical / SSc applications, those two omissions materially weaken
any deployment claim. v13 closes both gaps without retraining anything:
it loads v06's saved per-fold predictions, computes calibration curves,
Brier scores, and decision curves, and writes a single results JSON +
a plot.

## Metrics

| Metric                   | What it answers                                      |
|--------------------------|------------------------------------------------------|
| Brier score              | Mean-squared error of predicted probabilities         |
| Expected Calibration Error (ECE) | Bin-wise gap between predicted prob and observed freq |
| Maximum Calibration Error (MCE) | Worst-bin gap                                  |
| Sliced Wasserstein (10-bin) | Distribution-distance between predicted & empirical risks |
| Decision-curve net benefit at threshold t | (TP - FP * t / (1-t)) / N        |
| Platt rescaling improvement | Brier score change after sigmoid recalibration   |

## How to run

```powershell
cd GSE76809Examples\v13
# Re-uses v06 saved predictions (if present) or recomputes them:
python calibration.py --source v06         # default
python calibration.py --source v06 --model classical_xgb
```

## Files

| File              | Purpose                                                 |
|-------------------|----------------------------------------------------------|
| `calibration.py`  | Loads/regenerates per-fold predictions, computes metrics |
| `decision_curve.py` | Decision-curve analysis (Vickers 2006)                 |

## Output

- `results/v13_calibration.json` — per-model Brier / ECE / MCE / DCA
- `results/v13_calibration.png` — calibration plot + DCA plot

## Why it matters

A model with AUC 0.95 but ECE 0.20 is not deployable: its predicted
probabilities are systematically over- or under-confident. The v06
ensemble may well be in that regime because it averages classifiers with
very different output scales (sigmoid vs SVM decision-function vs
XGBoost). v13 quantifies that and reports whether Platt rescaling fixes
it.
