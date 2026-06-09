# v07 — Cross-Dataset Replication of v06's Findings

## Goal

v06 found one possibly-interesting effect on a *single* dataset (GSE76809):
the data-reuploading VQC degrades more gracefully than XGBoost / RBF-SVM
when training data is reduced to ~10 % (~21 samples). The natural follow-up
question is **does this signal reproduce on other systemic-sclerosis
expression datasets, or is it a property of this particular split?**

v07 answers that by running v06's exact pipeline on multiple datasets
without changing any model hyperparameters.

## Datasets

| GSE       | n total | platform / tissue                 | role in v07         | status in current run |
|-----------|---------|-----------------------------------|---------------------|-----------------------|
| GSE76809  | 266     | GPL6480, multi-tissue SSc         | reference (= v06)   | ✅ completed          |
| GSE9285   |  75     | Agilent, skin biopsies            | independent #1      | ⬜ not yet run        |
| GSE58095  | 102     | Illumina HT-12 V4, skin           | independent #2      | ⬜ not yet run        |
| GSE45536  | 123     | Affymetrix HG-U133+2, blood       | independent #3      | ⬜ not yet run        |

The three "independent" datasets are deliberately *smaller and from
different platforms* — exactly the regime where a small-data QML advantage
should be most visible.

> ⚠️ **Current status:** `results/v07_cross_dataset_results.json` contains
> **only GSE76809** so far. The three independent datasets have not been run
> yet (their labelers in `dataset_loaders.py` are still placeholders —
> see *Known limitations*). The cross-dataset replication question below
> therefore **cannot be answered yet**; what we have is a single-dataset
> sanity check that the v06 pipeline still reproduces on its reference data.

## Methodology

- For each dataset: ANOVA top-16 features, 5-fold stratified CV, per-fold
  SMOTE for VQC/MLP, per-fold scaler/PCA, identical v06 model code.
- No holdout split (each dataset's full labelled set goes into 5-fold CV).
- No learning-curve sweep (kept for v08).
- Adds **Logistic-Regression-L1** to the comparison set so every dataset
  has the missing canonical baseline.
- All seeds fixed to 2026 within a dataset; the random_state of v06 is
  reused so per-dataset runs are individually reproducible.

## How to run

```powershell
# Download the datasets first (one-time)
cd C:\dev\GeoML
python download_geo.py --gse GSE76809 GSE9285 GSE58095 GSE45536

# Run cross-dataset evaluation
cd GSE76809Examples\v07
python compare_models.py
```

Estimated runtime: the actual GSE76809 run took **~117 min** on CPU
(see `total_runtime_minutes` in the results JSON), so budget closer to
**~2 hours per dataset** (≈8 hours for all 4), not the earlier ~30 min
guess. The VQC's 5-fold CV dominates this cost.

## Files

| File                 | Purpose                                                 |
|----------------------|---------------------------------------------------------|
| `dataset_loaders.py` | Per-GSE labelling + platform-filter logic               |
| `preprocess.py`      | Generic preprocessing taking a `gse_id` argument        |
| `compare_models.py`  | Loops over datasets, runs CV, writes per-dataset JSON   |

The model files are **reused unchanged from v06** via a `sys.path` import.
The whole point of v07 is to test the v06 models on new data, not to
re-tune them.

## Interpreting the output

`results/v07_cross_dataset_results.json` contains, per dataset:
- 5-fold CV mean/std AUC, accuracy, F1 for each model
- Wilcoxon signed-rank + Nadeau-Bengio corrected paired t-test against LR-L1
- Holm-corrected p-values across the per-dataset comparisons

The replication question is:
> *Does the VQC's CV-AUC mean exceed the strongest classical baseline on
> at least one independent dataset, with Cohen's d > 0.5 and Holm-adjusted
> p < 0.05?*

If yes on **2+ datasets**, the small-data quantum-advantage claim is
supported. If no, the effect is at best dataset-specific to GSE76809.

## Results so far (GSE76809 only)

5-fold stratified CV, ANOVA top-16 features, n=266. Mean ± std AUC:

| Model              | Mean AUC | Std    | vs LR-L1 (Cohen's d) | Holm p | Significant? |
|--------------------|----------|--------|----------------------|--------|--------------|
| classical_xgb      | 0.899    | 0.101  | +0.563 (medium)      | 0.100  | no           |
| **classical_logreg (LR-L1)** | **0.832** | **0.112** | baseline | — | — |
| quantum_vqc        | 0.827    | 0.090  | −0.045 (negligible)  | 0.898  | no           |
| classical_svm      | 0.812    | 0.112  | −0.161 (negligible)  | 0.704  | no           |
| classical_mlp      | 0.787    | 0.074  | −0.427 (small)       | 0.769  | no           |
| quantum_kernel     | 0.570    | 0.129  | −1.944 (large)       | 0.151  | no           |

**Findings:**

- **No quantum advantage on the reference dataset.** XGBoost is the
  strongest model (0.899). The VQC (0.827) is essentially tied with the
  LR-L1 baseline (0.832) — a *negligible negative* effect (d = −0.045)
  that is not significant after Holm correction.
- **The quantum kernel collapses** to 0.570 AUC (large negative effect,
  d = −1.94), well below every classical baseline.
- After Holm correction across the five comparisons, **no model differs
  significantly** from LR-L1 on this dataset (smallest Holm p = 0.10 for
  XGBoost).
- Because the three independent datasets have **not been run**, the
  cross-dataset replication question is **still open**. On the evidence
  available, the v06 "graceful VQC" signal does **not** translate into a
  CV-AUC advantage even on GSE76809 itself, so the small-data
  quantum-advantage claim is currently **unsupported**.

**Next step:** label and run GSE9285 / GSE58095 / GSE45536, then revisit
the replication verdict above.

## Known limitations / TODO

- **Labelers are placeholders.** Each GSE has its own metadata schema.
  `dataset_loaders.py` ships heuristics that match common GEO conventions
  (`disease state`, `source`, `title` regexes). After downloading a new
  dataset, open `data/<GSE>/<GSE>_metadata.csv` and verify the produced
  labels with `python preprocess.py --inspect <GSE>`. Fix the labeler if
  the label distribution looks wrong before trusting the CV results.
- For platforms with very few non-NaN probes (e.g. small Agilent slices),
  the variance pre-filter top-50 % may produce <16 features. The code falls
  back to whatever is available; if you see "n_features=X < 16" in the
  preprocess output for a dataset, treat that dataset's results with extra
  caution.
- No noise/hardware simulation (saved for v10).
