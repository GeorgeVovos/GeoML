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

| GSE       | n total | platform / tissue                 | role in v07         |
|-----------|---------|-----------------------------------|---------------------|
| GSE76809  | 266     | GPL6480, multi-tissue SSc         | reference (= v06)   |
| GSE9285   |  75     | Agilent, skin biopsies            | independent #1      |
| GSE58095  | 102     | Illumina HT-12 V4, skin           | independent #2      |
| GSE45536  | 123     | Affymetrix HG-U133+2, blood       | independent #3      |

The three "independent" datasets are deliberately *smaller and from
different platforms* — exactly the regime where a small-data QML advantage
should be most visible.

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

Estimated runtime: **~30 min per dataset** (no LC, no holdout), so
**~2 hours total** for 4 datasets on CPU.

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
