# v12 — Strong Classical Baseline Pass

## Goal

v06's "classical baselines" were SVM-RBF, an MLP, and XGBoost. That is a
respectable set, but it omits the models that genuinely dominate
small-sample / high-dimensional gene-expression classification in the
biostatistics literature:

| Model                   | Why it matters on small-N omics                       |
|-------------------------|--------------------------------------------------------|
| **LR-L1**               | Sparse linear: tends to win on n≈p data; v06 added it. |
| **LR-Elastic-Net**      | Combines L1 sparsity with L2 stability — the GEO/TCGA standard. |
| **Linear SVM**          | Closed-form margin; no kernel-bandwidth nuisance.       |
| **Random Forest**       | Robust to feature collinearity; non-parametric.         |
| **Naive Bayes (GNB)**   | Famously hard to beat at n < 50.                        |
| **Nearest-shrunken-centroid** (PAM-style approximation) | Classic genomics baseline. |
| **Stacked ensemble**    | Logistic-regression meta-learner over the above.        |

v12 runs all of them on the **same** v06 fold layout and reports the
same paired stats. This ensures the quantum models are compared against
a properly competitive classical suite, addressing the most common
critique of QML benchmarking papers.

## How to run

```powershell
cd GSE76809Examples\v12
python compare_classical.py                # all baselines (~10 min)
python compare_classical.py --models lr_en rf gnb stacked
```

## Files

| File                     | Purpose                                          |
|--------------------------|---------------------------------------------------|
| `extra_baselines.py`     | LR-EN, linear SVM, RF, GNB, NSC, stacked          |
| `compare_classical.py`   | 5-fold CV driver, stats vs LR-L1 (small-data king)|

## Output

`results/v12_classical_pass.json` with per-model CV-AUC and Holm-
adjusted paired tests against LR-L1.

If **any** of these baselines beats v06's best quantum model with a
significant adjusted p-value, it must be reported openly — it does not
invalidate v07/v08/v09/v10/v11, but it does sharpen the final claim.
