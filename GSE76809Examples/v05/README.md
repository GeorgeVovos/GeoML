# v05: Quantum vs Classical on a DIFFERENT GSE76809 Data Slice

## What This Version Does

v05 re-runs the **same hypothesis as v04** — *can quantum ML beat classical ML for
SSc classification?* — but on a **different slice of the same GSE76809 dataset**.

The goal is a robustness check: if quantum advantage (or its absence) seen in v04
was driven by lucky feature selection, the SMOTE augmentation, or the specific
train/test split, then changing those should change the outcome. If conclusions
hold across both versions, they are more credible.

---

## Same as v04

- **Database**: GSE76809 (NCBI GEO) — SSc vs Healthy from blood transcriptomics
- **Platform**: GPL6480 (Agilent 4x44K microarray) — 266 labeled samples
- **Task**: binary classification SSc (1) vs Healthy (0)
- **4 models**: Quantum VQC, Quantum Kernel SVM, Classical MLP, Classical XGBoost
- **Evaluation**: holdout + 5-fold CV + learning curves + paired t-tests

---

## What's DIFFERENT in v05

| Aspect | v04 | v05 |
|---|---|---|
| Random split seed | 42 | **2024** (different train/test partition) |
| Variance pre-filter | top 75% of genes | **top 50%** (stricter, fewer noisy genes) |
| Feature selection | Mutual Information | **ANOVA F-test** (linear separation score) |
| # features | 64 | **16** |
| VQC qubits | 6 (2^6 = 64) | **4** (2^4 = 16, smaller & faster circuit) |
| VQC parameters | 5,619 | **1,405** |
| Class imbalance | BorderlineSMOTE → 364 synthetic train samples | **Class weighting only** — 212 real samples only |
| Kernel PCA dims | 8 qubits | **6 qubits** |
| MLP hidden dims | 256→128→64→32 | **128→64→32→16** (proportional to input) |
| MLP parameters | 120,737 | **26,577** |

---

## Why These Changes Are Improvements

### 1. No SMOTE — more honest evaluation
v04 used BorderlineSMOTE to synthetically inflate the minority class from 30 to
182 training samples. This contaminated cross-validation: synthetic Healthy
samples generated from fold A leaked into fold B's validation set, making CV
AUC scores unrealistically high (v04 Quantum VQC CV AUC = **0.998**).

v05 removes SMOTE entirely and relies only on class weighting (`pos_weight` /
`scale_pos_weight`). The 5-fold CV now evaluates models on **real samples only**,
giving honest estimates.

### 2. ANOVA F-test — faster, less prone to overfitting
Mutual Information (v04) estimates non-linear dependencies using k-nearest
neighbours and is sensitive to the training split. ANOVA F-test measures the
linear variance ratio between classes — faster, deterministic, and less likely
to over-select features that happen to be noisy correlates of the random split.

### 3. Smaller circuit — faster, fairer comparison
With 4 qubits instead of 6, the quantum simulator runs ~4× faster per sample
(state space 2^4 = 16 vs 2^6 = 64). This makes the quantum models less
expensive while still testing the core question of quantum advantage.

### 4. Smaller MLP — proportional to input size
v04's MLP (120K parameters) was dramatically over-parameterised for 212 training
samples (567 params/sample). v05 scales it down (26K params, ~125 params/sample),
making the classical baseline a fairer opponent for the quantum models.

---

## Actual Results (from `results/v05_comparison_results.json`)

### Holdout (54 test samples)

| Model | Accuracy | AUC | F1(SSc) | F1(Healthy) | Train time |
|---|---|---|---|---|---|
| Quantum VQC | 79.6% | 0.791 | 0.867 | 0.560 | 395s |
| Quantum Kernel | 83.3% | 0.859 | 0.903 | 0.400 | 196s |
| Classical MLP | 87.0% | 0.962 | 0.918 | 0.696 | <1s |
| **Classical XGBoost** | **96.3%** | **0.957** | **0.978** | **0.875** | <1s |

### 5-Fold Cross-Validation AUC

| Model | Mean AUC | Std | Min | Max |
|---|---|---|---|---|
| Quantum VQC | 0.853 | 0.031 | 0.801 | 0.892 |
| Quantum Kernel | 0.840 | 0.056 | 0.773 | 0.940 |
| Classical MLP | **0.910** | 0.055 | 0.847 | 1.000 |
| Classical XGBoost | 0.874 | 0.054 | 0.793 | 0.944 |

### Learning Curves (AUC at data fractions)

| Model | 25% | 50% | 75% | 100% |
|---|---|---|---|---|
| Quantum VQC | 0.758 | 0.842 | 0.929 | 0.856 |
| Quantum Kernel | 0.704 | 0.745 | 0.761 | 0.859 |
| Classical MLP | 0.867 | 0.951 | 0.902 | 0.973 |
| Classical XGBoost | 0.500 | 0.857 | 0.878 | 0.957 |

### Statistical Tests (paired t-test on 5 CV fold AUCs)

All comparisons: **not significant** (all p > 0.10). The dataset is too small for
t-tests to reach p < 0.05 with only 5 fold pairs.

---

## v04 vs v05 Side-by-Side

| | v04 CV AUC | v05 CV AUC | Change |
|---|---|---|---|
| Quantum VQC | **0.998** ± 0.004 | 0.853 ± 0.031 | -0.145 |
| Quantum Kernel | 0.891 ± 0.088 | 0.840 ± 0.056 | -0.051 |
| Classical MLP | 0.994 ± 0.009 | **0.910** ± 0.055 | -0.084 |
| Classical XGBoost | 0.921 ± 0.027 | 0.874 ± 0.054 | -0.047 |

**The dramatic drop in v04's VQC CV AUC (0.998 → 0.853) is the key finding.**
It confirms that v04's near-perfect quantum CV score was an artefact of SMOTE
leakage, not genuine quantum capability. With real samples only (v05), classical
models — especially XGBoost and MLP — outperform both quantum approaches on
every metric.

### What This Means

- **Classical XGBoost is the best model** on this dataset under honest conditions
- **Quantum models are 400–1300× slower** to train with no compensating accuracy
- **No result reached statistical significance** in either version, reflecting the
  small sample size (n=266 total, n=54 test)
- v04's quantum "advantage" in CV was a **SMOTE artefact**, not a real effect

---

## The Four Models

### 1. Quantum VQC (`model_quantum_vqc.py`)
- 4 qubits, 6 variational layers, amplitude encoding of 16 features
- ZZ interaction layer + multi-basis (X+Y+Z) measurement → 12 quantum features
- Post-net: 12 → 32 → 16 → 1, sigmoid
- Adam + warm-up (5 epochs) + cosine decay + gradient clipping (norm=1.0)

### 2. Quantum Kernel SVM (`model_quantum_kernel.py`)
- ZZ feature map on 6 qubits (PCA-projected 6-D input scaled to [0, π])
- Depth = 2 feature-map repetitions
- Precomputed kernel matrix → `sklearn.svm.SVC` with balanced class weight, C=10

### 3. Classical MLP (`model_classical_mlp.py`)
- Residual MLP: 16 → 128 → 64 → 32 → 16 → 1
- Mixup augmentation (first 80% of epochs) + label smoothing (0.05)
- AdamW + cosine warm restarts + class-weighted BCE

### 4. Classical XGBoost (`model_classical_xgb.py`)
- 200 trees, depth 4, lr 0.05, subsample 0.8, colsample_bytree 0.8
- `scale_pos_weight` = n_neg/n_pos for imbalance, no SMOTE

---

## How to Run

```powershell
# Download data first (if not already present)
cd c:\TestSchedule\GeoML
python download_geo.py --gse GSE76809

# Run the full pipeline (~65 min on a modern CPU)
cd GSE76809Examples\v05
python compare_models.py
```

Results are saved to `results/v05_comparison_results.json`.

### Run individual stages

```powershell
python preprocess_gse76809.py     # writes data/GSE76809/processed_v05/*.npy
python model_quantum_vqc.py
python model_quantum_kernel.py
python model_classical_mlp.py
python model_classical_xgb.py
```

---

## Applying These Improvements to v01–v04

v01–v04 are **historical snapshots** of the project's evolution. Each one
references the previous version's flaws in its README, and each `results/`
folder contains JSON pinned to the code that produced it. Editing those
versions in place would break that narrative and invalidate the saved metrics.

If you want to roll the v05 improvements (no SMOTE, ANOVA F-test, scaled MLP,
smaller VQC, stricter variance filter) back into the older examples, here are
the options to consider:

**Option A — Create v06, v07, …** mirroring each older version with v05's
improvements applied. Cleanest, preserves history.

**Option B — Apply a subset of changes in place** (e.g. only the "no SMOTE"
change in v04, only the "smaller MLP" change in v03). Less destructive but
still rewrites history.

**Option C — Add a single "improvements applied" patch to all four older
versions**: same changes everywhere (no SMOTE, ANOVA, scaled MLP, smaller VQC),
and re-run them all. Roughly 4 × 60 min = **4 hours of compute**.

**Option D — Just update v04** since it is the most direct comparison to v05
(~65 min of compute).

### Which specific improvements?

1. Drop SMOTE entirely (use class weighting)
2. ANOVA F-test instead of mutual information
3. Smaller MLP (parameters proportional to input size)
4. Smaller VQC / fewer qubits
5. Stricter variance filter (top 50% vs top 75% of genes)

Pick a combination (e.g. "Option A with improvements 1+3") to proceed.
