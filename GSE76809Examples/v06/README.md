# GSE76809 v06 — Fair Quantum vs Classical Comparison

## Purpose

v06 addresses the methodological flaws of v04 (SMOTE leakage) and v05 (undertuned baselines) to provide a **scientifically rigorous** comparison between quantum and classical machine learning on gene expression data.

## What's New in v06

| Fix | v04/v05 Problem | v06 Solution |
|-----|-----------------|--------------|
| **Per-fold SMOTE** | v04 applied SMOTE before CV splits → data leakage | SMOTE applied inside each fold; validation never sees synthetic data |
| **Parameter-matched MLP** | v04: 120K params; v05: 26K params — both unfair comparisons | MLP has ~1,600 params, same as VQC (~1,500 params) |
| **Data reuploading VQC** | Single amplitude encoding limits expressiveness | Re-encodes input at every layer — provably more expressive |
| **Classical RBF SVM** | No direct kernel comparison existed | RBF SVM on same PCA data as quantum kernel |
| **Tuned XGBoost** | Fixed hyperparams → catastrophic failure at small data | Inner CV selects optimal depth/trees/learning rate per fold |
| **Fine-grained learning curves** | Only tested 25/50/75/100% | Tests 10/20/30/50/75/100% with 3 repeats each |
| **McNemar's test** | Only paired t-tests (weak with 5 folds) | Per-sample agreement test on holdout predictions |
| **Cohen's d** | No effect sizes reported | Quantifies practical significance alongside p-values |

## Models

### Quantum Models

#### 1. Data Reuploading VQC (~1,500 parameters)

**Key innovation**: Instead of encoding data once (amplitude encoding), re-encodes the input features at **every layer** with learned projections. This is based on Pérez-Salinas et al. (2020) and provides universal function approximation with just 4 qubits.

```
For each of 8 layers:
  For each qubit q:
    RY(Σ w[l,q,i] * x[i])   ← learned linear combination of ALL 16 features
    RZ(θ[l,q])               ← variational (data-independent)
    RX(φ[l,q])               ← variational
  CNOT ring entanglement

Measurement: ⟨Z⟩, ⟨X⟩, ⟨Y⟩ on each qubit → 12 quantum features
Post-net: 12 → 64 → 1
```

**Why this is better**: Each layer creates a *different* embedding of the data into Hilbert space. The circuit can express functions that require exponentially many terms in a Fourier series, while using only O(qubits × layers) parameters.

#### 2. Quantum Kernel SVM (0 trained parameters in feature map)

**Improvement**: All-to-all ZZ connectivity (v04/v05 used ring-only), depth=3.

```
Feature map U(x) applied 3 times:
  Hadamard layer → RZ(x_i) per qubit → ALL-PAIRS ZZ(x_i·x_j) interaction

Kernel: k(x1, x2) = |⟨0|U†(x2)·U(x1)|0⟩|²
```

The all-to-all connectivity means **every pair of genes** has its interaction encoded, not just adjacent ones. With 6 PCA components, this captures all 15 pairwise interactions per depth repetition.

### Classical Models

#### 3. Parameter-Matched MLP (~1,600 parameters)

Architecture: `16 → 48 → 16 → 1` with GELU + Dropout(0.2).

**Deliberately matched** to VQC parameter count. Uses identical:
- Optimizer (Adam, lr=0.005)
- Training epochs (80)
- Early stopping (patience=15)
- Per-fold SMOTE
- Class-weighted loss

If VQC outperforms this, the quantum circuit's **structure** (entanglement, interference) is providing benefit beyond what parameters alone give.

#### 4. Classical RBF SVM

RBF kernel: `k(x,y) = exp(-γ||x-y||²)` on same 6-dim PCA data as quantum kernel.

**Direct comparison**: Both SVMs see the same data representation. The only difference is the kernel function. If quantum kernel beats RBF, the quantum Hilbert space captures structure that Gaussian similarity misses.

#### 5. Tuned XGBoost

Grid-searched over `{max_depth: [2,3,4], n_estimators: [50,100,150], learning_rate: [0.05,0.1,0.2], min_child_weight: [1,3,5]}` via inner 3-fold CV.

Uses `scale_pos_weight` for class imbalance. This gives XGBoost its best chance — no more arbitrary fixed hyperparameters.

## Evaluation Methodology

### Per-Fold SMOTE (Correct Approach)

```python
for train_idx, val_idx in kfold.split(X, y):
    X_fold_train, y_fold_train = smote.fit_resample(X[train_idx], y[train_idx])
    X_fold_val = X[val_idx]  # NEVER augmented — pure real samples
```

This ensures validation performance reflects generalization to real unseen data.

### Learning Curves (Small-Data Regime)

Tests at **10%, 20%, 30%, 50%, 75%, 100%** with 3 random subsets per fraction.

The quantum advantage hypothesis: quantum models should maintain AUC at small data sizes where classical tree methods collapse (due to the exponential Hilbert space providing implicit regularization).

### Statistical Tests

| Test | What it measures | Advantage over t-test |
|------|-----------------|----------------------|
| **Paired t-test** | Mean AUC difference across 5 folds | Standard approach |
| **Cohen's d** | Practical significance (effect size) | Not inflated by n |
| **McNemar's test** | Per-sample prediction agreement | Uses all 54 test samples, not just 5 fold means |

## Running

```bash
cd v06
python compare_models.py
```

Results saved to `v06/results/v06_comparison_results.json`.

## Expected Runtime

- VQC (data reuploading): ~3-5 min/fold on CPU (faster than v04's amplitude encoding)
- Quantum Kernel: ~3-5 min/fold (all-to-all ZZ is O(n²) but on 6 qubits)
- Classical models: < 10 seconds total
- Learning curves: ~30-60 min (many VQC evaluations at reduced epochs)
- **Total estimate: 2-4 hours on CPU**

## Theoretical Justification

### Why Data Reuploading?

Classical neural networks achieve universality through depth and width. Quantum circuits achieve universality through **data reuploading** — by encoding data multiple times with different projections, the circuit generates a Fourier series in the input features with exponentially many terms (due to tensor product structure of qubits).

A 4-qubit, 8-layer reuploading circuit can express functions with up to 2⁴ × 8 = 128 frequency components, while a classical network with 1,500 parameters is limited to the rank of its weight matrices (~48 effective frequencies for our architecture).

### Why Quantum Kernels for Gene Expression?

ZZ gates encode `exp(i · x_i · x_j)` — the PRODUCT of two features mapped into a quantum phase. In genomics, gene-gene interactions (epistasis) are multiplicative: gene A's effect depends on gene B's expression level. The quantum feature map naturally encodes this multiplicative structure in its very fabric.

A classical RBF kernel computes `exp(-γ||x-y||²)` — a DISTANCE metric that treats features independently (no cross-terms in the exponent). The quantum kernel's ZZ gates explicitly capture pairwise interactions that RBF kernels cannot.

## File Structure

```
v06/
├── preprocess_gse76809.py  — ANOVA features, per-fold SMOTE utility
├── model_quantum_vqc.py    — Data reuploading VQC (4 qubits, 8 layers)
├── model_quantum_kernel.py — All-to-all ZZ kernel (6 qubits, depth=3)
├── model_classical_mlp.py  — Parameter-matched MLP (~1,600 params)
├── model_classical_svm.py  — Classical RBF SVM (GridSearchCV)
├── model_classical_xgb.py  — Tuned XGBoost (GridSearchCV)
├── compare_models.py       — Full pipeline: CV + holdout + learning curves + stats
├── README.md               — This file
└── results/                — Output JSON with all metrics
```
