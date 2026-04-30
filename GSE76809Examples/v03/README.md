# GSE76809 v03 — Advanced Quantum vs Classical Classification

## Overview

Advanced SSc (systemic sclerosis) vs Healthy classification on [GSE76809](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76809) using amplitude-encoded dressed quantum circuits, a residual MLP with mixup augmentation, ensemble stacking, and 3-fold cross-validation.

## Key Improvements over v02

| Area | v02 | v03 |
|------|-----|-----|
| Quantum encoding | Angle embedding (100 features, 8 qubits) | Amplitude embedding (64 features → 6 qubits) |
| Quantum architecture | Flat VQC | Dressed circuit (pre/post classical layers) |
| Feature count | 100 (MI) | 64 (power-of-2 for amplitude encoding) |
| Class balancing | SMOTE | BorderlineSMOTE |
| Classical augmentation | None | Mixup (α=0.2) + label smoothing (0.05) |
| Threshold selection | Fixed 0.5 | Youden's J statistic optimization |
| Ensemble | None | Weighted average + logistic regression stacking |
| Validation | Single holdout | Holdout + 3-fold stratified CV |

## Results — Holdout Test Set

| Metric | Quantum v03 | Classical v03 | Ensemble v03 | Winner |
|--------|-------------|---------------|--------------|--------|
| **Accuracy** | **96.30%** | 94.44% | 94.44% | Quantum |
| **F1 (SSc)** | **97.87%** | 96.70% | 96.70% | Quantum |
| **F1 (Healthy)** | **85.71%** | 82.35% | 82.35% | Quantum |
| **AUC-ROC** | 91.30% | 95.11% | **95.65%** | Ensemble |
| Train Time | 1259.4s | **0.8s** | — | Classical |
| Parameters | 4,361 | 120,737 | — | Quantum (fewer) |

## Results — 3-Fold Cross-Validation

| Metric | Quantum (mean ± std) | Classical (mean ± std) |
|--------|----------------------|------------------------|
| **Accuracy** | **98.63% ± 1.03%** | 97.80% ± 1.03% |
| **F1 (SSc)** | **98.63% ± 1.03%** | 97.81% ± 1.02% |
| **F1 (Healthy)** | **98.63% ± 1.03%** | 97.80% ± 1.04% |
| **AUC-ROC** | **99.75% ± 0.24%** | 99.09% ± 0.83% |

### Per-Fold Detail

| Fold | Quantum Acc | Quantum AUC | Classical Acc | Classical AUC |
|------|-------------|-------------|---------------|---------------|
| 1 | 98.36% | 99.81% | 97.54% | 99.33% |
| 2 | 97.52% | 99.43% | 96.69% | 97.98% |
| 3 | **100.00%** | **100.00%** | 99.17% | 99.97% |

## Progression Across Versions

| Metric | Quantum v01 | Classical v01 | Quantum v02 | Classical v02 | Quantum v03 | Classical v03 |
|--------|-------------|---------------|-------------|---------------|-------------|---------------|
| Accuracy | 83.33% | 85.19% | 87.04% | 77.78% | **96.30%** | 94.44% |
| AUC-ROC | 57.07% | 83.70% | 88.04% | 94.02% | 91.30% | **95.11%** |
| F1 (SSc) | 90.72% | 91.67% | 92.13% | 85.71% | **97.87%** | 96.70% |
| F1 (Healthy) | 0.00% | 40.00% | 63.16% | 50.00% | **85.71%** | 82.35% |
| CV AUC | — | — | — | — | **99.75%** | 99.09% |

## Data Pipeline

```
GSE76809 (577 samples, 7 platforms)
  └─ Filter: GPL6480 platform only → 266 samples (228 SSc, 38 Healthy)
     └─ Train/Test split (80/20, stratified) → 212 train, 54 test
        └─ Variance filter (top 75%) → 6,782 genes
           └─ Mutual Information selection → 64 features (2^6 for amplitude encoding)
              └─ BorderlineSMOTE on training set → 364 samples (182:182)
                 └─ Unit-norm normalization (for amplitude encoding)
```

## Architecture Details

### Quantum Model — Dressed Circuit (PennyLane)
- **Qubits:** 6 (encodes 2^6 = 64 features via amplitude embedding)
- **Variational layers:** 4 (RY + RZ + RX rotations, circular CNOT entanglement)
- **Pre-net:** Linear(64 → 64) + Tanh (classical preprocessing)
- **Post-net:** Linear(6 → 16) + GELU + Linear(16 → 1) (classical postprocessing)
- **Optimizer:** Adam (LR=0.005) + Cosine annealing
- **Early stopping:** Patience=12 on AUC-ROC
- **Threshold:** Youden's J statistic
- **Parameters:** 4,361

### Classical Model — Residual MLP (PyTorch)
- **Architecture:** 64 → 256 → 128 → 64 → 32 → 1 with residual connections
- **Activation:** GELU + BatchNorm + Dropout (0.4)
- **Augmentation:** Mixup (α=0.2, first 80% of epochs)
- **Label smoothing:** 0.05
- **Optimizer:** AdamW (LR=0.001) + Cosine annealing with warm restarts
- **Early stopping:** Patience=15 on AUC-ROC
- **Threshold:** Youden's J statistic
- **Parameters:** 120,737

### Ensemble Model
- **Method 1:** Weighted average (grid search over quantum/classical weights)
- **Method 2:** Logistic regression stacking on OOF predictions
- **Best:** Weighted average with quantum_w=0.10, classical_w=0.90

## Files

| File | Description |
|------|-------------|
| `preprocess_gse76809.py` | MI feature selection (64), BorderlineSMOTE, 3-fold CV splits, unit-norm normalization |
| `model_quantum.py` | Dressed quantum circuit with amplitude encoding, threshold optimization |
| `model_classical.py` | Residual MLP with mixup, label smoothing, threshold optimization |
| `model_ensemble.py` | Weighted average + logistic regression stacking |
| `compare_models.py` | Full pipeline: preprocess → quantum → classical → ensemble → 3-fold CV → summary |
| `results/comparison.json` | All metrics in machine-readable format |

## Key Takeaways

- **Quantum model wins on all accuracy/F1 metrics** in both holdout and CV settings
- **CV confirms robustness:** Quantum achieves 99.75% mean AUC across 3 folds (±0.24%), with one perfect fold
- **Amplitude encoding is highly effective:** 64 features mapped to 6 qubits (vs 8-10 qubits in v02) with better results
- **Dressed circuit architecture** (classical pre/post layers around quantum circuit) significantly boosts performance
- **Ensemble slightly improves AUC** over individual models on the holdout set (95.65% vs 95.11% classical)
- **Quantum model uses 28× fewer parameters** than classical (4,361 vs 120,737) while achieving higher accuracy

## Environment

- Python 3.10.20 (conda env: GEO)
- PennyLane 0.42.3
- PyTorch 2.11.0
- scikit-learn 1.7.2
- imbalanced-learn 0.14.1

## Run

```bash
conda activate GEO
cd GSE76809Examples/v03
python compare_models.py
```
