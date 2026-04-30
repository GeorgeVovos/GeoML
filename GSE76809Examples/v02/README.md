# GSE76809 v02 — Improved Quantum vs Classical Classification

## Overview

Improved SSc (systemic sclerosis) vs Healthy classification on [GSE76809](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76809) using PennyLane quantum circuits and a PyTorch residual MLP. This version applies several improvements over v01 to address class imbalance and feature quality.

## Key Improvements over v01

| Area | v01 | v02 |
|------|-----|-----|
| Feature selection | Variance filter → top 50 | Mutual Information → top 100 |
| Class balancing | None (228 SSc vs 38 Healthy) | SMOTE oversampling (182:182) |
| Loss function | Standard BCE | Class-weighted BCE |
| Quantum circuit | 6 qubits, 2 layers, angle embedding | 8 qubits, 3 layers, data re-uploading |
| Classical model | 2-layer MLP, ReLU | Residual MLP (4 layers), GELU, BatchNorm |
| Normalization | Standard scaling | Quantile normalization + standard scaling |
| LR scheduling | None | Cosine annealing (quantum), warm restarts (classical) |
| Early stopping | None | Patience=10 on AUC-ROC |

## Results

| Metric | Quantum VQC v02 | Classical MLP v02 | Winner |
|--------|-----------------|-------------------|--------|
| **Accuracy** | **87.04%** | 77.78% | Quantum |
| **F1 (SSc)** | **92.13%** | 85.71% | Quantum |
| **F1 (Healthy)** | **63.16%** | 50.00% | Quantum |
| **AUC-ROC** | 88.04% | **94.02%** | Classical |
| Train Time | 638.3s | **1.5s** | Classical |
| Parameters | 3,729 | 139,169 | Quantum (fewer) |

### Improvement over v01 Baseline

| Metric | Quantum v01 → v02 | Classical v01 → v02 |
|--------|-------------------|---------------------|
| AUC-ROC | 0.5707 → **0.8804** (+0.31) | 0.8370 → **0.9402** (+0.10) |
| Accuracy | 0.8333 → **0.8704** (+0.04) | 0.8519 → 0.7778 (−0.07) |
| F1 (SSc) | 0.9072 → **0.9213** (+0.01) | 0.9167 → 0.8571 (−0.06) |

**Key takeaways:**
- Quantum AUC improved dramatically (+31 points) — SMOTE + class weighting fixed the minority class collapse seen in v01
- Quantum model now **wins on accuracy and both F1 scores**, outperforming the classical model on these metrics
- Classical model achieves higher AUC (better probability ranking) but lower accuracy due to threshold effects
- Both models show improved minority-class recall, with Healthy F1 scores meaningful (vs 0.0 in v01 quantum)

## Data Pipeline

```
GSE76809 (577 samples, 7 platforms)
  └─ Filter: GPL6480 platform only → 266 samples (228 SSc, 38 Healthy)
     └─ Train/Test split (80/20, stratified) → 212 train, 54 test
        └─ Variance filter (top 75%) → 6,782 genes
           └─ Mutual Information selection → 100 features
              └─ SMOTE on training set → 364 samples (182:182)
                 └─ Quantile normalization + Standard scaling
```

## Architecture Details

### Quantum Model (PennyLane)
- **Qubits:** 8
- **Circuit:** Data re-uploading (features re-encoded every layer)
- **Layers:** 3 (RY + RZ rotations + CNOT entanglement per layer)
- **Measurement:** Expectation values of PauliZ on all qubits → linear → sigmoid
- **Optimizer:** Adam (lr=0.008) + cosine annealing
- **Parameters:** 3,729

### Classical Model (PyTorch)
- **Architecture:** 100 → 256 → 128 → 64 → 32 → 1 (residual connections)
- **Activation:** GELU
- **Regularization:** BatchNorm + Dropout (0.4)
- **Optimizer:** AdamW (lr=0.001) + cosine warm restarts
- **Parameters:** 139,169

## Files

| File | Description |
|------|-------------|
| `preprocess_gse76809.py` | MI feature selection, SMOTE, quantile normalization |
| `model_quantum.py` | Data re-uploading VQC with class weights and LR scheduling |
| `model_classical.py` | Residual MLP with GELU, BatchNorm, warm restarts |
| `compare_models.py` | Orchestrates pipeline and generates comparison |
| `results/comparison.json` | Full metrics in JSON format |

## Reproduction

```bash
conda activate GEO
cd GSE76809Examples/v02
python compare_models.py
```

Requires: `pennylane`, `torch`, `scikit-learn`, `imbalanced-learn`, `pandas`, `numpy`

## Environment

- Python 3.10.20, PennyLane 0.42.3, PyTorch 2.11.0
- scikit-learn 1.7.2, imbalanced-learn 0.14.1
- CPU execution (no GPU/QPU)
