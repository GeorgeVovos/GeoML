# GSE76809 Classification: Quantum vs Classical Neural Networks (v01)

## Overview

This experiment compares a **Variational Quantum Circuit (VQC)** built with [PennyLane](https://pennylane.ai/) + PyTorch against a **Classical Multi-Layer Perceptron (MLP)** in PyTorch for binary classification of **Systemic Sclerosis (SSc) vs Healthy** patients using gene expression data from [GSE76809](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76809).

---

## Dataset: GSE76809

| Property | Value |
|----------|-------|
| **GEO Accession** | [GSE76809](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76809) |
| **Platform used** | GPL6480 (Agilent-014850 Whole Human Genome Microarray 4x44K) |
| **Total samples (all platforms)** | 577 |
| **Samples on GPL6480** | 280 |
| **Labeled samples (SSc + Healthy)** | 266 |
| **SSc samples** | 228 (85.7%) |
| **Healthy samples** | 38 (14.3%) |
| **Original probe count** | 171,146 |
| **Probes after NaN removal** | 9,043 |
| **Probes after variance filter** | 4,521 |
| **Final features selected** | 50 (top variance) |

### Labeling Strategy

Samples were labeled using metadata fields (`title`, `disease state`, `sample type`, `case/control`):

- **SSc (label=1):** Titles containing `dSSc`, `lSSc`, `RIT`, `NONRIT`, `Morph`; disease states with "sclerosis"/"scleroderma"; case/control = "case"
- **Healthy (label=0):** Titles starting with `Nor`, `NL`; disease states containing "normal"; sample type "Normal Control"; case/control = "control"
- **Excluded (label=-1):** Other diseases (IPAH, IPF, GERD, PPH) — 43 samples excluded to keep the task as pure SSc vs Healthy

### Train/Test Split

| Set | Total | SSc | Healthy |
|-----|-------|-----|---------|
| **Train** | 212 | 182 (85.8%) | 30 (14.2%) |
| **Test** | 54 | 46 (85.2%) | 8 (14.8%) |

- Split: 80/20 stratified
- Random seed: 42

---

## Preprocessing Pipeline

1. **Load** expression matrix (171,146 probes × 577 samples) and metadata
2. **Filter** to GPL6480 platform only (ensures probe consistency)
3. **Label** samples as SSc (1), Healthy (0), or Excluded (-1)
4. **Drop** probes with any NaN values → 9,043 remaining
5. **Log2 transform** with offset for zero/negative values
6. **Variance filter** — remove bottom 50% lowest-variance probes → 4,521 remaining
7. **Feature selection** — top 50 probes by variance
8. **StandardScaler** — zero mean, unit variance normalization
9. **Save** as numpy arrays (`X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`)

---

## Model 1: Quantum Variational Circuit (PennyLane + PyTorch)

### Architecture

```
Input (50 features)
    │
    ▼
Linear(50 → 8) + Tanh()          ← Classical pre-processing layer
    │
    ▼
┌──────────────────────────────┐
│   QUANTUM CIRCUIT (8 qubits) │
│                              │
│   Encoding: RX(x_i * π)     │  ← Angle encoding on each qubit
│                              │
│   Layer 1-3 (repeated):      │
│     RY(θ) + RZ(φ) per qubit │  ← Parameterized rotations
│     CNOT ring entanglement   │  ← Circular entanglement
│                              │
│   Measurement: ⟨Z⟩ per qubit│  ← 8 expectation values
└──────────────────────────────┘
    │
    ▼
Linear(8 → 1) + Sigmoid()        ← Classical post-processing
    │
    ▼
Output: P(SSc)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Qubits | 8 |
| Variational layers | 3 |
| Circuit parameters | 48 (3 layers × 8 qubits × 2 rotations) |
| Pre-net parameters | 408 (50×8 + 8 bias) |
| Post-net parameters | 9 (8×1 + 1 bias) |
| Total trainable parameters | ~465 |
| Optimizer | Adam |
| Learning rate | 0.005 |
| Loss function | Binary Cross-Entropy |
| Batch size | 16 |
| Epochs | 30 |
| Quantum device | `default.qubit` (state-vector simulator) |
| Differentiation | Backpropagation |

### Training Progression

| Epoch | Train Loss | Train Acc | Test Acc |
|-------|-----------|-----------|----------|
| 1 | 0.5365 | 85.85% | 85.19% |
| 5 | 0.4264 | 80.19% | 85.19% |
| 10 | 0.3343 | 87.74% | 85.19% |
| 15 | 0.3015 | 87.26% | 64.81% |
| 20 | 0.3060 | 87.74% | 81.48% |
| 25 | 0.2620 | 89.15% | 87.04% |
| 30 | 0.2525 | 87.26% | 83.33% |

### Observations

- Training shows loss reduction from 0.54 → 0.25 over 30 epochs
- Test accuracy fluctuates significantly (64%–87%), indicating the model is sensitive to the quantum parameter landscape
- The quantum model achieves reasonable train accuracy but struggles with generalization, particularly on the minority class

---

## Model 2: Classical Multi-Layer Perceptron (PyTorch)

### Architecture

```
Input (50 features)
    │
    ▼
Linear(50 → 128) + BatchNorm + ReLU + Dropout(0.3)
    │
    ▼
Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.3)
    │
    ▼
Linear(64 → 32) + BatchNorm + ReLU + Dropout(0.3)
    │
    ▼
Linear(32 → 1) + Sigmoid()
    │
    ▼
Output: P(SSc)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Hidden layer sizes | 128 → 64 → 32 |
| Activation | ReLU |
| Normalization | BatchNorm1d per hidden layer |
| Regularization | Dropout (p=0.3) |
| Total trainable parameters | 17,345 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Binary Cross-Entropy |
| Batch size | 32 |
| Epochs | 50 |

### Training Progression

| Epoch | Train Loss | Train Acc | Test Acc |
|-------|-----------|-----------|----------|
| 1 | 0.6431 | 68.40% | 85.19% |
| 10 | 0.4097 | 86.32% | 88.89% |
| 20 | 0.3125 | 86.79% | 88.89% |
| 30 | 0.2879 | 89.15% | 83.33% |
| 40 | 0.2544 | 88.68% | 85.19% |
| 50 | 0.2586 | 89.15% | 85.19% |

### Observations

- Much faster convergence — reaches ~87% train accuracy by epoch 10
- More stable test accuracy compared to quantum (83%–89% range)
- BatchNorm + Dropout provide effective regularization for this small dataset
- Peak test performance of 88.89% observed around epoch 10-20 (potential for early stopping)

---

## Final Results Comparison

### Performance Metrics (Test Set, n=54)

| Metric | Quantum VQC | Classical MLP | Δ (Classical − Quantum) |
|--------|-------------|---------------|-------------------------|
| **Accuracy** | 83.33% | 85.19% | +1.86% |
| **F1 Score** | 0.9072 | 0.9167 | +0.0095 |
| **AUC-ROC** | 0.5707 | 0.8370 | **+0.2663** |
| **Training Time** | 355.1 s | 1.2 s | **~296× faster** |

### Per-Class Classification Report

#### Quantum VQC
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Healthy | 0.33 | 0.12 | 0.18 | 8 |
| SSc | 0.86 | 0.96 | 0.91 | 46 |
| **Weighted Avg** | **0.78** | **0.83** | **0.80** | **54** |

#### Classical MLP
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Healthy | 0.50 | 0.25 | 0.33 | 8 |
| SSc | 0.88 | 0.96 | 0.92 | 46 |
| **Weighted Avg** | **0.82** | **0.85** | **0.83** | **54** |

---

## Analysis & Discussion

### Why Classical Outperforms Quantum Here

1. **Dataset size** — With only 212 training samples, the classical MLP has sufficient capacity (17K params) without overfitting thanks to BatchNorm + Dropout. The quantum model's expressibility is limited by 8 qubits.

2. **AUC-ROC gap** — The quantum model's AUC-ROC (0.57) is barely above random (0.50), meaning it struggles to rank healthy samples correctly. The classical model (0.84) has much better discrimination ability.

3. **Training instability** — Quantum circuits suffer from barren plateaus and parameter initialization sensitivity. The test accuracy fluctuation (65%–87%) reflects this.

4. **Class imbalance** — Both models predict the majority class (SSc) well but struggle with minority class (Healthy, n=8 in test). The quantum model essentially learns to predict SSc for everything (recall=0.12 for Healthy).

5. **Speed** — Quantum simulation on classical hardware is exponentially expensive. 8 qubits ≈ 256-dimensional Hilbert space, but each forward pass requires full state-vector evolution per sample.

### Quantum Model Strengths (Theoretical)

- **Parameter efficiency** — 465 trainable parameters vs 17,345 (37× fewer)
- **Expressibility** — Quantum circuits can represent correlations that may require exponentially more classical parameters in certain data distributions
- **Future hardware** — On real quantum hardware, circuit execution would be O(depth) regardless of qubit count

### Limitations of This Experiment

- **Simulation overhead** — Running on `default.qubit` simulator; real QPU would change the timing picture
- **Small test set** — Only 8 healthy samples in test makes per-class metrics noisy
- **No class weighting** — Neither model uses class weights or oversampling to address the 6:1 imbalance
- **Feature count** — 50 features projected to 8 qubits may lose too much information

---

## Potential Improvements (v02)

| Improvement | Expected Impact |
|-------------|----------------|
| Class-weighted loss (`pos_weight`) | Better minority-class recall |
| More qubits (10-12) | Higher quantum expressibility |
| Data re-encoding (multiple encoding layers) | Richer feature embedding |
| Early stopping on validation AUC | Prevent overfitting |
| Feature selection by mutual information | More discriminative features |
| SMOTE oversampling for Healthy class | Balanced training |
| Cross-validation (5-fold) | More robust metrics |
| Amplitude encoding | Encode more features per qubit |

---

## How to Run

### Prerequisites

```bash
conda activate GEO
pip install pennylane pennylane-lightning torch scikit-learn pandas numpy
```

### Run Full Comparison

```bash
cd GSE76809Examples/v01
python compare_models.py
```

### Run Individual Models

```bash
# Preprocessing only
python preprocess_gse76809.py --n-features 50 --platform GPL6480

# Quantum model
python model_quantum.py --n-qubits 8 --n-layers 3 --epochs 30 --lr 0.005 --batch-size 16

# Classical model
python model_classical.py --hidden 128 64 32 --dropout 0.3 --epochs 50 --lr 0.001 --batch-size 32
```

### CLI Arguments

| Script | Argument | Default | Description |
|--------|----------|---------|-------------|
| `preprocess_gse76809.py` | `--n-features` | 50 | Number of top genes to select |
| | `--platform` | GPL6480 | GEO microarray platform |
| | `--test-size` | 0.2 | Fraction for test split |
| | `--selection` | variance | `variance` or `mutual_info` |
| `model_quantum.py` | `--n-qubits` | 8 | Number of qubits |
| | `--n-layers` | 3 | Variational circuit depth |
| | `--epochs` | 30 | Training epochs |
| | `--lr` | 0.005 | Learning rate |
| | `--batch-size` | 16 | Batch size |
| `model_classical.py` | `--hidden` | 128 64 32 | Hidden layer sizes |
| | `--dropout` | 0.3 | Dropout probability |
| | `--epochs` | 50 | Training epochs |
| | `--lr` | 0.001 | Learning rate |
| | `--batch-size` | 32 | Batch size |

---

## File Structure

```
GSE76809Examples/v01/
├── README.md                  ← This file
├── preprocess_gse76809.py     ← Data loading, labeling, feature selection
├── model_quantum.py           ← PennyLane VQC + PyTorch hybrid classifier
├── model_classical.py         ← PyTorch MLP classifier
├── compare_models.py          ← Orchestrator: preprocess → train both → compare
└── results/
    ├── comparison.json        ← Combined metrics from both models
    ├── quantum_results.json   ← Quantum model final metrics
    ├── quantum_model.pt       ← Saved quantum model weights
    ├── quantum_history.npy    ← Per-epoch training metrics (quantum)
    ├── classical_results.json ← Classical model final metrics
    ├── classical_model.pt     ← Saved classical model weights
    └── classical_history.npy  ← Per-epoch training metrics (classical)
```

---

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.10.20 |
| PennyLane | 0.42.3 |
| PennyLane-Lightning | 0.42.0 |
| PyTorch | 2.11.0 |
| scikit-learn | 1.7.2 |
| pandas | 2.3.3 |
| NumPy | 2.2.6 |
| OS | Windows |
| Hardware | CPU (no GPU/QPU) |

---

## References

- **Dataset:** Milano et al. — GSE76809, a compendium of systemic sclerosis gene expression studies
- **PennyLane:** Bergholm et al., "PennyLane: Automatic differentiation of hybrid quantum-classical computations" (2018)
- **VQC for classification:** Schuld et al., "Circuit-centric quantum classifiers" (2020)
