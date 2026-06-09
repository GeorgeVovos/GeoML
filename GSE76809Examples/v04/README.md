# v04: Demonstrating Quantum Advantage in SSc Classification

## What This Version Does

This version is specifically designed to **show scenarios where quantum machine learning outperforms classical approaches**. It improves on v03 by:

1. Adding a **quantum kernel method** (completely different approach from variational circuits)
2. Enhancing the VQC with **multi-basis measurements** and **ZZ interactions**
3. Including **XGBoost** (the strongest off-the-shelf classical method for tabular data)
4. Using **5-fold cross-validation** for statistically robust comparison
5. Running **learning curves** to show quantum advantage at small sample sizes
6. Performing **paired t-tests** for statistical significance

> **Methodology update (May 2026):** the pipeline was overhauled to remove
> several leakage and pairing bugs that inflated earlier numbers. See the
> [Methodology fixes](#methodology-fixes-may-2026) section below for details.
> All numbers in the [Results](#results) section are from the corrected run.

---

## Why Quantum Might Beat Classical Here

### The Dataset's Properties Favor Quantum

Our dataset (GSE76809) has properties that align with quantum computing's strengths:

1. **Small sample size (266 samples)**: Quantum models have fewer parameters, so they're less likely to overfit on small datasets
2. **High dimensionality (64 features)**: Quantum feature maps can represent exponentially many feature interactions using only a few qubits
3. **Complex gene-gene interactions**: Quantum entanglement naturally captures correlated features that classical models handle sequentially

### Think of It This Way

Imagine you have 64 genes and want to check if every possible pair interacts. That's 64×63/2 = 2,016 pairs. A classical model needs explicit parameters for each. A quantum model with 6 entangled qubits implicitly considers all combinations simultaneously in its 2⁶ = 64 dimensional state space.

---

## The Four Models

### 1. Quantum VQC (Enhanced Dressed Circuit)

**What's new vs v03:**

| Feature | v03 | v04 |
|---------|-----|-----|
| Variational layers | 4 | 6 |
| Measurement | PauliZ only (6 outputs) | PauliX + PauliY + PauliZ (18 outputs) |
| Entanglement | Ring CNOT only | Alternating ring + ladder + long-range |
| Interactions | None after encoding | ZZ interactions (IsingZZ gates) |
| Post-net | 6→16→1 | 18→32→16→1 |
| LR schedule | Cosine only | Warm-up (5 epochs) then cosine |
| Gradient | No clipping | Clipped at norm 1.0 |

#### Multi-Basis Measurement Explained

In v03, we only measured each qubit in the Z-basis (spin up/down). That's like looking at data from only one angle.

In v04, we measure in **three bases** (X, Y, Z). Think of it like photographing an object from the front, side, and top — you get 3× more information about its shape.

- **PauliZ**: "Is the qubit more |0⟩ or |1⟩?" (computational basis)
- **PauliX**: "Is the qubit more |+⟩ or |-⟩?" (superposition basis)
- **PauliY**: "Is the qubit more |+i⟩ or |-i⟩?" (phase basis)

This gives us 18 features from 6 qubits, capturing information that Z-only measurement misses.

#### ZZ Interactions Explained

After encoding the gene expression data into qubits, we apply **IsingZZ gates** between neighboring qubits:

```
IsingZZ(θ) = exp(-i θ/2 Z⊗Z)
```

This creates **data-dependent correlations** between qubits. The θ parameter is learned during training, allowing the circuit to emphasize or suppress different qubit-qubit interactions.

**Analogy**: Imagine each qubit holds information about a gene. The ZZ gate is like asking "do these two genes tend to be active together?" — it encodes their correlation into the quantum state.

### 2. Quantum Kernel SVM (NEW in v04)

This is a completely different quantum approach:

#### How Classical Kernels Work

A kernel function measures similarity between two data points. The kernel "trick" maps data into a high-dimensional space where it becomes linearly separable, without explicitly computing the mapping.

Example: Two classes arranged in a circle pattern can't be separated by a line in 2D, but map them to 3D (using r² = x² + y²) and they become separable by a plane.

#### How the Quantum Kernel Works

Instead of a classical mapping, we use a **quantum circuit** (the ZZ feature map) to embed each data point as a quantum state:

```
|φ(x)⟩ = U(x)|0...0⟩
```

The kernel value between two samples is the **overlap of their quantum states**:

```
k(x₁, x₂) = |⟨φ(x₁)|φ(x₂)⟩|²
```

This measures: "How similar do these two gene expression profiles look in quantum Hilbert space?"

#### The ZZ Feature Map

For each data point x with 8 features (after PCA from 64):

1. **Hadamard**: Put all qubits in superposition
2. **RZ(xᵢ)**: Encode each feature as a phase rotation
3. **ZZ(xᵢ·xⱼ)**: Between neighboring qubits, apply a gate whose angle is the **product** of two features

Step 3 is crucial — it creates **pairwise feature interactions** within the quantum state. Classical RBF kernels only see individual features; the ZZ feature map sees all neighboring pairs simultaneously.

This is repeated twice (depth=2) for more expressiveness.

#### Why This Works for Gene Data

Gene expression is inherently about **interactions** — one gene activating another, regulatory networks, pathway cascades. The ZZ feature map naturally encodes these pairwise relationships into quantum entanglement.

#### Practical Details

- Features are projected from 64 → 8 using PCA (8 qubits)
- PCA features scaled to [0, π] for optimal angle encoding
- Uses **pre-SMOTE data** (212 samples) to keep kernel matrix manageable
- SVM handles class imbalance via `class_weight='balanced'`
- No parameter training needed — the kernel is **fixed** (no barren plateaus!)
- Total kernel evaluations: ~22,000 (takes ~2-5 minutes)

### 3. Classical MLP (Same as v03)

The residual MLP from v03 serves as the neural network baseline:
- Architecture: 256→128→64→32→1 with residual connections
- Mixup augmentation (α=0.2) for first 80% of epochs
- Label smoothing (0.05)
- BatchNorm + GELU + Dropout(0.4)
- ~120K parameters

### 4. Classical XGBoost (NEW in v04)

XGBoost is the gold standard for tabular machine learning:
- 200 trees, max depth 4
- Learning rate 0.05 with subsampling
- Handles imbalance via `scale_pos_weight`
- Uses **pre-SMOTE data** (it handles imbalance natively)
- L1 + L2 regularization

If quantum can beat XGBoost, that's a meaningful result.

---

## Evaluation Strategy

### 5-Fold Cross-Validation

Instead of a single train/test split (which is noisy), we split data 5 ways:

```
Fold 1: [TEST] [train] [train] [train] [train]
Fold 2: [train] [TEST] [train] [train] [train]
Fold 3: [train] [train] [TEST] [train] [train]
Fold 4: [train] [train] [train] [TEST] [train]
Fold 5: [train] [train] [train] [train] [TEST]
```

Each fold trains on 80% and tests on 20%. We report mean ± std across all 5 folds.

### Learning Curves

We train at 25%, 50%, 75%, and 100% of available training data. This reveals:
- **Quantum advantage at small data**: If quantum models maintain high AUC at 25% data while classical models drop significantly, that demonstrates quantum's sample efficiency
- **Convergence behavior**: How quickly each model reaches its peak performance

### Statistical Tests

We use a **paired t-test** on the 5 fold AUC scores:
- H₀: "The two models have the same expected AUC"
- If p < 0.05: The difference is statistically significant (not due to random chance)
- The "paired" aspect accounts for fold-to-fold variation

---

## Running This Example

### Prerequisites

```bash
conda activate GEO
pip install xgboost  # Only new dependency vs v03
```

### Run

```bash
cd GSE76809Examples/v04
python compare_models.py
```

### Expected Runtime

| Component | Approximate Time |
|-----------|-----------------|
| Preprocessing | ~30 seconds |
| Holdout evaluation (4 models) | ~10 minutes |
| 5-fold CV (4 models × 5 folds, per-fold preprocessing + per-fold SMOTE) | ~70 minutes |
| Learning curves (4 models × 4 fractions, per-subsample refit) | ~40 minutes |
| **Total** | **~2 hours** |

The quantum kernel costs ~2–3 min per fold for the Gram-matrix computation. The
VQC is the slowest per-fold (~5–10 min) due to per-sample circuit evaluation.

---

## File Structure

```
v04/
├── preprocess_gse76809.py    # Feature selection + SMOTE + PCA + 5-fold splits
├── model_quantum_vqc.py      # Enhanced VQC with multi-basis measurement
├── model_quantum_kernel.py   # ZZ feature map quantum kernel + SVM
├── model_classical_mlp.py    # Residual MLP baseline
├── model_classical_xgb.py    # XGBoost baseline
├── compare_models.py         # Full pipeline: holdout + CV + learning curves + stats
├── results/                  # Output directory for JSON results
└── README.md                 # This file
```

---

## Key Concepts for Beginners

### What is a "Kernel" in Machine Learning?

A kernel is a function that measures similarity between two data points without explicitly mapping them to a higher-dimensional space.

**Analogy**: Imagine comparing two recipes. Instead of listing every single molecular compound in each dish (expensive!), you taste both and say "these taste 80% similar." The kernel gives you the similarity score without computing the full representation.

The **kernel trick** works because SVM only needs similarity scores between pairs of samples, not their actual positions in high-dimensional space. This makes it computationally efficient.

### What is "Quantum Advantage"?

Quantum advantage means a quantum algorithm achieves better results than the best known classical algorithm for the same problem. There are different levels:

1. **Exponential speedup**: Quantum is exponentially faster (e.g., Shor's algorithm for factoring)
2. **Polynomial speedup**: Quantum is polynomially faster (e.g., Grover's search)
3. **Better accuracy**: Quantum achieves better prediction quality for the same amount of data

In this example, we target **#3**: showing that quantum models can classify SSc more accurately than classical models, especially when training data is limited.

### Why Small Datasets Favor Quantum

Classical deep learning (MLPs with 120K parameters) needs lots of data to avoid overfitting. When you have only 212 samples:

- **Classical MLP**: Has 120K parameters for 212 samples → 567 parameters per sample → high overfitting risk
- **Quantum VQC**: Has ~5K parameters for 212 samples → 24 parameters per sample → lower overfitting risk
- **Quantum Kernel**: Has 0 trainable parameters → zero overfitting in the feature map itself

The quantum kernel is particularly interesting because it cannot overfit at the feature map level — all overfitting control is in the SVM's regularization parameter C, which is well-understood.

### What are IsingZZ Gates?

The IsingZZ gate is a two-qubit gate that creates correlations between qubits proportional to a parameter θ:

```
IsingZZ(θ)|00⟩ = e^{-iθ/2}|00⟩
IsingZZ(θ)|01⟩ = e^{+iθ/2}|01⟩
IsingZZ(θ)|10⟩ = e^{+iθ/2}|10⟩
IsingZZ(θ)|11⟩ = e^{-iθ/2}|11⟩
```

Notice: states where both qubits agree (00, 11) get phase e^{-iθ/2}, while disagreeing states (01, 10) get e^{+iθ/2}. This preferentially correlates qubits that are in the same state — exactly what you want for capturing feature correlations.

### What is PCA?

Principal Component Analysis (PCA) finds the directions in the data with the most variance. It's like rotating your coordinate system so axis 1 has the most spread, axis 2 has the next most, etc.

We use PCA to reduce 64 features → 8 for the quantum kernel because:
1. 8 qubits is computationally feasible for kernel evaluation
2. PCA preserves the most informative directions
3. The top 8 PCA components typically capture 70-90% of the variance

---

## Understanding the Results

After running, check `results/v04_comparison_results.json` for:

### Holdout Results
Single train/test split performance — quick sanity check.

### Cross-Validation Results
The most reliable comparison. Look for:
- **Higher mean AUC** for quantum models → quantum advantage
- **Lower std** for quantum models → more consistent performance

### Learning Curves
If quantum AUC at 25% data > classical AUC at 25% data, that demonstrates **sample efficiency advantage**.

### Statistical Tests
If p < 0.05 for "VQC vs XGBoost" or "Kernel vs XGBoost", the quantum advantage is statistically significant.

---

## Results

Results below are from the corrected pipeline (see
[Methodology fixes](#methodology-fixes-may-2026)). All numbers come from
`results/v04_comparison_results.json`.

### Cross-Validation AUC (5-fold) — Most Reliable Comparison

| Model | Mean AUC | Std | Min | Max |
|-------|----------|-----|-----|-----|
| **Quantum VQC** | **0.9827** | ±0.0137 | 0.9630 | 1.0000 |
| Classical MLP | 0.9747 | ±0.0278 | 0.9324 | 1.0000 |
| Classical XGBoost | 0.9202 | ±0.0237 | 0.8843 | 0.9505 |
| Quantum Kernel | 0.8909 | ±0.0878 | 0.7342 | 1.0000 |

### Holdout Results (Single Train/Test Split)

| Model | Accuracy | AUC | F1(SSc) | F1(Healthy) |
|-------|----------|-----|---------|-------------|
| Classical MLP | 0.9074 | **0.9565** | 0.9438 | 0.7368 |
| Classical XGBoost | 0.8704 | 0.9158 | 0.9195 | 0.6667 |
| **Quantum VQC** | **0.9630** | 0.8967 | 0.9787 | 0.8571 |
| Quantum Kernel | 0.8889 | 0.7364 | 0.9362 | 0.5714 |

### Learning Curves (AUC at Different Data Fractions)

Feature selection / scaling / PCA are now refit on each subsample, and SMOTE is
applied **after** subsampling — so each row reflects what each model actually
learns from that much real data.

| Model | 25% data | 50% data | 75% data | 100% data |
|-------|----------|----------|----------|-----------|
| Quantum VQC | 0.8560 | 0.8859 | 0.9266 | 0.8723 |
| Classical MLP | 0.8234 | 0.8859 | 0.9565 | 0.9592 |
| Classical XGBoost | 0.5000 | 0.6739 | 0.8370 | 0.9212 |
| Quantum Kernel | 0.5027 | 0.8859 | 0.5625 | 0.7364 |

### Statistical Significance (Paired t-test on 5-fold AUCs)

| Comparison | t-statistic | p-value | Significant? |
|-----------|-------------|---------|--------------|
| **VQC vs XGBoost** | 4.12 | **0.0147** | **Yes** |
| VQC vs MLP | 0.58 | 0.5909 | No |
| Kernel vs MLP | -2.22 | 0.0910 | No |
| Kernel vs XGBoost | -0.59 | 0.5892 | No |

### Key Findings

1. **Quantum VQC has the highest mean CV AUC** (0.9827 ± 0.0137). With per-fold
   SMOTE and per-fold feature engineering, the previous near-perfect 0.998 score
   collapsed to a still-strong but realistic 0.983.

2. **Quantum VQC significantly outperforms XGBoost** (p=0.0147). XGBoost is the
   gold standard for tabular data, so beating it on a per-fold paired test is a
   meaningful result — and it now uses an honest fold construction.

3. **Quantum VQC vs Classical MLP is statistically a tie** (p=0.59). Both reach
   ~0.97–0.98 mean CV AUC, but the VQC does it with **5,619 parameters** vs the
   MLP's **120,737** — about 21× fewer parameters.

4. **Sample-efficiency advantage at 25% data**: VQC AUC = 0.86, MLP = 0.82,
   XGBoost = 0.50 (random). XGBoost completely collapses on 53 training samples,
   while the quantum and parameter-light models still discriminate.

5. **Best holdout AUC is the MLP** (0.9565). Holdout is one realization of an
   80/20 split (n=54 test); the per-fold CV estimate is more reliable for
   ranking models, and there VQC > MLP > XGBoost > Kernel.

6. **Quantum Kernel underperformed and is unstable** (CV std = 0.088, LC values
   bounce between 0.50 and 0.89). The aggressive PCA reduction to 8 qubits and
   fixed ZZ feature map are likely discarding discriminative information.

7. **Total runtime**: ~119 minutes on CPU simulation. The VQC dominates runtime
   due to per-sample circuit evaluation; the quantum kernel is second because of
   the O(n²) Gram-matrix computation.

---

## Methodology fixes (May 2026)

The earlier version of this pipeline contained several leakage and pairing bugs
that inflated CV AUCs (VQC was reported at 0.998). The current code fixes:

- **SMOTE-before-CV leakage** — synthetic minority samples generated from the
  full training set were being used in both train and validation folds. SMOTE
  is now applied *inside* each CV fold, on the training side only, after the
  split.
- **Misaligned fold pairing** — different models were getting different folds.
  A single `StratifiedKFold` is now built once on the raw training set and
  reused for every model, so paired t-tests are honest.
- **Feature-engineering leakage in learning curves** — mutual-information
  feature selection, `StandardScaler`, PCA, and `MinMaxScaler` were fit on the
  full training set and then applied to subsampled folds. They are now refit on
  each subsample via `build_fold_features()`.
- **Subsampling SMOTE'd data in learning curves** — "25% of training data"
  meant 25% of the SMOTE-augmented matrix, so small-data points were dominated
  by synthetic samples. The pipeline now subsamples *raw* rows first, then
  applies SMOTE per subsample.

Results stored in `results/v04_comparison_results.json` reflect the post-fix
run.

---

## Improvements Over v03

| Aspect | v03 | v04 |
|--------|-----|-----|
| Quantum approaches | 1 (VQC) | 2 (VQC + Kernel) |
| Classical baselines | 1 (MLP) | 2 (MLP + XGBoost) |
| CV folds | 3 | 5 |
| Statistical testing | None | Paired t-test |
| Learning curves | None | 4 data fractions |
| VQC measurements | PauliZ only (6) | X+Y+Z (18) |
| Entanglement | Ring CNOT | Ring + ladder + long-range + ZZ |
| Gradient handling | No clipping | Clipped + warm-up |
