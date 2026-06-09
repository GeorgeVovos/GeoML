# GSE76809 Classification: Quantum vs Classical Neural Networks (v01 — Baseline)

## What This Project Does

This project answers a simple question: **can a quantum computer help diagnose disease from blood samples?**

We take gene expression data from real patients — some with **Systemic Sclerosis (SSc)**, an autoimmune disease that hardens skin and organs, and some **Healthy** — and build two machine learning models to classify them:

1. **A Quantum Model** — uses a simulated quantum computer (via [PennyLane](https://pennylane.ai/))
2. **A Classical Model** — a traditional neural network (via [PyTorch](https://pytorch.org/))

We then compare: which one is better at telling sick patients from healthy ones?

> **Spoiler for v01:** The classical model wins, but not by much. Both models have fundamental limitations we fix in v02 and v03.

---

## Concepts You Need to Know

### What is Gene Expression Data?

Every cell in your body contains the same DNA, but different cells "turn on" different genes at different levels. **Gene expression** measures how active each gene is — think of it as a snapshot of which genes are "loud" or "quiet" in a particular tissue.

A **microarray** chip can measure thousands of genes simultaneously from a single blood sample. The result is a table where:
- Each **row** = one patient sample
- Each **column** = one gene's activity level (a number)
- The **label** = whether that patient has the disease

### What is Machine Learning Classification?

**Classification** = teaching a computer to sort things into categories based on examples.

We show the model many examples of "here are the gene levels for a sick patient" and "here are the gene levels for a healthy patient." The model learns to find patterns that distinguish the two groups, then uses those patterns to classify new patients it hasn't seen before.

This is called **supervised learning** because we provide the correct answers ("labels") during training.

### What is a Quantum Computer? (Simplified)

A regular computer stores information as **bits** — each bit is either 0 or 1. A quantum computer uses **qubits** that can be in a **superposition** — a combination of 0 and 1 at the same time, until you measure them.

Multiple qubits can be **entangled**, meaning they are correlated in ways impossible for classical bits — measuring one instantly tells you about another.

These properties theoretically let quantum computers explore many possibilities simultaneously. The hope is that quantum classifiers might detect subtle patterns in data that classical models miss.

> **Important:** We don't have a real quantum computer here. We're *simulating* one on a regular CPU using PennyLane. This means it runs slower than a real quantum chip would, but gives identical mathematical results.

### What is a Variational Quantum Circuit (VQC)?

A VQC is the quantum equivalent of a neural network. It works in four steps:

1. **Encode** the patient's data onto qubits (like feeding data into a neural network's input)
2. **Apply parameterized gates** — quantum operations with adjustable settings (like the "weights" in a neural network)
3. **Measure** the qubits to get numerical outputs (like a neural network's output layer)
4. **Adjust parameters** based on how wrong the prediction was (like backpropagation in classical ML)

The word "variational" means we **vary** (optimize) the gate parameters during training to make better predictions, just like adjusting weights in a neural net.

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

### Why This Dataset is Challenging

- **Extreme class imbalance** — 228 SSc vs only 38 Healthy (6:1 ratio). A model that always guesses "SSc" would score 86% accuracy just by cheating!
- **High dimensionality** — Thousands of genes but only 266 samples. More features than patients creates overfitting risk (the model memorizes noise instead of learning real patterns).
- **Small test set** — Only 8 healthy patients in the test set, making results noisy.

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

**Why 80/20?** This is a standard split. We need enough training data for the model to learn, but enough test data to reliably evaluate. With only 266 samples, 80/20 is a common compromise.

**Why stratified?** We maintain the same sick/healthy ratio in both sets. Without stratification, the test set might accidentally end up with 0 healthy patients, making evaluation impossible.

**Why random seed = 42?** A fixed seed makes the split identical every run. This means you and I get the same results (reproducibility). The number 42 is arbitrary — any number works.

---

## Preprocessing Pipeline — Step by Step

### Why Preprocess at All?

Raw gene data has 171,146 measurements per patient. Feeding all of these into a model would be:
- **Too slow** — especially for quantum circuits that process one feature at a time
- **Too noisy** — most genes have nothing to do with SSc
- **Overfitting-prone** — the model would memorize random noise instead of learning real disease patterns

So we aggressively reduce from 171,146 down to just 50 of the most informative genes.

### Pipeline Steps Explained

| Step | Action | Input → Output | Why? |
|------|--------|----------------|------|
| 1 | Load data | Raw GEO files → 171,146×577 matrix | Get the raw gene expression data |
| 2 | Filter to GPL6480 | 577 → 280 samples | Different platforms measure different genes; keep one for consistency |
| 3 | Label samples | Add SSc/Healthy/Excluded labels | Map metadata to training labels |
| 4 | Drop NaN probes | 171,146 → 9,043 probes | NaN = missing value — can't do math with missing numbers |
| 5 | Log2 transform | Gene values compressed | Raw values span 1–100,000. Log makes them manageable (0–17) |
| 6 | Variance filter | 9,043 → 4,521 probes | Genes that barely change across patients carry zero information |
| 7 | Top 50 by variance | 4,521 → 50 features | Keep only the most variable genes (most likely to differ between groups) |
| 8 | StandardScaler | Mean=0, StdDev=1 | Puts all genes on the same scale so no gene dominates by having bigger numbers |
| 9 | Save arrays | `.npy` files | Ready for models to load |

### What is StandardScaler?

StandardScaler transforms each gene so that across all patients:
- The **mean** (average) becomes 0
- The **standard deviation** (spread) becomes 1

**Why this matters:** If Gene A ranges from 0–100 and Gene B ranges from 0–0.01, the model would pay attention mostly to Gene A just because its numbers are bigger. Scaling fixes this so all genes get equal attention.

### What is Variance Filtering?

**Variance** measures how much a value varies across samples. A gene with high variance changes a lot between patients — it might be "on" in sick patients and "off" in healthy ones. A gene with near-zero variance is almost the same in everyone — useless for classification.

We keep the top 50% most variable genes, then narrow to the top 50.

> **Limitation (fixed in v02):** High variance doesn't guarantee disease-relevance. A gene might vary a lot for reasons unrelated to SSc. In v02, we use **Mutual Information** which directly measures relevance to the disease label.

---

## Model 1: Quantum Variational Circuit (PennyLane + PyTorch)

### How It Works — The Sandwich Analogy

Think of the quantum model as a three-layer sandwich:

1. **Classical bread (bottom)** — compress 50 gene values down to 8 numbers (one per qubit)
2. **Quantum filling** — the 8 numbers control qubits, which interact through quantum effects to discover patterns
3. **Classical bread (top)** — convert the quantum measurements into a single yes/no prediction

### Architecture Diagram

```
Input (50 gene expression values)
    │
    ▼
Linear(50 → 8) + Tanh()          ← CLASSICAL: Compress to qubit count
    │                                [Tanh squishes values to -1...+1 range]
    ▼
┌──────────────────────────────────────────────────────┐
│   QUANTUM CIRCUIT (8 qubits, simulated)              │
│                                                      │
│   Step 1 — ENCODING:                                 │
│     RX(x_i × π) on each qubit                       │
│     (Each data value becomes a qubit rotation)       │
│                                                      │
│   Step 2 — VARIATIONAL LAYERS (×3):                  │
│     RY(θ) + RZ(φ) on each qubit (tunable angles)    │
│     CNOT ring: qubit 0→1→2→3→4→5→6→7→0             │
│     (Creates quantum correlations between qubits)    │
│                                                      │
│   Step 3 — MEASUREMENT:                              │
│     Measure ⟨Z⟩ on each qubit → 8 numbers           │
│     (Collapse quantum state to classical values)     │
└──────────────────────────────────────────────────────┘
    │
    ▼
Linear(8 → 1) + Sigmoid()        ← CLASSICAL: Convert to probability
    │
    ▼
Output: P(SSc) — probability of disease (0.0 to 1.0)
```

### Understanding Each Component

#### Angle Encoding — How data gets into the quantum circuit

Each of the 8 compressed data values is converted to a rotation angle and applied to a qubit using an RX gate. If the value is 0, no rotation (qubit stays at |0⟩). If it's 1, the qubit rotates by π (180°) to |1⟩. Values in between create superpositions.

**Analogy:** Imagine 8 dials, each turned to a position based on one of your data values. That's angle encoding — your data becomes the "starting position" of the quantum system.

**Trade-off:** Simple and intuitive, but limited — you need one qubit per feature. With only 8 qubits, you can only encode 8 values directly (that's why we compress 50 → 8 first).

#### Variational Layers — The "learnable weights" of quantum ML

Each layer applies:
1. **RY and RZ rotations** with trainable angles (θ, φ) on each qubit — these are the quantum equivalent of neural network weights that get optimized during training
2. **CNOT gates** in a ring pattern — these create entanglement between neighboring qubits

**Why 3 layers?** Each layer adds more "expressivity" — the circuit can represent more complex functions. But too many layers on small datasets causes overfitting, and deeper circuits are prone to "barren plateaus" (explained below). 3 layers with 8 qubits gives 48 quantum parameters — a reasonable starting point.

#### CNOT Ring — Creating entanglement

CNOT (Controlled-NOT) gates connect qubit pairs so that one qubit's state influences another. In a ring pattern (0→1→2→...→7→0), information flows around all qubits.

**Why is this important?** Without entanglement, each qubit processes its input independently — you'd just have 8 separate tiny models running in parallel. Entanglement is what lets the quantum circuit discover *correlations between genes* — exactly what we want for finding disease patterns.

#### Measurement — Getting answers out of the quantum state

After the quantum operations, we measure each qubit's "Z expectation value" — a number between -1 and +1 that represents the average outcome if you measured that qubit many times. These 8 numbers are the quantum circuit's output, which then feeds into the final classical layer.

### Hyperparameters — Why These Values?

| Parameter | Value | Why This Value? |
|-----------|-------|-----------------|
| Qubits | 8 | More qubits = more powerful, but simulation cost doubles per qubit (2^n). 8 qubits = 256 amplitudes to track = manageable on a CPU |
| Layers | 3 | Enough to learn complex patterns, not so many it hits "barren plateaus" (vanishing gradients). Total params: 3×8×2 = 48 quantum angles |
| Learning rate | 0.005 | Higher than typical classical (0.001) because quantum gradients tend to be noisier — bigger steps help escape flat regions |
| Batch size | 16 | Quantum circuits process one sample at a time in simulation, so smaller batches = more frequent parameter updates |
| Epochs | 30 | Quantum training is slow (~12 sec/epoch) so we cap at 30 for practical reasons. More epochs don't help much given other limitations |
| Optimizer | Adam | Adapts learning rate per parameter — handles noisy quantum gradients well |
| Loss | BCE | Binary Cross-Entropy — standard loss function for "yes/no" classification. Measures how far the predicted probability is from the true label |
| Activation (pre-net) | Tanh | Squishes values to [-1, +1], which maps well to qubit rotation angles |

### Training Progression

| Epoch | Train Loss | Train Acc | Test Acc | What's Happening |
|-------|-----------|-----------|----------|------------------|
| 1 | 0.5365 | 85.85% | 85.19% | Just started — predicting majority class |
| 5 | 0.4264 | 80.19% | 85.19% | Still mostly guessing "SSc" |
| 10 | 0.3343 | 87.74% | 85.19% | Starting to learn some patterns |
| 15 | 0.3015 | 87.26% | 64.81% | ⚠️ Collapsed! Now guessing randomly |
| 20 | 0.3060 | 87.74% | 81.48% | Recovering, but unstable |
| 25 | 0.2620 | 89.15% | 87.04% | Best test performance so far |
| 30 | 0.2525 | 87.26% | 83.33% | Slightly worse — should have stopped at 25 |

**What the instability (epoch 15) tells us:** The quantum model is on a "barren plateau" — the loss landscape is nearly flat, so small parameter changes cause big swings in predictions. This is a known problem with variational circuits and is partially addressed in v02/v03.

---

## Model 2: Classical Multi-Layer Perceptron (PyTorch)

### How It Works — The Pipeline Analogy

A **Multi-Layer Perceptron (MLP)** is the most basic type of neural network. Think of it as a pipeline where data flows through successive stages of processing:

1. **Input** — raw gene values go in
2. **Hidden layers** — each layer learns to detect increasingly abstract patterns
3. **Output** — a single number representing "how likely is this patient to have SSc?"

Each layer applies three operations:
- **Linear transformation** — multiply by weights, add bias (like a weighted vote across all inputs)
- **Activation function** — introduce non-linearity (lets the model learn curved/complex boundaries)
- **Regularization** — prevent the model from memorizing the training data

### Architecture Diagram

```
Input (50 gene expression values)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: Linear(50 → 128) → BatchNorm → ReLU → Dropout(30%)│
│                                                             │
│ 128 neurons learn low-level combinations of genes.          │
│ Example: "Gene A high AND Gene B low → pattern #47"         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2: Linear(128 → 64) → BatchNorm → ReLU → Dropout(30%)│
│                                                             │
│ 64 neurons combine low-level patterns into medium-level.    │
│ Example: "Patterns #12 + #47 + #91 → pathway X active"     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3: Linear(64 → 32) → BatchNorm → ReLU → Dropout(30%) │
│                                                             │
│ 32 neurons distill the most disease-relevant information.   │
│ Example: "Pathways X + Y + Z → strong SSc signal"           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Linear(32 → 1) → Sigmoid()
    │
    ▼
Output: P(SSc) — probability between 0.0 and 1.0
         (> 0.5 → predict SSc, ≤ 0.5 → predict Healthy)
```

### Understanding Each Component

#### Linear Layers — The "weighted votes"
Each neuron in a layer computes: `output = (weight₁ × input₁) + (weight₂ × input₂) + ... + bias`. It's like a vote where each input gene gets a different importance weight. `Linear(50 → 128)` = 128 neurons, each looking at all 50 inputs with different weight sets = 50×128 + 128 = 6,528 parameters.

#### BatchNorm (Batch Normalization) — Keeping signals stable
During training, the outputs of each layer can drift to very large or very small numbers, which slows learning. BatchNorm rescales them to a standard range after each layer. Think of it as "recalibrating your instruments" between processing stages.

#### ReLU (Rectified Linear Unit) — Adding complexity
The simplest activation function: `output = max(0, input)`. Negative values become 0; positive values pass through unchanged.

**Why do we need this?** Without activation functions, stacking linear layers is useless — multiple linear transformations collapse into a single linear transformation. ReLU adds non-linearity, which lets the network learn complex, curved decision boundaries.

#### Dropout (p=0.3) — Preventing memorization
During each training step, 30% of neurons are randomly "turned off" (set to zero). This forces the network to spread knowledge across many neurons rather than relying on a few. It's like studying with a team where random members are absent — everyone has to be able to contribute independently.

**Why 0.3?** A standard starting value. Too low (0.1) = barely any regularization. Too high (0.7) = the network can't learn because too many neurons are disabled.

### Hyperparameters — Why These Values?

| Parameter | Value | Why This Value? |
|-----------|-------|-----------------|
| Architecture | 128 → 64 → 32 | "Funnel" shape forces information compression. Each layer extracts higher-level patterns from more neurons into fewer |
| Activation | ReLU | Simplest, fastest. Works well for most problems. More advanced alternatives (GELU) used in v02/v03 |
| Dropout | 0.3 | Standard regularization rate — enough to prevent overfitting without crippling learning |
| Learning rate | 0.001 | Standard starting point for Adam optimizer on classical networks |
| Batch size | 32 | Larger than quantum (16) because classical forward passes are instant — bigger batches give smoother gradient estimates |
| Epochs | 50 | Classical training is fast (~0.02 sec/epoch) so we can afford many passes through the data |
| Total params | 17,345 | 37× more than the quantum model, but still tiny by modern standards (GPT-4 has 1.7 trillion) |

### Training Progression

| Epoch | Train Loss | Train Acc | Test Acc | What's Happening |
|-------|-----------|-----------|----------|------------------|
| 1 | 0.6431 | 68.40% | 85.19% | Random weights, high test acc is luck + majority class |
| 10 | 0.4097 | 86.32% | 88.89% | ✅ Peak test accuracy! Model has learned the key patterns |
| 20 | 0.3125 | 86.79% | 88.89% | Holding steady — still at peak |
| 30 | 0.2879 | 89.15% | 83.33% | ⚠️ Starting to overfit (train↑, test↓) |
| 40 | 0.2544 | 88.68% | 85.19% | Slightly overfit |
| 50 | 0.2586 | 89.15% | 85.19% | Still overfit — should have stopped at epoch 10-20 |

**Key insight:** The model peaks at epoch 10-20 then slightly overfits. In v02, we add **early stopping** — automatically stop training when test performance stops improving.

---

## Final Results Comparison

### Performance Metrics (Test Set, n=54)

| Metric | Quantum VQC | Classical MLP | Winner |
|--------|-------------|---------------|--------|
| **Accuracy** | 83.33% | **85.19%** | Classical |
| **F1 Score (SSc)** | 0.9072 | **0.9167** | Classical |
| **AUC-ROC** | 0.5707 | **0.8370** | Classical |
| **Training Time** | 355.1s | **1.2s** | Classical (296× faster) |
| **Parameters** | **465** | 17,345 | Quantum (37× fewer) |

### Understanding These Metrics

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Accuracy** | % of all patients correctly classified | Can be misleading with imbalanced data — guessing "SSc" every time gives 85%! |
| **F1 Score** | Balance of precision and recall | Better than accuracy when classes are unequal. F1=1.0 is perfect |
| **AUC-ROC** | How well the model *ranks* patients by disease risk | **The single most important metric.** Measures discrimination ability independent of threshold. 0.5 = random guessing, 1.0 = perfect |
| **Training Time** | Wall-clock time to complete all epochs | Quantum simulation is inherently slow on classical CPUs |
| **Parameters** | Number of trainable values | Fewer params = simpler model = less overfitting risk (in theory) |

### Per-Class Detail — Where the Models Actually Fail

#### Quantum VQC
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Healthy | 0.33 | **0.12** | 0.18 | 8 |
| SSc | 0.86 | 0.96 | 0.91 | 46 |

#### Classical MLP
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Healthy | 0.50 | **0.25** | 0.33 | 8 |
| SSc | 0.88 | 0.96 | 0.92 | 46 |

### The Critical Failure: Minority Class Collapse

Look at the Healthy **recall** column: the quantum model only detects **12% of healthy patients** (about 1 out of 8). The classical model isn't much better at 25% (2 out of 8). Both models essentially learn to classify almost everything as SSc because that's the easy way to get high overall accuracy with a 6:1 imbalance.

This is called **majority class bias** — the model takes the shortcut of always predicting the majority class. It's the #1 problem we fix in v02.

**Quantum AUC = 0.57** — this is barely better than 0.50 (random coin flip). The quantum model genuinely cannot distinguish sick from healthy; it just predicts SSc for everyone.

---

## Why the Classical Model Wins in v01

| Problem | How It Hurts | Fixed in |
|---------|--------------|----------|
| **Class imbalance (6:1)** | Model just predicts "SSc" for everyone — gets 85% accuracy for free | v02 (SMOTE oversampling + class weights) |
| **Quantum AUC ≈ random (0.57)** | Circuit can't distinguish sick from healthy at all | v02 (data re-uploading circuit) |
| **No early stopping** | Model overfits past its peak performance | v02 (patience=10 on AUC) |
| **Variance-based feature selection** | High-variance ≠ disease-relevant (a gene can vary for reasons unrelated to SSc) | v02 (mutual information selection) |
| **Simple angle encoding** | Each feature is encoded only once — limited quantum advantage | v02 (data re-uploading = encode every layer) |

### What is a "Barren Plateau"?

The quantum model suffers from a known problem called **barren plateaus**: as quantum circuits get wider or deeper, the gradients (the "directions to improve") become exponentially small. The model appears stuck because adjusting any parameter produces almost zero change in the output.

**Analogy:** Imagine you're lost in a perfectly flat salt lake that stretches for miles. You can't tell which direction to walk because there are no hills, no valleys — just flat everywhere. That's what the quantum optimizer "sees" during a barren plateau.

**Why it matters here:** With 8 qubits and 3 layers, we're right at the edge where barren plateaus start appearing. This partially explains the training instability at epoch 15.

---

## How to Run

### Prerequisites

```bash
conda activate GEO
pip install pennylane pennylane-lightning torch scikit-learn pandas numpy
```

### Run Full Comparison (Recommended)

```bash
cd GSE76809Examples/v01
python compare_models.py
```

This runs all steps automatically: preprocessing → quantum training (~6 min) → classical training (~2 sec) → comparison summary.

### Run Individual Steps

```bash
# Preprocessing only (creates data/GSE76809/processed/ folder)
python preprocess_gse76809.py --n-features 50 --platform GPL6480

# Quantum model only (slow: ~6 minutes on CPU)
python model_quantum.py --n-qubits 8 --n-layers 3 --epochs 30 --lr 0.005 --batch-size 16

# Classical model only (fast: ~2 seconds)
python model_classical.py --hidden 128 64 32 --dropout 0.3 --epochs 50 --lr 0.001 --batch-size 32
```

### CLI Arguments Reference

| Script | Argument | Default | Description |
|--------|----------|---------|-------------|
| `preprocess_gse76809.py` | `--n-features` | 50 | Number of top genes to keep |
| | `--platform` | GPL6480 | GEO microarray platform |
| | `--test-size` | 0.2 | Fraction held back for testing |
| | `--selection` | variance | `variance` or `mutual_info` |
| `model_quantum.py` | `--n-qubits` | 8 | Number of qubits in the circuit |
| | `--n-layers` | 3 | Depth of variational circuit |
| | `--epochs` | 30 | Training iterations |
| | `--lr` | 0.005 | Learning rate |
| | `--batch-size` | 16 | Samples per training step |
| `model_classical.py` | `--hidden` | 128 64 32 | Neuron counts per hidden layer |
| | `--dropout` | 0.3 | Dropout probability |
| | `--epochs` | 50 | Training iterations |
| | `--lr` | 0.001 | Learning rate |
| | `--batch-size` | 32 | Samples per training step |

---

## File Structure

```
GSE76809Examples/v01/
├── README.md                  ← This file (you are here)
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
| Hardware | CPU only (no GPU/QPU) |

---

## What's Next? (Preview of v02 and v03)

The problems identified here are systematically fixed:

| Version | Key Fix | Result |
|---------|---------|--------|
| **v02** | SMOTE oversampling + class weights + mutual information features + data re-uploading circuit | Quantum AUC: 0.57 → 0.88 |
| **v03** | Amplitude encoding + dressed circuit + ensemble + cross-validation | Quantum AUC: 0.88 → 0.99 (CV) |

---

## References

- **Dataset:** Milano et al. — GSE76809, a compendium of systemic sclerosis gene expression studies
- **PennyLane:** Bergholm et al., "PennyLane: Automatic differentiation of hybrid quantum-classical computations" (2018)
- **VQC for classification:** Schuld et al., "Circuit-centric quantum classifiers" (2020)
- **Barren plateaus:** McClean et al., "Barren plateaus in quantum neural network training landscapes" (2018)
