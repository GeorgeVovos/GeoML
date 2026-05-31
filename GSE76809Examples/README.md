# GSE76809 — Quantum vs Classical Classification Examples

This folder contains a progression of experiments classifying **systemic
sclerosis (SSc) vs Healthy** patients from the GSE76809 gene-expression
dataset (266 samples on the GPL6480 platform; 228 SSc / 38 Healthy),
comparing **quantum machine-learning models** (variational quantum
classifiers and quantum kernels) against strong **classical baselines**
(MLP, XGBoost, SVM, logistic regression).

Each `vNN/` folder isolates one methodological change or one scientific
question and has its own detailed `README.md`. This file is a one-page
map of the first nine examples and what each one actually found.

> **Headline conclusion (v01–v09):** On GSE76809 there is **no robust
> quantum advantage**. After leakage fixes and proper cross-validation,
> the quantum VQC becomes *competitive* with classical models, but
> well-tuned classical methods (especially XGBoost) remain at least as
> strong. The quantum model's competitiveness traces almost entirely to
> the **data-reuploading encoding**, not to anything inherently quantum.

---

## Quick reference

| Ver | Focus | Headline finding |
|-----|-------|------------------|
| v01 | First naive quantum vs classical comparison | Classical MLP clearly wins; quantum barely beats random |
| v02 | Fix imbalance & encoding (MI features, SMOTE, data-reuploading) | Quantum AUC jumps 0.57→0.88; wins acc/F1, classical still wins AUC |
| v03 | Threshold tuning, CV, ensemble stacking | Quantum best holdout acc/F1; ensemble best AUC; CV near-perfect |
| v04 | Leakage-corrected cross-validation | VQC has highest mean CV AUC; beats XGBoost significantly |
| v05 | Re-run / re-check with tighter design | MLP becomes best CV model and significantly beats VQC |
| v06 | Tuned baselines + variance/learning-curve analysis | Tuned XGBoost wins AUC; VQC competitive with lowest variance |
| v07 | Cross-dataset replication of v06 | Only GSE76809 completed; replication question still open |
| v08 | Small-sample efficiency sweep | Negative result: no small-data quantum advantage |
| v09 | Quantum encoding ablation | Data-reuploading is best; dense_angle (2 feat/qubit) is worst — re-uploading drives performance |

---

## Detailed summaries

### v01 — Naive baseline comparison
First head-to-head with minimal tuning. The classical MLP clearly
outperforms the quantum model on AUC (**0.84 vs 0.57**), and the quantum
model fails to detect any Healthy patients (Healthy F1 = 0). Quantum
training is also far slower. **Takeaway:** out-of-the-box quantum is not
competitive here.

### v02 — Fixing imbalance and encoding
Adds mutual-information feature selection, SMOTE + class weighting, and a
**data-reuploading** quantum circuit. Quantum AUC jumps from 0.57 to
**0.88** and the model now wins on accuracy (87.0%) and both F1 scores,
while the classical MLP still has the higher AUC (**0.94** — better at
*ranking* patients). **Takeaway:** the encoding and balancing fixes make
quantum genuinely usable, but classical still leads on AUC.

### v03 — Thresholds, cross-validation, ensembling
Adds Youden-J threshold tuning, cross-validation, and a quantum+classical
stacking ensemble. The quantum model has the best holdout accuracy/F1,
the ensemble has the best AUC, and CV scores are near-perfect for both
models. **Caveat (motivates v04):** the near-perfect CV is a red flag for
data leakage in the CV pipeline.

### v04 — Leakage-corrected cross-validation
Rebuilds the CV so feature selection / scaling / SMOTE happen **inside**
each fold. With leakage removed, the quantum **VQC has the highest mean
CV AUC** and significantly beats XGBoost. Holdout AUC, however, is still
best for the MLP. **Takeaway:** the first methodologically clean signal
that quantum can be competitive — but it is split (CV favours VQC,
holdout favours MLP).

### v05 — Re-check under a tighter design
A re-run / re-evaluation flips the v04 ranking: the **MLP becomes the
best CV model** and significantly outperforms the VQC, with ordering
MLP > XGBoost > VQC > quantum-kernel. **Takeaway:** the apparent quantum
CV edge in v04 is fragile and does not survive a tighter design —
results are highly sensitive to pipeline details.

### v06 — Tuned baselines, variance & learning curves
Properly tunes the classical baselines and analyses variance and
small-data behaviour. **Tuned XGBoost wins AUC** (~0.90 CV / ~0.95
holdout); the VQC is competitive (~0.82 CV / ~0.91 holdout) with the
**lowest variance**, and a parameter-matched MLP is essentially tied with
the VQC. Small-data learning-curve hints favour quantum slightly but do
not overturn the classical lead. v06 is the reference pipeline reused by
later versions.

### v07 — Cross-dataset replication
Intended to replicate v06 across four datasets. **Only GSE76809 actually
completed** in the current run (the other three are placeholders in the
results JSON). On GSE76809, XGBoost again leads (mean CV AUC 0.899),
the VQC is competitive (0.827), and the quantum kernel is poor (0.570).
**Takeaway:** the README was corrected to state that the cross-dataset
replication question **remains open** — the single-dataset evidence does
not support a generalizable quantum advantage.

### v08 — Sample-efficiency sweep
Tests whether quantum wins when data is scarce. With a fixed test set and
multiple seeds, only `N_per_class = 10` and `20` could be evaluated (the
pool has just 30/class), and the quantum kernel was skipped for runtime.
At both sizes, **MLP and LR-L1 outperform the VQC**. **Takeaway:** an
honest **negative result** — no small-data quantum advantage survives the
tighter design. (The VQC reuses v06's 4-qubit data-reuploading circuit.)

### v09 — Encoding ablation
Holds the variational circuit fixed and swaps only the **data-encoding**
strategy. Mean CV AUCs: **data_reuploading 0.788** > iqp 0.648 >
amplitude 0.587 > angle 0.548 > dense_angle 0.487. By effect size the
gap is large (Cohen's d > 1.3), but after Holm correction **only
amplitude vs data_reuploading is statistically significant** (adjusted
p = 0.049). Notably, `dense_angle` (2 features per qubit, single-shot)
performs *worse* than plain `angle` (1 feature per qubit), showing that
packing more features into rotations without layer-wise repetition does
not help — it is the **re-uploading** that drives performance.
**Takeaway:** data-reuploading is the key encoding driver of the quantum
model's competitiveness. Even so, the best encoding still loses to tuned
XGBoost.

---

## Cross-cutting lessons

- **Methodology dominates conclusions.** Claims about quantum advantage
  flip between versions purely from pipeline changes (leakage fixes,
  tuning, CV design). v01–v03 are optimistic; v04 is mixed; v05–v09 show
  no robust advantage once the design is tightened.
- **Distinguish holdout vs CV vs ablation claims.** Different evaluation
  modes can rank the models differently; the per-version READMEs are
  explicit about which mode each conclusion comes from.
- **Encoding is the real quantum lever.** v09 shows the quantum model's
  performance hinges on data-reuploading encoding, not on entanglement or
  "quantumness" per se.
- **Negative results are reported honestly.** v05, v07, and v08 are
  neutral-to-negative for quantum and are documented as such.

## Dataset

| Property | Value |
|----------|-------|
| Series | GSE76809 (SSc vs Healthy) |
| Platform used | GPL6480 only |
| Samples | 266 (228 SSc / 38 Healthy) |
| Task | Binary classification |
| Primary metric | AUC-ROC (plus accuracy, F1, calibration in later versions) |

> Versions **v10–v13** extend this work (noise sweeps, alternative
> quantum kernels, extra classical baselines, and calibration /
> decision-curve analysis) and are documented in their own folders.
