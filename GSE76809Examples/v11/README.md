# v11 — Projected Quantum Kernels & Kernel-Target Alignment

## Goal

v06's quantum kernel was the worst model in the comparison (CV-AUC 0.570
vs RBF 0.812 vs XGBoost 0.899). The QML literature has two well-known
fixes for that, neither of which v01-v10 implemented:

1. **Projected Quantum Kernel (PQK)** — Huang et al. (2021), "Power of
   data in quantum machine learning". Instead of the full-state inner
   product `|⟨φ(x)|φ(y)⟩|²`, measure reduced single-qubit Pauli
   expectations, stack them into a classical feature vector, then run a
   classical RBF kernel on those. Sidesteps the exponential
   concentration / "kernel collapses to identity" problem that plagues
   inner-product kernels on small data.

2. **Quantum Kernel Alignment (QKA)** — Glick et al. (2021), Hubregtsen
   et al. (2021). Add trainable parameters to the feature map, then
   gradient-train them to maximise kernel-target alignment with the
   labels before fitting the SVM.

v11 implements both and compares them against:
- v06's plain inner-product ZZ quantum kernel (baseline)
- RBF SVM on the same PCA data
- L1 logistic regression (the small-data linear champion)

## How to run

```powershell
cd GSE76809Examples\v11
python compare_kernels.py                       # all kernels (~90 min)
python compare_kernels.py --kernels pqk rbf     # subset
```

## Files

| File                 | Purpose                                           |
|----------------------|----------------------------------------------------|
| `projected_kernel.py`| PQK: measure single-qubit Paulis, then RBF on them|
| `aligned_kernel.py`  | QKA: trainable ZZ feature map + SVM               |
| `compare_kernels.py` | 5-fold CV driver with Wilcoxon + Holm corrections |

The plain ZZ quantum kernel and RBF baselines are imported unchanged
from v06.

## Expected outcome

The QML literature suggests PQK should be **strictly better** than the
plain inner-product kernel on small data because it inherits the
exponential-concentration immunity of classical RBF on a low-dimensional
embedding. If v11 confirms that on GSE76809, v06's "the quantum kernel
loses" conclusion becomes "*this particular* quantum-kernel construction
loses; the projected variant wins" — a meaningful nuance.

If QKA outperforms PQK, you have evidence that the quantum kernel
hypothesis class is too rigid and needs trainable parameters to be
competitive.

## Caveats

- QKA's gradient training is itself an optimization problem; convergence
  on small CV folds is bumpy. We use 100 alignment-optimisation steps
  with Adam(lr=0.01); reduce / raise if you see plateaus.
- PQK measures 3 Pauli operators per qubit (X, Y, Z) = 18 classical
  features for 6 qubits. RBF is then trained on those 18 features with
  inner-CV C / gamma tuning.
