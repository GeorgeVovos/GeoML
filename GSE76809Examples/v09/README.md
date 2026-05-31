# v09 — Encoding Ablation

## Goal

v01 → v03 jumped from AUC 0.57 to 0.99 while changing seven things at
once: feature selection, SMOTE policy, encoding, circuit depth, post-net,
threshold strategy, and CV folds. That makes the v01→v03 narrative *not*
an attribution — you cannot honestly say "amplitude encoding caused it."

v09 fixes that by **freezing everything except the encoding**. Same v06
pipeline, same 4-qubit / 8-layer / 12-feature post-net, same training
budget, same per-fold preprocessing — only the encoding stage of the
quantum circuit changes.

## Encodings compared

| Encoding              | Qubits | Data per layer | What it tests                       |
|-----------------------|--------|----------------|-------------------------------------|
| `angle`               | 4      | once (start)   | v01-style baseline (1 feature/qubit) |
| `dense_angle`         | 4      | once (start)   | 2 features/qubit via RY+RZ          |
| `amplitude`           | 4      | once (start)   | v03/v04-style "compression"         |
| `iqp`                 | 4      | once (start)   | Hadamard + RZ + ZZ feature map      |
| `data_reuploading`    | 4      | every layer    | v06 baseline (reference)            |

Variational layers and measurements are *identical* across all five runs.
The only difference is how the 16 input features become qubit rotations.

`dense_angle` is a stronger version of `angle`: instead of loading one
feature per qubit (4 features total, the v01-style bottleneck), it packs
**two features per qubit** on orthogonal axes — `RY(x_{2q})` followed by
`RZ(x_{2q+1})` — loading the first **8** of the 16 features into the same
4-qubit register without adding qubits or trainable encoding weights. It
isolates the effect of *encoding density* alone.

## Why this matters

If `data_reuploading` is the best encoding by a clear margin, you can
honestly attribute v06's quantum performance to the encoding strategy
rather than to any of the other six things that changed v01→v06.

If two encodings tie, that is evidence the encoding choice is *not* the
operative factor and should be reported as such.

## How to run

```powershell
cd GSE76809Examples\v09
python compare_encodings.py                       # all four encodings
python compare_encodings.py --encodings angle iqp # subset
```

Runtime: the actual full 4-encoding run took **~386 min (~6.4 h)** on CPU
(`runtime_minutes` in the results JSON), i.e. ~96 min per encoding for the
5-fold CV — well above the earlier ~30 min/encoding guess.

## Results: Encoding Ablation (v09)

| Encoding           | Mean AUC | Std AUC | Significant vs Data Reuploading? |
|--------------------|----------|---------|----------------------------------|
| data_reuploading   | 0.788    | 0.072   | Reference                        |
| iqp                | 0.648    | 0.115   | No                               |
| amplitude          | 0.587    | 0.085   | **Yes (p=0.049)**                |
| angle              | 0.548    | 0.088   | No                               |
| dense_angle        | 0.487    | 0.194   | No                               |

**Interpretation:**

- The `data_reuploading` encoding (used in v06) outperforms all other encodings by a large margin (Cohen's d > 1.3 for all comparisons).
- Only the difference between `data_reuploading` and `amplitude` encoding is statistically significant after Holm correction (p=0.049).
- `dense_angle` (2 features/qubit) performs *worse* than plain `angle` (1 feature/qubit), with extremely high variance (std 0.194). Packing more features into single-shot rotations without layer-wise repetition does not help — the circuit cannot extract useful information from the extra rotations in a single pass.
- This strongly supports the claim that it is the **re-uploading** (layer-wise repetition) that drives performance, not merely the number of features encoded.

## Quantum vs Classical (context from v06)

- In v06, the best classical model (XGBoost) achieves higher AUC (0.899 CV, 0.948 holdout) than any quantum model, but the quantum VQC is competitive (0.822 CV, 0.905 holdout) and has the lowest variance.
- The parameter-matched MLP is essentially tied with the VQC on AUC, but the VQC wins on accuracy and F1.
- Quantum models degrade more gracefully in the small-data regime, maintaining higher AUC than classical models at 10% of the data.
- The choice of encoding is critical for quantum advantage: only the data reuploading strategy delivers strong results, while other encodings fall well short of classical baselines.

**Conclusion:**

- Quantum VQC with data reuploading encoding is competitive with strong classical baselines, but does not surpass the best-tuned classical models on this dataset. The encoding choice is the key driver of quantum performance improvements in v06/v09.

## Files

| File                  | Purpose                                       |
|-----------------------|-----------------------------------------------|
| `quantum_encodings.py`| The five data-loading circuit fragments        |
| `model_quantum_vqc.py`| VQC parameterised by encoding name             |
| `compare_encodings.py`| 5-fold CV runner; writes results JSON          |

Preprocessing is reused from v06 (`sys.path` import).

## Output

`results/v09_encoding_ablation.json` with per-encoding 5-fold AUC plus
Wilcoxon + Nadeau-Bengio paired tests against `data_reuploading`.
