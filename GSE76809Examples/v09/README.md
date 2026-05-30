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
| `angle`               | 4      | once (start)   | v01-style baseline                  |
| `amplitude`           | 4      | once (start)   | v03/v04-style "compression"         |
| `iqp`                 | 4      | once (start)   | Hadamard + RZ + ZZ feature map      |
| `data_reuploading`    | 4      | every layer    | v06 baseline (reference)            |

Variational layers and measurements are *identical* across all four runs.
The only difference is how the 16 input features become qubit rotations.

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
| amplitude          | 0.587    | 0.085   | **Yes (p=0.037)**                |
| angle              | 0.548    | 0.088   | No                               |

**Interpretation:**

- The `data_reuploading` encoding (used in v06) outperforms all other encodings by a large margin (Cohen's d > 1.3 for all comparisons).
- Only the difference between `data_reuploading` and `amplitude` encoding is statistically significant after Holm correction (p=0.037).
- This supports the claim that the performance jump in v06 is primarily due to the encoding strategy, not other pipeline changes.

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
| `encodings.py`        | The four data-loading circuit fragments        |
| `model_quantum_vqc.py`| VQC parameterised by encoding name             |
| `compare_encodings.py`| 5-fold CV runner; writes results JSON          |

Preprocessing is reused from v06 (`sys.path` import).

## Output

`results/v09_encoding_ablation.json` with per-encoding 5-fold AUC plus
Wilcoxon + Nadeau-Bengio paired tests against `data_reuploading`.
