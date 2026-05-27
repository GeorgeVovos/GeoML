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

Estimated runtime: **~30 min per encoding** (5-fold CV, no holdout, no
learning curve), so ~2 hours total on CPU.

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
