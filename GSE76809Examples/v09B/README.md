# v09B — Combined Encoding Ablation (base + data reuploading)

## Goal

v09 compared five **standalone** encodings and found `data_reuploading`
dominated. v09B asks the natural follow-up question:

> **Does prepending a structured base encoding to data reuploading help,
> hurt, or do nothing?**

In other words, instead of using the base encodings *on their own* (as v09
did), v09B **fuses** each one with data reuploading inside the same circuit:

1. `angle` + data reuploading
2. `dense_angle` + data reuploading
3. `amplitude` + data reuploading
4. `iqp` + data reuploading

The standalone `data_reuploading` encoding is kept as the **reference
baseline** so the paired statistics directly answer "did adding a base
encoding change anything?"

Everything else is **frozen from v06/v09**: same 4-qubit / 8-layer circuit,
same 12-output multi-basis measurement, same `12 → 64 → 1` post-net, same
per-fold preprocessing, same training budget (80 epochs, batch 24, lr 0.005,
Adam + cosine annealing, early stopping patience 15, per-fold BorderlineSMOTE,
seed 2026). The **only** thing that changes is the data-loading stage of the
quantum circuit.

## How the combination works

Each combined encoding splits the 8 circuit layers in two:

| Stage            | Layers   | Gates                                                |
|------------------|----------|------------------------------------------------------|
| Base encoding    | 0        | the base feature map (angle / dense_angle / amplitude / iqp) |
| Data reuploading | 1 … 7    | `RY(wᵀx)` per qubit — learned linear combo of all 16 features |

So the base encoding seeds the register once, then reuploading refines it at
every subsequent layer. Because reuploading is learnable, **all** v09B
encodings carry encoding weights of shape `(8, 4, 16)` — unlike v09 where
only `data_reuploading` did.

## Encodings compared

| Encoding            | Layer 0 base                          | Layers 1–7        |
|---------------------|---------------------------------------|-------------------|
| `data_reuploading`  | *(none — reuploading from layer 0)*   | reuploading       |
| `angle_reup`        | `RY(x_i)` (1 feature/qubit)           | reuploading       |
| `dense_angle_reup`  | `RY(x_{2q})` + `RZ(x_{2q+1})` (2/qubit)| reuploading      |
| `amplitude_reup`    | `AmplitudeEmbedding` (16 amplitudes)  | reuploading       |
| `iqp_reup`          | `H` + `RZ(x_i)` + ring `ZZ(x_i x_j)`  | reuploading       |

`amplitude_reup` requires unit-norm inputs for the layer-0 embedding; the
driver normalises `X` for that encoding (and the later reuploading layers
then dot the same unit-norm features with the learned weights).

## Why this matters

- If every combination ties with the `data_reuploading` reference, the base
  encoding is irrelevant once reuploading is present — reuploading does all
  the work.
- If a combination **beats** the reference, the base feature map adds
  information that reuploading alone cannot recover.
- If a combination **hurts**, the base map injects a bad inductive bias (e.g.
  amplitude's single-shot compression) that reuploading cannot fully undo.

## How to run

```powershell
conda activate GEO
cd GSE76809Examples\v09B

# Full ablation: reference + all four combined encodings, 5-fold CV
python compare_encodings.py

# Subset
python compare_encodings.py --encodings data_reuploading amplitude_reup

# One encoding at a time, checkpointed/resumable (recommended for long runs)
python run_one.py amplitude_reup
python run_one.py iqp_reup
```

Both runners checkpoint after every fold to
`results/v09b_encoding_ablation_partial.json`, so an interrupted run resumes
automatically. Expect a similar per-encoding cost to v09 (order of ~1.5 h of
CPU per encoding for the 5-fold CV; reuploading-heavy circuits are the slow
ones).

## Output

`results/v09b_encoding_ablation.json` with, per encoding, the 5-fold AUCs
(mean / std) plus Wilcoxon + Nadeau–Bengio paired tests against
`data_reuploading`, Holm-corrected across the four combined encodings.

## Results

_To be filled after running. Template:_

| Encoding           | Mean AUC | Std AUC | Significant vs Data Reuploading? |
|--------------------|----------|---------|----------------------------------|
| data_reuploading   | —        | —       | Reference                        |
| angle_reup         | —        | —       | —                                |
| dense_angle_reup   | —        | —       | —                                |
| amplitude_reup     | —        | —       | —                                |
| iqp_reup           | —        | —       | —                                |

## Files

| File                  | Purpose                                              |
|-----------------------|------------------------------------------------------|
| `quantum_encodings.py`| Combined data-loading fragments + reference          |
| `model_quantum_vqc.py`| VQC parameterised by encoding name (frozen from v09) |
| `compare_encodings.py`| 5-fold CV runner for all encodings; writes results JSON |
| `run_one.py`          | Checkpointed single-encoding runner (resumable)      |

Preprocessing is reused from v06 and statistics from `shared/` (both via
`sys.path` imports — nothing is copied or modified).

## Relation to v09

v09 is left **completely unchanged**. v09B is a sibling experiment (same
naming convention as v06 → v06B) that reuses v09's frozen architecture and
the identical `data_reuploading` reference, so the two result sets are
directly comparable: v09 tells you how each base encoding does *alone*, and
v09B tells you how it does *combined with* reuploading.
