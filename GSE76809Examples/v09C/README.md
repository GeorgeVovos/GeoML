# v09C — Per-Layer Encoding + Learned Reuploading (vs v09B)

## Goal

v09C is the sibling of v09B. Both combine a learned per-layer data-reuploading
module with four upstream encodings (angle, dense_angle, amplitude, iqp). The
**only** difference is *where the upstream encoding is applied*:

| | Upstream encoding applied | Reading of the template |
|---|---|---|
| **v09B** | layer 0 only (state preparation) | encoding "seeds" the circuit |
| **v09C** | **every layer** (interleaved) | encoding "applied in combination ... per-layer" |

v09C matches the canonical Pérez-Salinas data-reuploading architecture, where
the data is re-injected at every layer. Comparing the two isolates a single
architectural decision and shows whether it actually changes results.

## What is frozen (identical to v09B)

| Component | Value |
|-----------|-------|
| Preprocessing | v06 pipeline; per-fold SMOTE; 16 PCA features |
| Qubits | 4 |
| Layers | 8 |
| Variational block | RZ + RX per qubit, then CNOT ring |
| Measurement | PauliZ, PauliX, PauliY per qubit (12 outputs) |
| Post-net | Linear(12,64) → GELU → Dropout(0.1) → Linear(64,1) → Sigmoid |
| Optimiser | Adam, lr=0.005, CosineAnnealingLR(T_max=80) |
| Training | 80 epochs, batch 24, grad-clip 1.0, patience 15 (eval every 3 epochs) |
| CV | 5-fold from `preprocess()` |
| Seed | 2026 |
| Learned reuploading | `weights_enc` (8, 4, 16), trainable, every experiment |

## What varies vs v09B

The upstream encoding gate block fires at **every** layer instead of only layer 0.

| Experiment | Upstream encoding (every layer) | Learned reuploading (every layer) |
|---|---|---|
| `data_reuploading` | none — canonical reference | ✓ |
| `angle_combined` | RY(x_i · π), first 4 features | ✓ |
| `dense_angle_combined` | RY(x_{2q}·π) + RZ(x_{2q+1}·π), first 8 features | ✓ |
| `amplitude_combined` | AmplitudeEmbedding (unit-norm, 16 amplitudes) | ✓ |
| `iqp_combined` | IQP map: H, RZ, ZZ ring | ✓ |

## Circuit diagram (per layer l)

```
every layer l:
   if upstream encoding exists:
       upstream_gate_fn(x)              ← v09C: EVERY layer (v09B: l==0 only)

   for q in 0..3:
       RY( dot(weights_enc[l,q], x) )   ← learned reuploading

   for q in 0..3:
       RZ(weights_var[l,q,0])
       RX(weights_var[l,q,1])
   CNOT ring
```

## How to run

```powershell
# 1. Run the v09C ablation (~8 h on CPU, resumable)
python "GSE76809Examples\v09C\compare_encodings.py"

# Subset
python "GSE76809Examples\v09C\compare_encodings.py" --encodings data_reuploading iqp_combined

# 2. Compare v09C against v09B (instant; needs both result JSONs)
python "GSE76809Examples\v09C\compare_versions.py"
```

The runner is **resumable**: after every encoding/fold it writes
`results/v09C_encoding_ablation_partial.json`. Re-launch the same command to
resume; completed fold/encoding combinations are skipped automatically.

## How to interpret

### Within v09C (vs `data_reuploading` reference)
Same interpretation as v09B: a significant positive `mean_diff` means the
per-layer encoding adds information the learned reuploading cannot recover on
its own.

### v09C vs v09B (`compare_versions.py`)
This is the headline comparison. For each encoding it prints the mean AUC under
both architectures, Δ = v09C − v09B, Cohen's d, and a Wilcoxon paired test:

- **Δ > 0** — applying the encoding every layer helps; re-injecting data is
  beneficial (expected for `angle`, `dense_angle`, `iqp`).
- **Δ ≈ 0** — the architectural choice does not matter for that encoding.
- **Δ < 0** — layer-0-only is better. Strongly expected for `amplitude_combined`:
  `AmplitudeEmbedding` re-prepares the full statevector, so applying it every
  layer **overwrites** the accumulated computation, crippling the circuit. This
  is the clearest example of an architectural decision producing genuinely
  different results.

## Which is "more correct"?

For rotation-style encodings (angle, dense_angle, iqp), **v09C is the more
faithful implementation** of the template's "per-layer ... in combination"
wording and of standard data-reuploading. For `amplitude`, layer-0-only (v09B)
is the only sensible choice. `compare_versions.py` quantifies exactly how much
each interpretation matters on this dataset.

## Files

| File | Purpose |
|------|---------|
| `quantum_encodings.py` | Upstream gate functions + `UPSTREAM_GATES` dict |
| `model_quantum_vqc.py` | `CombinedEncodingVQC` (every-layer encoding) + `train_quantum_vqc` |
| `compare_encodings.py` | 5-fold CV runner; resumable; writes results JSON |
| `compare_versions.py` | Loads v09B + v09C JSONs; paired AUC comparison |
| `README.md` | This file |

## Output

- `results/v09C_encoding_ablation.json` — per-encoding 5-fold AUCs, mean/std,
  paired tests vs `data_reuploading` (Wilcoxon + Nadeau–Bengio, Holm-corrected).
- `results/v09C_encoding_ablation_partial.json` — rolling checkpoint.
- `results/v09B_vs_v09C_comparison.json` — per-encoding v09B-vs-v09C deltas.
