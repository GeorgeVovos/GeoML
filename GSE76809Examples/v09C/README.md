# v09C — Per-Layer Encoding + Learned Reuploading

## Goal

v09C combines a learned per-layer data-reuploading module with four upstream
encodings (angle, dense_angle, amplitude, iqp), applied at **every layer**
(interleaved) — the literal "applied in combination ... per-layer" reading of
the original template, and the canonical Pérez-Salinas data-reuploading
architecture where the data is re-injected at every layer.

Each combined encoding is compared against **pure learned reuploading**
(`data_reuploading`, no upstream gates) as the reference, to test whether
stacking a fixed encoding on top of the learned reuploading at every layer
adds anything on GSE76809.

## What is frozen (identical to v06/v09)

| Component | Value |
|-----------|-------|
| Preprocessing | v06 pipeline; per-fold SMOTE; **16 ANOVA-selected features, StandardScaler'd (z-scores)** fed to the VQC. (PCA→6 is also computed by the pipeline but is consumed only by kernel methods — the VQC uses the raw 16-d scaled features, *not* PCA.) |
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

## What varies across experiments

The upstream encoding gate block fires at **every** layer (or not at all, for
the `data_reuploading` reference).

| Experiment | Upstream encoding (every layer) | Learned reuploading (every layer) |
|---|---|---|
| `data_reuploading` | none — canonical reference | ✓ |
| `angle_combined` | RY(x_i · π), first 4 features | ✓ |
| `dense_angle_combined` | RY(x_{2q}·π) + RZ(x_{2q+1}·π), first 8 features | ✓ |
| `amplitude_combined` | AmplitudeEmbedding (unit-norm, 16 amplitudes) | ✓ |
| `iqp_combined` | IQP map: H, RZ, ZZ ring | ✓ |

## Results (5-fold CV, seed 2026)

Full run: 25 model trainings (5 encodings × 5 folds), **771.9 min (~12.9 h)** on CPU.
Results file: [results/v09C_encoding_ablation.json](results/v09C_encoding_ablation.json).

### Per-encoding AUC (sorted best → worst)

| Encoding | Per-fold AUC | Mean ± std |
|---|---|---|
| `data_reuploading` (ref) | 0.761, 0.779, 0.889, 0.977, 0.653 | **0.812 ± 0.111** |
| `iqp_combined` | 0.572, 0.802, 0.505, 0.595, 0.815 | 0.658 ± 0.127 |
| `amplitude_combined` | 0.507, 0.545, 0.718, 0.660, 0.507 | 0.587 ± 0.086 |
| `angle_combined` | 0.446, 0.414, 0.606, 0.657, 0.569 | 0.539 ± 0.094 |
| `dense_angle_combined` | 0.369, 0.423, 0.616, 0.292, 0.454 | 0.431 ± 0.108 |

### Paired tests vs `data_reuploading`

| Encoding | mean Δ | Wilcoxon p | NB-corrected p | Cohen's d | Holm p | Holm sig? |
|---|---|---|---|---|---|---|
| `amplitude_combined` | −0.225 | 0.0625 | 0.0079 | −2.02 (large) | 0.0317 | **yes** |
| `angle_combined` | −0.273 | 0.0625 | 0.0207 | −2.37 (large) | 0.0622 | no |
| `dense_angle_combined` | −0.381 | 0.0625 | 0.0375 | −3.11 (large) | 0.0751 | no |
| `iqp_combined` | −0.154 | 0.3125 | 0.3981 | −1.16 (large) | 0.3981 | no |

### Headline finding

**Every upstream encoding applied at every layer *reduces* AUC relative to plain
learned data-reuploading.** Stacking a fixed encoding on top of the learned
reuploading at every layer never helps on GSE76809; reuploading alone is best
(0.812). Only `amplitude_combined`'s drop is Holm-significant. This corroborates
v09 and confirms the README's prediction that re-applying an encoding every
layer tends to overwrite the accumulated computation.

## Problems / unexpected findings & improvements

1. **All encodings underperform the baseline.** Even the least-degraded
   (`iqp_combined`) loses ~0.15 AUC. The "combination" never adds information
   the learned reuploading could not already recover on its own.
2. **Sub-chance AUCs.** `dense_angle_combined` averages **0.431 (< 0.5)**, with
   folds dipping to **0.292**; `angle_combined` has two folds at 0.41–0.45.
   Worse-than-random ranking means these every-layer encodings *actively
   scramble* the signal, not merely add noise — the amplitude-overwrite warning
   generalises to the rotation encodings.
3. **Input-scaling mismatch (likely root cause).** The fixed upstream encodings
   hard-code `RY(x·π)`, `RZ(x·π)`, `RZ(x_i·x_j)`, but the VQC is fed
   **StandardScaler z-scores (≈ −3…3), not [0,1]/[0,π]-bounded values.** So
   `angle`/`dense_angle`/`iqp` see arbitrary, wrap-around rotation angles,
   whereas the learned reuploading absorbs the scale through trainable weights.
   A fairer test would MinMax-scale inputs to [0,1] (or [0,π]) *before* the fixed
   encodings — the current result partly measures a scaling bug, not just the
   architecture.
4. **Significance ranks opposite to effect size.** `amplitude_combined` is the
   *only* Holm-significant degradation (p_adj 0.032) despite the **smallest**
   effect (d −2.02, Δ −0.225); `dense_angle_combined` has the **largest** effect
   (d −3.11) yet is not significant. The Nadeau–Bengio/Holm pipeline rewards the
   *consistency* (low variance) of the per-fold differences, not their
   magnitude. Read "significant" here as *most consistent degradation*, not
   *largest*.
5. **Wilcoxon is underpowered at n = 5.** The minimum two-sided Wilcoxon p for 5
   paired folds is **0.0625**, so *no* encoding can reach p < 0.05 by Wilcoxon.
   The Holm "significant" flag rests entirely on the parametric NB-corrected
   t-test — treat it cautiously.
6. **Validation fold doubles as model-selection and reporting set.** Early
   stopping keeps the epoch with the best *validation* AUC, then the **same**
   fold is scored — so all reported AUCs are optimistically biased and are *not*
   a held-out estimate. The 54-sample holdout (`X_test`) built by `preprocess()`
   is never used here. Improvement: select on an inner split (or report the
   untouched holdout).
7. **High fold-to-fold variance.** `data_reuploading` swings 0.653–0.977. With
   only ~8 healthy (minority) samples per validation fold, single-fold AUCs are
   very noisy and the baseline's "win" leans on folds 3–4. Repeated CV / more
   folds / a larger minority class would stabilise this.

## Cross-machine reproducibility

The ablation was run independently on two PCs with the **same `seed: 2026`**.
The results files are [results/v09C_encoding_ablation.json](results/v09C_encoding_ablation.json)
(PC1, 771.9 min) and [resultsPC2/v09C_encoding_ablation.json](resultsPC2/v09C_encoding_ablation.json)
(PC2, 403.7 min).

| Encoding | PC1 mean AUC | PC2 mean AUC | Δ |
|---|---|---|---|
| `data_reuploading` (ref) | 0.812 | 0.782 | −0.030 |
| `iqp_combined` | 0.658 | 0.666 | +0.009 |
| `amplitude_combined` | 0.587 | 0.671 | +0.084 |
| `angle_combined` | 0.539 | 0.573 | +0.034 |
| `dense_angle_combined` | 0.431 | 0.480 | +0.049 |

**What reproduced:** the qualitative story is stable — `data_reuploading` wins
on both, *every* upstream encoding degrades AUC, all effect sizes are "large",
and `dense_angle_combined` is the worst (or tied-worst), sitting at/below 0.5.

**What did NOT reproduce:** the per-fold AUCs differ entirely, and the single
Holm-significant encoding **flips** — `amplitude_combined` (p_adj 0.032) on PC1
vs `dense_angle_combined` (p_adj 0.019) on PC2. This is a concrete demonstration
of problem #4/#5 above: at n = 5 the "significant" flag is fragile and tracks
per-fold *consistency*, not a stable truth.

**Root cause (now fixed).** The metadata `seed` only fixed *preprocessing*
(numpy/sklearn split + SMOTE). Model weight init (`torch.randn`) and batch
shuffling (`torch.randperm`) were **unseeded**, so init/training order — and the
borderline statistics — drifted between machines and library versions.
`compare_encodings.py` now calls `seed_everything(2026)` (seeds `random`,
`numpy`, and `torch`) and passes a deterministic per-(fold, encoding)
`init_seed` to `train_quantum_vqc`, making future runs reproducible. The two
result sets above predate that fix; re-running both machines now should match.

## Circuit diagram (per layer l)

```
every layer l:
   if upstream encoding exists:
       upstream_gate_fn(x)              ← v09C: applied at EVERY layer

   for q in 0..3:
       RY( dot(weights_enc[l,q], x) )   ← learned reuploading

   for q in 0..3:
       RZ(weights_var[l,q,0])
       RX(weights_var[l,q,1])
   CNOT ring
```

## How to run

```powershell
# Run the v09C ablation (~8 h on CPU, resumable)
python "GSE76809Examples\v09C\compare_encodings.py"

# Subset
python "GSE76809Examples\v09C\compare_encodings.py" --encodings data_reuploading iqp_combined
```

The runner is **resumable**: after every encoding/fold it writes
`results/v09C_encoding_ablation_partial.json`. Re-launch the same command to
resume; completed fold/encoding combinations are skipped automatically.

## How to interpret

Each combined encoding is compared against the `data_reuploading` reference: a
significant positive `mean_diff` means the per-layer encoding adds information
the learned reuploading cannot recover on its own. On GSE76809 every combined
encoding instead *reduces* AUC (see Results) — stacking a fixed encoding on top
of the learned reuploading never helps. `amplitude_combined` is expected to be
the most damaged: `AmplitudeEmbedding` re-prepares the full statevector, so
applying it every layer **overwrites** the accumulated computation.

## A note on the every-layer choice

For rotation-style encodings (angle, dense_angle, iqp), applying the encoding
at every layer is the faithful implementation of the template's "per-layer ...
in combination" wording and of standard data-reuploading. For `amplitude`,
applying `AmplitudeEmbedding` at every layer is architecturally destructive
(it re-prepares the statevector each layer), which is why `amplitude_combined`
degrades the most.

## Alternative architecture: layer-0-only encoding (v09B)

v09C is the *every-layer* reading of "base encoding combined with reuploading".
The natural sibling design — historically prototyped as **v09B** — instead
applies the upstream base encoding at **layer 0 only** (one-shot state
preparation), then runs the learned reuploading on layers 1–7:

| | Upstream encoding applied | Layers 1–7 |
|---|---|---|
| **v09B** (layer-0-only) | layer 0 only — seeds the register once | learned reuploading refines |
| **v09C** (this example) | **every layer**, interleaved | learned reuploading every layer |

Both share an identical frozen architecture (4 qubits, 8 layers, 12-output
measurement, `12 → 64 → 1` post-net, v06 preprocessing, 80 epochs / batch 24 /
lr 0.005, seed 2026) and the same `data_reuploading` reference. The **only**
difference is *where* the upstream gates fire.

### Scenarios where v09B (layer-0-only) is expected to do better

1. **`amplitude` — the clearest win.** `AmplitudeEmbedding` re-prepares the full
   statevector from scratch. Applied at every layer (v09C) it **overwrites** the
   accumulated computation each time, crippling the circuit — exactly why
   `amplitude_combined` is the most Holm-damaged encoding here. As a one-shot
   layer-0 state preparation (v09B) it is used as intended, so the reuploading
   layers can build on it instead of being repeatedly erased. v09B should
   recover most of the AUC lost by `amplitude_combined`.
2. **Rotation encodings under mis-scaled inputs.** The fixed `RY(x·π)` /
   `RZ(x·π)` / `RZ(x_i·x_j)` gates receive StandardScaler z-scores (≈ −3…3),
   not [0,1]/[0,π] values (problem #3). At every layer (v09C) that wrap-around
   angle error compounds 8×; injected once (v09B) it is a single perturbation
   the learned reuploading can largely correct. v09B should suffer less from the
   scaling bug for `angle`/`dense_angle`/`iqp`.
3. **Encodings that add a fixed inductive bias once.** When the base map carries
   a useful prior that reuploading cannot reconstruct, seeding it once and
   letting trainable layers refine (v09B) avoids forcing the same fixed
   transform on top of every learned step (v09C), which tends to dominate and
   scramble the representation.

### Scenarios where v09C (every-layer) is the more faithful choice

- For rotation encodings with **properly bounded inputs**, re-injecting the data
  every layer is the canonical Pérez-Salinas data-reuploading architecture and
  the literal "applied in combination ... per-layer" reading — the design v09C
  is built to test. If the scaling issue in problem #3 is fixed, v09C is the
  theoretically motivated variant for those encodings.

In short: v09B is expected to be strictly better for `amplitude` and more robust
for the rotation encodings *as currently scaled*, while v09C is the intended
architecture for rotation encodings once inputs are bounded. A direct head-to-head
would require running the layer-0-only variant; it is not present in this
workspace.

## Files

| File | Purpose |
|------|---------|
| `quantum_encodings.py` | Upstream gate functions + `UPSTREAM_GATES` dict |
| `model_quantum_vqc.py` | `CombinedEncodingVQC` (every-layer encoding) + `train_quantum_vqc` |
| `compare_encodings.py` | 5-fold CV runner; resumable; writes results JSON |
| `README.md` | This file |

## Output

- `results/v09C_encoding_ablation.json` — per-encoding 5-fold AUCs, mean/std,
  paired tests vs `data_reuploading` (Wilcoxon + Nadeau–Bengio, Holm-corrected).
- `results/v09C_encoding_ablation_partial.json` — rolling checkpoint.
