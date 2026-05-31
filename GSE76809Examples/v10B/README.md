# v10B — Noisy Simulation of the v09 Encoding Ablation

## Goal

Every v01–v09 result is on a **noiseless** statevector simulator. v10B
gives a defensible first-order answer to a specific NISQ question: **is
the encoding ranking that v09 measured a noiseless artifact, or does it
survive realistic gate error?**

v10B re-runs **v09's encoding ablation** on PennyLane's `default.mixed`
density-matrix simulator with a parameterised depolarising channel after
every gate, crossing v09's encoding axis with a noise axis.

## Why v09? (vs v06 / v06B / v07 / v08)

A noise study is only informative when it degrades a quantity that is
genuinely *quantum-internal*. Of the five candidates:

| Example | Core variable | Fit for a noise study |
|---------|---------------|------------------------|
| v06   | VQC vs tuned classical on full data | Weak: the v06 headline is "tuned XGBoost wins", so noise just makes a losing model lose harder. |
| v06B  | Classical feature extractor (ANOVA+PCA vs learned) | Wrong axis: the variable is the *classical* front-end; noise degrades all extractors roughly equally. |
| v07   | Cross-dataset transfer | Incomplete grid (only GSE76809 ran); noise adds an axis to an unfinished one. |
| v08   | Small-data sample efficiency | Strong — covered separately in **v10A**. |
| **v09** | **Encoding ablation** (`data_reuploading` is the measured winner, mean CV AUC 0.788) | **Chosen here.** Encoding is the cleanest *quantum-internal* lever, and noise directly tests whether the re-uploading advantage is real or a noiseless artifact. |

v09 is the right target for an *encoding* noise study because it is the
one experiment that freezes everything except the data-loading stage.
The interesting hypotheses are sharp:

- **Re-uploading is fragile.** It touches the data at every layer, so it
  accumulates the most noisy gates — its advantage may vanish first.
- **Encodings converge under decoherence.** As `noise_p` grows, every
  encoding may collapse toward chance (AUC 0.5) together, erasing the
  v09 ranking.

## Experimental design

For each cell in the grid

```
(encoding)  x  (noise_p)   evaluated with v06's 5-fold CV
```

we run v09's exact 5-fold CV on the frozen v06 pipeline, recording the 5
per-fold AUCs. `noise_p = 0` reproduces v09's noiseless ranking (modulo
density-matrix vs statevector representation).

Per noise level we run a paired **Wilcoxon** + **Nadeau-Bengio corrected
resampled t-test** of each encoding against **the same encoding at
`noise_p = 0`**, so we can name the error rate at which each encoding
becomes statistically indistinguishable from its own noiseless run.

The depolarising channel is inserted after **every** gate, including the
single-qubit rotations and CNOTs emitted *inside* the encoding fragment
(e.g. `iqp`'s Hadamard/RZ/CNOT block), so encodings with deeper loading
stages correctly pay a larger noise penalty.

## Noise model

| Setting              | Value                                                   |
|----------------------|---------------------------------------------------------|
| Device               | `default.mixed` (density-matrix simulator)              |
| Channel              | Depolarising (`DepolarizingChannel`)                    |
| Single-qubit error p | swept: 0, 1e-4, 1e-3, 5e-3, 1e-2                        |
| Two-qubit error      | `min(10 × p, 0.75)` (per NISQ folklore, capped)         |

This is **a model, not the model**. Real superconducting / trapped-ion
devices have correlated, time-dependent, qubit-specific errors. The
depolarising model is the simplest credible stand-in and is the standard
"first noisy experiment" in QML papers.

## Reduced design (compute-aware)

`default.mixed` is ~5–10× slower than `default.qubit`, so v10B is leaner
than v09:

- 5-fold CV (same as v09) but 40 training epochs (vs 80).
- Default to 3 encodings — `data_reuploading` (the winner), `amplitude`
  (the v03/v04 "compression" encoding), and `angle` (the v01-style
  bottleneck) — to keep the grid tractable. Add more with `--encodings`.
- Only the VQC is run; no classical baselines.

## How to run

```powershell
cd GSE76809Examples\v10B
python noise_sweep.py                                       # 3 encodings x 5 noise levels
python noise_sweep.py --noise 0 1e-3                        # 2-level subset
python noise_sweep.py --encodings data_reuploading angle    # encoding subset
```

The run is **resumable**: a checkpoint
(`results/v10B_encoding_noise_partial.json`) is written after every
`(encoding, noise, fold)` cell and reloaded on startup.

## Output

`results/v10B_encoding_noise.json` — per `(encoding, noise_p)` cell:
5-fold mean / std AUC, plus paired Wilcoxon + Nadeau-Bengio tests of each
noisy cell against the **same encoding's noiseless run**.

The headline number to report: **at what single-qubit error rate does the
`data_reuploading` advantage over `angle`/`amplitude` measured in v09
stop being statistically distinguishable, and does re-uploading degrade
faster than the shallower encodings?**

## Files

| File                          | Purpose                                                            |
|-------------------------------|-------------------------------------------------------------------|
| `model_quantum_vqc_noisy.py`  | v09 `EncodingVQC` ported to `default.mixed`; noise follows every gate, including those inside the encoding fragment |
| `noise_sweep.py`              | Driver — sweeps `(encoding × noise_p)` over 5-fold CV, resumable   |

## Caveats

- We do **not** include measurement (shot) noise here; deferred to a
  follow-up.
- `noise_p = 0` on `default.mixed` is numerically equivalent to v09's
  `default.qubit` run but not bit-identical (different backend).
- Backprop on `default.mixed` is supported but slow; consider
  parameter-shift if you only want a single noise level.
