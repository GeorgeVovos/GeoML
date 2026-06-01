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

### Resume behaviour

The run is **resumable**. After every `(encoding, noise_p, fold)` cell the
driver writes a checkpoint to `results/v10B_encoding_noise_partial.json`.
On startup, if that file exists it is loaded and any cell where
`len(aucs) > fold_idx` is skipped instantly. At most **one cell's worth
of compute** (~10–40 min depending on `--epochs`) is lost on a crash —
the partial cell is simply retried on the next run.

> **Epoch-upgrade caveat.** The resume logic skips a cell based on the
> number of completed folds, not the epoch count used. If you ran
> `--epochs 10` on night 1 and want `--epochs 40` on night 2, delete the
> partial checkpoint first so all cells are retrained at the higher
> budget. Otherwise only *new* cells (new encodings or noise levels) will
> use the higher epoch count.

### Tuning the run length

There are three independent axes to dial up, each with a different
scientific payoff:

**`--epochs` — training quality per cell**

| Value | Time / cell | Notes |
|-------|-------------|-------|
| 10    | ~10 min     | Fast signal check; models underfit but noise signal is visible |
| 20    | ~20 min     | Recommended minimum for a reportable result |
| 40    | ~40 min     | Matches v09's reduced budget; best convergence |
| 80    | ~80 min     | Matches v09 exactly; overkill under high noise |

Most important for the `noise_p = 0` column — you want that reference to
sit close to v09's noiseless 0.788.

**`--noise` — resolution of the noise curve**

| Config | Levels | What you learn |
|--------|--------|----------------|
| `0 1e-3 1e-2` | 3 | Does it survive at 1e-3? Does it die at 1e-2? |
| `0 1e-4 1e-3 5e-3 1e-2` | 5 | Full curve — where exactly the cliff is |
| `0 1e-4 5e-4 1e-3 5e-3 1e-2` | 6 | Fine-grained around the 1e-4–1e-3 region |

More levels let you pinpoint the critical error rate at which
re-uploading's advantage disappears — this is the headline number.

**`--encodings` — breadth of the comparison**

| Config | Value |
|--------|-------|
| `data_reuploading amplitude angle` | 3 — winner + two contrasts (default) |
| `data_reuploading amplitude angle iqp` | 4 — adds IQP, which has CNOTs *inside* the encoding stage and may degrade faster |
| `data_reuploading amplitude angle iqp dense_angle` | 5 — full v09 replication |

`iqp` is the most scientifically interesting addition: its encoding emits
extra CNOTs (2-qubit gates carry a 10× noise penalty), so it may degrade
faster than `angle` despite being the stronger noiseless encoding.

### Runtime estimates (5 folds, measured on CPU)

| `--epochs` | noise levels | encodings | cells | Est. wall time |
|------------|-------------|-----------|-------|----------------|
| 10  | 3 | 3 | 45  | ~7–8 h  ← overnight minimum |
| 20  | 5 | 3 | 75  | ~25 h   |
| 40  | 5 | 3 | 75  | ~50 h   |
| 20  | 5 | 5 | 125 | ~42 h   |
| 40  | 5 | 5 | 125 | ~83 h   |

### Recommended multi-night progression

Because the run is resumable, you can build the full dataset incrementally —
each night extends the previous checkpoint.

**Night 1** — fast signal check (~7–8 h):
```powershell
python noise_sweep.py --noise 0 1e-3 1e-2 --encodings data_reuploading amplitude angle --epochs 10
```

**Night 2** — fill in the full noise curve with better training
(skips night-1 cells, only runs new noise levels at `--epochs 20`):
```powershell
python noise_sweep.py --noise 0 1e-4 1e-3 5e-3 1e-2 --encodings data_reuploading amplitude angle --epochs 20
```
> Delete `v10B_encoding_noise_partial.json` first if you also want to
> retrain the night-1 cells at 20 epochs.

**Night 3** (optional) — add `iqp`, match v09's epoch budget:
```powershell
python noise_sweep.py --noise 0 1e-4 1e-3 5e-3 1e-2 --encodings data_reuploading amplitude angle iqp --epochs 40
```

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
