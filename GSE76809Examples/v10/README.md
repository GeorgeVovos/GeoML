# v10 — Noisy Simulation (NISQ Realism Check)

## Goal

Every v01-v09 result so far is on a noiseless statevector simulator.
The open question is whether any of it survives on real hardware. A
noisy simulation with realistic NISQ gate-error rates provides a
defensible first-order answer without requiring real-device access.

v10 re-runs v06's data-reuploading VQC on `default.mixed` with a
parameterised single-qubit depolarising channel inserted after every
single-qubit gate and a two-qubit depolarising channel after every CNOT.

## Noise model

| Setting              | Value                                                  |
|----------------------|--------------------------------------------------------|
| Device               | `default.mixed` (density-matrix simulator)             |
| Channel              | Depolarising (`DepolarizingChannel`)                    |
| Single-qubit error p | swept: 0, 1e-4, 1e-3, 5e-3, 1e-2                       |
| Two-qubit error      | 10 × single-qubit p (per NISQ folklore)                 |

This is **a model, not the model**. Real superconducting / trapped-ion
devices have correlated, time-dependent, qubit-specific errors. The
depolarising model is the simplest credible stand-in and is the standard
"first noisy experiment" in QML papers.

## Reduced design (compute-aware)

- Only the VQC is rerun (the quantum kernel under noise would dominate
  runtime).
- 5-fold CV, no holdout, no learning curve.
- Reduced epoch budget (40 vs 80) — under noise, training plateaus
  sooner and the gradients are estimator-style biased anyway.
- 4 qubits / 8 layers (matches v06).

## How to run

```powershell
cd GSE76809Examples\v10
python noise_sweep.py                          # full sweep (~3-4 hours)
python noise_sweep.py --noise 0 1e-3           # subset (~80 min)
```

## Output

`results/v10_noise_sweep.json` — per noise level: 5-fold CV mean/std AUC,
plus a paired Nadeau-Bengio test against the noiseless run.

The headline number to report: **at what noise level does the v06
small-data quantum advantage from v06/v08 disappear?**

## Files

| File                      | Purpose                                       |
|---------------------------|------------------------------------------------|
| `model_quantum_vqc_noisy.py` | v06 VQC ported to default.mixed with depolarising channels |
| `noise_sweep.py`             | Driver — sweeps noise levels, runs 5-fold CV  |

## Caveats

- `default.mixed` is slower than `default.qubit` by ~5-10x because it
  tracks the full density matrix (2^(2n) entries instead of 2^n). Each
  fold takes ~10-15 min vs ~3 min noiseless.
- We do not include measurement noise (sampling shot noise) here. Adding
  shot noise via `qml.sample`+`shots=N` is straightforward but costs
  another order of magnitude in wall time; defer to a follow-up.
- Backpropagation works for `default.mixed` but is slow. We use
  `diff_method="backprop"` to keep training stable; consider
  parameter-shift if you want a single-noise-level evaluation only.
