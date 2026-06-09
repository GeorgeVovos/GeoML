"""v10A noise sweep — does the v08 small-data behaviour survive NISQ noise?

v08 measured how the data-reuploading VQC's holdout AUC scales with the
number of training samples (1:1 stratified subsamples evaluated on the
fixed v06 holdout). v10A re-runs *only that VQC* under depolarising
noise, crossing the v08 sample-size axis with a noise axis:

	(N_per_class)  x  (subsample_seed)  x  (noise_p)

For each cell we record the holdout AUC; for noise_p = 0 we reproduce
v08's noiseless small-data curve, and the larger noise levels show how
fast decoherence erodes the small-N vs large-N AUC.

Compute-aware reduced design (default.mixed is ~5-10x slower):
- Reduced N grid and fewer subsample seeds than v08.
- Single VQC init per subsample (v08 used 3).
- 40 training epochs (v08 used 80) — under noise, training plateaus
  sooner and the gradients are biased anyway.

Resumable: a checkpoint JSON is written after every cell, and an existing
checkpoint is loaded on startup so a crashed run continues where it
stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))           # shared.*
sys.path.insert(0, str(_THIS.parent / "v06"))   # v06 preprocess helpers
sys.path.insert(0, str(_THIS.parent / "v08"))   # v08 preprocess + fold harness

from shared.subsampling import stratified_subsample  # noqa: E402
from preprocess import preprocess, build_fold_features, apply_smote_to_fold  # noqa: E402

from model_quantum_vqc_noisy import train_noisy_vqc  # noqa: E402


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer): return int(obj)
		if isinstance(obj, np.floating): return float(obj)
		if isinstance(obj, np.bool_): return bool(obj)
		if isinstance(obj, np.ndarray): return obj.tolist()
		return super().default(obj)


DEFAULT_NOISE = [0.0, 1e-4, 1e-3, 5e-3, 1e-2]
DEFAULT_N_PER_CLASS = [10, 20, 50]
DEFAULT_SUBSAMPLE_SEEDS = 5


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, rng_seed: int = 2026):
	"""Percentile bootstrap CI of the mean (same as v08)."""
	arr = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
	if len(arr) < 2:
		return (float("nan"), float("nan"))
	rng = np.random.RandomState(rng_seed)
	means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
	return (float(np.percentile(means, 100 * alpha / 2)),
			float(np.percentile(means, 100 * (1 - alpha / 2))))


def make_fold(data, train_idx, n_features: int = 16, pca_components: int = 6,
			  random_state: int = 2026):
	"""Leakage-safe fold built from only the subsample's training rows (v08 logic)."""
	X_train_raw = data["X_train_raw"][train_idx]
	y_train = data["y_train"][train_idx]
	fold = build_fold_features(
		X_train_raw, y_train, data["X_test_raw"],
		n_features=min(n_features, X_train_raw.shape[1]),
		pca_components=min(pca_components, X_train_raw.shape[1]),
		random_state=random_state,
	)
	fold["y_val"] = data["y_test"]
	return fold


def apply_smote_if_needed(fold, random_state):
	"""SMOTE'd copy for the VQC (train loop is called with use_smote=False)."""
	try:
		X_aug, y_aug = apply_smote_to_fold(fold["X_train"], fold["y_train"],
										   random_state=random_state)
	except Exception:
		return fold
	out = dict(fold)
	out["X_train"] = X_aug
	out["y_train"] = y_aug
	return out


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--noise", type=float, nargs="+", default=DEFAULT_NOISE)
	ap.add_argument("--n-per-class", type=int, nargs="+", default=DEFAULT_N_PER_CLASS)
	ap.add_argument("--subsample-seeds", type=int, default=DEFAULT_SUBSAMPLE_SEEDS)
	ap.add_argument("--epochs", type=int, default=40)
	args = ap.parse_args()

	print(f"Noise levels    : {args.noise}")
	print(f"N_per_class grid: {args.n_per_class}")
	print(f"Subsample seeds : {args.subsample_seeds}")

	start = time.time()
	data = preprocess()

	out_dir = _THIS / "results"
	out_dir.mkdir(exist_ok=True)
	ckpt_path = out_dir / "v10A_noise_sample_efficiency_partial.json"

	# raw[noise_p][n_per_class] = list of holdout AUCs over subsample seeds
	if ckpt_path.exists():
		with open(ckpt_path, "r") as f:
			raw = json.load(f)
		print(f"Resuming from checkpoint: {ckpt_path}")
	else:
		raw = {}
	for p in args.noise:
		raw.setdefault(str(p), {})
		for n in args.n_per_class:
			raw[str(p)].setdefault(str(n), [])

	rng = np.random.RandomState(2026)
	sub_seeds = rng.randint(0, 10**6, size=args.subsample_seeds)
	y_train = data["y_train"]
	class_min = int(np.bincount(y_train.astype(int)).min())

	for n in args.n_per_class:
		if n > class_min:
			print(f"\n[N_per_class={n}] SKIP — class minimum is {class_min}")
			continue
		print(f"\n{'='*72}\n[N_per_class={n}  total={2*n}]\n{'='*72}")
		for s_idx, sub_seed in enumerate(sub_seeds):
			sub_rng = np.random.RandomState(int(sub_seed))
			try:
				idx = stratified_subsample(y_train, n_per_class=n, rng=sub_rng)
			except ValueError as exc:
				print(f"  seed {sub_seed}: {exc}")
				continue
			print(f"\n  -- subsample seed {sub_seed} ({s_idx+1}/{len(sub_seeds)}) --")

			fold_raw = make_fold(data, idx, random_state=int(sub_seed))
			fold_smote = apply_smote_if_needed(fold_raw, random_state=int(sub_seed))

			for p in args.noise:
				bucket = raw[str(p)][str(n)]
				# Resume: skip cells already filled for this (n, p).
				if len(bucket) > s_idx:
					print(f"    [resume] skip noise_p={p}")
					continue
				t0 = time.time()
				try:
					res = train_noisy_vqc(
						fold_data=fold_smote, noise_p=p,
						epochs=args.epochs, use_smote=False,
						random_state=int(sub_seed), init_seed=0,
					)
					auc = res["auc_roc"]
				except Exception as exc:
					print(f"    [noise_p={p}] FAILED: {exc}")
					auc = float("nan")
				bucket.append(auc)
				print(f"    [noise_p={p:<8}] AUC={auc:.4f}  ({time.time()-t0:.1f}s)")
				with open(ckpt_path, "w") as f:
					json.dump(raw, f, indent=2, cls=NumpyEncoder)

	# Summarise.
	summary = {}
	for p in args.noise:
		summary[str(p)] = {}
		for n in args.n_per_class:
			vals = raw[str(p)].get(str(n), [])
			arr = np.asarray([v for v in vals if not np.isnan(v)], dtype=float)
			if len(arr) == 0:
				summary[str(p)][str(n)] = {"n_samples": 0}
				continue
			lo, hi = bootstrap_ci(arr)
			summary[str(p)][str(n)] = {
				"n_samples": int(len(arr)),
				"mean": float(arr.mean()),
				"std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
				"min": float(arr.min()),
				"max": float(arr.max()),
				"ci_low": lo,
				"ci_high": hi,
				"aucs": arr.tolist(),
			}

	print("\nNoise x sample-size AUC (mean):")
	for p in args.noise:
		row = "  ".join(
			f"N={n}:{summary[str(p)][str(n)].get('mean', float('nan')):.3f}"
			for n in args.n_per_class
		)
		print(f"  noise_p={p:<8}  {row}")

	elapsed_min = (time.time() - start) / 60
	out = {
		"metadata": {
			"version": "v10A",
			"targets": "v08",
			"purpose": "depolarising-noise sweep crossed with v08's small-data "
					   "sample-size axis (data-reuploading VQC on default.mixed)",
			"noise_levels": args.noise,
			"n_per_class": args.n_per_class,
			"subsample_seeds": args.subsample_seeds,
			"epochs": args.epochs,
			"device": "default.mixed",
			"test_set": "v06 GSE76809 holdout (54 samples, stratified 80/20, seed=2026)",
			"runtime_minutes": elapsed_min,
		},
		"summary": summary,
	}
	out_path = out_dir / "v10A_noise_sample_efficiency.json"
	with open(out_path, "w") as f:
		json.dump(out, f, indent=2, cls=NumpyEncoder)
	print(f"\nSaved {out_path} ({elapsed_min:.1f} min)")


if __name__ == "__main__":
	main()
