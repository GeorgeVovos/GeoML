"""v08 — pure sample-efficiency sweep.

For each N_per_class in the chosen grid, for each of `subsample_seeds`
random 1:1 stratified subsamples, train every selected model on the
subsample and evaluate on the FIXED v06 holdout test set. VQC additionally
gets `vqc_init_seeds` independent random initialisations per subsample.

Records per-cell AUC distributions, mean, std, and a bootstrap 95 % CI.

This is the experiment that directly tests the small-data quantum advantage
hypothesis. Defaults are heavy (designed to be run overnight). Use
``--quick`` / ``--skip-quantum-kernel`` to iterate.
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
sys.path.insert(0, str(_THIS.parent / "v06"))   # v06 models

from shared.subsampling import stratified_subsample  # noqa: E402
from shared.model_classical_logreg import train_classical_logreg  # noqa: E402

from preprocess import preprocess, build_fold_features, apply_smote_to_fold  # noqa: E402

from model_quantum_vqc import train_quantum_vqc  # noqa: E402
from model_quantum_kernel import train_quantum_kernel  # noqa: E402
from model_classical_mlp import train_classical_mlp  # noqa: E402
from model_classical_svm import train_classical_svm  # noqa: E402
from model_classical_xgb import train_classical_xgb  # noqa: E402


MODEL_REGISTRY = {
	# name             : (fn, needs_smote_in_call, supports_init_seed)
	"quantum_vqc":      (train_quantum_vqc, True, True),
	"quantum_kernel":   (train_quantum_kernel, False, False),
	"classical_mlp":    (train_classical_mlp, True, False),
	"classical_svm":    (train_classical_svm, False, False),
	"classical_xgb":    (train_classical_xgb, False, False),
	"classical_logreg": (train_classical_logreg, False, False),
}

DEFAULT_N_PER_CLASS = [5, 10, 15, 20, 30, 50, 75, 100]
DEFAULT_SUBSAMPLE_SEEDS = 20
DEFAULT_VQC_INIT_SEEDS = 3


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		if isinstance(obj, np.floating):
			return float(obj)
		if isinstance(obj, np.bool_):
			return bool(obj)
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return super().default(obj)


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05,
				  rng_seed: int = 2026):
	"""Percentile bootstrap CI of the mean."""
	if len(values) < 2:
		return (float("nan"), float("nan"))
	rng = np.random.RandomState(rng_seed)
	arr = np.asarray(values, dtype=float)
	arr = arr[~np.isnan(arr)]
	if len(arr) < 2:
		return (float("nan"), float("nan"))
	means = [rng.choice(arr, size=len(arr), replace=True).mean()
			 for _ in range(n_boot)]
	return (float(np.percentile(means, 100 * alpha / 2)),
			float(np.percentile(means, 100 * (1 - alpha / 2))))


def make_fold(data, train_idx, n_features: int = 16, pca_components: int = 6,
			  random_state: int = 2026):
	"""Build a leakage-safe fold dict using only the subsample's training rows."""
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
	"""SMOTE'd copy for VQC/MLP (their training loop expects already-balanced data
	when use_smote=False is passed). Falls back to original if SMOTE fails on
	very small N."""
	try:
		X_aug, y_aug = apply_smote_to_fold(
			fold["X_train"], fold["y_train"], random_state=random_state,
		)
	except Exception:
		return fold
	out = dict(fold)
	out["X_train"] = X_aug
	out["y_train"] = y_aug
	out["X_train_norm"] = X_aug / (np.linalg.norm(X_aug, axis=1, keepdims=True) + 1e-10)
	return out


def run_cell(model_name, fold_for_classical, fold_for_deep, init_seed):
	"""Train one model on one subsample once. Returns AUC."""
	fn, needs_smote, supports_init = MODEL_REGISTRY[model_name]
	fold = fold_for_deep if needs_smote else fold_for_classical
	kwargs = {}
	if supports_init:
		kwargs["init_seed"] = init_seed
	if needs_smote:
		# SMOTE was already applied (if applicable) in fold_for_deep, so tell
		# the model not to apply it again.
		kwargs["use_smote"] = False
	return fn(fold_data=fold, **kwargs)["auc_roc"]


def sweep(data, n_per_class_grid, subsample_seeds, vqc_init_seeds, models,
		   random_state: int = 2026):
	"""Return nested dict: {model: {n_per_class: [aucs over all repeats]}}."""
	out = {m: {n: [] for n in n_per_class_grid} for m in models}
	y_train = data["y_train"]

	rng = np.random.RandomState(random_state)
	# Pre-generate subsample seeds so models see identical subsamples
	sub_seeds = rng.randint(0, 10**6, size=subsample_seeds)

	for n in n_per_class_grid:
		# Skip N values where either class doesn't have enough samples.
		class_counts = np.bincount(y_train.astype(int))
		if n > class_counts.min():
			print(f"\n[N_per_class={n}] SKIP — class minimum is {class_counts.min()}")
			continue

		print(f"\n{'='*72}\n[N_per_class={n}  total={2*n}]\n{'='*72}")
		for s_idx, sub_seed in enumerate(sub_seeds):
			sub_rng = np.random.RandomState(sub_seed)
			try:
				idx = stratified_subsample(y_train, n_per_class=n, rng=sub_rng)
			except ValueError as exc:
				print(f"  seed {sub_seed}: {exc}")
				continue

			print(f"\n  -- subsample seed {sub_seed} ({s_idx+1}/{len(sub_seeds)}) --")

			fold_raw = make_fold(data, idx, random_state=int(sub_seed))
			fold_smote = apply_smote_if_needed(fold_raw, random_state=int(sub_seed))

			for m in models:
				if m == "quantum_kernel" and n < 15:
					out[m][n].append(float("nan"))
					continue
				supports_init = MODEL_REGISTRY[m][2]
				init_seeds = (list(range(vqc_init_seeds))
							  if supports_init else [None])
				for init in init_seeds:
					t0 = time.time()
					try:
						auc = run_cell(m, fold_raw, fold_smote, init_seed=init)
					except Exception as exc:
						print(f"    [{m}] FAILED: {exc}")
						auc = float("nan")
					out[m][n].append(auc)
					label = f"init={init}" if init is not None else "-"
					print(f"    [{m:<18}] {label:<8} AUC={auc if isinstance(auc, float) else auc:.4f}"
						  f"  ({time.time()-t0:.1f}s)")
	return out


def summarise(sweep_out):
	summary = {}
	for m, ndict in sweep_out.items():
		summary[m] = {}
		for n, aucs in ndict.items():
			arr = np.asarray([a for a in aucs if not np.isnan(a)], dtype=float)
			if len(arr) == 0:
				summary[m][n] = {"n_samples": 0}
				continue
			lo, hi = bootstrap_ci(arr)
			summary[m][n] = {
				"n_samples": int(len(arr)),
				"mean": float(arr.mean()),
				"std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
				"min": float(arr.min()),
				"max": float(arr.max()),
				"ci_low": lo,
				"ci_high": hi,
				"aucs": arr.tolist(),
			}
	return summary


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--quick", action="store_true",
					help="5 subsample seeds, only N in {10,20,50,100}")
	ap.add_argument("--skip-quantum-kernel", action="store_true")
	ap.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
					help="subset of models to evaluate")
	ap.add_argument("--n-per-class", type=int, nargs="+", default=DEFAULT_N_PER_CLASS)
	ap.add_argument("--subsample-seeds", type=int, default=DEFAULT_SUBSAMPLE_SEEDS)
	ap.add_argument("--vqc-init-seeds", type=int, default=DEFAULT_VQC_INIT_SEEDS)
	args = ap.parse_args()

	if args.quick:
		args.subsample_seeds = 5
		args.vqc_init_seeds = 2
		args.n_per_class = [10, 20, 50, 100]
	if args.skip_quantum_kernel and "quantum_kernel" in args.models:
		args.models = [m for m in args.models if m != "quantum_kernel"]

	bad = [m for m in args.models if m not in MODEL_REGISTRY]
	if bad:
		raise SystemExit(f"Unknown models: {bad}. Known: {list(MODEL_REGISTRY)}")

	print(f"Models: {args.models}")
	print(f"N_per_class grid: {args.n_per_class}")
	print(f"Subsample seeds: {args.subsample_seeds}, VQC init seeds: {args.vqc_init_seeds}")

	start = time.time()
	data = preprocess()
	raw = sweep(
		data,
		n_per_class_grid=args.n_per_class,
		subsample_seeds=args.subsample_seeds,
		vqc_init_seeds=args.vqc_init_seeds,
		models=args.models,
	)
	summary = summarise(raw)

	elapsed_min = (time.time() - start) / 60
	print(f"\nTotal runtime: {elapsed_min:.1f} minutes")

	out = {
		"metadata": {
			"version": "v08",
			"purpose": "pure sample-efficiency sweep with 1:1 stratified subsampling",
			"n_per_class": args.n_per_class,
			"subsample_seeds": args.subsample_seeds,
			"vqc_init_seeds": args.vqc_init_seeds,
			"models": args.models,
			"test_set": "v06 GSE76809 holdout (54 samples, stratified 80/20, seed=2026)",
			"runtime_minutes": elapsed_min,
		},
		"summary": summary,
	}
	out_dir = _THIS / "results"
	out_dir.mkdir(exist_ok=True)
	out_path = out_dir / "v08_sample_efficiency.json"
	with open(out_path, "w") as f:
		json.dump(out, f, indent=2, cls=NumpyEncoder)
	print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
	main()
