"""v07 cross-dataset replication driver.

Runs v06's exact model code on multiple SSc datasets to test whether the
small-data quantum-advantage signal seen on GSE76809 reproduces elsewhere.

Reuses (NOT copies) v06's model files via sys.path. Adds LR-L1 from
shared/. Stats use the corrected Nadeau-Bengio test + Holm correction so
the cross-dataset claims are honestly thresholded.

CLI:
	python compare_models.py                       # all 4 datasets
	python compare_models.py --datasets GSE9285    # subset
	python compare_models.py --skip-quantum-kernel # faster, skips ~5 min/dataset
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))               # for shared.*
sys.path.insert(0, str(_THIS.parent / "v06"))       # for v06 model_*.py

from shared.stats_utils import (  # noqa: E402
	cohens_d, effect_size_label, wilcoxon_signed_rank,
	corrected_resampled_ttest, holm_bonferroni,
)
from shared.model_classical_logreg import train_classical_logreg  # noqa: E402

from preprocess import preprocess  # noqa: E402

# v06 models (imported unchanged — that is the whole point of v07)
from model_quantum_vqc import train_quantum_vqc  # noqa: E402
from model_quantum_kernel import train_quantum_kernel  # noqa: E402
from model_classical_mlp import train_classical_mlp  # noqa: E402
from model_classical_svm import train_classical_svm  # noqa: E402
from model_classical_xgb import train_classical_xgb  # noqa: E402


DEFAULT_DATASETS = ["GSE76809", "GSE9285", "GSE58095", "GSE45536"]


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


def run_cv_for_dataset(data, skip_quantum_kernel: bool = False) -> dict:
	"""5-fold CV for all models on one preprocessed dataset."""
	model_keys = ["quantum_vqc", "classical_mlp", "classical_xgb",
				  "classical_svm", "classical_logreg"]
	if not skip_quantum_kernel:
		model_keys.append("quantum_kernel")

	results = {m: {"aucs": [], "accs": [], "f1s": []} for m in model_keys}

	for i, fold in enumerate(data["folds"]):
		print(f"\n  -- fold {i+1}/{len(data['folds'])} --")
		results["quantum_vqc"]["aucs"].append(
			train_quantum_vqc(fold_data=fold)["auc_roc"])
		results["classical_mlp"]["aucs"].append(
			train_classical_mlp(fold_data=fold)["auc_roc"])
		results["classical_xgb"]["aucs"].append(
			train_classical_xgb(fold_data=fold)["auc_roc"])
		results["classical_svm"]["aucs"].append(
			train_classical_svm(fold_data=fold)["auc_roc"])
		results["classical_logreg"]["aucs"].append(
			train_classical_logreg(fold_data=fold)["auc_roc"])
		if not skip_quantum_kernel:
			results["quantum_kernel"]["aucs"].append(
				train_quantum_kernel(fold_data=fold)["auc_roc"])

	return results


def per_dataset_stats(cv_results: dict, n_train_approx: int, n_test_approx: int):
	"""Comparisons against LR-L1 (the small-data linear baseline).

	Reports paired-t, Wilcoxon, Nadeau-Bengio corrected t, Cohen's d, and
	Holm-adjusted p-values across the comparisons in this dataset.
	"""
	if "classical_logreg" not in cv_results:
		return []

	baseline = "classical_logreg"
	others = [m for m in cv_results if m != baseline]
	base_aucs = np.asarray(cv_results[baseline]["aucs"])

	rows = []
	p_for_holm = []
	for model in others:
		aucs = np.asarray(cv_results[model]["aucs"])
		if len(aucs) < 2:
			continue
		w_stat, w_p = wilcoxon_signed_rank(aucs, base_aucs)
		nb_t, nb_p = corrected_resampled_ttest(
			aucs, base_aucs, n_train=n_train_approx, n_test=n_test_approx,
		)
		d = cohens_d(aucs, base_aucs)
		rows.append({
			"model_a": model, "model_b": baseline,
			"mean_a": float(aucs.mean()), "mean_b": float(base_aucs.mean()),
			"wilcoxon_p": w_p,
			"nb_corrected_p": nb_p,
			"cohens_d": d,
			"effect_size": effect_size_label(d),
		})
		p_for_holm.append(nb_p)

	adj = holm_bonferroni(p_for_holm)
	for r, a in zip(rows, adj):
		r["holm_adjusted_p"] = a
		r["holm_significant"] = a < 0.05
	return rows


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
	ap.add_argument("--skip-quantum-kernel", action="store_true",
					help="skip the slowest model (~5 min/dataset)")
	args = ap.parse_args()

	start = time.time()
	all_results = {"metadata": {
		"version": "v07",
		"purpose": "cross-dataset replication of v06",
		"datasets": args.datasets,
		"seed": 2026,
	}, "datasets": {}}

	for gse in args.datasets:
		print(f"\n{'#'*72}\n# DATASET: {gse}\n{'#'*72}")
		try:
			data = preprocess(gse)
		except Exception as exc:
			print(f"  preprocess failed for {gse}: {exc}")
			all_results["datasets"][gse] = {"error": str(exc)}
			continue

		cv = run_cv_for_dataset(data, skip_quantum_kernel=args.skip_quantum_kernel)
		n = len(data["X_train"])
		stats_rows = per_dataset_stats(
			cv, n_train_approx=int(n * 0.8), n_test_approx=int(n * 0.2),
		)

		print(f"\n  -- {gse} CV summary --")
		for m, r in sorted(cv.items(),
						   key=lambda kv: np.mean(kv[1]["aucs"]) if kv[1]["aucs"] else 0,
						   reverse=True):
			aucs = r["aucs"]
			if aucs:
				print(f"    {m:<22} mean AUC {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")

		all_results["datasets"][gse] = {
			"cv": {m: {
				"aucs": r["aucs"],
				"mean_auc": float(np.mean(r["aucs"])) if r["aucs"] else None,
				"std_auc": float(np.std(r["aucs"])) if r["aucs"] else None,
			} for m, r in cv.items()},
			"stats_vs_lr_l1": stats_rows,
			"n_samples": int(len(data["X_train"]) + len(data["X_test"])),
			"n_features": int(data["n_features"]),
		}

	elapsed_min = (time.time() - start) / 60
	all_results["metadata"]["total_runtime_minutes"] = elapsed_min
	print(f"\nTotal runtime: {elapsed_min:.1f} minutes")

	out_dir = _THIS / "results"
	out_dir.mkdir(exist_ok=True)
	out_path = out_dir / "v07_cross_dataset_results.json"
	with open(out_path, "w") as f:
		json.dump(all_results, f, indent=2, cls=NumpyEncoder)
	print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
	main()
