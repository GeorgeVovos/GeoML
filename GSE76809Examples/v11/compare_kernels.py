"""v11 driver — compare PQK, QKA, plain quantum kernel, RBF, and LR-L1
on the same v06 preprocessing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS.parent / "v06"))

from shared.stats_utils import (  # noqa: E402
	cohens_d, effect_size_label, wilcoxon_signed_rank,
	corrected_resampled_ttest, holm_bonferroni,
)
from shared.model_classical_logreg import train_classical_logreg  # noqa: E402

from preprocess_gse76809 import preprocess  # noqa: E402
from model_quantum_kernel import train_quantum_kernel  # noqa: E402
from model_classical_svm import train_classical_svm  # noqa: E402

from projected_kernel import train_projected_kernel  # noqa: E402
from aligned_kernel import train_aligned_kernel  # noqa: E402


REGISTRY = {
	"pqk": train_projected_kernel,
	"qka": train_aligned_kernel,
	"plain": train_quantum_kernel,
	"rbf": train_classical_svm,
	"lr_l1": train_classical_logreg,
}


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer): return int(obj)
		if isinstance(obj, np.floating): return float(obj)
		if isinstance(obj, np.bool_): return bool(obj)
		if isinstance(obj, np.ndarray): return obj.tolist()
		return super().default(obj)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--kernels", nargs="+", default=list(REGISTRY.keys()))
	args = ap.parse_args()
	bad = [k for k in args.kernels if k not in REGISTRY]
	if bad:
		raise SystemExit(f"unknown kernels {bad}; known {list(REGISTRY)}")

	start = time.time()
	data = preprocess()

	cv = {k: {"aucs": [], "accs": [], "f1s": []} for k in args.kernels}
	for fold_idx, fold in enumerate(data["folds"]):
		print(f"\n=== Fold {fold_idx+1}/{len(data['folds'])} ===")
		for k in args.kernels:
			fn = REGISTRY[k]
			try:
				res = fn(fold_data=fold)
				cv[k]["aucs"].append(res["auc_roc"])
				cv[k]["accs"].append(res["accuracy"])
				cv[k]["f1s"].append(res["f1_score"])
			except Exception as exc:
				print(f"  [{k}] FAILED: {exc}")
				cv[k]["aucs"].append(float("nan"))
				cv[k]["accs"].append(float("nan"))
				cv[k]["f1s"].append(float("nan"))

	print("\nKernel comparison CV-AUC:")
	for k, r in sorted(cv.items(),
					   key=lambda kv: np.nanmean(kv[1]["aucs"]) if kv[1]["aucs"] else 0,
					   reverse=True):
		a = np.asarray(r["aucs"], dtype=float)
		a = a[~np.isnan(a)]
		if len(a) == 0:
			print(f"  {k:<8}  all NaN")
		else:
			print(f"  {k:<8}  mean={a.mean():.4f}  +/- {a.std():.4f}")

	# Paired tests vs RBF (the strongest classical kernel baseline)
	stats_rows = []
	if "rbf" in cv:
		ref = np.asarray(cv["rbf"]["aucs"], dtype=float)
		ps = []
		for k in args.kernels:
			if k == "rbf":
				continue
			a = np.asarray(cv[k]["aucs"], dtype=float)
			mask = ~(np.isnan(a) | np.isnan(ref))
			if mask.sum() < 2:
				continue
			_, wp = wilcoxon_signed_rank(a[mask], ref[mask])
			_, nbp = corrected_resampled_ttest(
				a[mask], ref[mask],
				n_train=len(data["X_train"]) * 4 // 5,
				n_test=len(data["X_train"]) // 5,
			)
			d = cohens_d(a[mask], ref[mask])
			stats_rows.append({
				"kernel": k, "vs": "rbf",
				"mean_diff": float(a[mask].mean() - ref[mask].mean()),
				"wilcoxon_p": wp, "nb_corrected_p": nbp,
				"cohens_d": d, "effect_size": effect_size_label(d),
			})
			ps.append(nbp)
		for row, adj in zip(stats_rows, holm_bonferroni(ps)):
			row["holm_adjusted_p"] = adj
			row["holm_significant"] = adj < 0.05

	elapsed_min = (time.time() - start) / 60
	out = {
		"metadata": {
			"version": "v11", "purpose": "PQK and QKA vs plain quantum kernel",
			"kernels": args.kernels, "runtime_minutes": elapsed_min,
		},
		"cv": {k: {
			"aucs": r["aucs"],
			"mean_auc": float(np.nanmean(r["aucs"])) if r["aucs"] else None,
			"std_auc": float(np.nanstd(r["aucs"])) if r["aucs"] else None,
		} for k, r in cv.items()},
		"paired_stats_vs_rbf": stats_rows,
	}
	out_dir = _THIS / "results"
	out_dir.mkdir(exist_ok=True)
	out_path = out_dir / "v11_kernels_comparison.json"
	with open(out_path, "w") as f:
		json.dump(out, f, indent=2, cls=NumpyEncoder)
	print(f"\nSaved {out_path} ({elapsed_min:.1f} min)")


if __name__ == "__main__":
	main()
