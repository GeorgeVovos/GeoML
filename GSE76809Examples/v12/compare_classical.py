"""v12 — strong classical baseline pass on the v06 folds.

Adds LR-EN, linear SVM, RF, GNB, NSC, and a stacked ensemble to the
v06 LR-L1 / RBF / MLP / XGBoost / VQC / quantum-kernel comparison and
reports paired stats vs the small-data linear champion (LR-L1).
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

from extra_baselines import (  # noqa: E402
	train_lr_elasticnet, train_linear_svm, train_random_forest,
	train_gnb, train_nsc, train_stacked,
)

REGISTRY = {
	"lr_l1": train_classical_logreg,
	"lr_en": train_lr_elasticnet,
	"lin_svm": train_linear_svm,
	"rf": train_random_forest,
	"gnb": train_gnb,
	"nsc": train_nsc,
	"stacked": train_stacked,
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
	ap.add_argument("--models", nargs="+", default=list(REGISTRY.keys()))
	args = ap.parse_args()
	bad = [m for m in args.models if m not in REGISTRY]
	if bad:
		raise SystemExit(f"unknown models {bad}; known {list(REGISTRY)}")

	start = time.time()
	data = preprocess()

	cv = {m: {"aucs": [], "accs": [], "f1s": []} for m in args.models}
	for fold_idx, fold in enumerate(data["folds"]):
		print(f"\n=== Fold {fold_idx+1}/{len(data['folds'])} ===")
		for m in args.models:
			fn = REGISTRY[m]
			try:
				res = fn(fold_data=fold)
				cv[m]["aucs"].append(res["auc_roc"])
				cv[m]["accs"].append(res["accuracy"])
				cv[m]["f1s"].append(res["f1_score"])
				print(f"  [{m}] auc={res['auc_roc']:.4f}")
			except Exception as exc:
				print(f"  [{m}] FAILED: {exc}")
				cv[m]["aucs"].append(float("nan"))
				cv[m]["accs"].append(float("nan"))
				cv[m]["f1s"].append(float("nan"))

	print("\nClassical-pass CV-AUC:")
	for m, r in sorted(cv.items(),
					   key=lambda kv: np.nanmean(kv[1]["aucs"]) if kv[1]["aucs"] else 0,
					   reverse=True):
		a = np.asarray(r["aucs"], dtype=float)
		a = a[~np.isnan(a)]
		if len(a):
			print(f"  {m:<10}  mean={a.mean():.4f}  +/- {a.std():.4f}")

	stats_rows = []
	if "lr_l1" in cv:
		ref = np.asarray(cv["lr_l1"]["aucs"], dtype=float)
		ps = []
		for m in args.models:
			if m == "lr_l1":
				continue
			a = np.asarray(cv[m]["aucs"], dtype=float)
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
				"model": m, "vs": "lr_l1",
				"mean_diff": float(a[mask].mean() - ref[mask].mean()),
				"wilcoxon_p": wp, "nb_corrected_p": nbp,
				"cohens_d": d, "effect_size": effect_size_label(d),
			})
			ps.append(nbp)
		for row, adj in zip(stats_rows, holm_bonferroni(ps)):
			row["holm_adjusted_p"] = adj
			row["holm_significant"] = adj < 0.05

	elapsed = (time.time() - start) / 60
	out = {
		"metadata": {"version": "v12", "purpose": "strong classical baseline pass",
					  "models": args.models, "runtime_minutes": elapsed},
		"cv": {m: {
			"aucs": r["aucs"],
			"mean_auc": float(np.nanmean(r["aucs"])) if r["aucs"] else None,
			"std_auc": float(np.nanstd(r["aucs"])) if r["aucs"] else None,
		} for m, r in cv.items()},
		"paired_stats_vs_lr_l1": stats_rows,
	}
	out_dir = _THIS / "results"
	out_dir.mkdir(exist_ok=True)
	out_path = out_dir / "v12_classical_pass.json"
	with open(out_path, "w") as f:
		json.dump(out, f, indent=2, cls=NumpyEncoder)
	print(f"\nSaved {out_path} ({elapsed:.1f} min)")


if __name__ == "__main__":
	main()
