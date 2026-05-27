"""v09 encoding-ablation driver.

5-fold CV for each encoding on the same v06 GSE76809 preprocessing.
Reports per-encoding AUC mean/std and Wilcoxon + Nadeau-Bengio paired
tests against the data_reuploading baseline (so we can say whether the
v01→v06 narrative "amplitude/reuploading caused the jump" is actually
supported by paired evidence).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))           # shared
sys.path.insert(0, str(_THIS.parent / "v06"))   # preprocess

from shared.stats_utils import (  # noqa: E402
	cohens_d, effect_size_label, wilcoxon_signed_rank,
	corrected_resampled_ttest, holm_bonferroni,
)

from preprocess_gse76809 import preprocess  # noqa: E402
from model_quantum_vqc import train_quantum_vqc  # noqa: E402
from encodings import ENCODINGS  # noqa: E402


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer): return int(obj)
		if isinstance(obj, np.floating): return float(obj)
		if isinstance(obj, np.bool_): return bool(obj)
		if isinstance(obj, np.ndarray): return obj.tolist()
		return super().default(obj)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--encodings", nargs="+", default=list(ENCODINGS.keys()))
	args = ap.parse_args()
	bad = [e for e in args.encodings if e not in ENCODINGS]
	if bad:
		raise SystemExit(f"unknown encodings {bad}; known {list(ENCODINGS)}")

	start = time.time()
	data = preprocess()

	cv = {enc: {"aucs": [], "accs": [], "f1s": []} for enc in args.encodings}
	for fold_idx, fold in enumerate(data["folds"]):
		print(f"\n--- Fold {fold_idx+1}/{len(data['folds'])} ---")
		for enc in args.encodings:
			res = train_quantum_vqc(fold_data=fold, encoding=enc)
			cv[enc]["aucs"].append(res["auc_roc"])
			cv[enc]["accs"].append(res["accuracy"])
			cv[enc]["f1s"].append(res["f1_score"])

	print("\nPer-encoding 5-fold AUC:")
	for enc, r in sorted(cv.items(), key=lambda kv: np.mean(kv[1]["aucs"]), reverse=True):
		print(f"  {enc:<20}  mean={np.mean(r['aucs']):.4f}  +/- {np.std(r['aucs']):.4f}")

	# Paired tests vs the canonical reference (data_reuploading)
	ref = "data_reuploading"
	stats_rows = []
	if ref in cv:
		ref_aucs = np.asarray(cv[ref]["aucs"])
		ps = []
		for enc in args.encodings:
			if enc == ref:
				continue
			a = np.asarray(cv[enc]["aucs"])
			_, wp = wilcoxon_signed_rank(a, ref_aucs)
			_, nbp = corrected_resampled_ttest(
				a, ref_aucs,
				n_train=len(data["X_train"]) * 4 // 5,
				n_test=len(data["X_train"]) // 5,
			)
			d = cohens_d(a, ref_aucs)
			stats_rows.append({
				"encoding": enc, "vs": ref,
				"mean_diff": float(a.mean() - ref_aucs.mean()),
				"wilcoxon_p": wp, "nb_corrected_p": nbp,
				"cohens_d": d, "effect_size": effect_size_label(d),
			})
			ps.append(nbp)
		for row, adj in zip(stats_rows, holm_bonferroni(ps)):
			row["holm_adjusted_p"] = adj
			row["holm_significant"] = adj < 0.05

		print("\nPaired tests vs data_reuploading (Holm-adjusted):")
		for row in stats_rows:
			flag = " ***" if row["holm_significant"] else ""
			print(f"  {row['encoding']:<20} d={row['cohens_d']:+.2f} "
				  f"p_adj={row['holm_adjusted_p']:.4f}{flag}")

	elapsed_min = (time.time() - start) / 60
	out = {
		"metadata": {
			"version": "v09", "purpose": "encoding ablation on frozen v06 pipeline",
			"encodings": args.encodings, "seed": 2026,
			"runtime_minutes": elapsed_min,
		},
		"cv": {enc: {
			"aucs": r["aucs"],
			"mean_auc": float(np.mean(r["aucs"])),
			"std_auc": float(np.std(r["aucs"])),
		} for enc, r in cv.items()},
		"paired_stats_vs_data_reuploading": stats_rows,
	}
	out_dir = _THIS / "results"
	out_dir.mkdir(exist_ok=True)
	out_path = out_dir / "v09_encoding_ablation.json"
	with open(out_path, "w") as f:
		json.dump(out, f, indent=2, cls=NumpyEncoder)
	print(f"\nSaved {out_path} (runtime {elapsed_min:.1f} min)")


if __name__ == "__main__":
	main()
