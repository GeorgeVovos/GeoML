"""v10 noise sweep — re-run v06 VQC under increasing depolarising noise."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))           # shared
sys.path.insert(0, str(_THIS.parent / "v06"))

from shared.stats_utils import (  # noqa: E402
	wilcoxon_signed_rank, corrected_resampled_ttest, cohens_d, effect_size_label,
)
from preprocess_gse76809 import preprocess  # noqa: E402
from model_quantum_vqc_noisy import train_noisy_vqc  # noqa: E402


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer): return int(obj)
		if isinstance(obj, np.floating): return float(obj)
		if isinstance(obj, np.bool_): return bool(obj)
		if isinstance(obj, np.ndarray): return obj.tolist()
		return super().default(obj)


DEFAULT_NOISE = [0.0, 1e-4, 1e-3, 5e-3, 1e-2]


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--noise", type=float, nargs="+", default=DEFAULT_NOISE)
	args = ap.parse_args()

	start = time.time()
	data = preprocess()

	cv = {p: {"aucs": [], "accs": [], "f1s": []} for p in args.noise}
	for fold_idx, fold in enumerate(data["folds"]):
		print(f"\n=== Fold {fold_idx+1}/{len(data['folds'])} ===")
		for p in args.noise:
			print(f"  noise_p={p}")
			res = train_noisy_vqc(fold_data=fold, noise_p=p)
			cv[p]["aucs"].append(res["auc_roc"])
			cv[p]["accs"].append(res["accuracy"])
			cv[p]["f1s"].append(res["f1_score"])

	print("\nNoise sweep CV-AUC:")
	for p in args.noise:
		a = cv[p]["aucs"]
		print(f"  p={p:<10}  mean={np.mean(a):.4f}  +/- {np.std(a):.4f}")

	# Paired tests vs noiseless
	ref = 0.0
	stats_rows = []
	if ref in cv:
		ref_a = np.asarray(cv[ref]["aucs"])
		for p in args.noise:
			if p == ref:
				continue
			a = np.asarray(cv[p]["aucs"])
			_, wp = wilcoxon_signed_rank(a, ref_a)
			_, nbp = corrected_resampled_ttest(
				a, ref_a,
				n_train=len(data["X_train"]) * 4 // 5,
				n_test=len(data["X_train"]) // 5,
			)
			d = cohens_d(a, ref_a)
			stats_rows.append({
				"noise_p": p,
				"mean_diff_vs_noiseless": float(a.mean() - ref_a.mean()),
				"wilcoxon_p": wp,
				"nb_corrected_p": nbp,
				"cohens_d": d,
				"effect_size": effect_size_label(d),
			})

	elapsed_min = (time.time() - start) / 60
	out = {
		"metadata": {
			"version": "v10",
			"purpose": "depolarising-noise sweep for v06 data-reuploading VQC",
			"noise_levels": args.noise,
			"device": "default.mixed",
			"runtime_minutes": elapsed_min,
		},
		"cv": {str(p): {
			"aucs": cv[p]["aucs"],
			"mean_auc": float(np.mean(cv[p]["aucs"])),
			"std_auc": float(np.std(cv[p]["aucs"])),
		} for p in args.noise},
		"paired_stats_vs_noiseless": stats_rows,
	}
	out_dir = _THIS / "results"
	out_dir.mkdir(exist_ok=True)
	out_path = out_dir / "v10_noise_sweep.json"
	with open(out_path, "w") as f:
		json.dump(out, f, indent=2, cls=NumpyEncoder)
	print(f"\nSaved {out_path} ({elapsed_min:.1f} min)")


if __name__ == "__main__":
	main()
