"""v10B noise sweep — does the v09 encoding ranking survive NISQ noise?

v09 froze everything except the *encoding* and found that
``data_reuploading`` is the best data-loading strategy on the v06
pipeline (5-fold CV). v10B re-runs that same ablation under depolarising
noise, crossing the v09 encoding axis with a noise axis:

	(encoding)  x  (noise_p)   evaluated with v06's 5-fold CV

For each (encoding, noise_p) cell we record the 5 per-fold AUCs. The
``noise_p = 0`` column reproduces v09's noiseless ranking; the higher
noise levels test whether the re-uploading advantage is a noiseless
artifact or whether all encodings collapse toward chance together.

Per noise level we run a paired Wilcoxon + Nadeau-Bengio corrected
resampled t-test of each encoding against the **same encoding at
noise_p = 0**, so we can name the error rate at which each encoding stops
being distinguishable from its own noiseless run.

Compute-aware reduced design (default.mixed is ~5-10x slower):
- 5-fold CV (same as v09) but 40 epochs (v09 used 80).
- Only the VQC is run; no classical baselines.

Resumable: a checkpoint JSON is written after every (encoding, noise, fold)
cell and reloaded on startup.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent / "v06"))   # preprocess
sys.path.insert(0, str(_THIS.parent))           # shared
sys.path.insert(0, str(_THIS.parent / "v09"))   # quantum_encodings
sys.path.insert(0, str(_THIS))                  # local model

from shared.stats_utils import (  # noqa: E402
	cohens_d, effect_size_label, wilcoxon_signed_rank, corrected_resampled_ttest,
)
from preprocess_gse76809 import preprocess  # noqa: E402
from quantum_encodings import ENCODINGS  # noqa: E402
from model_quantum_vqc_noisy import train_noisy_vqc  # noqa: E402


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer): return int(obj)
		if isinstance(obj, np.floating): return float(obj)
		if isinstance(obj, np.bool_): return bool(obj)
		if isinstance(obj, np.ndarray): return obj.tolist()
		return super().default(obj)


DEFAULT_NOISE = [0.0, 1e-4, 1e-3, 5e-3, 1e-2]
DEFAULT_ENCODINGS = ["data_reuploading", "amplitude", "angle"]


def _key(enc, p):
	return f"{enc}@{p}"


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--noise", type=float, nargs="+", default=DEFAULT_NOISE)
	ap.add_argument("--encodings", nargs="+", default=DEFAULT_ENCODINGS)
	ap.add_argument("--epochs", type=int, default=40)
	args = ap.parse_args()

	bad = [e for e in args.encodings if e not in ENCODINGS]
	if bad:
		raise SystemExit(f"unknown encodings {bad}; known {list(ENCODINGS)}")

	print(f"Noise levels: {args.noise}")
	print(f"Encodings   : {args.encodings}")

	start = time.time()
	data = preprocess()

	out_dir = _THIS / "results"
	out_dir.mkdir(exist_ok=True)
	ckpt_path = out_dir / "v10B_encoding_noise_partial.json"

	# cv[enc@p] = {"aucs": [...per fold...]}
	if ckpt_path.exists():
		with open(ckpt_path, "r") as f:
			cv = json.load(f)
		print(f"Resuming from checkpoint: {ckpt_path}")
	else:
		cv = {}
	for enc in args.encodings:
		for p in args.noise:
			cv.setdefault(_key(enc, p), {"aucs": [], "accs": [], "f1s": []})

	for fold_idx, fold in enumerate(data["folds"]):
		print(f"\n--- Fold {fold_idx+1}/{len(data['folds'])} ---")
		for enc in args.encodings:
			for p in args.noise:
				k = _key(enc, p)
				if len(cv[k]["aucs"]) > fold_idx:
					print(f"  [resume] skip {k} fold {fold_idx+1}")
					continue
				t0 = time.time()
				try:
					res = train_noisy_vqc(
						fold_data=fold, encoding=enc, noise_p=p,
						epochs=args.epochs, init_seed=fold_idx,
					)
					auc = res["auc_roc"]
					acc = res["accuracy"]
					f1 = res["f1_score"]
				except Exception as exc:
					print(f"  [{k}] FAILED: {exc}")
					auc, acc, f1 = float("nan"), float("nan"), float("nan")
				cv[k]["aucs"].append(auc)
				cv[k]["accs"].append(acc)
				cv[k]["f1s"].append(f1)
				print(f"  [{k:<28}] AUC={auc:.4f}  ({time.time()-t0:.1f}s)")
				with open(ckpt_path, "w") as f:
					json.dump(cv, f, indent=2, cls=NumpyEncoder)

	print("\nPer (encoding, noise) 5-fold AUC:")
	for enc in args.encodings:
		for p in args.noise:
			a = cv[_key(enc, p)]["aucs"]
			print(f"  {enc:<18} p={p:<8}  mean={np.mean(a):.4f}  +/- {np.std(a):.4f}")

	# Paired tests: each (encoding, noise_p>0) vs the SAME encoding at noise_p=0.
	n_train = len(data["X_train"])
	stats_rows = []
	ref_p = 0.0
	if ref_p in args.noise:
		for enc in args.encodings:
			ref_a = np.asarray(cv[_key(enc, ref_p)]["aucs"])
			for p in args.noise:
				if p == ref_p:
					continue
				a = np.asarray(cv[_key(enc, p)]["aucs"])
				_, wp = wilcoxon_signed_rank(a, ref_a)
				_, nbp = corrected_resampled_ttest(
					a, ref_a, n_train=n_train * 4 // 5, n_test=n_train // 5,
				)
				d = cohens_d(a, ref_a)
				stats_rows.append({
					"encoding": enc,
					"noise_p": p,
					"mean_diff_vs_noiseless": float(a.mean() - ref_a.mean()),
					"wilcoxon_p": wp,
					"nb_corrected_p": nbp,
					"cohens_d": d,
					"effect_size": effect_size_label(d),
				})

		print("\nPaired tests vs each encoding's own noiseless run:")
		for row in stats_rows:
			print(f"  {row['encoding']:<18} p={row['noise_p']:<8} "
				  f"d={row['cohens_d']:+.2f} nb_p={row['nb_corrected_p']:.4f}")

	elapsed_min = (time.time() - start) / 60
	out = {
		"metadata": {
			"version": "v10B",
			"targets": "v09",
			"purpose": "depolarising-noise sweep crossed with v09's encoding "
					   "ablation (EncodingVQC on default.mixed)",
			"noise_levels": args.noise,
			"encodings": args.encodings,
			"epochs": args.epochs,
			"device": "default.mixed",
			"seed": 2026,
			"runtime_minutes": elapsed_min,
		},
		"cv": {k: {
			"aucs": v["aucs"],
			"mean_auc": float(np.mean(v["aucs"])) if v["aucs"] else float("nan"),
			"std_auc": float(np.std(v["aucs"])) if v["aucs"] else float("nan"),
		} for k, v in cv.items()},
		"paired_stats_vs_noiseless": stats_rows,
	}
	out_path = out_dir / "v10B_encoding_noise.json"
	with open(out_path, "w") as f:
		json.dump(out, f, indent=2, cls=NumpyEncoder)
	print(f"\nSaved {out_path} ({elapsed_min:.1f} min)")


if __name__ == "__main__":
	main()
