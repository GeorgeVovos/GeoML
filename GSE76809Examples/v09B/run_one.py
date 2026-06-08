"""Standalone single-encoding runner for v09B — robust to interruptions.

Runs the 5-fold CV for ONE encoding at a time and checkpoints after every
fold, so a long run can be killed and resumed without losing work. Useful
because each combined encoding takes ~1.5 h of CPU for its 5 folds.

Usage:
	conda activate GEO
	cd GSE76809Examples\\v09B
	python run_one.py amplitude_reup        # one encoding
	python run_one.py iqp_reup --epochs 80  # override training budget

After all 5 folds for the chosen encoding complete, the encoding's entry in
results/v09b_encoding_ablation.json is created/updated and (when the
data_reuploading reference is available) its paired test vs the reference is
recomputed. Run compare_encodings.py afterwards for the full Holm-corrected
comparison across every encoding.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS.parent / "v06"))

from quantum_encodings import ENCODINGS  # noqa: E402
from model_quantum_vqc import train_quantum_vqc  # noqa: E402
from preprocess_gse76809 import preprocess  # noqa: E402
from shared.stats_utils import (  # noqa: E402
	cohens_d, effect_size_label, wilcoxon_signed_rank,
	corrected_resampled_ttest,
)


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer): return int(obj)
		if isinstance(obj, np.floating): return float(obj)
		if isinstance(obj, np.bool_): return bool(obj)
		if isinstance(obj, np.ndarray): return obj.tolist()
		return super().default(obj)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("encoding", choices=list(ENCODINGS.keys()),
					help="which encoding to run all 5 folds for")
	ap.add_argument("--epochs", type=int, default=80)
	args = ap.parse_args()
	enc = args.encoding

	start = time.time()
	out_dir = _THIS / "results"
	out_dir.mkdir(exist_ok=True)
	checkpoint_path = out_dir / "v09b_encoding_ablation_partial.json"

	# Load checkpoint
	if checkpoint_path.exists():
		with open(checkpoint_path, "r") as f:
			cv = json.load(f)
		if enc not in cv:
			cv[enc] = {"aucs": [], "accs": [], "f1s": []}
		done = len(cv[enc]["aucs"])
		print(f"Checkpoint loaded: {done}/5 folds already complete for {enc}")
	else:
		cv = {enc: {"aucs": [], "accs": [], "f1s": []}}
		done = 0

	# Nadeau-Bengio per-fold train/test sizes. Derived from the actual
	# training-set length when data is loaded; otherwise fall back to the
	# GSE76809 80/20 split (193 train samples -> 5-fold gives ~154/38).
	nb_n_train, nb_n_test = 154, 38

	if done >= 5:
		print(f"All 5 folds already complete for {enc} — skipping to results.")
	else:
		data = preprocess()
		folds = data["folds"]
		n_total = len(data["X_train"])
		nb_n_train, nb_n_test = n_total * 4 // 5, n_total // 5
		for fold_idx in range(done, 5):
			print(f"\n--- {enc} fold {fold_idx + 1}/5 ---")
			fold = folds[fold_idx]
			res = train_quantum_vqc(fold_data=fold, encoding=enc, epochs=args.epochs)
			cv[enc]["aucs"].append(res["auc_roc"])
			cv[enc]["accs"].append(res["accuracy"])
			cv[enc]["f1s"].append(res["f1_score"])
			with open(checkpoint_path, "w") as f:
				json.dump(cv, f, indent=2, cls=NumpyEncoder)
			print(f"  Fold {fold_idx + 1} done: AUC={res['auc_roc']:.4f} "
				  f"(checkpoint saved)")

	# --- Merge this encoding into the final results JSON ---
	print("\nUpdating final results JSON...")
	final_path = out_dir / "v09b_encoding_ablation.json"
	if final_path.exists():
		with open(final_path, "r") as f:
			results = json.load(f)
	else:
		results = {
			"metadata": {
				"version": "v09B",
				"purpose": "combined base-encoding + data_reuploading ablation on frozen v06 pipeline",
				"seed": 2026,
			},
			"cv": {},
			"paired_stats_vs_data_reuploading": [],
		}

	results["metadata"]["encodings_updated"] = time.strftime("%Y-%m-%d %H:%M")
	results["metadata"][f"{enc}_runtime_minutes"] = (time.time() - start) / 60

	aucs = cv[enc]["aucs"]
	results["cv"][enc] = {
		"aucs": aucs,
		"mean_auc": float(np.mean(aucs)),
		"std_auc": float(np.std(aucs)),
	}

	# Paired test vs data_reuploading (if reference available, and not self)
	ref = "data_reuploading"
	ref_aucs = None
	if enc != ref:
		if ref in cv and len(cv[ref]["aucs"]) == 5:
			ref_aucs = np.array(cv[ref]["aucs"])
		elif ref in results.get("cv", {}) and len(results["cv"][ref]["aucs"]) == 5:
			ref_aucs = np.array(results["cv"][ref]["aucs"])

	if ref_aucs is not None and len(aucs) == 5:
		a = np.array(aucs)
		_, wp = wilcoxon_signed_rank(a, ref_aucs)
		_, nbp = corrected_resampled_ttest(a, ref_aucs, n_train=nb_n_train, n_test=nb_n_test)
		d = cohens_d(a, ref_aucs)
		row = {
			"encoding": enc, "vs": ref,
			"mean_diff": float(a.mean() - ref_aucs.mean()),
			"wilcoxon_p": wp, "nb_corrected_p": nbp,
			"cohens_d": d, "effect_size": effect_size_label(d),
		}
		# Replace any existing row for this encoding
		rows = [r for r in results.get("paired_stats_vs_data_reuploading", [])
				if r.get("encoding") != enc]
		rows.append(row)
		results["paired_stats_vs_data_reuploading"] = rows
		print(f"  {enc} vs {ref}: d={d:+.2f} nb_p={nbp:.4f}")

	with open(final_path, "w") as f:
		json.dump(results, f, indent=2, cls=NumpyEncoder)
	print(f"\nSaved {final_path}")
	print("Note: run compare_encodings.py for the full Holm-corrected table.")


if __name__ == "__main__":
	main()
