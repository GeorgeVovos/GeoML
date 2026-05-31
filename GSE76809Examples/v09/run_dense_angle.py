"""Standalone runner for the dense_angle encoding — robust to interruptions.

Usage:
	conda activate GEO
	cd GSE76809Examples\v09
	python run_dense_angle.py

Resumes from checkpoint automatically. Writes final results into
results/v09_encoding_ablation.json when all 5 folds are complete.
"""

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
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "model_quantum_vqc_v09", str(_THIS / "model_quantum_vqc.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
train_quantum_vqc = _mod.train_quantum_vqc

from preprocess_gse76809 import preprocess  # noqa: E402
from shared.stats_utils import (  # noqa: E402
	cohens_d, effect_size_label, wilcoxon_signed_rank,
	corrected_resampled_ttest, holm_bonferroni,
)


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer): return int(obj)
		if isinstance(obj, np.floating): return float(obj)
		if isinstance(obj, np.bool_): return bool(obj)
		if isinstance(obj, np.ndarray): return obj.tolist()
		return super().default(obj)


def main():
	start = time.time()
	ENC = "dense_angle"
	out_dir = _THIS / "results"
	out_dir.mkdir(exist_ok=True)
	checkpoint_path = out_dir / "v09_encoding_ablation_partial.json"

	# Load checkpoint
	if checkpoint_path.exists():
		with open(checkpoint_path, "r") as f:
			cv = json.load(f)
		if ENC not in cv:
			cv[ENC] = {"aucs": [], "accs": [], "f1s": []}
		done = len(cv[ENC]["aucs"])
		print(f"Checkpoint loaded: {done}/5 folds already complete for {ENC}")
	else:
		cv = {ENC: {"aucs": [], "accs": [], "f1s": []}}
		done = 0

	if done >= 5:
		print("All 5 folds already complete — skipping to results generation.")
	else:
		data = preprocess()
		folds = data["folds"]
		for fold_idx in range(done, 5):
			print(f"\n--- Fold {fold_idx + 1}/5 ---")
			fold = folds[fold_idx]
			res = train_quantum_vqc(fold_data=fold, encoding=ENC)
			cv[ENC]["aucs"].append(res["auc_roc"])
			cv[ENC]["accs"].append(res["accuracy"])
			cv[ENC]["f1s"].append(res["f1_score"])
			# Save checkpoint immediately
			with open(checkpoint_path, "w") as f:
				json.dump(cv, f, indent=2, cls=NumpyEncoder)
			print(f"  Fold {fold_idx + 1} done: AUC={res['auc_roc']:.4f} "
				  f"(checkpoint saved)")

	# --- Generate updated final results JSON ---
	print("\nGenerating final results JSON with all encodings...")

	# Load the existing final results to merge dense_angle into it
	final_path = out_dir / "v09_encoding_ablation.json"
	if final_path.exists():
		with open(final_path, "r") as f:
			results = json.load(f)
	else:
		results = {"metadata": {}, "cv": {}, "paired_stats_vs_data_reuploading": []}

	# Update metadata
	results["metadata"]["encodings_updated"] = time.strftime("%Y-%m-%d %H:%M")
	results["metadata"]["dense_angle_runtime_minutes"] = (time.time() - start) / 60

	# Add/update dense_angle in cv section
	aucs = cv[ENC]["aucs"]
	results["cv"][ENC] = {
		"aucs": aucs,
		"mean_auc": float(np.mean(aucs)),
		"std_auc": float(np.std(aucs)),
	}

	# Compute paired test vs data_reuploading if available
	if "data_reuploading" in cv and len(cv["data_reuploading"]["aucs"]) == 5:
		ref_aucs = np.array(cv["data_reuploading"]["aucs"])
	elif "data_reuploading" in results.get("cv", {}):
		ref_aucs = np.array(results["cv"]["data_reuploading"]["aucs"])
	else:
		ref_aucs = None

	if ref_aucs is not None and len(aucs) == 5:
		a = np.array(aucs)
		_, wp = wilcoxon_signed_rank(a, ref_aucs)
		_, nbp = corrected_resampled_ttest(a, ref_aucs, n_train=170, n_test=42)
		d = cohens_d(a, ref_aucs)

		dense_stat = {
			"encoding": ENC, "vs": "data_reuploading",
			"mean_diff": float(a.mean() - ref_aucs.mean()),
			"wilcoxon_p": wp, "nb_corrected_p": nbp,
			"cohens_d": d, "effect_size": effect_size_label(d),
		}

		# Add Holm correction across ALL existing stats rows
		existing_stats = results.get("paired_stats_vs_data_reuploading", [])
		# Remove any old dense_angle entry
		existing_stats = [r for r in existing_stats if r.get("encoding") != ENC]
		existing_stats.append(dense_stat)
		# Re-do Holm on all p-values
		ps = [r["nb_corrected_p"] for r in existing_stats]
		for row, adj in zip(existing_stats, holm_bonferroni(ps)):
			row["holm_adjusted_p"] = adj
			row["holm_significant"] = adj < 0.05
		results["paired_stats_vs_data_reuploading"] = existing_stats

	with open(final_path, "w") as f:
		json.dump(results, f, indent=2, cls=NumpyEncoder)

	elapsed = (time.time() - start) / 60
	print(f"\nDone! Saved {final_path} (this run: {elapsed:.1f} min)")
	print(f"dense_angle: mean_auc={np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")


if __name__ == "__main__":
	main()
