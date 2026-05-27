"""v13 — calibration & decision-curve analysis on v06 predictions.

Strategy:
- Re-run the cheap v06 models (LR, RBF, MLP, XGBoost) on v06 folds so we
  have per-fold probability vectors. Quantum models can be added later
  by importing v06.model_quantum_vqc / model_quantum_kernel, but those
  are slow; default is classical-only for fast feedback.
- For each model: pool out-of-fold probabilities and compute Brier, ECE,
  MCE, and DCA against the pooled out-of-fold truth.
- Also fit Platt scaling on a leave-one-fold-out basis to estimate
  whether recalibration helps.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS.parent / "v06"))

from preprocess_gse76809 import preprocess  # noqa: E402
from model_classical_mlp import train_classical_mlp  # noqa: E402
from model_classical_svm import train_classical_svm  # noqa: E402
from model_classical_xgb import train_classical_xgb  # noqa: E402
from shared.model_classical_logreg import train_classical_logreg  # noqa: E402

from decision_curve import net_benefit  # noqa: E402

REGISTRY = {
	"lr_l1": train_classical_logreg,
	"rbf": train_classical_svm,
	"mlp": train_classical_mlp,
	"xgb": train_classical_xgb,
}


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer): return int(obj)
		if isinstance(obj, np.floating): return float(obj)
		if isinstance(obj, np.bool_): return bool(obj)
		if isinstance(obj, np.ndarray): return obj.tolist()
		return super().default(obj)


def _ece_mce(y_true, y_prob, n_bins: int = 10):
	bins = np.linspace(0, 1, n_bins + 1)
	idx = np.digitize(y_prob, bins) - 1
	idx = np.clip(idx, 0, n_bins - 1)
	n = len(y_true)
	ece, mce = 0.0, 0.0
	for b in range(n_bins):
		mask = idx == b
		if not mask.any():
			continue
		conf = y_prob[mask].mean()
		acc = y_true[mask].mean()
		gap = abs(conf - acc)
		ece += (mask.sum() / n) * gap
		mce = max(mce, gap)
	return float(ece), float(mce)


def _scale(probs):
	"""Min-max squeeze to [0,1] for models with non-probability scores."""
	lo, hi = probs.min(), probs.max()
	if hi - lo < 1e-9:
		return np.full_like(probs, 0.5, dtype=float)
	return (probs - lo) / (hi - lo)


def _platt_loo(model_probs_per_fold, model_truths_per_fold):
	"""Leave-one-fold-out Platt rescaling. Returns recalibrated pooled probs."""
	recalibrated = []
	for held_out_idx in range(len(model_probs_per_fold)):
		train_p = np.concatenate(
			[model_probs_per_fold[i] for i in range(len(model_probs_per_fold))
			 if i != held_out_idx])
		train_y = np.concatenate(
			[model_truths_per_fold[i] for i in range(len(model_truths_per_fold))
			 if i != held_out_idx])
		platt = LogisticRegression(max_iter=2000)
		platt.fit(train_p.reshape(-1, 1), train_y)
		rec = platt.predict_proba(
			model_probs_per_fold[held_out_idx].reshape(-1, 1))[:, 1]
		recalibrated.append(rec)
	return recalibrated


def _analyse(name, probs_per_fold, truths_per_fold):
	pooled_p = np.concatenate(probs_per_fold)
	pooled_y = np.concatenate(truths_per_fold)
	# Some models return decision_function not probability -> squash
	if pooled_p.min() < 0 or pooled_p.max() > 1:
		pooled_p_use = _scale(pooled_p)
		probs_per_fold_use = [_scale(p) for p in probs_per_fold]
	else:
		pooled_p_use = pooled_p
		probs_per_fold_use = probs_per_fold

	brier = brier_score_loss(pooled_y, pooled_p_use)
	ece, mce = _ece_mce(pooled_y, pooled_p_use)
	frac_pos, mean_pred = calibration_curve(pooled_y, pooled_p_use, n_bins=10,
											  strategy="quantile")

	rec_per_fold = _platt_loo(probs_per_fold_use, truths_per_fold)
	rec_pooled = np.concatenate(rec_per_fold)
	brier_after = brier_score_loss(pooled_y, rec_pooled)
	ece_after, _ = _ece_mce(pooled_y, rec_pooled)

	dca = net_benefit(pooled_y, pooled_p_use)
	return {
		"model": name,
		"brier": float(brier),
		"ece": float(ece),
		"mce": float(mce),
		"calibration_curve": {
			"fraction_positives": frac_pos.tolist(),
			"mean_predicted": mean_pred.tolist(),
		},
		"after_platt": {
			"brier": float(brier_after),
			"ece": float(ece_after),
			"delta_brier": float(brier_after - brier),
		},
		"decision_curve": {
			"thresholds": dca["thresholds"].tolist(),
			"net_benefit": dca["net_benefit"].tolist(),
			"net_benefit_treat_all": dca["net_benefit_treat_all"].tolist(),
			"prevalence": dca["prevalence"],
		},
	}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--models", nargs="+", default=list(REGISTRY.keys()))
	args = ap.parse_args()
	bad = [m for m in args.models if m not in REGISTRY]
	if bad:
		raise SystemExit(f"unknown models {bad}; known {list(REGISTRY)}")

	start = time.time()
	data = preprocess()

	fold_probs = {m: [] for m in args.models}
	fold_truths = {m: [] for m in args.models}
	for fold_idx, fold in enumerate(data["folds"]):
		print(f"\n=== Fold {fold_idx+1}/{len(data['folds'])} ===")
		for m in args.models:
			res = REGISTRY[m](fold_data=fold)
			fold_probs[m].append(np.asarray(res["predictions"], dtype=float))
			fold_truths[m].append(np.asarray(res["y_true"], dtype=int))
			print(f"  [{m}] AUC={res['auc_roc']:.4f}")

	report = []
	for m in args.models:
		report.append(_analyse(m, fold_probs[m], fold_truths[m]))

	print("\nCalibration summary (lower Brier/ECE = better):")
	for r in sorted(report, key=lambda x: x["brier"]):
		print(f"  {r['model']:<8}  Brier={r['brier']:.4f}  ECE={r['ece']:.4f}  "
			  f"after-Platt Brier={r['after_platt']['brier']:.4f}")

	elapsed = (time.time() - start) / 60
	out = {
		"metadata": {"version": "v13",
					  "purpose": "calibration and decision-curve analysis",
					  "models": args.models, "runtime_minutes": elapsed},
		"per_model": report,
	}
	out_dir = _THIS / "results"
	out_dir.mkdir(exist_ok=True)
	out_path = out_dir / "v13_calibration.json"
	with open(out_path, "w") as f:
		json.dump(out, f, indent=2, cls=NumpyEncoder)
	print(f"\nSaved {out_path} ({elapsed:.1f} min)")


if __name__ == "__main__":
	main()
