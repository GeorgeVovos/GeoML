"""v06B driver — Feature extraction / embedding comparison.

Question: does a *learned* embedding (supervised MLP encoder or
unsupervised autoencoder) beat the v06 ANOVA+PCA feature pipeline as the
front-end for quantum and classical classifiers on GSE76809?

Design (leakage-free):
  1. Reuse v06.preprocess() to get the variance-filtered + quantile-
	 normalised raw gene matrix and the holdout train/test split.
  2. Build ONE set of StratifiedKFold indices so every extractor sees
	 the SAME fold splits.
  3. For each extractor in {anova_pca, mlp_encoder, autoencoder}:
	   - fit the extractor on fold-train rows only -> 16-d embedding
	   - run the same downstream models (VQC, MLP, XGBoost, SVM)
		that v06 uses, on those embeddings.
  4. Report per-(extractor, model) 5-fold AUC and paired tests comparing
	 each learned extractor against the anova_pca baseline (per model).

Robust to interruption: per-(extractor, model) results are checkpointed
to results/v06B_feature_extraction_partial.json after every fold-set, so
re-running resumes where it left off.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# The reused v06 trainers print a Unicode arrow (U+2192) in their result
# lines. On a legacy Windows (cp1252) console this raises UnicodeEncodeError
# and would kill the multi-hour run on the first fold. Force UTF-8 stdout.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

import numpy as np
from sklearn.model_selection import StratifiedKFold

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS.parent / "v06"))

from feature_extractors import EXTRACTORS, EMB_DIM  # noqa: E402

from preprocess_gse76809 import preprocess  # noqa: E402
from model_quantum_vqc import train_quantum_vqc  # noqa: E402
from model_classical_mlp import train_classical_mlp  # noqa: E402
from model_classical_xgb import train_classical_xgb  # noqa: E402
# v06B uses a local, inversion-safe SVM trainer (see module docstring); the
# shared v06 SVM trainer's predict_proba ranking inverts on tiny imbalanced
# inner folds, which corrupted the anova_pca SVM baseline.
from model_classical_svm_v06b import train_classical_svm  # noqa: E402

from shared.stats_utils import (  # noqa: E402
	cohens_d, effect_size_label, wilcoxon_signed_rank,
	corrected_resampled_ttest, holm_bonferroni,
)

N_FOLDS = 5
RANDOM_STATE = 2026

# Downstream models keyed by name -> trainer callable
MODELS = {
	"quantum_vqc": train_quantum_vqc,
	"classical_mlp": train_classical_mlp,
	"classical_xgb": train_classical_xgb,
	"classical_svm": train_classical_svm,
}


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer): return int(obj)
		if isinstance(obj, np.floating): return float(obj)
		if isinstance(obj, np.bool_): return bool(obj)
		if isinstance(obj, np.ndarray): return obj.tolist()
		return super().default(obj)


def _key(extractor, model):
	return f"{extractor}::{model}"


def _atomic_save(cv, checkpoint: Path):
	"""Crash-safe checkpoint write.

	Write to a temp file, fsync, then os.replace() over the real file
	(atomic on Windows + POSIX). A previous good checkpoint is kept as
	`<name>.bak` so a kill *during* the write can never destroy progress.
	"""
	tmp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
	with open(tmp, "w", encoding="utf-8") as f:
		json.dump(cv, f, indent=2, cls=NumpyEncoder)
		f.flush()
		os.fsync(f.fileno())
	if checkpoint.exists():
		try:
			os.replace(checkpoint, checkpoint.with_suffix(checkpoint.suffix + ".bak"))
		except OSError:
			pass
	os.replace(tmp, checkpoint)


def _load_checkpoint(checkpoint: Path):
	"""Load the checkpoint, tolerating a corrupt/partial primary file by
	falling back to the `.bak` copy. Returns {} if nothing usable exists."""
	for path in (checkpoint, checkpoint.with_suffix(checkpoint.suffix + ".bak")):
		if path.exists():
			try:
				with open(path, "r", encoding="utf-8") as f:
					data = json.load(f)
				print(f"Resuming from {path}")
				return data
			except (json.JSONDecodeError, OSError) as exc:
				print(f"WARNING: {path} unreadable ({exc}); trying fallback")
	return {}


def main():
	start = time.time()
	out_dir = _THIS / "results"
	out_dir.mkdir(exist_ok=True)
	checkpoint = out_dir / "v06B_feature_extraction_partial.json"

	# --- 1. raw matrix + holdout split + shared fold indices ---------------
	data = preprocess(n_features=EMB_DIM, n_folds=N_FOLDS,
					  random_state=RANDOM_STATE)
	X_raw = data["X_train_raw"]          # variance-filtered + QT, train rows
	y = data["y_train"]

	skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
						  random_state=RANDOM_STATE)
	fold_indices = list(skf.split(X_raw, y))

	# --- 2. load / init checkpoint -----------------------------------------
	cv = _load_checkpoint(checkpoint)

	# --- 3. run every (extractor, model) combination -----------------------
	for extractor_name, extractor_fn in EXTRACTORS.items():
		for model_name, trainer in MODELS.items():
			k = _key(extractor_name, model_name)
			cv.setdefault(k, {"aucs": [], "accs": [], "f1s": []})

			for fold_idx, (tr_idx, va_idx) in enumerate(fold_indices):
				if len(cv[k]["aucs"]) > fold_idx:
					continue  # already done (resume)

				print(f"\n[{extractor_name} | {model_name}] "
					  f"fold {fold_idx + 1}/{N_FOLDS}")

				# Fit extractor on fold-train ONLY, transform val
				X_tr_emb, X_va_emb = extractor_fn(
					X_raw[tr_idx], y[tr_idx], X_raw[va_idx],
					emb_dim=EMB_DIM, random_state=RANDOM_STATE,
				)
				fold_data = {
					"X_train": X_tr_emb,
					"X_val": X_va_emb,
					# SVM trainer reads the *_pca keys; feed it the embedding
					"X_train_pca": X_tr_emb,
					"X_val_pca": X_va_emb,
					"y_train": y[tr_idx],
					"y_val": y[va_idx],
				}
				try:
					res = trainer(fold_data)
				except Exception as exc:
					# Save whatever is complete so a re-launch resumes from
					# THIS fold (not from zero), then surface the error.
					_atomic_save(cv, checkpoint)
					print(f"    FOLD FAILED [{k} fold {fold_idx + 1}]: "
						  f"{type(exc).__name__}: {exc}")
					print("    Progress saved. Re-run the script to retry "
						  "this fold and continue.")
					raise
				cv[k]["aucs"].append(res["auc_roc"])
				cv[k]["accs"].append(res["accuracy"])
				cv[k]["f1s"].append(res["f1_score"])

				_atomic_save(cv, checkpoint)
				print(f"    AUC={res['auc_roc']:.4f} (checkpoint saved)")

	# --- 4. summaries + paired tests vs anova_pca (per model) --------------
	print("\n=== Mean 5-fold AUC by (extractor, model) ===")
	summary = {}
	for extractor_name in EXTRACTORS:
		for model_name in MODELS:
			k = _key(extractor_name, model_name)
			aucs = cv[k]["aucs"]
			summary[k] = {
				"extractor": extractor_name,
				"model": model_name,
				"aucs": aucs,
				"mean_auc": float(np.mean(aucs)),
				"std_auc": float(np.std(aucs)),
			}
			print(f"  {k:<34} {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

	stats_rows = []
	n_tr = len(fold_indices[0][0])
	n_va = len(fold_indices[0][1])
	for model_name in MODELS:
		base = np.asarray(cv[_key("anova_pca", model_name)]["aucs"])
		ps, rows = [], []
		for extractor_name in EXTRACTORS:
			if extractor_name == "anova_pca":
				continue
			a = np.asarray(cv[_key(extractor_name, model_name)]["aucs"])
			_, wp = wilcoxon_signed_rank(a, base)
			_, nbp = corrected_resampled_ttest(a, base, n_train=n_tr, n_test=n_va)
			d = cohens_d(a, base)
			row = {
				"model": model_name,
				"extractor": extractor_name,
				"vs": "anova_pca",
				"mean_diff": float(a.mean() - base.mean()),
				"wilcoxon_p": wp,
				"nb_corrected_p": nbp,
				"cohens_d": d,
				"effect_size": effect_size_label(d),
			}
			rows.append(row)
			ps.append(nbp)
		for row, adj in zip(rows, holm_bonferroni(ps)):
			row["holm_adjusted_p"] = adj
			row["holm_significant"] = adj < 0.05
		stats_rows.extend(rows)

	print("\n=== Learned extractor vs anova_pca (Holm-adjusted) ===")
	for row in stats_rows:
		flag = " ***" if row["holm_significant"] else ""
		print(f"  {row['model']:<16} {row['extractor']:<12} "
			  f"dAUC={row['mean_diff']:+.4f} d={row['cohens_d']:+.2f} "
			  f"p_adj={row['holm_adjusted_p']:.4f}{flag}")

	elapsed = (time.time() - start) / 60
	out = {
		"metadata": {
			"version": "v06B",
			"purpose": "feature extraction: ANOVA+PCA vs MLP encoder vs autoencoder",
			"extractors": list(EXTRACTORS.keys()),
			"models": list(MODELS.keys()),
			"emb_dim": EMB_DIM,
			"n_folds": N_FOLDS,
			"seed": RANDOM_STATE,
			"n_samples": int(len(y)),
			"runtime_minutes": elapsed,
		},
		"cv": summary,
		"paired_stats_vs_anova_pca": stats_rows,
	}
	out_path = out_dir / "v06B_feature_extraction.json"
	with open(out_path, "w") as f:
		json.dump(out, f, indent=2, cls=NumpyEncoder)
	print(f"\nSaved {out_path} (runtime {elapsed:.1f} min)")


if __name__ == "__main__":
	main()
