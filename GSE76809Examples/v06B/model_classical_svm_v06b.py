"""v06B-local RBF SVM trainer (fixes the probability-inversion artifact).

Why this exists
---------------
The shared v06 SVM trainer (`v06/model_classical_svm.py`) ranks validation
samples by `SVC.predict_proba(...)[:, 1]`. With `probability=True`, scikit-learn
fits an *internal* Platt-scaling cross-validation to produce those
probabilities. On v06B's tiny, heavily imbalanced inner folds (~8 Healthy
per 3-fold split) this Platt model can become anti-correlated with the SVM
decision boundary, producing AUCs that are the *inverted mirror* of the true
ranking (e.g. 0.116 instead of ~0.88). That artifact corrupted the
`anova_pca::classical_svm` baseline (mean AUC 0.319) and inflated every
learned-extractor delta on SVM.

Fixes here (v06B only; v06 results untouched):
  1. Rank by `decision_function` (monotonic, never inverted) for AUC.
  2. Use a guarded StratifiedKFold for the inner grid search so each inner
	 fold keeps both classes; fall back to a default SVM when a fold is too
	 small to stratify into 3.
  3. Keep `class_weight='balanced'`, RBF kernel, and the same param grid so
	 the comparison to v06 stays apples-to-apples in every other respect.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def train_classical_svm(fold_data, C_values=None):
	"""RBF SVM with inversion-safe AUC scoring for the v06B comparison."""
	X_train = fold_data["X_train_pca"]
	X_val = fold_data["X_val_pca"]
	y_train = np.asarray(fold_data["y_train"])
	y_val = np.asarray(fold_data["y_val"])

	if C_values is None:
		C_values = [0.1, 1.0, 10.0, 100.0]

	param_grid = {
		"C": C_values,
		"gamma": ["scale", "auto", 0.1, 1.0],
	}

	# decision_function ranks are immune to the Platt-scaling inversion, so we
	# do NOT need probability=True for AUC.
	svm_base = SVC(
		kernel="rbf",
		class_weight="balanced",
		random_state=2026,
	)

	# Guard the inner CV: only grid-search if the minority class can populate
	# all 3 stratified folds; otherwise use a sane default SVM.
	min_class = int(np.min(np.bincount(y_train.astype(int))))
	if min_class >= 3:
		inner_cv = StratifiedKFold(
			n_splits=3, shuffle=True, random_state=2026
		)
		grid = GridSearchCV(
			svm_base, param_grid,
			cv=inner_cv, scoring="roc_auc",
			n_jobs=-1, refit=True,
		)
		grid.fit(X_train, y_train)
		best_svm = grid.best_estimator_
		print(f"    RBF SVM best params: C={grid.best_params_['C']}, "
			  f"gamma={grid.best_params_['gamma']}")
	else:
		best_svm = svm_base.set_params(C=1.0, gamma="scale")
		best_svm.fit(X_train, y_train)
		print("    RBF SVM: minority class < 3 — using default C=1, gamma=scale")

	# Rank by signed distance to the boundary (monotonic, never inverted).
	val_scores = best_svm.decision_function(X_val)
	val_preds = best_svm.predict(X_val)

	try:
		auc = roc_auc_score(y_val, val_scores)
	except ValueError:
		auc = 0.5

	acc = accuracy_score(y_val, val_preds)
	f1 = f1_score(y_val, val_preds, zero_division=0)

	print(f"    -> AUC: {auc:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")

	return {
		"auc_roc": auc,
		"accuracy": acc,
		"f1_score": f1,
		"predictions": val_scores,
		"pred_binary": val_preds,
		"y_true": y_val,
	}
