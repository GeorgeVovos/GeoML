"""L1-regularised Logistic Regression — the canonical small-data, high-dim
tabular baseline (and the missing classical competitor in v01-v06).

Used by v06 (additive baseline), v07 (cross-dataset), v08 (sample efficiency).

Why this matters: on gene-expression data with n ~ 200 and p ~ 16-64, an
L1-regularised linear model is historically the model to beat. Comparing
QML to MLP / XGBoost without also comparing to LR-L1 risks beating only
the wrong baselines. This file plugs that gap.

API matches the v06 model files: train_classical_logreg(fold_data, ...).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def train_classical_logreg(fold_data, random_state: int = 2026,
							penalty: str = "l1", Cs=None):
	"""Train L1-regularised logistic regression on a fold.

	No SMOTE — uses class_weight='balanced' which is the standard, well-
	behaved imbalance fix for linear models. Inner 3-fold CV picks ``C``.
	"""
	X_train = fold_data["X_train"]
	X_val = fold_data["X_val"]
	y_train = fold_data["y_train"]
	y_val = fold_data["y_val"]

	if Cs is None:
		Cs = [0.01, 0.1, 1.0, 10.0]

	base = LogisticRegression(
		penalty=penalty,
		solver="liblinear" if penalty in ("l1", "l2") else "saga",
		class_weight="balanced",
		max_iter=2000,
		random_state=random_state,
	)

	# Inner CV only if we have enough samples in the minority class;
	# otherwise just use C=1.0 to avoid spurious failures.
	minority = int(min(np.bincount(y_train.astype(int))))
	if minority >= 3:
		grid = GridSearchCV(base, {"C": Cs}, cv=3, scoring="roc_auc", n_jobs=-1)
		grid.fit(X_train, y_train)
		clf = grid.best_estimator_
		best_C = grid.best_params_["C"]
	else:
		base.C = 1.0
		base.fit(X_train, y_train)
		clf = base
		best_C = 1.0

	val_probs = clf.predict_proba(X_val)[:, 1]
	val_preds = clf.predict(X_val)

	try:
		auc = roc_auc_score(y_val, val_probs)
	except ValueError:
		auc = 0.5
	acc = accuracy_score(y_val, val_preds)
	f1 = f1_score(y_val, val_preds, zero_division=0)

	print(f"    LR-{penalty.upper()} best C={best_C} → AUC: {auc:.4f}, "
		  f"Acc: {acc:.4f}, F1: {f1:.4f}")

	return {
		"auc_roc": auc,
		"accuracy": acc,
		"f1_score": f1,
		"predictions": val_probs,
		"pred_binary": val_preds,
		"y_true": y_val,
	}
