"""Extra strong classical baselines for v12.

Each function takes a v06-style ``fold_data`` dict and returns the same
output schema as v06 model trainers (auc_roc / accuracy / f1_score /
predictions / pred_binary / y_true).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC


def _pack(y_true, probs, preds):
	try:
		auc = roc_auc_score(y_true, probs)
	except ValueError:
		auc = 0.5
	return {
		"auc_roc": auc,
		"accuracy": accuracy_score(y_true, preds),
		"f1_score": f1_score(y_true, preds, zero_division=0),
		"predictions": probs,
		"pred_binary": preds,
		"y_true": y_true,
	}


def train_lr_elasticnet(fold_data, Cs=None):
	if Cs is None:
		Cs = [0.01, 0.1, 1.0, 10.0]
	grid = GridSearchCV(
		LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000,
							class_weight="balanced", random_state=2026),
		{"C": Cs, "l1_ratio": [0.2, 0.5, 0.8]},
		cv=3, scoring="roc_auc", n_jobs=-1,
	)
	grid.fit(fold_data["X_train"], fold_data["y_train"])
	clf = grid.best_estimator_
	probs = clf.predict_proba(fold_data["X_val"])[:, 1]
	preds = clf.predict(fold_data["X_val"])
	return _pack(fold_data["y_val"], probs, preds)


def train_linear_svm(fold_data, Cs=None):
	if Cs is None:
		Cs = [0.01, 0.1, 1.0, 10.0]
	grid = GridSearchCV(
		LinearSVC(class_weight="balanced", random_state=2026, max_iter=5000),
		{"C": Cs}, cv=3, scoring="roc_auc", n_jobs=-1,
	)
	grid.fit(fold_data["X_train"], fold_data["y_train"])
	clf = grid.best_estimator_
	# LinearSVC has no predict_proba — use decision_function as probability score
	scores = clf.decision_function(fold_data["X_val"])
	preds = clf.predict(fold_data["X_val"])
	return _pack(fold_data["y_val"], scores, preds)


def train_random_forest(fold_data):
	grid = GridSearchCV(
		RandomForestClassifier(class_weight="balanced", random_state=2026, n_jobs=-1),
		{"n_estimators": [200, 500], "max_depth": [None, 5, 10]},
		cv=3, scoring="roc_auc", n_jobs=-1,
	)
	grid.fit(fold_data["X_train"], fold_data["y_train"])
	clf = grid.best_estimator_
	probs = clf.predict_proba(fold_data["X_val"])[:, 1]
	preds = clf.predict(fold_data["X_val"])
	return _pack(fold_data["y_val"], probs, preds)


def train_gnb(fold_data):
	clf = GaussianNB()
	clf.fit(fold_data["X_train"], fold_data["y_train"])
	probs = clf.predict_proba(fold_data["X_val"])[:, 1]
	preds = clf.predict(fold_data["X_val"])
	return _pack(fold_data["y_val"], probs, preds)


def train_nsc(fold_data):
	"""Nearest-shrunken-centroid approximation (PAM-style baseline).

	sklearn's NearestCentroid with `shrink_threshold` is the canonical
	light-weight stand-in for Tibshirani's PAM. No probability output;
	we use signed distance to the positive centroid as the score.
	"""
	clf = NearestCentroid(shrink_threshold=0.2)
	clf.fit(fold_data["X_train"], fold_data["y_train"])
	preds = clf.predict(fold_data["X_val"])
	# Score: negative distance to positive class centroid
	pos_centroid = clf.centroids_[list(clf.classes_).index(1)]
	diffs = fold_data["X_val"] - pos_centroid
	scores = -np.linalg.norm(diffs, axis=1)
	return _pack(fold_data["y_val"], scores, preds)


def train_stacked(fold_data):
	"""Stacked ensemble: LR-EN + linear SVM + RF + GNB, LR meta-learner."""
	estimators = [
		("lr_en", LogisticRegression(penalty="elasticnet", solver="saga",
									  l1_ratio=0.5, C=1.0, max_iter=5000,
									  class_weight="balanced", random_state=2026)),
		("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
									   random_state=2026, n_jobs=-1)),
		("gnb", GaussianNB()),
	]
	clf = StackingClassifier(
		estimators=estimators,
		final_estimator=LogisticRegression(class_weight="balanced", max_iter=2000),
		cv=3, n_jobs=-1, passthrough=False,
	)
	clf.fit(fold_data["X_train"], fold_data["y_train"])
	probs = clf.predict_proba(fold_data["X_val"])[:, 1]
	preds = clf.predict(fold_data["X_val"])
	return _pack(fold_data["y_val"], probs, preds)
