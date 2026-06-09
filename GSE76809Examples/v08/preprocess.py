"""v08 preprocessing — same v06 GSE76809 pipeline, but exposing the *raw*
80%-train / 20%-test split for stratified subsampling.

Returns a dict with:
- X_train_raw / y_train  : variance-filtered+QT matrix used for subsampling
- X_test_raw / y_test    : held-out 54-sample test set (NEVER touched)
- variance-filter-only feature names
- random_state           : the seed used for the holdout split

v06's build_fold_features() is re-used to refit ANOVA/StandardScaler/PCA
on every subsample so each (N, seed) cell is leakage-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif  # noqa: F401  (sanity)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent / "v06"))
from preprocess_gse76809 import (  # noqa: E402
	DATA_DIR, assign_labels, build_fold_features, apply_smote_to_fold,
)


def preprocess(random_state: int = 2026, test_size: float = 0.2):
	"""Return the GSE76809 80/20 split as raw matrices (no per-fold work)."""
	print("Loading GSE76809...")
	expr = pd.read_csv(DATA_DIR / "GSE76809_expression_matrix.csv",
					   index_col=0, low_memory=False)
	meta = pd.read_csv(DATA_DIR / "GSE76809_metadata.csv")
	meta = assign_labels(meta)

	labelled = meta[(meta["platform"] == "GPL6480") & (meta["label"] >= 0)]
	available = [s for s in labelled["sample_id"] if s in expr.columns]
	X = expr[available].T.dropna(axis=1)
	y = labelled.set_index("sample_id").loc[available, "label"].values
	print(f"  usable: {len(X)} samples, {X.shape[1]} genes after NaN drop")

	# Log2 + quantile-normal — same as v06.
	X_min = X.min().min()
	if X_min <= 0:
		X = X - X_min + 1
	X = np.log2(X + 1)
	qt = QuantileTransformer(
		n_quantiles=min(100, X.shape[0]),
		output_distribution="normal",
		random_state=random_state,
	)
	X = pd.DataFrame(qt.fit_transform(X), index=X.index, columns=X.columns)

	# Top-50% variance — same as v06.
	variances = X.var(axis=0)
	X = X.loc[:, variances > variances.quantile(0.50)]
	print(f"  after variance filter: {X.shape[1]} genes")

	X_train_df, X_test_df, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=random_state, stratify=y
	)

	return {
		"X_train_raw": X_train_df.values,
		"y_train": np.asarray(y_train),
		"X_test_raw": X_test_df.values,
		"y_test": np.asarray(y_test),
		"feature_names_var": X_train_df.columns.tolist(),
		"random_state": random_state,
	}


# Re-export helpers for the driver script.
__all__ = ["preprocess", "build_fold_features", "apply_smote_to_fold"]


if __name__ == "__main__":
	data = preprocess()
	print(f"OK: train={len(data['X_train_raw'])}, test={len(data['X_test_raw'])}")
