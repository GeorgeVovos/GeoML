"""v07 generic preprocessing: same pipeline as v06, parameterised by gse_id.

This file re-implements v06.preprocess_gse76809.preprocess with two changes:
1. The dataset id is an argument (not hard-coded to GSE76809).
2. The labelling function is looked up from dataset_loaders.REGISTRY.

Otherwise the steps are identical to v06: log2 -> quantile-normal -> top-50 %
variance filter -> 80/20 stratified split -> ANOVA top-N selection
(per-fold!) -> StandardScaler -> PCA (per-fold!) -> unit-norm.

The function returns the same dict shape as v06, so v06's model files work
unchanged when imported by v07/compare_models.py.

CLI:
	python preprocess.py --inspect GSE9285   # print labelling diagnostics
	python preprocess.py --run     GSE9285   # run preprocessing once
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler

# Reuse v06's per-fold helpers verbatim — they're already leakage-safe.
_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent / "v06"))
from preprocess_gse76809 import apply_smote_to_fold, build_fold_features  # noqa: E402

from dataset_loaders import get_labeler, get_platform


_ROOT = _THIS.parent.parent
DATA_DIR = _ROOT / "data"


def load_dataset(gse_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Load expression matrix + metadata for ``gse_id``."""
	series_dir = DATA_DIR / gse_id
	expr_path = series_dir / f"{gse_id}_expression_matrix.csv"
	meta_path = series_dir / f"{gse_id}_metadata.csv"
	if not expr_path.exists():
		raise FileNotFoundError(
			f"{expr_path} not found. Run: python download_geo.py --gse {gse_id}"
		)
	expr = pd.read_csv(expr_path, index_col=0, low_memory=False)
	meta = pd.read_csv(meta_path)
	return expr, meta


def inspect_labels(gse_id: str) -> None:
	"""Print the label distribution for a dataset — sanity-check before CV."""
	_, meta = load_dataset(gse_id)
	labeler = get_labeler(gse_id)
	labels = labeler(meta)
	print(f"{gse_id} metadata columns: {list(meta.columns)}")
	print(f"{gse_id} label distribution: {labels.value_counts().to_dict()}")
	if (labels == -1).sum() > 0:
		print(f"  ({(labels == -1).sum()} samples excluded as ambiguous)")


def preprocess(gse_id: str, n_features: int = 16, n_folds: int = 5,
				pca_components: int = 6, random_state: int = 2026,
				test_size: float = 0.2):
	"""Same return contract as v06's preprocess(), but for arbitrary GSE."""
	print(f"[v07] Loading {gse_id}...")
	expr, meta = load_dataset(gse_id)

	labeler = get_labeler(gse_id)
	meta = meta.copy()
	meta["label"] = labeler(meta)
	print(f"  label distribution: {meta['label'].value_counts().to_dict()}")

	# Platform filter (if specified)
	platform = get_platform(gse_id)
	if platform is not None and "platform" in meta.columns:
		meta = meta[meta["platform"] == platform]
		print(f"  filtered to platform {platform}: {len(meta)} samples")

	labelled = meta[meta["label"] >= 0]
	available = [s for s in labelled["sample_id"] if s in expr.columns]
	if len(available) < 20:
		raise RuntimeError(
			f"{gse_id}: only {len(available)} labelled samples with expression "
			f"data — too few to fit a 5-fold CV. Check the labeller."
		)

	X = expr[available].T.dropna(axis=1)
	y = labelled.set_index("sample_id").loc[available, "label"].values
	print(f"  usable samples: {len(X)}, genes after NaN drop: {X.shape[1]}")

	# Log2 + quantile-normal (same as v06)
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

	# Top-50 % variance
	variances = X.var(axis=0)
	X = X.loc[:, variances > variances.quantile(0.50)]
	print(f"  after top-50% variance: {X.shape[1]} genes")

	# If the dataset is too small to support n_features after variance filter
	n_features_effective = min(n_features, X.shape[1])
	pca_components_eff = min(pca_components, n_features_effective)
	if n_features_effective < n_features:
		print(f"  WARNING: only {n_features_effective} features available, "
			  f"requested {n_features}")

	X_train_df, X_test_df, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=random_state, stratify=y
	)

	# Holdout-level ANOVA / scaler / PCA — only used for the holdout call.
	f_scores, _ = f_classif(X_train_df.values, y_train)
	f_scores = np.nan_to_num(f_scores, nan=0.0)
	top_idx = np.argsort(f_scores)[-n_features_effective:]
	top_features = X_train_df.columns[top_idx].tolist()

	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train_df[top_features].values)
	X_test_scaled = scaler.transform(X_test_df[top_features].values)

	pca = PCA(n_components=pca_components_eff, random_state=random_state)
	X_train_pca = pca.fit_transform(X_train_scaled)
	X_test_pca = pca.transform(X_test_scaled)
	pca_scaler = MinMaxScaler(feature_range=(0, np.pi))
	X_train_pca = pca_scaler.fit_transform(X_train_pca)
	X_test_pca = pca_scaler.transform(X_test_pca)

	X_train_norm = X_train_scaled / (np.linalg.norm(X_train_scaled, axis=1, keepdims=True) + 1e-10)
	X_test_norm = X_test_scaled / (np.linalg.norm(X_test_scaled, axis=1, keepdims=True) + 1e-10)

	# Per-fold splits (leakage-safe via build_fold_features)
	X_train_raw_all = X_train_df.values
	skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
	folds = []
	for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_raw_all, y_train)):
		fold = build_fold_features(
			X_train_raw_all[train_idx], y_train[train_idx],
			X_train_raw_all[val_idx],
			n_features=n_features_effective,
			pca_components=pca_components_eff,
			random_state=random_state,
		)
		fold["y_val"] = y_train[val_idx]
		folds.append(fold)
		print(f"    fold {fold_idx+1}: train={len(train_idx)} "
			  f"(pos={int((y_train[train_idx]==1).sum())}, "
			  f"neg={int((y_train[train_idx]==0).sum())}), val={len(val_idx)}")

	return {
		"gse_id": gse_id,
		"X_train": X_train_scaled, "X_test": X_test_scaled,
		"y_train": y_train, "y_test": y_test,
		"X_train_norm": X_train_norm, "X_test_norm": X_test_norm,
		"X_train_pca": X_train_pca, "X_test_pca": X_test_pca,
		"folds": folds,
		"feature_names": top_features,
		"n_features": n_features_effective,
		"pca_components": pca_components_eff,
		"random_state": random_state,
		"X_train_raw": X_train_df.values,
		"X_test_raw": X_test_df.values,
	}


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--inspect", metavar="GSE_ID", help="print label distribution and exit")
	ap.add_argument("--run", metavar="GSE_ID", help="run preprocess and print shapes")
	args = ap.parse_args()
	if args.inspect:
		inspect_labels(args.inspect)
	elif args.run:
		data = preprocess(args.run)
		print(f"OK: {data['gse_id']}, n_features={data['n_features']}, folds={len(data['folds'])}")
	else:
		ap.print_help()
