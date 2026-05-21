"""
Preprocess GSE76809 for binary classification: SSc vs Healthy (v05).

Same dataset as v04 (GSE76809 / GPL6480) but a DIFFERENT data slice:
1. Different train/test split seed (2024 vs 42) -> different sample partition
2. ANOVA F-test feature selection (vs mutual information in v04)
3. Fewer features (16 vs 64) -> 4 qubits instead of 6
4. NO SMOTE -> rely on class weighting / scale_pos_weight only
5. PCA to 6 components (vs 8) for the quantum kernel
6. 5-fold CV preserved for direct comparability with v04
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA


_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _ROOT / "data" / "GSE76809"
OUTPUT_DIR = _ROOT / "data" / "GSE76809" / "processed_v05"


def assign_labels(meta: pd.DataFrame) -> pd.DataFrame:
    """Assign binary labels: 1=SSc, 0=Healthy."""

    def _classify(row):
        title = str(row["title"]).lower()
        disease = str(row["disease state"]).lower()
        sample_type = str(row["sample type"]).lower()
        case_ctrl = str(row["case/control"]).lower()

        if "normal" in disease or "normal" in sample_type or case_ctrl == "control":
            return 0
        if re.match(r"(nor|nl)\d", title):
            return 0

        if any(k in title for k in ["ssc", "dssc", "lssc", "rit", "nonrit"]):
            return 1
        if "sclerosis" in disease or "scleroderma" in disease or "ssc" in disease:
            return 1
        if case_ctrl == "case" or "ssc" in sample_type:
            return 1
        if "morph" in title:
            return 1

        return -1

    meta = meta.copy()
    meta["label"] = meta.apply(_classify, axis=1)
    return meta


def build_fold_features(X_train_raw, y_train, X_val_raw, n_features=16,
                        n_pca=6, random_state=2024):
    """Fit ANOVA selection + StandardScaler + PCA on the fold's training rows ONLY.

    All feature-engineering steps that use the labels or per-feature statistics
    are refit inside each fold to prevent validation-row leakage. Returns the
    full fold dict expected by the v05 models.
    """
    # ANOVA F-test on training rows of this fold only
    f_scores, _ = f_classif(X_train_raw, y_train)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    top_idx = np.argsort(f_scores)[-n_features:]

    X_tr_sel = X_train_raw[:, top_idx]
    X_va_sel = X_val_raw[:, top_idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_sel)
    X_va = scaler.transform(X_va_sel)

    pca = PCA(n_components=n_pca, random_state=random_state)
    X_tr_pca = pca.fit_transform(X_tr)
    X_va_pca = pca.transform(X_va)
    pca_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_tr_pca = pca_scaler.fit_transform(X_tr_pca)
    X_va_pca = pca_scaler.transform(X_va_pca)

    X_tr_norm = X_tr / (np.linalg.norm(X_tr, axis=1, keepdims=True) + 1e-10)
    X_va_norm = X_va / (np.linalg.norm(X_va, axis=1, keepdims=True) + 1e-10)

    return {
        "X_train": X_tr, "X_val": X_va,
        "X_train_norm": X_tr_norm, "X_val_norm": X_va_norm,
        "X_train_pca": X_tr_pca, "X_val_pca": X_va_pca,
        "y_train": y_train, "y_val": None,  # caller sets y_val
    }


def preprocess(
    n_features: int = 16,
    platform: str = "GPL6480",
    test_size: float = 0.2,
    random_state: int = 2024,
    n_folds: int = 5,
    n_pca: int = 6,
):
    """
    v05 preprocessing: ANOVA F-test selection, NO SMOTE, smaller feature set.

    Returns:
        dict with keys:
            'X_train', 'X_test', 'y_train', 'y_test' - holdout split (scaled, no SMOTE)
            'X_train_norm', 'X_test_norm' - unit-norm for amplitude encoding
            'X_train_pca', 'X_test_pca' - PCA-projected (n_pca components) for kernel
            'folds' - list of fold dicts for CV (no SMOTE)
            'folds_pre_smote' - identical to 'folds' here (kept for API compatibility)
            'n_features' - number of features used
    """
    print("Loading data...")
    expr = pd.read_csv(DATA_DIR / "GSE76809_expression_matrix.csv", index_col=0, low_memory=False)
    meta = pd.read_csv(DATA_DIR / "GSE76809_metadata.csv")

    meta = assign_labels(meta)
    print(f"Label distribution (all): {meta['label'].value_counts().to_dict()}")

    labeled_meta = meta[(meta["platform"] == platform) & (meta["label"] >= 0)]
    print(f"Platform {platform}: {len(labeled_meta)} labeled samples")

    sample_ids = labeled_meta["sample_id"].values
    labels = labeled_meta["label"].values

    available = [s for s in sample_ids if s in expr.columns]
    print(f"Samples with expression data: {len(available)}")

    X = expr[available].T
    y = labels[:len(available)]

    X = X.dropna(axis=1)
    print(f"Genes after NaN removal: {X.shape[1]}")

    X_min = X.min().min()
    if X_min <= 0:
        X = X - X_min + 1
    X = np.log2(X + 1)

    qt = QuantileTransformer(
        n_quantiles=min(100, X.shape[0]),
        output_distribution="normal",
        random_state=random_state,
    )
    X_qt = pd.DataFrame(qt.fit_transform(X), index=X.index, columns=X.columns)

    # Stricter variance filter than v04 (top 50% instead of 75%)
    variances = X_qt.var(axis=0)
    high_var_mask = variances > variances.quantile(0.50)
    X_qt = X_qt.loc[:, high_var_mask]
    print(f"Genes after variance filter (top 50%): {X_qt.shape[1]}")

    # Holdout split with DIFFERENT seed than v04
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_qt, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ANOVA F-test feature selection on train set (different from v04's MI)
    print(f"Selecting top {n_features} features by ANOVA F-test...")
    f_scores, _ = f_classif(X_train_df.values, y_train)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    f_series = pd.Series(f_scores, index=X_train_df.columns)
    top_features = f_series.nlargest(n_features).index.tolist()

    X_train_sel = X_train_df[top_features].values
    X_test_sel = X_test_df[top_features].values

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    # PCA projection for quantum kernel
    pca = PCA(n_components=n_pca, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    # Scale PCA features to [0, pi] for angle encoding in the kernel
    pca_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_pca = pca_scaler.fit_transform(X_train_pca)
    X_test_pca = pca_scaler.transform(X_test_pca)

    print(f"PCA ({n_pca} comps) explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"NO SMOTE - Train: {X_train_scaled.shape[0]} "
          f"(SSc={sum(y_train==1)}, Healthy={sum(y_train==0)})")
    print(f"Test: {X_test_scaled.shape[0]} (SSc={sum(y_test==1)}, Healthy={sum(y_test==0)})")

    # Unit-norm for amplitude encoding
    X_train_norm = X_train_scaled / (np.linalg.norm(X_train_scaled, axis=1, keepdims=True) + 1e-10)
    X_test_norm = X_test_scaled / (np.linalg.norm(X_test_scaled, axis=1, keepdims=True) + 1e-10)

    # 5-fold CV splits (folds use per-fold ANOVA/scaling/PCA — no leakage)
    print(f"\nGenerating {n_folds}-fold CV splits (per-fold feature engineering)...")
    X_train_raw_all = X_train_df.values  # variance-filtered, unselected, unscaled
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_raw_all, y_train)):
        fold = build_fold_features(
            X_train_raw_all[train_idx],
            y_train[train_idx],
            X_train_raw_all[val_idx],
            n_features=n_features,
            n_pca=n_pca,
            random_state=random_state,
        )
        fold["y_val"] = y_train[val_idx]
        folds.append(fold)
        print(f"  Fold {fold_idx+1}: train={len(train_idx)}, val={len(val_idx)}")

    # Same fold list serves the 'pre_smote' role for API compatibility with v04 patterns
    folds_pre_smote = folds

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "X_train.npy", X_train_scaled)
    np.save(OUTPUT_DIR / "X_test.npy", X_test_scaled)
    np.save(OUTPUT_DIR / "X_train_norm.npy", X_train_norm)
    np.save(OUTPUT_DIR / "X_test_norm.npy", X_test_norm)
    np.save(OUTPUT_DIR / "X_train_pca.npy", X_train_pca)
    np.save(OUTPUT_DIR / "X_test_pca.npy", X_test_pca)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)
    pd.Series(top_features).to_csv(OUTPUT_DIR / "selected_features.csv", index=False)

    print(f"\nSaved to {OUTPUT_DIR}/")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "X_train_norm": X_train_norm,
        "X_test_norm": X_test_norm,
        "X_train_pre_smote": X_train_scaled,
        "X_test_pca": X_test_pca,
        "X_train_pca": X_train_pca,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pre_smote": y_train,
        "folds": folds,
        "folds_pre_smote": folds_pre_smote,
        "n_features": n_features,
        "n_pca": n_pca,
        "random_state": random_state,
        # Raw (variance-filtered + QT) matrices for per-subsample refitting in
        # learning curves. These do NOT include ANOVA selection / scaling / PCA.
        "X_train_raw": X_train_df.values,
        "X_test_raw": X_test_df.values,
    }


if __name__ == "__main__":
    preprocess()
