"""
GSE76809 Preprocessing for v06 — Fair Quantum vs Classical Comparison.

Key methodological fixes over v04/v05:
1. Per-fold SMOTE (no data leakage)
2. ANOVA F-test feature selection (deterministic)
3. 16 features, seed=2026 for a fresh split
4. PCA→6 for kernel methods
5. Provides utility for per-fold SMOTE application
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import BorderlineSMOTE, SMOTE


_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _ROOT / "data" / "GSE76809"


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


def apply_smote_to_fold(X_train, y_train, random_state=2026):
    """
    Apply SMOTE to a single fold's training data.
    This is the CORRECT way — SMOTE inside the fold, never before splitting.
    
    Returns augmented X_train, y_train.
    """
    minority_count = min(np.bincount(y_train.astype(int)))
    k_neighbors = min(5, minority_count - 1)
    k_neighbors = max(1, k_neighbors)
    
    try:
        smote = BorderlineSMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_aug, y_aug = smote.fit_resample(X_train, y_train)
    except ValueError:
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_aug, y_aug = smote.fit_resample(X_train, y_train)
    
    return X_aug, y_aug


def build_fold_features(X_train_raw, y_train, X_val_raw, n_features=16,
                        pca_components=6, random_state=2026):
    """Fit ANOVA selection + StandardScaler + PCA on the fold's training rows ONLY.

    Refitting these inside every CV fold prevents validation rows from
    influencing which features and PCA components are chosen.
    """
    f_scores, _ = f_classif(X_train_raw, y_train)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    top_idx = np.argsort(f_scores)[-n_features:]

    X_tr_sel = X_train_raw[:, top_idx]
    X_va_sel = X_val_raw[:, top_idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_sel)
    X_va = scaler.transform(X_va_sel)

    pca = PCA(n_components=pca_components, random_state=random_state)
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
        "y_train": y_train, "y_val": None,
    }


def preprocess(n_features=16, n_folds=5, random_state=2026, pca_components=6):
    """
    Preprocess GSE76809 with correct per-fold SMOTE methodology.
    
    Returns:
        dict with:
            'X_train', 'X_test', 'y_train', 'y_test' — holdout (NO SMOTE)
            'X_train_norm', 'X_test_norm' — unit-norm for amplitude encoding
            'X_train_pca', 'X_test_pca' — PCA-projected for kernel methods
            'folds' — list of fold dicts (raw, NO SMOTE applied yet)
            'feature_names' — selected feature names
            'n_features' — number of features
            'pca_components' — number of PCA components
    """
    print("Loading GSE76809 dataset...")
    expr = pd.read_csv(DATA_DIR / "GSE76809_expression_matrix.csv", index_col=0, low_memory=False)
    meta = pd.read_csv(DATA_DIR / "GSE76809_metadata.csv")

    meta = assign_labels(meta)
    print(f"Label distribution (all): {meta['label'].value_counts().to_dict()}")

    # Filter to GPL6480 platform, labeled samples
    labeled_meta = meta[(meta["platform"] == "GPL6480") & (meta["label"] >= 0)]
    print(f"Platform GPL6480: {len(labeled_meta)} labeled samples")

    sample_ids = labeled_meta["sample_id"].values
    labels = labeled_meta["label"].values

    available = [s for s in sample_ids if s in expr.columns]
    print(f"Samples with expression data: {len(available)}")

    X = expr[available].T
    y = labels[:len(available)]

    X = X.dropna(axis=1)
    feature_names_all = X.columns.tolist()
    print(f"Genes after NaN removal: {X.shape[1]}")

    # Log2 transform
    X_min = X.min().min()
    if X_min <= 0:
        X = X - X_min + 1
    X = np.log2(X + 1)

    # Quantile normalization
    qt = QuantileTransformer(
        n_quantiles=min(100, X.shape[0]),
        output_distribution="normal",
        random_state=random_state,
    )
    X_qt = pd.DataFrame(qt.fit_transform(X), index=X.index, columns=X.columns)

    # Variance filter (top 50%)
    variances = X_qt.var(axis=0)
    high_var_mask = variances > variances.quantile(0.50)
    X_qt = X_qt.loc[:, high_var_mask]
    feature_names_var = X_qt.columns.tolist()
    print(f"Genes after variance filter (top 50%): {X_qt.shape[1]}")

    # Train/test split (80/20)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_qt, y, test_size=0.2, random_state=random_state, stratify=y
    )
    print(f"\nHoldout split: Train={len(X_train_df)}, Test={len(X_test_df)}")
    print(f"  Train: SSc={sum(y_train==1)}, Healthy={sum(y_train==0)}")
    print(f"  Test:  SSc={sum(y_test==1)}, Healthy={sum(y_test==0)}")

    # ANOVA F-test feature selection (on train only)
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

    # PCA projection for kernel methods
    pca = PCA(n_components=pca_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Scale PCA to [0, π] for angle encoding
    pca_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_pca = pca_scaler.fit_transform(X_train_pca)
    X_test_pca = pca_scaler.transform(X_test_pca)
    
    print(f"PCA ({pca_components} comps) explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # Unit-norm for amplitude encoding
    X_train_norm = X_train_scaled / (np.linalg.norm(X_train_scaled, axis=1, keepdims=True) + 1e-10)
    X_test_norm = X_test_scaled / (np.linalg.norm(X_test_scaled, axis=1, keepdims=True) + 1e-10)

    print(f"NO SMOTE applied — will be applied per-fold during training")
    print(f"Train: {X_train_scaled.shape[0]} (SSc={sum(y_train==1)}, Healthy={sum(y_train==0)})")
    print(f"Test: {X_test_scaled.shape[0]} (SSc={sum(y_test==1)}, Healthy={sum(y_test==0)})")

    # Generate 5-fold CV splits with per-fold ANOVA/scaling/PCA (no leakage).
    # SMOTE is also applied per-fold by the models that need it.
    print(f"\nGenerating {n_folds}-fold CV splits (per-fold feature engineering)...")
    X_train_raw_all = X_train_df.values  # variance-filtered, unselected, unscaled
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_raw_all, y_train)):
        fold_data = build_fold_features(
            X_train_raw_all[train_idx],
            y_train[train_idx],
            X_train_raw_all[val_idx],
            n_features=n_features,
            pca_components=pca_components,
            random_state=random_state,
        )
        fold_data["y_val"] = y_train[val_idx]
        folds.append(fold_data)
        print(f"  Fold {fold_idx+1}: train={len(train_idx)} "
              f"(SSc={sum(y_train[train_idx]==1)}, H={sum(y_train[train_idx]==0)}), "
              f"val={len(val_idx)}")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_norm": X_train_norm,
        "X_test_norm": X_test_norm,
        "X_train_pca": X_train_pca,
        "X_test_pca": X_test_pca,
        "folds": folds,
        "feature_names": top_features,
        "n_features": n_features,
        "pca_components": pca_components,
        "random_state": random_state,
        # Raw (variance-filtered + QT) matrices for per-subsample refitting in
        # learning curves. These do NOT include ANOVA selection / scaling / PCA.
        "X_train_raw": X_train_df.values,
        "X_test_raw": X_test_df.values,
    }


if __name__ == "__main__":
    data = preprocess()
    print("\nPreprocessing complete.")
    print(f"Features: {data['n_features']}")
    print(f"Train samples: {len(data['X_train'])}")
    print(f"Test samples: {len(data['X_test'])}")
