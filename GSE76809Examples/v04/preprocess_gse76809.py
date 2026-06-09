"""
Preprocess GSE76809 for binary classification: SSc vs Healthy (v04).

Improvements over v03:
1. 5-fold CV for more robust evaluation
2. Save pre-SMOTE data for quantum kernel methods
3. PCA projection (64→8) for kernel feature map
4. Learning curve subsampling support
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from imblearn.over_sampling import BorderlineSMOTE, SMOTE


_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _ROOT / "data" / "GSE76809"
OUTPUT_DIR = _ROOT / "data" / "GSE76809" / "processed_v04"


def apply_smote_to_fold(X_train, y_train, random_state=42):
    """Apply BorderlineSMOTE (or fall back to SMOTE) to a single fold's training data.

    This is the CORRECT way to use SMOTE: synthetic samples are only ever
    generated from the training portion of a fold, never from validation rows.
    """
    minority_count = int(min(np.bincount(y_train.astype(int))))
    k_neighbors = max(1, min(5, minority_count - 1))
    try:
        sm = BorderlineSMOTE(random_state=random_state, k_neighbors=k_neighbors)
        return sm.fit_resample(X_train, y_train)
    except ValueError:
        sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        return sm.fit_resample(X_train, y_train)


def build_fold_features(X_train_raw, y_train, X_val_raw, n_features=64,
                        n_pca=8, random_state=42):
    """Fit MI selection + StandardScaler + PCA on the fold's training rows ONLY.

    Used by both CV folds and learning-curve subsamples so that small-data
    points are not given features chosen on the full training set.
    """
    mi_scores = mutual_info_classif(
        X_train_raw, y_train, random_state=random_state, n_neighbors=5
    )
    mi_scores = np.nan_to_num(mi_scores, nan=0.0)
    top_idx = np.argsort(mi_scores)[-n_features:]

    X_tr_sel = X_train_raw[:, top_idx]
    X_va_sel = X_val_raw[:, top_idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_sel)
    X_va = scaler.transform(X_va_sel)

    pca = PCA(n_components=n_pca, random_state=random_state)
    X_tr_pca = pca.fit_transform(X_tr)
    X_va_pca = pca.transform(X_va)
    from sklearn.preprocessing import MinMaxScaler
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


def preprocess(
    n_features: int = 64,
    platform: str = "GPL6480",
    test_size: float = 0.2,
    random_state: int = 42,
    n_folds: int = 5,
):
    """
    Preprocessing with BorderlineSMOTE + MI selection + 5-fold CV splits.

    Returns:
        dict with keys:
            'X_train', 'X_test', 'y_train', 'y_test' — holdout split (SMOTE'd)
            'X_train_pre_smote', 'y_train_pre_smote' — pre-SMOTE for kernel methods
            'X_test_scaled' — test set (scaled, not unit-norm)
            'X_train_norm', 'X_test_norm' — unit-norm versions for amplitude encoding
            'X_train_pca', 'X_test_pca' — PCA-projected (8 components) for kernel
            'folds' — list of fold dicts for CV
            'folds_pre_smote' — CV folds without SMOTE (for kernel)
            'n_features' — number of features used
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

    variances = X_qt.var(axis=0)
    high_var_mask = variances > variances.quantile(0.25)
    X_qt = X_qt.loc[:, high_var_mask]
    print(f"Genes after variance filter (top 75%): {X_qt.shape[1]}")

    # Holdout split
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_qt, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # MI feature selection on train set
    print(f"Selecting top {n_features} features by mutual information...")
    mi_scores = mutual_info_classif(
        X_train_df.values, y_train, random_state=random_state, n_neighbors=5
    )
    mi_series = pd.Series(mi_scores, index=X_train_df.columns)
    top_features = mi_series.nlargest(n_features).index.tolist()

    X_train_sel = X_train_df[top_features].values
    X_test_sel = X_test_df[top_features].values

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    # Save pre-SMOTE data (for kernel methods)
    X_train_pre_smote = X_train_scaled.copy()
    y_train_pre_smote = y_train.copy()

    # PCA projection for quantum kernel (64→8 features)
    pca = PCA(n_components=8, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    # Scale PCA features to [0, π] range for angle encoding in kernel
    from sklearn.preprocessing import MinMaxScaler
    pca_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_pca = pca_scaler.fit_transform(X_train_pca)
    X_test_pca = pca_scaler.transform(X_test_pca)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"Before SMOTE — Train: {X_train_scaled.shape[0]} "
          f"(SSc={sum(y_train==1)}, Healthy={sum(y_train==0)})")

    # BorderlineSMOTE
    k_neighbors = min(5, sum(y_train == 0) - 1)
    try:
        smote = BorderlineSMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
        smote_type = "BorderlineSMOTE"
    except ValueError:
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
        smote_type = "SMOTE (fallback)"

    print(f"After {smote_type} — Train: {X_train_final.shape[0]} "
          f"(SSc={sum(y_train_final==1)}, Healthy={sum(y_train_final==0)})")
    print(f"Test: {X_test_scaled.shape[0]} (SSc={sum(y_test==1)}, Healthy={sum(y_test==0)})")

    # Unit-norm for amplitude encoding
    X_train_norm = X_train_final / (np.linalg.norm(X_train_final, axis=1, keepdims=True) + 1e-10)
    X_test_norm = X_test_scaled / (np.linalg.norm(X_test_scaled, axis=1, keepdims=True) + 1e-10)

    # Generate 5-fold CV splits on the RAW (pre-SMOTE) training set.
    # All models share the same fold indices, so paired statistics are valid.
    # SMOTE is applied per-fold by `apply_smote_to_fold` inside the CV loop.
    print(f"\nGenerating {n_folds}-fold CV splits (raw, no SMOTE)...")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_pre_smote, y_train_pre_smote)):
        X_f_train = X_train_pre_smote[train_idx]
        X_f_val = X_train_pre_smote[val_idx]
        y_f_train = y_train_pre_smote[train_idx]
        y_f_val = y_train_pre_smote[val_idx]

        # Unit-norm versions (for amplitude encoding) computed from the raw fold rows
        X_f_train_norm = X_f_train / (np.linalg.norm(X_f_train, axis=1, keepdims=True) + 1e-10)
        X_f_val_norm = X_f_val / (np.linalg.norm(X_f_val, axis=1, keepdims=True) + 1e-10)

        folds.append({
            "X_train": X_f_train, "X_val": X_f_val,
            "y_train": y_f_train, "y_val": y_f_val,
            "X_train_norm": X_f_train_norm, "X_val_norm": X_f_val_norm,
            "X_train_pca": X_train_pca[train_idx], "X_val_pca": X_train_pca[val_idx],
        })
        print(f"  Fold {fold_idx+1}: train={len(train_idx)}, val={len(val_idx)}")

    # Kept for backward compatibility with code that referenced both fold lists.
    folds_pre_smote = folds

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "X_train.npy", X_train_final)
    np.save(OUTPUT_DIR / "X_test.npy", X_test_scaled)
    np.save(OUTPUT_DIR / "X_train_norm.npy", X_train_norm)
    np.save(OUTPUT_DIR / "X_test_norm.npy", X_test_norm)
    np.save(OUTPUT_DIR / "X_train_pre_smote.npy", X_train_pre_smote)
    np.save(OUTPUT_DIR / "X_train_pca.npy", X_train_pca)
    np.save(OUTPUT_DIR / "X_test_pca.npy", X_test_pca)
    np.save(OUTPUT_DIR / "y_train.npy", y_train_final)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)
    np.save(OUTPUT_DIR / "y_train_pre_smote.npy", y_train_pre_smote)
    pd.Series(top_features).to_csv(OUTPUT_DIR / "selected_features.csv", index=False)

    print(f"\nSaved to {OUTPUT_DIR}/")

    return {
        "X_train": X_train_final,
        "X_test": X_test_scaled,
        "X_train_norm": X_train_norm,
        "X_test_norm": X_test_norm,
        "X_train_pre_smote": X_train_pre_smote,
        "X_test_pca": X_test_pca,
        "X_train_pca": X_train_pca,
        "y_train": y_train_final,
        "y_test": y_test,
        "y_train_pre_smote": y_train_pre_smote,
        "folds": folds,
        "folds_pre_smote": folds_pre_smote,
        "n_features": n_features,
        # Raw (variance-filtered + QT) matrices for per-subsample refitting in
        # learning curves. These do NOT include MI selection / scaling / PCA.
        "X_train_raw": X_train_df.values,
        "X_test_raw": X_test_df.values,
    }


if __name__ == "__main__":
    preprocess()
