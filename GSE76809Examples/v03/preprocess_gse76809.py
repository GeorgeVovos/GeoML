"""
Preprocess GSE76809 for binary classification: SSc vs Healthy (v03).

Improvements over v02:
1. BorderlineSMOTE (better synthetic samples near decision boundary)
2. 64 features (power of 2 for amplitude encoding: 2^6 = 64)
3. Stratified 3-fold CV for robust evaluation
4. PCA whitening option for amplitude encoding compatibility
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
OUTPUT_DIR = _ROOT / "data" / "GSE76809" / "processed_v03"


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
    n_folds: int = 3,
):
    """
    Preprocessing with BorderlineSMOTE + MI selection + 3-fold CV splits.

    Returns:
        dict with keys:
            'X_train', 'X_test', 'y_train', 'y_test' — holdout split
            'folds' — list of (X_train_fold, X_val_fold, y_train_fold, y_val_fold) for CV
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

    qt = QuantileTransformer(n_quantiles=min(100, X.shape[0]), output_distribution="normal", random_state=random_state)
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
    mi_scores = mutual_info_classif(X_train_df.values, y_train, random_state=random_state, n_neighbors=5)
    mi_series = pd.Series(mi_scores, index=X_train_df.columns)
    top_features = mi_series.nlargest(n_features).index.tolist()

    X_train_sel = X_train_df[top_features].values
    X_test_sel = X_test_df[top_features].values

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    print(f"Before SMOTE — Train: {X_train_scaled.shape[0]} (SSc={sum(y_train==1)}, Healthy={sum(y_train==0)})")

    # BorderlineSMOTE
    k_neighbors = min(5, sum(y_train == 0) - 1)
    try:
        smote = BorderlineSMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
        smote_type = "BorderlineSMOTE"
    except ValueError:
        # Fall back to regular SMOTE if borderline fails
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
        smote_type = "SMOTE (fallback)"

    print(f"After {smote_type} — Train: {X_train_final.shape[0]} (SSc={sum(y_train_final==1)}, Healthy={sum(y_train_final==0)})")
    print(f"Test: {X_test_scaled.shape[0]} (SSc={sum(y_test==1)}, Healthy={sum(y_test==0)})")
    print(f"Final feature count: {n_features}")

    # Normalize for amplitude encoding (unit norm per sample)
    X_train_norm = X_train_final / np.linalg.norm(X_train_final, axis=1, keepdims=True)
    X_test_norm = X_test_scaled / np.linalg.norm(X_test_scaled, axis=1, keepdims=True)

    # Generate 3-fold CV splits (on the SMOTE'd training data)
    print(f"\nGenerating {n_folds}-fold CV splits...")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_final, y_train_final)):
        X_f_train = X_train_final[train_idx]
        X_f_val = X_train_final[val_idx]
        y_f_train = y_train_final[train_idx]
        y_f_val = y_train_final[val_idx]
        # Also normalized versions for amplitude encoding
        X_f_train_norm = X_train_norm[train_idx]
        X_f_val_norm = X_train_norm[val_idx]
        folds.append({
            "X_train": X_f_train, "X_val": X_f_val,
            "y_train": y_f_train, "y_val": y_f_val,
            "X_train_norm": X_f_train_norm, "X_val_norm": X_f_val_norm,
        })
        print(f"  Fold {fold_idx+1}: train={len(train_idx)}, val={len(val_idx)}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "X_train.npy", X_train_final)
    np.save(OUTPUT_DIR / "X_test.npy", X_test_scaled)
    np.save(OUTPUT_DIR / "X_train_norm.npy", X_train_norm)
    np.save(OUTPUT_DIR / "X_test_norm.npy", X_test_norm)
    np.save(OUTPUT_DIR / "y_train.npy", y_train_final)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)
    pd.Series(top_features).to_csv(OUTPUT_DIR / "selected_features.csv", index=False)

    n_pos = sum(y_train == 1)
    n_neg = sum(y_train == 0)
    pos_weight = n_neg / n_pos
    neg_weight = n_pos / n_neg
    print(f"\nClass weights — Healthy (0): {neg_weight:.2f}, SSc (1): {pos_weight:.2f}")
    print(f"Saved to {OUTPUT_DIR}/")

    return {
        "X_train": X_train_final,
        "X_test": X_test_scaled,
        "X_train_norm": X_train_norm,
        "X_test_norm": X_test_norm,
        "y_train": y_train_final,
        "y_test": y_test,
        "folds": folds,
        "n_features": n_features,
        "pos_weight": pos_weight,
        "neg_weight": neg_weight,
    }


if __name__ == "__main__":
    preprocess()
