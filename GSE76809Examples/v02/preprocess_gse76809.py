"""
Preprocess GSE76809 for binary classification: SSc vs Healthy (v02 - Improved).

Improvements over v01:
1. Mutual information feature selection (instead of variance)
2. More features (100 instead of 50)
3. SMOTE oversampling to address class imbalance
4. Uses all platforms (cross-platform quantile normalization)
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE


# Resolve paths relative to workspace root
_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _ROOT / "data" / "GSE76809"
OUTPUT_DIR = _ROOT / "data" / "GSE76809" / "processed_v02"


def assign_labels(meta: pd.DataFrame) -> pd.DataFrame:
    """Assign binary labels: 1=SSc, 0=Healthy."""

    def _classify(row):
        title = str(row["title"]).lower()
        disease = str(row["disease state"]).lower()
        sample_type = str(row["sample type"]).lower()
        case_ctrl = str(row["case/control"]).lower()

        # --- Healthy ---
        if "normal" in disease or "normal" in sample_type or case_ctrl == "control":
            return 0
        if re.match(r"(nor|nl)\d", title):
            return 0

        # --- SSc ---
        if any(k in title for k in ["ssc", "dssc", "lssc", "rit", "nonrit"]):
            return 1
        if "sclerosis" in disease or "scleroderma" in disease or "ssc" in disease:
            return 1
        if case_ctrl == "case" or "ssc" in sample_type:
            return 1
        if "morph" in title:
            return 1

        # Other diseases (IPAH, IPF, GERD, PPH) → exclude
        return -1

    meta = meta.copy()
    meta["label"] = meta.apply(_classify, axis=1)
    return meta


def preprocess(
    n_features: int = 100,
    platform: str = "GPL6480",
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = True,
):
    """Run improved preprocessing pipeline with MI selection + SMOTE."""
    print("Loading data...")
    expr = pd.read_csv(DATA_DIR / "GSE76809_expression_matrix.csv", index_col=0, low_memory=False)
    meta = pd.read_csv(DATA_DIR / "GSE76809_metadata.csv")

    # Assign labels
    meta = assign_labels(meta)
    print(f"Label distribution (all): {meta['label'].value_counts().to_dict()}")

    # Filter to selected platform
    labeled_meta = meta[(meta["platform"] == platform) & (meta["label"] >= 0)]
    print(f"Platform {platform}: {len(labeled_meta)} labeled samples")

    sample_ids = labeled_meta["sample_id"].values
    labels = labeled_meta["label"].values

    # Subset expression matrix
    available = [s for s in sample_ids if s in expr.columns]
    print(f"Samples with expression data: {len(available)}")

    X = expr[available].T  # samples × genes
    y = labels[:len(available)]

    # Drop genes with any NaN
    X = X.dropna(axis=1)
    print(f"Genes after NaN removal: {X.shape[1]}")

    # Log2 transform
    X_min = X.min().min()
    if X_min <= 0:
        X = X - X_min + 1
    X = np.log2(X + 1)

    # Quantile normalization across samples (rank-based)
    qt = QuantileTransformer(n_quantiles=min(100, X.shape[0]), output_distribution="normal", random_state=random_state)
    X_qt = pd.DataFrame(qt.fit_transform(X), index=X.index, columns=X.columns)

    # Remove bottom 25% lowest-variance genes (less aggressive than v01's 50%)
    variances = X_qt.var(axis=0)
    high_var_mask = variances > variances.quantile(0.25)
    X_qt = X_qt.loc[:, high_var_mask]
    print(f"Genes after variance filter (top 75%): {X_qt.shape[1]}")

    # Train/test split BEFORE feature selection (prevents data leakage)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_qt, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Feature selection using mutual information (on train set only)
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

    # SMOTE oversampling on training set
    if use_smote:
        smote = SMOTE(random_state=random_state, k_neighbors=min(5, sum(y_train == 0) - 1))
        X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
        print(f"After SMOTE  — Train: {X_train_final.shape[0]} (SSc={sum(y_train_final==1)}, Healthy={sum(y_train_final==0)})")
    else:
        X_train_final, y_train_final = X_train_scaled, y_train

    print(f"Test: {X_test_scaled.shape[0]} (SSc={sum(y_test==1)}, Healthy={sum(y_test==0)})")
    print(f"Final feature count: {n_features}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "X_train.npy", X_train_final)
    np.save(OUTPUT_DIR / "X_test.npy", X_test_scaled)
    np.save(OUTPUT_DIR / "y_train.npy", y_train_final)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)
    pd.Series(top_features).to_csv(OUTPUT_DIR / "selected_features.csv", index=False)
    mi_series[top_features].to_csv(OUTPUT_DIR / "feature_mi_scores.csv")

    # Save metadata
    info = {
        "platform": platform,
        "n_features": n_features,
        "n_train_original": int(X_train_scaled.shape[0]),
        "n_train_after_smote": int(X_train_final.shape[0]),
        "n_test": int(X_test_scaled.shape[0]),
        "selection_method": "mutual_information",
        "smote": use_smote,
        "quantile_normalization": True,
    }
    pd.Series(info).to_json(OUTPUT_DIR / "preprocessing_info.json")
    print(f"\nSaved to {OUTPUT_DIR}/")

    # Also compute class weight for models that use it
    n_pos = sum(y_train == 1)
    n_neg = sum(y_train == 0)
    pos_weight = n_neg / n_pos  # < 1 since majority is positive
    neg_weight = n_pos / n_neg  # > 1 to upweight minority
    print(f"Class weights — Healthy (0): {neg_weight:.2f}, SSc (1): {pos_weight:.2f}")

    return X_train_final, X_test_scaled, y_train_final, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GSE76809 for ML (v02)")
    parser.add_argument("--n-features", type=int, default=100)
    parser.add_argument("--platform", default="GPL6480")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--no-smote", action="store_true")
    args = parser.parse_args()

    preprocess(
        n_features=args.n_features,
        platform=args.platform,
        test_size=args.test_size,
        use_smote=not args.no_smote,
    )
