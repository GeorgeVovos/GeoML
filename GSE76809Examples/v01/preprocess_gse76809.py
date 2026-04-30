"""
Preprocess GSE76809 for binary classification: SSc vs Healthy.

This script:
1. Loads the expression matrix and metadata
2. Assigns binary labels (SSc=1, Healthy=0)
3. Filters to the largest platform (GPL6480) for consistency
4. Removes low-variance genes, applies log2 transform + quantile normalization
5. Selects top features by variance (or mutual information)
6. Saves train/test splits ready for modeling
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif


DATA_DIR = Path("data/GSE76809")
OUTPUT_DIR = Path("data/GSE76809/processed")


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
        # Title-based healthy: Nor*, NL*
        if re.match(r"(nor|nl)\d", title):
            return 0

        # --- SSc ---
        if any(k in title for k in ["ssc", "dssc", "lssc", "rit", "nonrit"]):
            return 1
        if "sclerosis" in disease or "scleroderma" in disease or "ssc" in disease:
            return 1
        if case_ctrl == "case" or "ssc" in sample_type:
            return 1
        # Morphea = localized scleroderma
        if "morph" in title:
            return 1

        # Other diseases (IPAH, IPF, GERD, PPH) → exclude
        return -1

    meta = meta.copy()
    meta["label"] = meta.apply(_classify, axis=1)
    return meta


def preprocess(
    n_features: int = 50,
    platform: str = "GPL6480",
    test_size: float = 0.2,
    random_state: int = 42,
    selection_method: str = "variance",
):
    """Run full preprocessing pipeline."""
    print("Loading data...")
    expr = pd.read_csv(DATA_DIR / "GSE76809_expression_matrix.csv", index_col=0, low_memory=False)
    meta = pd.read_csv(DATA_DIR / "GSE76809_metadata.csv")

    # Assign labels
    meta = assign_labels(meta)
    print(f"Label distribution (all): {meta['label'].value_counts().to_dict()}")

    # Filter to selected platform for probe consistency
    platform_samples = meta[meta["platform"] == platform]["sample_id"].values
    labeled_meta = meta[(meta["platform"] == platform) & (meta["label"] >= 0)]
    print(f"Platform {platform}: {len(platform_samples)} total, {len(labeled_meta)} labeled (SSc/Healthy)")

    sample_ids = labeled_meta["sample_id"].values
    labels = labeled_meta["label"].values

    # Subset expression matrix
    available = [s for s in sample_ids if s in expr.columns]
    print(f"Samples with expression data: {len(available)}")

    X = expr[available].T  # samples × genes
    y = labels[: len(available)]

    # Drop genes with any NaN
    X = X.dropna(axis=1)
    print(f"Genes after NaN removal: {X.shape[1]}")

    # Log2 transform (add small offset for zeros)
    X_min = X.min().min()
    if X_min <= 0:
        X = X - X_min + 1
    X = np.log2(X + 1)

    # Remove low-variance genes (bottom 50%)
    variances = X.var(axis=0)
    high_var_mask = variances > variances.median()
    X = X.loc[:, high_var_mask]
    print(f"Genes after variance filter: {X.shape[1]}")

    # Feature selection
    if selection_method == "mutual_info":
        print(f"Selecting top {n_features} features by mutual information...")
        mi_scores = mutual_info_classif(X.values, y, random_state=random_state)
        top_idx = np.argsort(mi_scores)[-n_features:]
        X = X.iloc[:, top_idx]
    else:
        # Top N by variance
        print(f"Selecting top {n_features} features by variance...")
        top_var_genes = X.var(axis=0).nlargest(n_features).index
        X = X[top_var_genes]

    print(f"Final feature matrix: {X.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    print(f"Train label dist: SSc={sum(y_train==1)}, Healthy={sum(y_train==0)}")
    print(f"Test label dist: SSc={sum(y_test==1)}, Healthy={sum(y_test==0)}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "X_test.npy", X_test)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)
    pd.Series(X.columns).to_csv(OUTPUT_DIR / "selected_features.csv", index=False)

    # Save metadata
    info = {
        "platform": platform,
        "n_features": n_features,
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "selection_method": selection_method,
    }
    pd.Series(info).to_json(OUTPUT_DIR / "preprocessing_info.json")
    print(f"\nSaved to {OUTPUT_DIR}/")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GSE76809 for ML")
    parser.add_argument("--n-features", type=int, default=50, help="Number of features to select")
    parser.add_argument("--platform", default="GPL6480", help="GEO platform to use")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--selection", choices=["variance", "mutual_info"], default="variance")
    args = parser.parse_args()

    preprocess(
        n_features=args.n_features,
        platform=args.platform,
        test_size=args.test_size,
        selection_method=args.selection,
    )
