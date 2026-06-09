"""
Classical XGBoost baseline (v04): Strong gradient boosting classifier.

XGBoost is widely considered the best off-the-shelf method for tabular data.
If quantum can match or beat XGBoost on small biomedical datasets, that
demonstrates genuine quantum advantage.
"""

import time
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report, roc_curve
)
from sklearn.model_selection import StratifiedKFold

_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _ROOT / "data" / "GSE76809" / "processed_v04"


def find_optimal_threshold(y_true, y_prob):
    """Find optimal threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


def train_classical_xgb(fold_data=None):
    """
    Train XGBoost classifier with scale_pos_weight for class imbalance.

    Uses pre-SMOTE data since XGBoost handles imbalance natively via
    scale_pos_weight parameter (no need for synthetic oversampling).
    """

    if fold_data is not None:
        X_train = fold_data["X_train"]
        X_test = fold_data["X_val"]
        y_train = fold_data["y_train"]
        y_test = fold_data["y_val"]
    else:
        # Use pre-SMOTE data for XGBoost (handles imbalance natively)
        X_train = np.load(DATA_DIR / "X_train_pre_smote.npy")
        X_test = np.load(DATA_DIR / "X_test.npy")
        y_train = np.load(DATA_DIR / "y_train_pre_smote.npy")
        y_test = np.load(DATA_DIR / "y_test.npy")

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    print(f"{'='*60}")
    print(f"CLASSICAL XGBoost (Gradient Boosting)")
    print(f"  Features: {X_train.shape[1]}, Trees: 200, Depth: 4")
    print(f"  Scale pos weight: {scale_pos_weight:.2f}")
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"{'='*60}")

    start_time = time.time()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    train_time = time.time() - start_time

    # Predict
    test_probs = model.predict_proba(X_test)[:, 1]

    # Threshold optimization
    opt_threshold = find_optimal_threshold(y_test, test_probs)
    test_preds = (test_probs >= opt_threshold).astype(int)

    acc = accuracy_score(y_test, test_preds)
    f1_ssc = f1_score(y_test, test_preds, pos_label=1)
    f1_healthy = f1_score(y_test, test_preds, pos_label=0)
    try:
        auc = roc_auc_score(y_test, test_probs)
    except ValueError:
        auc = 0.5

    # Feature importance
    importances = model.feature_importances_
    top_5_idx = np.argsort(importances)[-5:][::-1]
    top_5_imp = importances[top_5_idx]

    print(f"\n  FINAL (threshold={opt_threshold:.4f}):")
    print(f"  Accuracy: {acc:.4f} | F1(SSc): {f1_ssc:.4f} | "
          f"F1(Healthy): {f1_healthy:.4f} | AUC: {auc:.4f}")
    print(f"  Train time: {train_time:.1f}s")
    print(f"  Top 5 feature importances: {top_5_imp.round(4)}")
    print(classification_report(y_test, test_preds, target_names=["Healthy", "SSc"]))

    return {
        "accuracy": float(acc),
        "f1_score": float(f1_ssc),
        "f1_healthy": float(f1_healthy),
        "auc_roc": float(auc),
        "train_time_seconds": float(train_time),
        "n_params": int(model.n_estimators * (2 ** model.max_depth)),  # approximate
        "threshold": float(opt_threshold),
        "test_probs": test_probs,
    }


if __name__ == "__main__":
    train_classical_xgb()
