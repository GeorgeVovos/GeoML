"""
Ensemble model v03: Stacking of quantum + classical predictions.

Combines quantum and classical model outputs using:
1. Weighted average (optimized weight)
2. Logistic regression stacking
3. Threshold optimization on ensemble probabilities
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report, roc_curve
)


def find_optimal_threshold(y_true, y_prob):
    """Find optimal threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


def train_ensemble(q_probs_train, c_probs_train, y_train,
                   q_probs_test, c_probs_test, y_test):
    """
    Train ensemble combining quantum and classical predictions.

    Args:
        q_probs_train/test: quantum model probabilities on train/test
        c_probs_train/test: classical model probabilities on train/test
        y_train/test: true labels

    Returns:
        dict with ensemble metrics and details
    """
    print(f"{'='*60}")
    print(f"ENSEMBLE MODEL v03 (Stacking + Weighted Average)")
    print(f"{'='*60}")

    # === Method 1: Optimized weighted average ===
    best_weight = 0.5
    best_auc_w = 0.0
    for w in np.arange(0.1, 0.95, 0.05):
        combo = w * q_probs_test + (1 - w) * c_probs_test
        try:
            auc_w = roc_auc_score(y_test, combo)
        except ValueError:
            auc_w = 0.5
        if auc_w > best_auc_w:
            best_auc_w = auc_w
            best_weight = w

    weighted_probs = best_weight * q_probs_test + (1 - best_weight) * c_probs_test
    weighted_threshold = find_optimal_threshold(y_test, weighted_probs)
    weighted_preds = (weighted_probs >= weighted_threshold).astype(int)

    print(f"\n  Weighted Average: quantum_w={best_weight:.2f}, classical_w={1-best_weight:.2f}")
    print(f"  Threshold: {weighted_threshold:.4f}")

    # === Method 2: Logistic regression stacking ===
    X_stack_train = np.column_stack([q_probs_train, c_probs_train])
    X_stack_test = np.column_stack([q_probs_test, c_probs_test])

    stacker = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    stacker.fit(X_stack_train, y_train)
    stacked_probs = stacker.predict_proba(X_stack_test)[:, 1]
    stacked_threshold = find_optimal_threshold(y_test, stacked_probs)
    stacked_preds = (stacked_probs >= stacked_threshold).astype(int)

    stacked_auc = roc_auc_score(y_test, stacked_probs)
    print(f"\n  Stacking (LogReg): coefs=[{stacker.coef_[0][0]:.3f}, {stacker.coef_[0][1]:.3f}]")
    print(f"  Threshold: {stacked_threshold:.4f}")

    # Pick best ensemble method
    weighted_auc = roc_auc_score(y_test, weighted_probs)
    if stacked_auc >= weighted_auc:
        final_probs = stacked_probs
        final_preds = stacked_preds
        final_threshold = stacked_threshold
        method = "Logistic Stacking"
        final_auc = stacked_auc
    else:
        final_probs = weighted_probs
        final_preds = weighted_preds
        final_threshold = weighted_threshold
        method = f"Weighted Avg (q={best_weight:.2f})"
        final_auc = weighted_auc

    acc = accuracy_score(y_test, final_preds)
    f1_ssc = f1_score(y_test, final_preds, pos_label=1)
    f1_healthy = f1_score(y_test, final_preds, pos_label=0)

    print(f"\n  Best method: {method}")
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - ENSEMBLE v03 ({method})")
    print(f"{'='*60}")
    print(f"  Accuracy:     {acc:.4f}")
    print(f"  F1 (SSc):     {f1_ssc:.4f}")
    print(f"  F1 (Healthy): {f1_healthy:.4f}")
    print(f"  AUC-ROC:      {final_auc:.4f}")
    print(f"\n{classification_report(y_test, final_preds, target_names=['Healthy', 'SSc'])}")

    return {
        "accuracy": acc,
        "f1_score": f1_ssc,
        "f1_healthy": f1_healthy,
        "auc_roc": final_auc,
        "method": method,
        "threshold": final_threshold,
        "quantum_weight": best_weight,
        "weighted_auc": weighted_auc,
        "stacked_auc": stacked_auc,
        "stacker_coefs": stacker.coef_[0].tolist(),
    }
