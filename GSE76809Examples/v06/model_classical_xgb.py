"""
Tuned XGBoost for v06.

Key improvement over v04/v05: Uses cross-validated hyperparameter tuning
instead of fixed config. This gives XGBoost a fair chance — the v04/v05
fixed config (200 trees, depth 4) was suboptimal for small data.

Tuning strategy:
- Inner 3-fold CV on training data
- Search over: max_depth, n_estimators, learning_rate, min_child_weight
- Optimizes for AUC

This also fixes the 25%-data catastrophic failure by:
- Searching for simpler models (fewer trees, shallower) at small data sizes
- Using min_child_weight to prevent overfitting on tiny samples
"""

import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def train_classical_xgb(fold_data, random_state=2026):
    """
    Train tuned XGBoost on a single fold.
    
    No SMOTE — XGBoost handles imbalance via scale_pos_weight.
    Hyperparameters selected via inner 3-fold CV.
    """
    X_train = fold_data["X_train"]
    X_val = fold_data["X_val"]
    y_train = fold_data["y_train"]
    y_val = fold_data["y_val"]

    # Compute class weight
    n_pos = sum(y_train == 1)
    n_neg = sum(y_train == 0)
    scale_pos_weight = n_neg / max(n_pos, 1)

    # Parameter grid — adapted for small dataset
    param_grid = {
        'max_depth': [2, 3, 4],
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.05, 0.1, 0.2],
        'min_child_weight': [1, 3, 5],
    }

    xgb_base = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        eval_metric='logloss',
        verbosity=0,
    )

    # Inner CV for hyperparameter selection
    grid = GridSearchCV(
        xgb_base, param_grid,
        cv=3, scoring='roc_auc',
        n_jobs=-1, refit=True
    )
    grid.fit(X_train, y_train)

    best_xgb = grid.best_estimator_
    print(f"    XGBoost best: depth={grid.best_params_['max_depth']}, "
          f"n_est={grid.best_params_['n_estimators']}, "
          f"lr={grid.best_params_['learning_rate']}, "
          f"mcw={grid.best_params_['min_child_weight']}")

    # Predict on validation
    val_probs = best_xgb.predict_proba(X_val)[:, 1]
    val_preds = best_xgb.predict(X_val)

    try:
        auc = roc_auc_score(y_val, val_probs)
    except ValueError:
        auc = 0.5
    
    acc = accuracy_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds, zero_division=0)

    print(f"    → AUC: {auc:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")

    return {
        "auc_roc": auc,
        "accuracy": acc,
        "f1_score": f1,
        "predictions": val_probs,
        "pred_binary": val_preds,
        "y_true": y_val,
    }
