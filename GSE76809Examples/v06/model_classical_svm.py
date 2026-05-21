"""
Classical RBF SVM for v06.

This is the FAIR classical comparison to the quantum kernel SVM.
Both operate on the same PCA-projected data (6 components).
Both use SVM with class_weight='balanced'.

The only difference: RBF kernel k(x,y) = exp(-γ||x-y||²) vs
quantum kernel k(x,y) = |⟨0|U†(y)U(x)|0⟩|².

If quantum kernel beats RBF on the SAME data representation,
it provides evidence that the quantum Hilbert space captures
structure that classical kernels miss.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def train_classical_svm(fold_data, C_values=None):
    """
    Train RBF SVM on the same PCA-projected data as the quantum kernel.
    
    Uses grid search on C and gamma for fair tuning.
    """
    X_train_pca = fold_data["X_train_pca"]
    X_val_pca = fold_data["X_val_pca"]
    y_train = fold_data["y_train"]
    y_val = fold_data["y_val"]

    if C_values is None:
        C_values = [0.1, 1.0, 10.0, 100.0]

    # Grid search for best C and gamma (inner validation via stratified 3-fold)
    param_grid = {
        'C': C_values,
        'gamma': ['scale', 'auto', 0.1, 1.0],
    }

    svm_base = SVC(
        kernel='rbf',
        class_weight='balanced',
        probability=True,
        random_state=2026,
    )

    # Use inner CV for hyperparameter selection
    grid = GridSearchCV(
        svm_base, param_grid,
        cv=3, scoring='roc_auc',
        n_jobs=-1, refit=True
    )
    grid.fit(X_train_pca, y_train)

    best_svm = grid.best_estimator_
    print(f"    RBF SVM best params: C={grid.best_params_['C']}, "
          f"gamma={grid.best_params_['gamma']}")

    # Predict on validation
    val_probs = best_svm.predict_proba(X_val_pca)[:, 1]
    val_preds = best_svm.predict(X_val_pca)

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
