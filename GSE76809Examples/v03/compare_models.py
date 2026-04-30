"""
Compare quantum, classical, and ensemble models for GSE76809 (v03).

Pipeline:
1. Preprocess with MI + BorderlineSMOTE + 3-fold CV splits
2. Train dressed quantum circuit (amplitude encoding)
3. Train residual MLP (mixup + label smoothing)
4. Ensemble (stacking + weighted average)
5. Cross-validation for robust metrics
6. Compare all models vs v01/v02 baselines
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess_gse76809 import preprocess
from model_quantum import train_quantum
from model_classical import train_classical
from model_ensemble import train_ensemble

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_cv(data, n_folds=3):
    """Run 3-fold CV for both models and return mean metrics."""
    print(f"\n{'='*70}")
    print(f"  CROSS-VALIDATION ({n_folds}-fold)")
    print(f"{'='*70}")

    q_metrics = {"accuracy": [], "f1_score": [], "f1_healthy": [], "auc_roc": []}
    c_metrics = {"accuracy": [], "f1_score": [], "f1_healthy": [], "auc_roc": []}

    for fold_idx, fold in enumerate(data["folds"]):
        print(f"\n--- Fold {fold_idx+1}/{n_folds} ---")

        print(f"\n  [Quantum]")
        q_res = train_quantum(n_qubits=6, n_layers=4, epochs=30, lr=0.005, patience=10, fold_data=fold)
        for k in q_metrics:
            q_metrics[k].append(q_res[k])

        print(f"\n  [Classical]")
        c_res = train_classical(epochs=80, lr=0.001, patience=12, fold_data=fold)
        for k in c_metrics:
            c_metrics[k].append(c_res[k])

    print(f"\n{'='*70}")
    print(f"  CV RESULTS (mean ± std over {n_folds} folds)")
    print(f"{'='*70}")
    print(f"{'Metric':<15} {'Quantum':>20} {'Classical':>20}")
    print("-" * 60)
    for k in q_metrics:
        q_mean, q_std = np.mean(q_metrics[k]), np.std(q_metrics[k])
        c_mean, c_std = np.mean(c_metrics[k]), np.std(c_metrics[k])
        print(f"  {k:<13} {q_mean:.4f} ± {q_std:.4f}    {c_mean:.4f} ± {c_std:.4f}")

    return {
        "quantum_cv": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": [float(x) for x in v]} for k, v in q_metrics.items()},
        "classical_cv": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": [float(x) for x in v]} for k, v in c_metrics.items()},
    }


def main():
    print("=" * 70)
    print("  SSc vs HEALTHY CLASSIFICATION v03 — ADVANCED")
    print("  Amplitude Encoding + Dressed Circuit + Ensemble + CV")
    print("=" * 70)

    # Step 1: Preprocessing
    print(f"\nSTEP 1: PREPROCESSING (MI + BorderlineSMOTE + 3-fold splits)")
    print("-" * 70)
    data = preprocess(n_features=64, n_folds=3)
    print()

    # Step 2: Quantum model (holdout)
    print(f"\nSTEP 2: QUANTUM MODEL (dressed circuit, amplitude encoding, 6 qubits)")
    print("-" * 70)
    q_results = train_quantum(n_qubits=6, n_layers=4, epochs=50, lr=0.005, patience=12)
    print()

    # Step 3: Classical model (holdout)
    print(f"\nSTEP 3: CLASSICAL MODEL (residual MLP + mixup + threshold opt)")
    print("-" * 70)
    c_results = train_classical(epochs=100, lr=0.001, patience=15)
    print()

    # Step 4: Ensemble
    print(f"\nSTEP 4: ENSEMBLE (stacking + weighted average)")
    print("-" * 70)

    # Get train probs for stacking (re-run on train set for stacking input)
    from model_quantum import DressedQuantumCircuit, DATA_DIR as Q_DATA_DIR
    from model_classical import ResidualMLP
    import torch

    X_train = np.load(Q_DATA_DIR / "X_train_norm.npy")
    X_test = np.load(Q_DATA_DIR / "X_test_norm.npy")
    y_train = np.load(Q_DATA_DIR / "y_train.npy")
    y_test = np.load(Q_DATA_DIR / "y_test.npy")

    # Use the test probabilities from individual models
    q_probs_test = q_results["test_probs"]
    c_probs_test = c_results["test_probs"]

    # For stacking training, use cross-validated predictions on train set
    # Simple approach: use the train probabilities (slight optimistic bias but practical)
    q_probs_train = q_probs_test  # placeholder - will use test probs for simple stacking
    c_probs_train = c_probs_test

    # Better: generate train predictions via internal CV
    # For simplicity, we'll do a quick 2-fold on train to get OOF predictions
    from sklearn.model_selection import StratifiedKFold
    X_train_std = np.load(Q_DATA_DIR / "X_train.npy")
    q_oof = np.zeros(len(y_train))
    c_oof = np.zeros(len(y_train))

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X_train_std, y_train):
        fold = {
            "X_train": X_train_std[train_idx],
            "X_val": X_train_std[val_idx],
            "X_train_norm": X_train[train_idx],
            "X_val_norm": X_train[val_idx],
            "y_train": y_train[train_idx],
            "y_val": y_train[val_idx],
        }
        # Quick quantum on fold for OOF
        q_fold = train_quantum(n_qubits=6, n_layers=3, epochs=20, lr=0.005, patience=8, fold_data=fold)
        q_oof[val_idx] = q_fold["test_probs"].flatten()

        # Quick classical on fold for OOF
        c_fold = train_classical(epochs=50, lr=0.001, patience=10, fold_data=fold)
        c_oof[val_idx] = c_fold["test_probs"].flatten()

    e_results = train_ensemble(q_oof, c_oof, y_train,
                               q_probs_test.flatten(), c_probs_test.flatten(), y_test)
    print()

    # Step 5: Cross-validation
    print(f"\nSTEP 5: 3-FOLD CROSS-VALIDATION")
    print("-" * 70)
    cv_results = run_cv(data, n_folds=3)
    print()

    # Step 6: Summary
    print("\n" + "=" * 70)
    print("  v03 COMPARISON SUMMARY (Holdout Test Set)")
    print("=" * 70)
    print(f"{'Metric':<20} {'Quantum v03':<20} {'Classical v03':<20} {'Ensemble v03':<20} {'Winner':<12}")
    print("-" * 92)

    metrics_compare = [
        ("Accuracy", q_results["accuracy"], c_results["accuracy"], e_results["accuracy"]),
        ("F1 Score (SSc)", q_results["f1_score"], c_results["f1_score"], e_results["f1_score"]),
        ("F1 (Healthy)", q_results["f1_healthy"], c_results["f1_healthy"], e_results["f1_healthy"]),
        ("AUC-ROC", q_results["auc_roc"], c_results["auc_roc"], e_results["auc_roc"]),
    ]

    for name, q, c, e in metrics_compare:
        best = max(q, c, e)
        winner = "Quantum" if q == best else ("Classical" if c == best else "Ensemble")
        print(f"{name:<20} {q:<20.4f} {c:<20.4f} {e:<20.4f} {winner:<12}")

    print(f"{'Train Time (s)':<20} {q_results['train_time_seconds']:<20.1f} {c_results['train_time_seconds']:<20.1f} {'—':<20} {'Classical':<12}")

    # Baselines
    print(f"\n  BASELINES:")
    print(f"  {'Metric':<20} {'Quantum v01':<15} {'Classical v01':<15} {'Quantum v02':<15} {'Classical v02':<15}")
    print(f"  {'Accuracy':<20} {'0.8333':<15} {'0.8519':<15} {'0.8704':<15} {'0.7778':<15}")
    print(f"  {'AUC-ROC':<20} {'0.5707':<15} {'0.8370':<15} {'0.8804':<15} {'0.9402':<15}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "quantum_v03": {
            "model": "Quantum v03 (Dressed Circuit + Amplitude Encoding)",
            "accuracy": q_results["accuracy"],
            "f1_score": q_results["f1_score"],
            "f1_healthy": q_results["f1_healthy"],
            "auc_roc": q_results["auc_roc"],
            "train_time_seconds": q_results["train_time_seconds"],
            "n_qubits": 6,
            "n_layers": 4,
            "n_params": q_results["n_params"],
            "threshold": q_results["threshold"],
            "epochs_run": q_results["epochs_run"],
        },
        "classical_v03": {
            "model": "Classical v03 (Residual MLP + Mixup + Threshold Opt)",
            "accuracy": c_results["accuracy"],
            "f1_score": c_results["f1_score"],
            "f1_healthy": c_results["f1_healthy"],
            "auc_roc": c_results["auc_roc"],
            "train_time_seconds": c_results["train_time_seconds"],
            "n_params": c_results["n_params"],
            "threshold": c_results["threshold"],
            "epochs_run": c_results["epochs_run"],
        },
        "ensemble_v03": {
            "model": f"Ensemble v03 ({e_results['method']})",
            "accuracy": e_results["accuracy"],
            "f1_score": e_results["f1_score"],
            "f1_healthy": e_results["f1_healthy"],
            "auc_roc": e_results["auc_roc"],
            "method": e_results["method"],
            "quantum_weight": e_results["quantum_weight"],
            "stacker_coefs": e_results["stacker_coefs"],
        },
        "cross_validation": cv_results,
        "v01_baseline": {
            "quantum": {"accuracy": 0.8333, "f1_score": 0.9072, "auc_roc": 0.5707, "train_time_seconds": 355.1},
            "classical": {"accuracy": 0.8519, "f1_score": 0.9167, "auc_roc": 0.837, "train_time_seconds": 1.2},
        },
        "v02_baseline": {
            "quantum": {"accuracy": 0.8704, "f1_score": 0.9213, "auc_roc": 0.8804, "train_time_seconds": 638.3},
            "classical": {"accuracy": 0.7778, "f1_score": 0.8571, "auc_roc": 0.9402, "train_time_seconds": 1.5},
        },
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to {RESULTS_DIR}/comparison.json")


if __name__ == "__main__":
    main()
