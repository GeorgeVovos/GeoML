"""
Compare all models v04: Full pipeline with 5-fold CV, learning curves, and statistical tests.

This script orchestrates:
1. Preprocessing (64 MI features, SMOTE, PCA for kernel)
2. Quantum VQC (enhanced dressed circuit with multi-basis measurement)
3. Quantum Kernel SVM (ZZ feature map + SVM)
4. Classical MLP (residual network + mixup)
5. Classical XGBoost (gradient boosting)
6. 5-fold CV for robust comparison
7. Learning curves (quantum advantage at small data)
8. Paired t-test for statistical significance
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_gse76809 import preprocess, apply_smote_to_fold, build_fold_features
from model_quantum_vqc import train_quantum_vqc
from model_quantum_kernel import train_quantum_kernel
from model_classical_mlp import train_classical_mlp
from model_classical_xgb import train_classical_xgb


def _smote_fold(fold, random_state=42):
    """Return a copy of `fold` with SMOTE applied to the training rows only.

    Validation rows are passed through unchanged so they remain real samples.
    """
    X_aug, y_aug = apply_smote_to_fold(fold["X_train"], fold["y_train"], random_state=random_state)
    X_aug_norm = X_aug / (np.linalg.norm(X_aug, axis=1, keepdims=True) + 1e-10)
    new_fold = dict(fold)
    new_fold["X_train"] = X_aug
    new_fold["y_train"] = y_aug
    new_fold["X_train_norm"] = X_aug_norm
    return new_fold

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_cv(data, n_folds=5):
    """
    Run 5-fold cross-validation for all 4 models.

    Returns dict with per-fold results for each model.
    """
    folds = data["folds"]

    results = {
        "quantum_vqc": {"aucs": [], "accs": [], "f1s": []},
        "quantum_kernel": {"aucs": [], "accs": [], "f1s": []},
        "classical_mlp": {"aucs": [], "accs": [], "f1s": []},
        "classical_xgb": {"aucs": [], "accs": [], "f1s": []},
    }

    for fold_idx in range(n_folds):
        print(f"\n{'#'*70}")
        print(f"# FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'#'*70}\n")

        fold_raw = folds[fold_idx]
        # Per-fold SMOTE for models that need balanced training (VQC, MLP)
        fold_smote = _smote_fold(fold_raw, random_state=42 + fold_idx)

        # 1. Quantum VQC (uses per-fold SMOTE'd data)
        print(f"\n--- Fold {fold_idx+1}: Quantum VQC ---")
        vqc_res = train_quantum_vqc(fold_data=fold_smote)
        results["quantum_vqc"]["aucs"].append(vqc_res["auc_roc"])
        results["quantum_vqc"]["accs"].append(vqc_res["accuracy"])
        results["quantum_vqc"]["f1s"].append(vqc_res["f1_score"])

        # 2. Quantum Kernel (no SMOTE — SVM handles imbalance via class_weight)
        print(f"\n--- Fold {fold_idx+1}: Quantum Kernel SVM ---")
        kernel_res = train_quantum_kernel(fold_data=fold_raw)
        results["quantum_kernel"]["aucs"].append(kernel_res["auc_roc"])
        results["quantum_kernel"]["accs"].append(kernel_res["accuracy"])
        results["quantum_kernel"]["f1s"].append(kernel_res["f1_score"])

        # 3. Classical MLP (uses per-fold SMOTE'd data)
        print(f"\n--- Fold {fold_idx+1}: Classical MLP ---")
        mlp_res = train_classical_mlp(fold_data=fold_smote)
        results["classical_mlp"]["aucs"].append(mlp_res["auc_roc"])
        results["classical_mlp"]["accs"].append(mlp_res["accuracy"])
        results["classical_mlp"]["f1s"].append(mlp_res["f1_score"])

        # 4. Classical XGBoost (no SMOTE — scale_pos_weight handles imbalance)
        print(f"\n--- Fold {fold_idx+1}: Classical XGBoost ---")
        xgb_res = train_classical_xgb(fold_data=fold_raw)
        results["classical_xgb"]["aucs"].append(xgb_res["auc_roc"])
        results["classical_xgb"]["accs"].append(xgb_res["accuracy"])
        results["classical_xgb"]["f1s"].append(xgb_res["f1_score"])

    return results


def run_learning_curves(data, fractions=(0.25, 0.50, 0.75, 1.0)):
    """
    Train models at different data fractions to show quantum advantage
    at small sample sizes.

    Subsamples real (pre-SMOTE) holdout-train rows, then for each subsample
    REFITS MI selection / StandardScaler / PCA and applies SMOTE per subsample
    so the small-data points genuinely reflect a model trained on only that
    much data.
    """
    print(f"\n{'#'*70}")
    print(f"# LEARNING CURVES")
    print(f"{'#'*70}\n")

    X_train_raw = data["X_train_raw"]      # variance-filtered, QT'd holdout-train
    X_test_raw = data["X_test_raw"]
    y_train_full = data["y_train_pre_smote"]
    y_test = data["y_test"]
    n_features = data["n_features"]

    lc_results = {model: [] for model in ["quantum_vqc", "quantum_kernel", "classical_mlp", "classical_xgb"]}

    rng = np.random.RandomState(42)

    for frac in fractions:
        print(f"\n{'='*60}")
        print(f"  Data fraction: {frac*100:.0f}%")
        print(f"{'='*60}")

        n = int(len(X_train_raw) * frac)
        if frac < 1.0:
            idx = rng.choice(len(X_train_raw), n, replace=False)
        else:
            idx = np.arange(len(X_train_raw))

        # Refit feature engineering using ONLY the subsample
        fold_raw = build_fold_features(
            X_train_raw[idx], y_train_full[idx], X_test_raw,
            n_features=n_features, n_pca=8, random_state=42,
        )
        fold_raw["y_val"] = y_test

        # Per-subsample SMOTE for VQC / MLP
        X_aug, y_aug = apply_smote_to_fold(fold_raw["X_train"], fold_raw["y_train"], random_state=42)
        X_aug_norm = X_aug / (np.linalg.norm(X_aug, axis=1, keepdims=True) + 1e-10)
        fold_smote = dict(fold_raw)
        fold_smote["X_train"] = X_aug
        fold_smote["y_train"] = y_aug
        fold_smote["X_train_norm"] = X_aug_norm

        print(f"\n  [LC {frac*100:.0f}%] Quantum VQC (n={n} raw → {len(X_aug)} SMOTE'd)...")
        vqc = train_quantum_vqc(epochs=40, patience=10, fold_data=fold_smote)
        lc_results["quantum_vqc"].append({"frac": frac, "n_train": n, "auc": vqc["auc_roc"]})

        print(f"\n  [LC {frac*100:.0f}%] Quantum Kernel (n={n})...")
        kernel = train_quantum_kernel(fold_data=fold_raw)
        lc_results["quantum_kernel"].append({"frac": frac, "n_train": n, "auc": kernel["auc_roc"]})

        print(f"\n  [LC {frac*100:.0f}%] Classical MLP (n={n} raw → {len(X_aug)} SMOTE'd)...")
        mlp = train_classical_mlp(epochs=60, patience=10, fold_data=fold_smote)
        lc_results["classical_mlp"].append({"frac": frac, "n_train": n, "auc": mlp["auc_roc"]})

        print(f"\n  [LC {frac*100:.0f}%] Classical XGBoost (n={n})...")
        xgb = train_classical_xgb(fold_data=fold_raw)
        lc_results["classical_xgb"].append({"frac": frac, "n_train": n, "auc": xgb["auc_roc"]})

    return lc_results


def statistical_tests(cv_results):
    """Perform paired t-tests between quantum and classical models."""
    print(f"\n{'#'*70}")
    print(f"# STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'#'*70}\n")

    tests = {}

    # VQC vs MLP
    t_stat, p_val = stats.ttest_rel(
        cv_results["quantum_vqc"]["aucs"],
        cv_results["classical_mlp"]["aucs"]
    )
    tests["vqc_vs_mlp"] = {"t_stat": float(t_stat), "p_value": float(p_val)}
    print(f"  VQC vs MLP: t={t_stat:.4f}, p={p_val:.4f} "
          f"{'*SIGNIFICANT*' if p_val < 0.05 else '(not significant)'}")

    # VQC vs XGBoost
    t_stat, p_val = stats.ttest_rel(
        cv_results["quantum_vqc"]["aucs"],
        cv_results["classical_xgb"]["aucs"]
    )
    tests["vqc_vs_xgb"] = {"t_stat": float(t_stat), "p_value": float(p_val)}
    print(f"  VQC vs XGBoost: t={t_stat:.4f}, p={p_val:.4f} "
          f"{'*SIGNIFICANT*' if p_val < 0.05 else '(not significant)'}")

    # Kernel vs MLP
    t_stat, p_val = stats.ttest_rel(
        cv_results["quantum_kernel"]["aucs"],
        cv_results["classical_mlp"]["aucs"]
    )
    tests["kernel_vs_mlp"] = {"t_stat": float(t_stat), "p_value": float(p_val)}
    print(f"  Kernel vs MLP: t={t_stat:.4f}, p={p_val:.4f} "
          f"{'*SIGNIFICANT*' if p_val < 0.05 else '(not significant)'}")

    # Kernel vs XGBoost
    t_stat, p_val = stats.ttest_rel(
        cv_results["quantum_kernel"]["aucs"],
        cv_results["classical_xgb"]["aucs"]
    )
    tests["kernel_vs_xgb"] = {"t_stat": float(t_stat), "p_value": float(p_val)}
    print(f"  Kernel vs XGBoost: t={t_stat:.4f}, p={p_val:.4f} "
          f"{'*SIGNIFICANT*' if p_val < 0.05 else '(not significant)'}")

    # Best quantum vs best classical
    best_q = np.maximum(
        cv_results["quantum_vqc"]["aucs"],
        cv_results["quantum_kernel"]["aucs"]
    )
    best_c = np.maximum(
        cv_results["classical_mlp"]["aucs"],
        cv_results["classical_xgb"]["aucs"]
    )
    t_stat, p_val = stats.ttest_rel(best_q, best_c)
    tests["best_quantum_vs_best_classical"] = {"t_stat": float(t_stat), "p_value": float(p_val)}
    print(f"\n  Best Quantum vs Best Classical: t={t_stat:.4f}, p={p_val:.4f} "
          f"{'*SIGNIFICANT*' if p_val < 0.05 else '(not significant)'}")

    return tests


def main():
    """Run the full v04 comparison pipeline."""
    total_start = time.time()

    print("=" * 70)
    print("   v04: QUANTUM ADVANTAGE IN SSc CLASSIFICATION")
    print("   Enhanced VQC + Quantum Kernel vs MLP + XGBoost")
    print("=" * 70)

    # Step 1: Preprocessing
    print("\n\n" + "=" * 70)
    print("STEP 1: PREPROCESSING")
    print("=" * 70)
    data = preprocess(n_features=64, n_folds=5)

    # Step 2: Holdout evaluation (single train/test split)
    print("\n\n" + "=" * 70)
    print("STEP 2: HOLDOUT EVALUATION")
    print("=" * 70)

    print("\n--- Quantum VQC ---")
    vqc_results = train_quantum_vqc()
    print("\n--- Quantum Kernel SVM ---")
    kernel_results = train_quantum_kernel()
    print("\n--- Classical MLP ---")
    mlp_results = train_classical_mlp()
    print("\n--- Classical XGBoost ---")
    xgb_results = train_classical_xgb()

    # Step 3: 5-fold Cross-Validation
    print("\n\n" + "=" * 70)
    print("STEP 3: 5-FOLD CROSS-VALIDATION")
    print("=" * 70)
    cv_results = run_cv(data, n_folds=5)

    # Step 4: Learning Curves
    print("\n\n" + "=" * 70)
    print("STEP 4: LEARNING CURVES")
    print("=" * 70)
    lc_results = run_learning_curves(data)

    # Step 5: Statistical Tests
    print("\n\n" + "=" * 70)
    print("STEP 5: STATISTICAL TESTS")
    print("=" * 70)
    stat_tests = statistical_tests(cv_results)

    # Summary
    total_time = time.time() - total_start

    print("\n\n" + "=" * 70)
    print("   FINAL SUMMARY")
    print("=" * 70)

    print("\n  HOLDOUT RESULTS:")
    print(f"  {'Model':<25} {'Accuracy':>10} {'AUC':>10} {'F1(SSc)':>10} {'F1(Heal.)':>10}")
    print(f"  {'-'*65}")
    for name, res in [
        ("Quantum VQC", vqc_results),
        ("Quantum Kernel", kernel_results),
        ("Classical MLP", mlp_results),
        ("Classical XGBoost", xgb_results),
    ]:
        print(f"  {name:<25} {res['accuracy']:>10.4f} {res['auc_roc']:>10.4f} "
              f"{res['f1_score']:>10.4f} {res['f1_healthy']:>10.4f}")

    print(f"\n  CROSS-VALIDATION AUC (5-fold):")
    print(f"  {'Model':<25} {'Mean AUC':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*65}")
    for name, key in [
        ("Quantum VQC", "quantum_vqc"),
        ("Quantum Kernel", "quantum_kernel"),
        ("Classical MLP", "classical_mlp"),
        ("Classical XGBoost", "classical_xgb"),
    ]:
        aucs = cv_results[key]["aucs"]
        print(f"  {name:<25} {np.mean(aucs):>10.4f} {np.std(aucs):>10.4f} "
              f"{np.min(aucs):>10.4f} {np.max(aucs):>10.4f}")

    print(f"\n  LEARNING CURVES (AUC at different data fractions):")
    print(f"  {'Model':<25} {'25%':>8} {'50%':>8} {'75%':>8} {'100%':>8}")
    print(f"  {'-'*60}")
    for name, key in [
        ("Quantum VQC", "quantum_vqc"),
        ("Quantum Kernel", "quantum_kernel"),
        ("Classical MLP", "classical_mlp"),
        ("Classical XGBoost", "classical_xgb"),
    ]:
        aucs = [r["auc"] for r in lc_results[key]]
        print(f"  {name:<25} {aucs[0]:>8.4f} {aucs[1]:>8.4f} {aucs[2]:>8.4f} {aucs[3]:>8.4f}")

    print(f"\n  Total pipeline time: {total_time/60:.1f} minutes")

    # Save all results
    all_results = {
        "holdout": {
            "quantum_vqc": {k: v for k, v in vqc_results.items() if k != "test_probs"},
            "quantum_kernel": {k: v for k, v in kernel_results.items() if k != "test_probs"},
            "classical_mlp": {k: v for k, v in mlp_results.items() if k != "test_probs"},
            "classical_xgb": {k: v for k, v in xgb_results.items() if k != "test_probs"},
        },
        "cross_validation": {
            key: {
                "aucs": vals["aucs"],
                "accs": vals["accs"],
                "f1s": vals["f1s"],
                "mean_auc": float(np.mean(vals["aucs"])),
                "std_auc": float(np.std(vals["aucs"])),
            }
            for key, vals in cv_results.items()
        },
        "learning_curves": lc_results,
        "statistical_tests": stat_tests,
        "metadata": {
            "n_features": 64,
            "n_folds": 5,
            "total_time_seconds": float(total_time),
        },
    }

    results_path = RESULTS_DIR / "v04_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
