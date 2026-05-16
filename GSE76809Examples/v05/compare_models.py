"""
Compare all models v05: same hypothesis as v04 on a different data slice.

Pipeline:
1. Preprocess GSE76809 (16 ANOVA features, NO SMOTE, seed=2024, PCA->6)
2. Holdout evaluation for all 4 models
3. 5-fold cross-validation
4. Learning curves at 25/50/75/100% data fractions
5. Paired t-tests (quantum vs classical)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_gse76809 import preprocess
from model_quantum_vqc import train_quantum_vqc
from model_quantum_kernel import train_quantum_kernel
from model_classical_mlp import train_classical_mlp
from model_classical_xgb import train_classical_xgb

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_cv(data, n_folds=5):
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

        fold = folds[fold_idx]

        print(f"\n--- Fold {fold_idx+1}: Quantum VQC ---")
        vqc_res = train_quantum_vqc(fold_data=fold)
        results["quantum_vqc"]["aucs"].append(vqc_res["auc_roc"])
        results["quantum_vqc"]["accs"].append(vqc_res["accuracy"])
        results["quantum_vqc"]["f1s"].append(vqc_res["f1_score"])

        print(f"\n--- Fold {fold_idx+1}: Quantum Kernel SVM ---")
        kernel_res = train_quantum_kernel(fold_data=fold)
        results["quantum_kernel"]["aucs"].append(kernel_res["auc_roc"])
        results["quantum_kernel"]["accs"].append(kernel_res["accuracy"])
        results["quantum_kernel"]["f1s"].append(kernel_res["f1_score"])

        print(f"\n--- Fold {fold_idx+1}: Classical MLP ---")
        mlp_res = train_classical_mlp(fold_data=fold)
        results["classical_mlp"]["aucs"].append(mlp_res["auc_roc"])
        results["classical_mlp"]["accs"].append(mlp_res["accuracy"])
        results["classical_mlp"]["f1s"].append(mlp_res["f1_score"])

        print(f"\n--- Fold {fold_idx+1}: Classical XGBoost ---")
        xgb_res = train_classical_xgb(fold_data=fold)
        results["classical_xgb"]["aucs"].append(xgb_res["auc_roc"])
        results["classical_xgb"]["accs"].append(xgb_res["accuracy"])
        results["classical_xgb"]["f1s"].append(xgb_res["f1_score"])

    return results


def run_learning_curves(data, fractions=(0.25, 0.50, 0.75, 1.0)):
    print(f"\n{'#'*70}")
    print(f"# LEARNING CURVES")
    print(f"{'#'*70}\n")

    X_train_norm = data["X_train_norm"]
    X_train = data["X_train"]
    X_train_pca = data["X_train_pca"]
    X_test_norm = data["X_test_norm"]
    X_test = data["X_test"]
    X_test_pca = data["X_test_pca"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    lc_results = {model: [] for model in ["quantum_vqc", "quantum_kernel", "classical_mlp", "classical_xgb"]}

    for frac in fractions:
        print(f"\n{'='*60}")
        print(f"  Data fraction: {frac*100:.0f}%")
        print(f"{'='*60}")

        n = int(len(X_train) * frac)

        if frac < 1.0:
            rng = np.random.RandomState(2024)
            idx = rng.choice(len(X_train), n, replace=False)
        else:
            idx = np.arange(len(X_train))

        fold = {
            "X_train_norm": X_train_norm[idx],
            "X_val_norm": X_test_norm,
            "X_train": X_train[idx],
            "X_val": X_test,
            "X_train_pca": X_train_pca[idx],
            "X_val_pca": X_test_pca,
            "y_train": y_train[idx],
            "y_val": y_test,
        }

        print(f"\n  [LC {frac*100:.0f}%] Quantum VQC (n={n})...")
        vqc = train_quantum_vqc(epochs=40, patience=10, fold_data=fold)
        lc_results["quantum_vqc"].append({"frac": frac, "n_train": n, "auc": vqc["auc_roc"]})

        print(f"\n  [LC {frac*100:.0f}%] Quantum Kernel (n={n})...")
        kernel = train_quantum_kernel(fold_data=fold)
        lc_results["quantum_kernel"].append({"frac": frac, "n_train": n, "auc": kernel["auc_roc"]})

        print(f"\n  [LC {frac*100:.0f}%] Classical MLP (n={n})...")
        mlp = train_classical_mlp(epochs=60, patience=10, fold_data=fold)
        lc_results["classical_mlp"].append({"frac": frac, "n_train": n, "auc": mlp["auc_roc"]})

        print(f"\n  [LC {frac*100:.0f}%] Classical XGBoost (n={n})...")
        xgb = train_classical_xgb(fold_data=fold)
        lc_results["classical_xgb"].append({"frac": frac, "n_train": n, "auc": xgb["auc_roc"]})

    return lc_results


def statistical_tests(cv_results):
    print(f"\n{'#'*70}")
    print(f"# STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'#'*70}\n")

    tests = {}

    pairs = [
        ("vqc_vs_mlp", "quantum_vqc", "classical_mlp"),
        ("vqc_vs_xgb", "quantum_vqc", "classical_xgb"),
        ("kernel_vs_mlp", "quantum_kernel", "classical_mlp"),
        ("kernel_vs_xgb", "quantum_kernel", "classical_xgb"),
    ]
    for key, a, b in pairs:
        t_stat, p_val = stats.ttest_rel(cv_results[a]["aucs"], cv_results[b]["aucs"])
        tests[key] = {"t_stat": float(t_stat), "p_value": float(p_val)}
        print(f"  {key}: t={t_stat:.4f}, p={p_val:.4f} "
              f"{'*SIGNIFICANT*' if p_val < 0.05 else '(not significant)'}")

    best_q = np.maximum(cv_results["quantum_vqc"]["aucs"], cv_results["quantum_kernel"]["aucs"])
    best_c = np.maximum(cv_results["classical_mlp"]["aucs"], cv_results["classical_xgb"]["aucs"])
    t_stat, p_val = stats.ttest_rel(best_q, best_c)
    tests["best_quantum_vs_best_classical"] = {"t_stat": float(t_stat), "p_value": float(p_val)}
    print(f"\n  Best Quantum vs Best Classical: t={t_stat:.4f}, p={p_val:.4f} "
          f"{'*SIGNIFICANT*' if p_val < 0.05 else '(not significant)'}")

    return tests


def main():
    total_start = time.time()

    print("=" * 70)
    print("   v05: QUANTUM vs CLASSICAL on a DIFFERENT GSE76809 DATA SLICE")
    print("   16 ANOVA features | seed=2024 | NO SMOTE | 4-qubit VQC")
    print("=" * 70)

    print("\n\n" + "=" * 70)
    print("STEP 1: PREPROCESSING")
    print("=" * 70)
    data = preprocess(n_features=16, n_folds=5, n_pca=6)

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

    print("\n\n" + "=" * 70)
    print("STEP 3: 5-FOLD CROSS-VALIDATION")
    print("=" * 70)
    cv_results = run_cv(data, n_folds=5)

    print("\n\n" + "=" * 70)
    print("STEP 4: LEARNING CURVES")
    print("=" * 70)
    lc_results = run_learning_curves(data)

    print("\n\n" + "=" * 70)
    print("STEP 5: STATISTICAL TESTS")
    print("=" * 70)
    stat_tests = statistical_tests(cv_results)

    total_time = time.time() - total_start

    print("\n\n" + "=" * 70)
    print("   FINAL SUMMARY (v05)")
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
            "n_features": 16,
            "n_qubits_vqc": 4,
            "n_qubits_kernel": 6,
            "n_folds": 5,
            "random_state": 2024,
            "feature_selection": "ANOVA F-test",
            "imbalance_handling": "class_weighting_only",
            "total_time_seconds": float(total_time),
        },
    }

    results_path = RESULTS_DIR / "v05_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
