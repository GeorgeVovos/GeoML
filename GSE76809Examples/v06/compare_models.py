"""
GSE76809 v06 — Fair Quantum vs Classical Comparison Pipeline.

Evaluation methodology improvements:
1. Per-fold SMOTE (no leakage)
2. 5-fold CV with fine-grained learning curves
3. Parameter-matched comparisons (VQC ~1,500 params ↔ MLP ~1,600 params)
4. Quantum kernel vs Classical RBF SVM on same PCA data
5. McNemar's test for statistical significance
6. Cohen's d effect sizes
7. Learning curves at 10%, 20%, 30%, 50%, 75%, 100% data fractions

Models compared:
- Quantum VQC (data reuploading, ~1,500 params)
- Quantum Kernel SVM (all-to-all ZZ, depth=3)
- Classical MLP (parameter-matched, ~1,600 params)
- Classical RBF SVM (same PCA data as quantum kernel)
- Classical XGBoost (CV-tuned hyperparameters)
"""

import json
import time
import numpy as np
from pathlib import Path
from scipy import stats

from preprocess_gse76809 import preprocess, apply_smote_to_fold, build_fold_features
from model_quantum_vqc import train_quantum_vqc
from model_quantum_kernel import train_quantum_kernel
from model_classical_mlp import train_classical_mlp
from model_classical_svm import train_classical_svm
from model_classical_xgb import train_classical_xgb


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def cohens_d(group1, group2):
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def mcnemars_test(y_true, preds_a, preds_b):
    """
    McNemar's test: tests whether two classifiers have the same error rate.
    Uses the contingency table of disagreements.

    Uses the exact binomial test when the total disagreement count is small
    (b+c < 25), where the chi-square approximation is unreliable.
    """
    # Correct/incorrect for each model
    correct_a = (preds_a == y_true).astype(int)
    correct_b = (preds_b == y_true).astype(int)

    # Contingency: cases where A is right & B wrong, and vice versa
    b_count = int(np.sum((correct_a == 1) & (correct_b == 0)))  # A right, B wrong
    c_count = int(np.sum((correct_a == 0) & (correct_b == 1)))  # A wrong, B right

    n = b_count + c_count
    if n == 0:
        return 1.0  # No disagreements

    if n < 25:
        # Exact two-sided binomial test on min(b, c) successes out of n trials
        # under H0: p = 0.5.
        try:
            res = stats.binomtest(min(b_count, c_count), n=n, p=0.5, alternative='two-sided')
            return float(res.pvalue)
        except AttributeError:  # older scipy
            return float(stats.binom_test(min(b_count, c_count), n=n, p=0.5))

    # Chi-square approximation with continuity correction
    statistic = (abs(b_count - c_count) - 1) ** 2 / n
    return float(1 - stats.chi2.cdf(statistic, df=1))


def run_cv(data, n_folds=5):
    """Run 5-fold CV for all 5 models with per-fold SMOTE."""
    folds = data["folds"]

    results = {
        "quantum_vqc": {"aucs": [], "accs": [], "f1s": []},
        "quantum_kernel": {"aucs": [], "accs": [], "f1s": []},
        "classical_mlp": {"aucs": [], "accs": [], "f1s": []},
        "classical_svm": {"aucs": [], "accs": [], "f1s": []},
        "classical_xgb": {"aucs": [], "accs": [], "f1s": []},
    }

    for fold_idx in range(n_folds):
        print(f"\n{'#'*70}")
        print(f"# FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'#'*70}\n")

        fold = folds[fold_idx]

        # 1. Quantum VQC (data reuploading, per-fold SMOTE)
        print(f"\n--- Fold {fold_idx+1}: Quantum VQC (Data Reuploading) ---")
        vqc_res = train_quantum_vqc(fold_data=fold)
        results["quantum_vqc"]["aucs"].append(vqc_res["auc_roc"])
        results["quantum_vqc"]["accs"].append(vqc_res["accuracy"])
        results["quantum_vqc"]["f1s"].append(vqc_res["f1_score"])

        # 2. Quantum Kernel SVM (PCA data, no SMOTE)
        print(f"\n--- Fold {fold_idx+1}: Quantum Kernel SVM ---")
        kernel_res = train_quantum_kernel(fold_data=fold)
        results["quantum_kernel"]["aucs"].append(kernel_res["auc_roc"])
        results["quantum_kernel"]["accs"].append(kernel_res["accuracy"])
        results["quantum_kernel"]["f1s"].append(kernel_res["f1_score"])

        # 3. Classical MLP (parameter-matched, per-fold SMOTE)
        print(f"\n--- Fold {fold_idx+1}: Classical MLP (Parameter-Matched) ---")
        mlp_res = train_classical_mlp(fold_data=fold)
        results["classical_mlp"]["aucs"].append(mlp_res["auc_roc"])
        results["classical_mlp"]["accs"].append(mlp_res["accuracy"])
        results["classical_mlp"]["f1s"].append(mlp_res["f1_score"])

        # 4. Classical RBF SVM (same PCA data as quantum kernel)
        print(f"\n--- Fold {fold_idx+1}: Classical RBF SVM ---")
        svm_res = train_classical_svm(fold_data=fold)
        results["classical_svm"]["aucs"].append(svm_res["auc_roc"])
        results["classical_svm"]["accs"].append(svm_res["accuracy"])
        results["classical_svm"]["f1s"].append(svm_res["f1_score"])

        # 5. Classical XGBoost (tuned, no SMOTE)
        print(f"\n--- Fold {fold_idx+1}: Classical XGBoost (Tuned) ---")
        xgb_res = train_classical_xgb(fold_data=fold)
        results["classical_xgb"]["aucs"].append(xgb_res["auc_roc"])
        results["classical_xgb"]["accs"].append(xgb_res["accuracy"])
        results["classical_xgb"]["f1s"].append(xgb_res["f1_score"])

    return results


def run_holdout(data):
    """Run holdout evaluation on all models. Returns predictions for McNemar's."""
    print(f"\n{'='*70}")
    print("HOLDOUT EVALUATION (80/20 split)")
    print(f"{'='*70}\n")

    holdout_fold = {
        "X_train": data["X_train"],
        "X_val": data["X_test"],
        "y_train": data["y_train"],
        "y_val": data["y_test"],
        "X_train_norm": data["X_train_norm"],
        "X_val_norm": data["X_test_norm"],
        "X_train_pca": data["X_train_pca"],
        "X_val_pca": data["X_test_pca"],
    }

    results = {}
    timings = {}

    # Quantum VQC (holdout — no SMOTE, matches metadata claim)
    print("\n--- Holdout: Quantum VQC ---")
    t0 = time.time()
    vqc_res = train_quantum_vqc(fold_data=holdout_fold, use_smote=False)
    timings["quantum_vqc"] = time.time() - t0
    results["quantum_vqc"] = vqc_res

    # Quantum Kernel
    print("\n--- Holdout: Quantum Kernel SVM ---")
    t0 = time.time()
    kernel_res = train_quantum_kernel(fold_data=holdout_fold)
    timings["quantum_kernel"] = time.time() - t0
    results["quantum_kernel"] = kernel_res

    # Classical MLP (holdout — no SMOTE, matches metadata claim)
    print("\n--- Holdout: Classical MLP ---")
    t0 = time.time()
    mlp_res = train_classical_mlp(fold_data=holdout_fold, use_smote=False)
    timings["classical_mlp"] = time.time() - t0
    results["classical_mlp"] = mlp_res

    # Classical RBF SVM
    print("\n--- Holdout: Classical RBF SVM ---")
    t0 = time.time()
    svm_res = train_classical_svm(fold_data=holdout_fold)
    timings["classical_svm"] = time.time() - t0
    results["classical_svm"] = svm_res

    # Classical XGBoost
    print("\n--- Holdout: Classical XGBoost ---")
    t0 = time.time()
    xgb_res = train_classical_xgb(fold_data=holdout_fold)
    timings["classical_xgb"] = time.time() - t0
    results["classical_xgb"] = xgb_res

    return results, timings


def run_learning_curves(data, fractions=None, n_repeats=3):
    """
    Fine-grained learning curves at small data fractions.

    For each subsample, ANOVA selection / StandardScaler / PCA are REFIT on
    that subsample only, and SMOTE is applied per subsample for the deep
    models. This way the small-data points genuinely reflect what each model
    learns from only that much data.
    """
    if fractions is None:
        fractions = [0.10, 0.20, 0.30, 0.50, 0.75, 1.0]

    print(f"\n{'='*70}")
    print(f"LEARNING CURVES — Data fractions: {fractions}")
    print(f"{'='*70}\n")

    X_train_raw = data["X_train_raw"]      # variance-filtered, QT'd holdout-train
    X_test_raw = data["X_test_raw"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    n_features = data["n_features"]
    pca_components = data["pca_components"]
    random_state = data["random_state"]

    results = {model: {str(f): [] for f in fractions}
               for model in ["quantum_vqc", "classical_mlp", "classical_xgb",
                             "classical_svm", "quantum_kernel"]}

    rng = np.random.RandomState(random_state)

    for frac in fractions:
        n_samples = int(len(X_train_raw) * frac)
        print(f"\n--- Data fraction: {frac*100:.0f}% ({n_samples} samples) ---")

        for rep in range(n_repeats):
            if frac < 1.0:
                idx = rng.choice(len(X_train_raw), n_samples, replace=False)
            else:
                idx = np.arange(len(X_train_raw))

            # Refit ANOVA / scaler / PCA on the subsample only
            fold_raw = build_fold_features(
                X_train_raw[idx], y_train[idx], X_test_raw,
                n_features=n_features, pca_components=pca_components,
                random_state=random_state,
            )
            fold_raw["y_val"] = y_test

            # Per-subsample SMOTE for VQC / MLP
            X_aug, y_aug = apply_smote_to_fold(
                fold_raw["X_train"], fold_raw["y_train"], random_state=random_state
            )
            X_aug_norm = X_aug / (np.linalg.norm(X_aug, axis=1, keepdims=True) + 1e-10)
            fold_smote = dict(fold_raw)
            fold_smote["X_train"] = X_aug
            fold_smote["y_train"] = y_aug
            fold_smote["X_train_norm"] = X_aug_norm

            print(f"\n  Repeat {rep+1}/{n_repeats} at {frac*100:.0f}%:")

            # Quantum VQC (SMOTE'd subsample)
            print(f"    [VQC]", end=" ")
            vqc_r = train_quantum_vqc(fold_data=fold_smote, epochs=60, patience=12, use_smote=False)
            results["quantum_vqc"][str(frac)].append(vqc_r["auc_roc"])

            # Classical MLP (SMOTE'd subsample)
            print(f"    [MLP]", end=" ")
            mlp_r = train_classical_mlp(fold_data=fold_smote, epochs=60, patience=12, use_smote=False)
            results["classical_mlp"][str(frac)].append(mlp_r["auc_roc"])

            # Classical XGBoost (raw subsample — handles imbalance natively)
            print(f"    [XGB]", end=" ")
            xgb_r = train_classical_xgb(fold_data=fold_raw)
            results["classical_xgb"][str(frac)].append(xgb_r["auc_roc"])

            # Classical RBF SVM (raw subsample)
            print(f"    [SVM]", end=" ")
            svm_r = train_classical_svm(fold_data=fold_raw)
            results["classical_svm"][str(frac)].append(svm_r["auc_roc"])

            # Quantum Kernel (only at 30%+ to keep runtime manageable)
            if frac >= 0.30:
                print(f"    [QKernel]", end=" ")
                qk_r = train_quantum_kernel(fold_data=fold_raw)
                results["quantum_kernel"][str(frac)].append(qk_r["auc_roc"])
            else:
                results["quantum_kernel"][str(frac)].append(np.nan)

    return results


def run_statistical_tests(cv_results):
    """
    Run statistical tests on CV results.
    
    - Paired t-tests (traditional)
    - Cohen's d effect sizes
    - Interpretation guidelines
    """
    print(f"\n{'='*70}")
    print("STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'='*70}\n")

    models = list(cv_results.keys())
    comparisons = [
        ("quantum_vqc", "classical_mlp", "VQC vs Matched-MLP (same params)"),
        ("quantum_vqc", "classical_xgb", "VQC vs Tuned-XGBoost"),
        ("quantum_vqc", "classical_svm", "VQC vs RBF-SVM"),
        ("quantum_kernel", "classical_svm", "Q-Kernel vs RBF-SVM (same data)"),
        ("quantum_kernel", "classical_xgb", "Q-Kernel vs Tuned-XGBoost"),
        ("quantum_vqc", "quantum_kernel", "VQC vs Q-Kernel"),
    ]

    stat_results = []
    for model_a, model_b, label in comparisons:
        aucs_a = np.array(cv_results[model_a]["aucs"])
        aucs_b = np.array(cv_results[model_b]["aucs"])

        # Paired t-test
        t_stat, p_val = stats.ttest_rel(aucs_a, aucs_b)
        
        # Cohen's d
        d = cohens_d(aucs_a, aucs_b)
        
        # Effect size interpretation
        if abs(d) < 0.2:
            effect = "negligible"
        elif abs(d) < 0.5:
            effect = "small"
        elif abs(d) < 0.8:
            effect = "medium"
        else:
            effect = "large"

        significant = p_val < 0.05
        print(f"  {label}:")
        print(f"    t={t_stat:.3f}, p={p_val:.4f} {'*** SIGNIFICANT' if significant else ''}")
        print(f"    Cohen's d={d:.3f} ({effect} effect)")
        print(f"    Mean diff: {np.mean(aucs_a) - np.mean(aucs_b):.4f}")
        print()

        stat_results.append({
            "comparison": label,
            "model_a": model_a,
            "model_b": model_b,
            "t_statistic": t_stat,
            "p_value": p_val,
            "cohens_d": d,
            "effect_size": effect,
            "significant": significant,
            "mean_a": float(np.mean(aucs_a)),
            "mean_b": float(np.mean(aucs_b)),
        })

    return stat_results


def run_mcnemar_tests(holdout_results):
    """Run McNemar's tests on holdout predictions."""
    print(f"\n{'='*70}")
    print("McNEMAR'S TESTS (Holdout predictions)")
    print(f"{'='*70}\n")

    y_true = holdout_results["quantum_vqc"]["y_true"]
    mcnemar_results = []

    comparisons = [
        ("quantum_vqc", "classical_mlp", "VQC vs Matched-MLP"),
        ("quantum_vqc", "classical_xgb", "VQC vs Tuned-XGBoost"),
        ("quantum_kernel", "classical_svm", "Q-Kernel vs RBF-SVM"),
    ]

    for model_a, model_b, label in comparisons:
        preds_a = holdout_results[model_a]["pred_binary"]
        preds_b = holdout_results[model_b]["pred_binary"]

        p_val = mcnemars_test(y_true, preds_a, preds_b)
        significant = p_val < 0.05

        print(f"  {label}: p={p_val:.4f} {'*** SIGNIFICANT' if significant else ''}")
        mcnemar_results.append({
            "comparison": label,
            "p_value": p_val,
            "significant": significant,
        })

    return mcnemar_results


def main():
    """Run the full v06 comparison pipeline."""
    start_time = time.time()

    print("=" * 70)
    print("GSE76809 v06 — Fair Quantum vs Classical Comparison")
    print("   16 ANOVA features | seed=2026 | Per-fold SMOTE | Data Reuploading VQC")
    print("   5 models: VQC, Q-Kernel, Matched-MLP, RBF-SVM, Tuned-XGBoost")
    print("=" * 70)

    # 1. Preprocess
    print("\n[1/5] PREPROCESSING")
    data = preprocess()

    # 2. Holdout evaluation
    print("\n[2/5] HOLDOUT EVALUATION")
    holdout_results, timings = run_holdout(data)

    # 3. Cross-validation
    print("\n[3/5] 5-FOLD CROSS-VALIDATION")
    cv_results = run_cv(data)

    # 4. Learning curves
    print("\n[4/5] LEARNING CURVES")
    lc_results = run_learning_curves(data)

    # 5. Statistical tests
    print("\n[5/5] STATISTICAL ANALYSIS")
    stat_results = run_statistical_tests(cv_results)
    mcnemar_results = run_mcnemar_tests(holdout_results)

    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}\n")

    print("5-Fold Cross-Validation AUC:")
    print(f"{'Model':<25} {'Mean AUC':<12} {'Std':<10} {'Min':<8} {'Max':<8}")
    print("-" * 63)
    for model, res in sorted(cv_results.items(),
                             key=lambda x: np.mean(x[1]["aucs"]), reverse=True):
        aucs = res["aucs"]
        name = model.replace("_", " ").title()
        print(f"{name:<25} {np.mean(aucs):.4f}      ±{np.std(aucs):.4f}    "
              f"{min(aucs):.4f}  {max(aucs):.4f}")

    print(f"\nHoldout Results:")
    print(f"{'Model':<25} {'AUC':<10} {'Acc':<10} {'F1':<10} {'Time (s)':<10}")
    print("-" * 65)
    for model, res in sorted(holdout_results.items(),
                             key=lambda x: x[1]["auc_roc"], reverse=True):
        name = model.replace("_", " ").title()
        print(f"{name:<25} {res['auc_roc']:.4f}    {res['accuracy']:.4f}    "
              f"{res['f1_score']:.4f}    {timings[model]:.1f}")

    print(f"\nLearning Curves (mean AUC per fraction):")
    fracs = sorted(lc_results["quantum_vqc"].keys(), key=float)
    header = f"{'Model':<20}" + "".join(f"{float(f)*100:>8.0f}%" for f in fracs)
    print(header)
    print("-" * len(header))
    for model in ["quantum_vqc", "classical_mlp", "classical_xgb",
                  "classical_svm", "quantum_kernel"]:
        name = model.replace("_", " ").title()[:18]
        row = f"{name:<20}"
        for f in fracs:
            vals = [v for v in lc_results[model][f] if not np.isnan(v)]
            if vals:
                row += f"{np.mean(vals):>9.4f}"
            else:
                row += f"{'N/A':>9}"
        print(row)

    print(f"\nTotal runtime: {total_time/60:.1f} minutes")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    all_results = {
        "metadata": {
            "version": "v06",
            "dataset": "GSE76809",
            "n_features": 16,
            "feature_selection": "ANOVA F-test",
            "seed": 2026,
            "smote": "per-fold (correct methodology)",
            "total_runtime_minutes": total_time / 60,
        },
        "cross_validation": {
            model: {
                "aucs": res["aucs"],
                "mean_auc": float(np.mean(res["aucs"])),
                "std_auc": float(np.std(res["aucs"])),
                "accs": res["accs"],
                "f1s": res["f1s"],
            }
            for model, res in cv_results.items()
        },
        "holdout": {
            model: {
                "auc": res["auc_roc"],
                "accuracy": res["accuracy"],
                "f1": res["f1_score"],
                "time_seconds": timings[model],
            }
            for model, res in holdout_results.items()
        },
        "learning_curves": {
            model: {f: vals for f, vals in frac_results.items()}
            for model, frac_results in lc_results.items()
        },
        "statistical_tests": {
            "paired_t_tests": stat_results,
            "mcnemar_tests": mcnemar_results,
        },
    }

    results_path = results_dir / "v06_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
