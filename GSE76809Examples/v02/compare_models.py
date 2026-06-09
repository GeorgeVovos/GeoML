"""
Compare Quantum vs Classical models for SSc classification (v02 - Improved).

Includes v01 baseline results for comparison.
"""

import json
import sys
from pathlib import Path

# Add v02 directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from preprocess_gse76809 import preprocess
from model_quantum import train_model as train_quantum
from model_classical import train_model as train_classical

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_comparison():
    """Run full improved pipeline: preprocess → train both → compare."""
    print("=" * 70)
    print("  SSc vs HEALTHY CLASSIFICATION v02 — IMPROVED")
    print("  Quantum vs Classical with MI features, SMOTE, class weights")
    print("=" * 70)
    print()

    # Step 1: Preprocess
    print("STEP 1: PREPROCESSING (Mutual Info + SMOTE)")
    print("-" * 70)
    preprocess(n_features=100, platform="GPL6480", use_smote=True)
    print()

    # Step 2: Quantum model
    print("\nSTEP 2: QUANTUM MODEL (data re-uploading, 8 qubits, 3 layers)")
    print("-" * 70)
    q_results = train_quantum(n_qubits=8, n_layers=3, epochs=40, lr=0.008, batch_size=16, patience=10)
    print()

    # Step 3: Classical model
    print("\nSTEP 3: CLASSICAL MODEL (residual MLP, GELU, class weights)")
    print("-" * 70)
    c_results = train_classical(hidden_sizes=(256, 128, 64, 32), epochs=80, lr=0.001, batch_size=32, patience=15)
    print()

    # Step 4: Comparison
    print("\n" + "=" * 70)
    print("  v02 COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<20} {'Quantum VQC v02':<20} {'Classical MLP v02':<20} {'Winner':<12}")
    print("-" * 70)

    metrics = [
        ("Accuracy", "accuracy"),
        ("F1 Score (SSc)", "f1_score"),
        ("F1 (Healthy)", "f1_healthy"),
        ("AUC-ROC", "auc_roc"),
        ("Train Time (s)", "train_time_seconds"),
    ]

    for label, key in metrics:
        qv = q_results[key]
        cv = c_results[key]
        if key == "train_time_seconds":
            winner = "Quantum" if qv < cv else "Classical"
            print(f"{label:<20} {qv:<20.2f} {cv:<20.2f} {winner:<12}")
        else:
            winner = "Quantum" if qv > cv else "Classical" if cv > qv else "Tie"
            print(f"{label:<20} {qv:<20.4f} {cv:<20.4f} {winner:<12}")

    print("-" * 70)

    # v01 baseline for reference
    print("\n  v01 BASELINE (for reference):")
    print(f"  {'Metric':<20} {'Quantum v01':<18} {'Classical v01':<18}")
    print(f"  {'Accuracy':<20} {'0.8333':<18} {'0.8519':<18}")
    print(f"  {'F1 Score':<20} {'0.9072':<18} {'0.9167':<18}")
    print(f"  {'AUC-ROC':<20} {'0.5707':<18} {'0.8370':<18}")
    print()

    # Delta improvement
    print("  IMPROVEMENT over v01:")
    print(f"  {'Quantum AUC':<20} {q_results['auc_roc'] - 0.5707:+.4f}")
    print(f"  {'Classical AUC':<20} {c_results['auc_roc'] - 0.8370:+.4f}")
    print()

    # Save comparison
    comparison = {
        "quantum_v02": q_results,
        "classical_v02": c_results,
        "v01_baseline": {
            "quantum": {"accuracy": 0.8333, "f1_score": 0.9072, "auc_roc": 0.5707, "train_time_seconds": 355.1},
            "classical": {"accuracy": 0.8519, "f1_score": 0.9167, "auc_roc": 0.8370, "train_time_seconds": 1.2},
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Results saved to {RESULTS_DIR}/comparison.json")


if __name__ == "__main__":
    run_comparison()
