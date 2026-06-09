"""
Compare Quantum vs Classical models for SSc classification.

Runs both models, collects metrics, and prints a comparison table.
"""

import json
from pathlib import Path

from preprocess_gse76809 import preprocess
from model_quantum import train_model as train_quantum
from model_classical import train_model as train_classical

RESULTS_DIR = Path("results")


def run_comparison():
    """Run full pipeline: preprocess → train both models → compare."""
    print("=" * 70)
    print("  SSc vs HEALTHY CLASSIFICATION - QUANTUM vs CLASSICAL COMPARISON")
    print("=" * 70)
    print()

    # Step 1: Preprocess
    print("STEP 1: PREPROCESSING")
    print("-" * 70)
    preprocess(n_features=50, platform="GPL6480", selection_method="variance")
    print()

    # Step 2: Train quantum model
    print("\nSTEP 2: QUANTUM MODEL")
    print("-" * 70)
    q_results = train_quantum(n_qubits=8, n_layers=3, epochs=30, lr=0.005, batch_size=16)
    print()

    # Step 3: Train classical model
    print("\nSTEP 3: CLASSICAL MODEL")
    print("-" * 70)
    c_results = train_classical(hidden_sizes=(128, 64, 32), epochs=50, lr=0.001, batch_size=32)
    print()

    # Step 4: Comparison
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<20} {'Quantum VQC':<18} {'Classical MLP':<18} {'Winner':<12}")
    print("-" * 70)

    metrics = [
        ("Accuracy", "accuracy"),
        ("F1 Score", "f1_score"),
        ("AUC-ROC", "auc_roc"),
        ("Train Time (s)", "train_time_seconds"),
    ]

    for label, key in metrics:
        qv = q_results[key]
        cv = c_results[key]
        if key == "train_time_seconds":
            winner = "Quantum" if qv < cv else "Classical"
            print(f"{label:<20} {qv:<18.2f} {cv:<18.2f} {winner:<12}")
        else:
            winner = "Quantum" if qv > cv else "Classical"
            print(f"{label:<20} {qv:<18.4f} {cv:<18.4f} {winner:<12}")

    print("-" * 70)
    print()

    # Save combined comparison
    comparison = {"quantum": q_results, "classical": c_results}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Results saved to {RESULTS_DIR}/comparison.json")


if __name__ == "__main__":
    run_comparison()
