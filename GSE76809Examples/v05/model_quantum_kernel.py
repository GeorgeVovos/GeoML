"""
Quantum Kernel SVM classifier v05: ZZ feature map with 6 qubits.

Same approach as v04 (fixed ZZ feature map + precomputed kernel SVM) but
operates on a 6-component PCA projection of the v05 data slice.
"""

import time
from pathlib import Path

import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report, roc_curve
)

_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _ROOT / "data" / "GSE76809" / "processed_v05"


def find_optimal_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


class QuantumKernel:
    """Quantum kernel using the ZZ feature map (see v04 docs)."""

    def __init__(self, n_qubits=6, depth=2):
        self.n_qubits = n_qubits
        self.depth = depth
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self._build_circuit()

    def _build_circuit(self):
        n_q = self.n_qubits
        depth = self.depth

        @qml.qnode(self.dev, interface="numpy")
        def kernel_circuit(x1, x2):
            for d in range(depth):
                for i in range(n_q):
                    qml.Hadamard(wires=i)
                for i in range(n_q):
                    qml.RZ(x1[i], wires=i)
                for i in range(n_q):
                    j = (i + 1) % n_q
                    qml.CNOT(wires=[i, j])
                    qml.RZ(x1[i] * x1[j], wires=j)
                    qml.CNOT(wires=[i, j])

            for d in range(depth - 1, -1, -1):
                for i in range(n_q - 1, -1, -1):
                    j = (i + 1) % n_q
                    qml.CNOT(wires=[i, j])
                    qml.RZ(-x2[i] * x2[j], wires=j)
                    qml.CNOT(wires=[i, j])
                for i in range(n_q - 1, -1, -1):
                    qml.RZ(-x2[i], wires=i)
                for i in range(n_q - 1, -1, -1):
                    qml.Hadamard(wires=i)

            return qml.probs(wires=range(n_q))

        self._circuit = kernel_circuit

    def evaluate(self, x1, x2):
        probs = self._circuit(x1, x2)
        return probs[0]

    def compute_kernel_matrix(self, X1, X2=None, verbose=True):
        n1 = len(X1)
        symmetric = X2 is None
        if symmetric:
            X2 = X1
        n2 = len(X2)

        K = np.zeros((n1, n2))
        total = n1 * n2 if not symmetric else n1 * (n1 + 1) // 2
        count = 0
        start = time.time()

        if symmetric:
            for i in range(n1):
                K[i, i] = 1.0
                for j in range(i + 1, n2):
                    K[i, j] = self.evaluate(X1[i], X2[j])
                    K[j, i] = K[i, j]
                    count += 1
                    if verbose and count % 500 == 0:
                        elapsed = time.time() - start
                        rate = count / elapsed
                        remaining = (total - count) / rate
                        print(f"    Kernel: {count}/{total} ({count/total*100:.1f}%) "
                              f"| ETA: {remaining:.0f}s")
        else:
            for i in range(n1):
                for j in range(n2):
                    K[i, j] = self.evaluate(X1[i], X2[j])
                    count += 1
                    if verbose and count % 500 == 0:
                        elapsed = time.time() - start
                        rate = count / elapsed
                        remaining = (total - count) / rate
                        print(f"    Kernel: {count}/{total} ({count/total*100:.1f}%) "
                              f"| ETA: {remaining:.0f}s")

        elapsed = time.time() - start
        if verbose:
            print(f"    Kernel computation done: {count} evaluations in {elapsed:.1f}s "
                  f"({count/elapsed:.0f} eval/s)")
        return K


def train_quantum_kernel(fold_data=None):
    if fold_data is not None:
        X_train_pca = fold_data["X_train_pca"]
        X_test_pca = fold_data["X_val_pca"]
        y_train = fold_data["y_train"]
        y_test = fold_data["y_val"]
    else:
        X_train_pca = np.load(DATA_DIR / "X_train_pca.npy")
        X_test_pca = np.load(DATA_DIR / "X_test_pca.npy")
        y_train = np.load(DATA_DIR / "y_train.npy")
        y_test = np.load(DATA_DIR / "y_test.npy")

    n_qubits = X_train_pca.shape[1]  # 6

    print(f"{'='*60}")
    print(f"QUANTUM KERNEL SVM v05 (ZZ Feature Map, {n_qubits} qubits)")
    print(f"  Train: {len(X_train_pca)}, Test: {len(X_test_pca)}")
    print(f"  Feature range: [{X_train_pca.min():.4f}, {X_train_pca.max():.4f}]")
    print(f"{'='*60}")

    start_time = time.time()

    kernel = QuantumKernel(n_qubits=n_qubits, depth=2)

    print(f"\n  Computing training kernel matrix ({len(X_train_pca)}x{len(X_train_pca)})...")
    K_train = kernel.compute_kernel_matrix(X_train_pca)

    print(f"\n  Computing test kernel matrix ({len(X_test_pca)}x{len(X_train_pca)})...")
    K_test = kernel.compute_kernel_matrix(X_test_pca, X_train_pca)

    K_train += 1e-6 * np.eye(len(K_train))

    print(f"\n  Training SVM with precomputed quantum kernel...")
    svm = SVC(
        kernel="precomputed",
        C=10.0,
        class_weight="balanced",
        probability=True,
        random_state=2024,
    )
    svm.fit(K_train, y_train)

    test_probs = svm.predict_proba(K_test)[:, 1]
    test_preds = svm.predict(K_test)

    train_time = time.time() - start_time

    acc = accuracy_score(y_test, test_preds)
    f1_ssc = f1_score(y_test, test_preds, pos_label=1)
    f1_healthy = f1_score(y_test, test_preds, pos_label=0)
    try:
        auc = roc_auc_score(y_test, test_probs)
    except ValueError:
        auc = 0.5

    print(f"\n  FINAL:")
    print(f"  Accuracy: {acc:.4f} | F1(SSc): {f1_ssc:.4f} | "
          f"F1(Healthy): {f1_healthy:.4f} | AUC: {auc:.4f}")
    print(f"  Train time: {train_time:.1f}s")
    print(classification_report(y_test, test_preds, target_names=["Healthy", "SSc"]))

    return {
        "accuracy": float(acc),
        "f1_score": float(f1_ssc),
        "f1_healthy": float(f1_healthy),
        "auc_roc": float(auc),
        "train_time_seconds": float(train_time),
        "n_params": 0,
        "test_probs": test_probs,
    }


if __name__ == "__main__":
    train_quantum_kernel()
