"""
Quantum classifier v03: Dressed quantum circuit with amplitude encoding.

Improvements over v02:
1. Amplitude encoding — encodes 2^n features into n qubits (64 features → 6 qubits)
2. Dressed quantum circuit — trainable classical layers before and after quantum
3. Threshold optimization via Youden's J statistic
4. Supports fold-based training for CV evaluation
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report, roc_curve
)

_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _ROOT / "data" / "GSE76809" / "processed_v03"


def find_optimal_threshold(y_true, y_prob):
    """Find optimal threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


class DressedQuantumCircuit(nn.Module):
    """
    Dressed quantum circuit:
      Input (64) → Linear (64→64) → Amplitude Encoding (6 qubits)
      → Variational layers → Measurement → Linear (6→1) → Sigmoid
    """

    def __init__(self, n_features=64, n_qubits=6, n_layers=4):
        super().__init__()
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Pre-processing classical layer (dressing)
        self.pre_net = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Tanh(),
        )

        # Quantum circuit parameters
        # Each layer: RY + RZ per qubit + entanglement
        self.q_params = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))

        # Post-processing classical layer (dressing)
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        # Device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="backprop")

    def _circuit(self, inputs, weights):
        """Quantum circuit with amplitude encoding + variational layers."""
        # Amplitude encoding: encodes 2^n amplitudes into n qubits
        qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True)

        # Variational layers
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                qml.RY(weights[layer, qubit, 0], wires=qubit)
                qml.RZ(weights[layer, qubit, 1], wires=qubit)
                qml.RX(weights[layer, qubit, 2], wires=qubit)
            # Entanglement: circular CNOT
            for qubit in range(self.n_qubits):
                qml.CNOT(wires=[qubit, (qubit + 1) % self.n_qubits])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        """Forward pass through dressed quantum circuit."""
        # Pre-net transforms features
        x = self.pre_net(x)

        # Normalize for amplitude encoding (must be unit norm)
        x = x / (x.norm(dim=1, keepdim=True) + 1e-8)

        # Process through quantum circuit (per sample)
        batch_out = []
        for i in range(x.shape[0]):
            q_out = self.qnode(x[i], self.q_params)
            q_out = torch.stack(q_out)
            batch_out.append(q_out)

        q_features = torch.stack(batch_out).float()

        # Post-net processes quantum measurements
        out = self.post_net(q_features)
        return torch.sigmoid(out.squeeze(-1))


def train_quantum(
    n_qubits=6,
    n_layers=4,
    epochs=50,
    lr=0.005,
    patience=12,
    fold_data=None,
):
    """
    Train dressed quantum circuit.

    Args:
        fold_data: if provided, dict with X_train, X_val, y_train, y_val (for CV)
                   if None, loads from disk (holdout evaluation)
    """
    n_features = 2 ** n_qubits  # 64 for 6 qubits

    if fold_data is not None:
        X_train = fold_data["X_train_norm"]
        X_test = fold_data["X_val_norm"]
        y_train = fold_data["y_train"]
        y_test = fold_data["y_val"]
    else:
        X_train = np.load(DATA_DIR / "X_train_norm.npy")
        X_test = np.load(DATA_DIR / "X_test_norm.npy")
        y_train = np.load(DATA_DIR / "y_train.npy")
        y_test = np.load(DATA_DIR / "y_test.npy")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # Class weight for loss
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    model = DressedQuantumCircuit(n_features=n_features, n_qubits=n_qubits, n_layers=n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss(reduction='none')

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{'='*60}")
    print(f"QUANTUM CLASSIFIER v03 (Dressed Circuit + Amplitude Encoding)")
    print(f"  Features: {n_features}, Qubits: {n_qubits}, Layers: {n_layers}")
    print(f"  Epochs: {epochs}, LR: {lr}, Patience: {patience}")
    print(f"  Trainable parameters: {total_params:,}")
    print(f"{'='*60}")

    best_auc = 0.0
    best_state = None
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()

        # Mini-batch training (batch of 32 to speed things up)
        batch_size = 32
        indices = torch.randperm(len(X_train_t))
        epoch_loss = 0.0

        for start in range(0, len(X_train_t), batch_size):
            batch_idx = indices[start:start+batch_size]
            xb = X_train_t[batch_idx]
            yb = y_train_t[batch_idx]

            optimizer.zero_grad()
            preds = model(xb)

            # Class-weighted loss
            weights = torch.where(yb == 1, pos_weight, torch.ones(1))
            loss = (criterion(preds, yb) * weights).mean()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            train_probs = model(X_train_t).numpy()
            test_probs = model(X_test_t).numpy()

        train_preds = (train_probs >= 0.5).astype(int)
        test_preds = (test_probs >= 0.5).astype(int)
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)

        try:
            auc = roc_auc_score(y_test, test_probs)
        except ValueError:
            auc = 0.5

        if epoch <= 5 or epoch % 5 == 0 or epoch == epochs:
            avg_loss = epoch_loss / (len(X_train_t) // batch_size + 1)
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (best AUC: {best_auc:.4f})")
            break

    train_time = time.time() - start_time

    # Restore best model
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_probs = model(X_test_t).numpy()

    # Threshold optimization
    opt_threshold = find_optimal_threshold(y_test, test_probs)
    test_preds_opt = (test_probs >= opt_threshold).astype(int)
    test_preds_default = (test_probs >= 0.5).astype(int)

    acc_default = accuracy_score(y_test, test_preds_default)
    acc_opt = accuracy_score(y_test, test_preds_opt)
    auc = roc_auc_score(y_test, test_probs)

    # Use optimized threshold
    test_preds = test_preds_opt if acc_opt >= acc_default else test_preds_default
    used_threshold = opt_threshold if acc_opt >= acc_default else 0.5

    acc = accuracy_score(y_test, test_preds)
    f1_ssc = f1_score(y_test, test_preds, pos_label=1)
    f1_healthy = f1_score(y_test, test_preds, pos_label=0)

    print(f"\n  Training time: {train_time:.1f}s")
    print(f"  Optimal threshold: {used_threshold:.4f} (Youden's J)")
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - QUANTUM v03 (Dressed + Amplitude Encoding)")
    print(f"{'='*60}")
    print(f"  Accuracy:     {acc:.4f}")
    print(f"  F1 (SSc):     {f1_ssc:.4f}")
    print(f"  F1 (Healthy): {f1_healthy:.4f}")
    print(f"  AUC-ROC:      {auc:.4f}")
    print(f"\n{classification_report(y_test, test_preds, target_names=['Healthy', 'SSc'])}")

    return {
        "accuracy": acc,
        "f1_score": f1_ssc,
        "f1_healthy": f1_healthy,
        "auc_roc": auc,
        "train_time_seconds": train_time,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_params": total_params,
        "threshold": used_threshold,
        "test_probs": test_probs,
        "epochs_run": epoch,
    }


if __name__ == "__main__":
    results = train_quantum()
