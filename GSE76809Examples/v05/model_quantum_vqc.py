"""
Quantum VQC classifier v05: Enhanced dressed circuit, 4 qubits / 16 features.

Same architectural family as v04 (multi-basis measurement + ZZ interactions +
amplitude encoding) but scaled to the v05 data slice (16 features -> 4 qubits).
This lets us check whether the v04 quantum-vs-classical conclusions hold up on
a different data slice with a smaller feature set.
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
DATA_DIR = _ROOT / "data" / "GSE76809" / "processed_v05"


def find_optimal_threshold(y_true, y_prob):
    """Find optimal threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


class EnhancedDressedQuantumCircuit(nn.Module):
    """
    Dressed quantum circuit v05:
      Input (16) -> Pre-Net -> Amplitude Encoding (4 qubits)
      -> ZZ Interaction Layer -> Variational layers (x6)
      -> Multi-basis Measurement (12 values) -> Post-Net -> Sigmoid
    """

    def __init__(self, n_features=16, n_qubits=4, n_layers=6):
        super().__init__()
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_measurements = 3 * n_qubits  # X + Y + Z per qubit

        self.pre_net = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.LayerNorm(n_features),
            nn.Tanh(),
        )

        self.q_params = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))
        self.zz_params = nn.Parameter(0.1 * torch.randn(n_qubits))

        self.post_net = nn.Sequential(
            nn.Linear(self.n_measurements, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(
            self._circuit, self.dev, interface="torch", diff_method="backprop"
        )

    def _circuit(self, inputs, weights, zz_params):
        n_q = self.n_qubits

        qml.AmplitudeEmbedding(inputs, wires=range(n_q), normalize=True)

        for i in range(n_q):
            j = (i + 1) % n_q
            qml.IsingZZ(zz_params[i], wires=[i, j])

        for layer in range(self.n_layers):
            for qubit in range(n_q):
                qml.RY(weights[layer, qubit, 0], wires=qubit)
                qml.RZ(weights[layer, qubit, 1], wires=qubit)
                qml.RX(weights[layer, qubit, 2], wires=qubit)

            if layer % 2 == 0:
                for qubit in range(n_q):
                    qml.CNOT(wires=[qubit, (qubit + 1) % n_q])
            else:
                for qubit in range(0, n_q - 1, 2):
                    qml.CNOT(wires=[qubit, qubit + 1])
                for qubit in range(1, n_q - 1, 2):
                    qml.CNOT(wires=[qubit, qubit + 1])
                qml.CNOT(wires=[0, n_q - 1])

        measurements = []
        for i in range(n_q):
            measurements.append(qml.expval(qml.PauliZ(i)))
        for i in range(n_q):
            measurements.append(qml.expval(qml.PauliX(i)))
        for i in range(n_q):
            measurements.append(qml.expval(qml.PauliY(i)))

        return measurements

    def forward(self, x):
        x = self.pre_net(x)
        x = x / (x.norm(dim=1, keepdim=True) + 1e-8)

        batch_out = []
        for i in range(x.shape[0]):
            q_out = self.qnode(x[i], self.q_params, self.zz_params)
            q_out = torch.stack(q_out)
            batch_out.append(q_out)

        q_features = torch.stack(batch_out).float()
        out = self.post_net(q_features)
        return torch.sigmoid(out.squeeze(-1))


def train_quantum_vqc(
    n_qubits=4,
    n_layers=6,
    epochs=60,
    lr=0.003,
    patience=15,
    fold_data=None,
):
    n_features = 2 ** n_qubits  # 16

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

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    model = EnhancedDressedQuantumCircuit(
        n_features=n_features, n_qubits=n_qubits, n_layers=n_layers
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.BCELoss(reduction='none')

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{'='*60}")
    print(f"QUANTUM VQC v05 (Dressed + Multi-Basis + ZZ, smaller circuit)")
    print(f"  Features: {n_features}, Qubits: {n_qubits}, Layers: {n_layers}")
    print(f"  Measurements: {model.n_measurements} (Z+X+Y)")
    print(f"  Epochs: {epochs}, LR: {lr}, Patience: {patience}")
    print(f"  Trainable parameters: {total_params:,}")
    print(f"{'='*60}")

    best_auc = 0.0
    best_state = None
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()

        batch_size = 32
        indices = torch.randperm(len(X_train_t))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(X_train_t), batch_size):
            batch_idx = indices[start:start + batch_size]
            xb = X_train_t[batch_idx]
            yb = y_train_t[batch_idx]

            optimizer.zero_grad()
            preds = model(xb)

            weights = torch.where(yb == 1, pos_weight, torch.ones(1))
            loss = (criterion(preds, yb) * weights).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

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
            avg_loss = epoch_loss / max(n_batches, 1)
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
    epochs_run = epoch

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_probs = model(X_test_t).numpy()

    opt_threshold = find_optimal_threshold(y_test, test_probs)
    test_preds = (test_probs >= opt_threshold).astype(int)

    acc = accuracy_score(y_test, test_preds)
    f1_ssc = f1_score(y_test, test_preds, pos_label=1)
    f1_healthy = f1_score(y_test, test_preds, pos_label=0)
    auc = roc_auc_score(y_test, test_probs)

    print(f"\n  FINAL (threshold={opt_threshold:.4f}):")
    print(f"  Accuracy: {acc:.4f} | F1(SSc): {f1_ssc:.4f} | "
          f"F1(Healthy): {f1_healthy:.4f} | AUC: {auc:.4f}")
    print(f"  Train time: {train_time:.1f}s | Epochs: {epochs_run}")
    print(classification_report(y_test, test_preds, target_names=["Healthy", "SSc"]))

    return {
        "accuracy": float(acc),
        "f1_score": float(f1_ssc),
        "f1_healthy": float(f1_healthy),
        "auc_roc": float(auc),
        "train_time_seconds": float(train_time),
        "n_params": total_params,
        "threshold": float(opt_threshold),
        "epochs_run": epochs_run,
        "test_probs": test_probs,
    }


if __name__ == "__main__":
    train_quantum_vqc()
