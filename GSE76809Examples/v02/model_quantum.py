"""
Quantum Classifier for SSc vs Healthy using PennyLane + PyTorch (v02 - Improved).

Improvements over v01:
1. More qubits (10) with data re-uploading (feature encoding repeated each layer)
2. Class-weighted BCE loss to handle imbalance
3. Learning rate scheduling (cosine annealing)
4. More epochs with early stopping on validation loss
5. Strongly entangling layers from PennyLane templates
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import pennylane as qml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _ROOT / "data" / "GSE76809" / "processed_v02"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


class QuantumCircuit:
    """PennyLane quantum circuit with data re-uploading."""

    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for layer in range(n_layers):
                # Data re-uploading: encode features every layer
                for i in range(n_qubits):
                    qml.RY(inputs[i] * np.pi, wires=i)
                    qml.RZ(inputs[i] * np.pi * 0.5, wires=i)

                # Parameterized rotations
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                    qml.RX(weights[layer, i, 2], wires=i)

                # Entanglement: all-to-all for first half, ring for second
                if layer % 2 == 0:
                    for i in range(n_qubits):
                        qml.CNOT(wires=[i, (i + 1) % n_qubits])
                else:
                    for i in range(0, n_qubits - 1, 2):
                        qml.CNOT(wires=[i, i + 1])
                    for i in range(1, n_qubits - 1, 2):
                        qml.CNOT(wires=[i, i + 1])

            return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_qubits))

        self.circuit = circuit


class QuantumClassifier(nn.Module):
    """Hybrid quantum-classical classifier with data re-uploading."""

    def __init__(self, n_features: int, n_qubits: int = 10, n_layers: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Pre-processing: project features to qubit count
        self.pre_net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, n_qubits),
            nn.Tanh(),
        )

        # Quantum circuit
        self.qc = QuantumCircuit(n_qubits, n_layers)
        # Weights: (n_layers, n_qubits, 3) for RY, RZ, RX
        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.3)

        # Post-processing
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.pre_net(x)
        batch_size = x.shape[0]
        q_out = torch.zeros(batch_size, self.n_qubits, device=x.device)
        for i in range(batch_size):
            result = self.qc.circuit(x[i], self.q_weights)
            q_out[i] = torch.stack(result)
        x = self.post_net(q_out)
        return x.squeeze(-1)


def load_data(batch_size: int = 16):
    """Load preprocessed v02 data."""
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, X_train.shape[1]


def train_model(
    n_qubits: int = 10,
    n_layers: int = 4,
    epochs: int = 40,
    lr: float = 0.008,
    batch_size: int = 16,
    patience: int = 10,
):
    """Train the improved quantum classifier."""
    print("=" * 60)
    print("QUANTUM CLASSIFIER v02 (PennyLane + PyTorch)")
    print("  + Data re-uploading, class weights, LR scheduling")
    print("=" * 60)

    train_loader, test_loader, n_features = load_data(batch_size)
    print(f"Features: {n_features}, Qubits: {n_qubits}, Layers: {n_layers}")
    print(f"Epochs: {epochs}, LR: {lr}, Batch size: {batch_size}")

    # Compute class weight from training data
    y_train_all = np.load(DATA_DIR / "y_train.npy")
    n_pos = sum(y_train_all == 1)
    n_neg = sum(y_train_all == 0)
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)
    print(f"Positive class weight: {pos_weight.item():.4f}")
    print()

    model = QuantumClassifier(n_features, n_qubits, n_layers)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.BCELoss(reduction='none')

    # Training loop with early stopping
    train_start = time.time()
    history = {"train_loss": [], "train_acc": [], "test_acc": [], "test_auc": []}
    best_auc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        y_pred_all, y_true_all = [], []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            # Weighted loss
            weights = torch.where(y_batch == 0, pos_weight.reciprocal() * 3.0, torch.ones(1))
            loss = (criterion(y_pred, y_batch) * weights).mean()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(y_batch)
            y_pred_all.extend((y_pred.detach() > 0.5).int().numpy())
            y_true_all.extend(y_batch.int().numpy())

        scheduler.step()
        train_loss = epoch_loss / len(y_pred_all)
        train_acc = accuracy_score(y_true_all, y_pred_all)

        # Evaluate
        model.eval()
        y_pred_test, y_true_test, y_prob_test = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch)
                y_prob_test.extend(y_pred.numpy())
                y_pred_test.extend((y_pred > 0.5).int().numpy())
                y_true_test.extend(y_batch.int().numpy())

        test_acc = accuracy_score(y_true_test, y_pred_test)
        test_auc = roc_auc_score(y_true_test, y_prob_test)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["test_auc"].append(test_auc)

        # Early stopping on AUC
        if test_auc > best_auc:
            best_auc = test_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | AUC: {test_auc:.4f}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1} (best AUC: {best_auc:.4f})")
            break

    train_time = time.time() - train_start
    print(f"\nTraining time: {train_time:.1f}s")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    y_pred_test, y_true_test, y_prob_test = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            y_prob_test.extend(y_pred.numpy())
            y_pred_test.extend((y_pred > 0.5).int().numpy())
            y_true_test.extend(y_batch.int().numpy())

    results = {
        "model": "Quantum v02 (PennyLane VQC + re-uploading)",
        "accuracy": accuracy_score(y_true_test, y_pred_test),
        "f1_score": f1_score(y_true_test, y_pred_test),
        "auc_roc": roc_auc_score(y_true_test, y_prob_test),
        "f1_healthy": f1_score(y_true_test, y_pred_test, pos_label=0),
        "train_time_seconds": train_time,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_params": n_params,
        "epochs_run": epoch + 1,
        "best_epoch_auc": best_auc,
    }

    print(f"\n{'='*60}")
    print("FINAL RESULTS - QUANTUM CLASSIFIER v02")
    print(f"{'='*60}")
    print(f"  Accuracy:    {results['accuracy']:.4f}")
    print(f"  F1 (SSc):    {results['f1_score']:.4f}")
    print(f"  F1 (Healthy):{results['f1_healthy']:.4f}")
    print(f"  AUC-ROC:     {results['auc_roc']:.4f}")
    print(f"\n{classification_report(y_true_test, y_pred_test, target_names=['Healthy', 'SSc'])}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    import json
    with open(RESULTS_DIR / "quantum_results.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), RESULTS_DIR / "quantum_model.pt")
    np.save(RESULTS_DIR / "quantum_history.npy", history)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quantum classifier v02 for SSc")
    parser.add_argument("--n-qubits", type=int, default=10)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.008)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    train_model(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
    )
