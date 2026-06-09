"""
Quantum Classifier for SSc vs Healthy using PennyLane + PyTorch.

Uses a variational quantum circuit (VQC) as the classification model:
- Angle encoding to embed features into qubit rotations
- Parameterized circuit layers (strongly entangling layers)
- Measurement expectation values fed to a final linear layer
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

DATA_DIR = Path("data/GSE76809/processed")
RESULTS_DIR = Path("results")


class QuantumCircuit:
    """Wrapper for PennyLane quantum circuit that handles batching manually."""

    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # Angle encoding
            for i in range(n_qubits):
                qml.RX(inputs[i] * np.pi, wires=i)
            # Variational layers
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_qubits))

        self.circuit = circuit


class QuantumClassifier(nn.Module):
    """Hybrid quantum-classical classifier."""

    def __init__(self, n_features: int, n_qubits: int = 8, n_layers: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # If features > qubits, use a linear projection first
        self.pre_net = None
        if n_features > n_qubits:
            self.pre_net = nn.Sequential(
                nn.Linear(n_features, n_qubits),
                nn.Tanh(),  # bound inputs to [-1, 1] for angle encoding
            )

        # Quantum circuit
        self.qc = QuantumCircuit(n_qubits, n_layers)
        # Trainable quantum weights
        self.q_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 2) * 0.1
        )

        # Post-processing classical layer
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if self.pre_net is not None:
            x = self.pre_net(x)

        # Process each sample through the quantum circuit
        batch_size = x.shape[0]
        q_out = torch.zeros(batch_size, self.n_qubits, device=x.device)
        for i in range(batch_size):
            result = self.qc.circuit(x[i], self.q_weights)
            q_out[i] = torch.stack(result)

        x = self.post_net(q_out)
        return x.squeeze(-1)


def load_data(batch_size: int = 32):
    """Load preprocessed data."""
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
    n_qubits: int = 8,
    n_layers: int = 3,
    epochs: int = 30,
    lr: float = 0.005,
    batch_size: int = 16,
):
    """Train the quantum classifier."""
    print("=" * 60)
    print("QUANTUM CLASSIFIER (PennyLane + PyTorch)")
    print("=" * 60)

    train_loader, test_loader, n_features = load_data(batch_size)
    print(f"Features: {n_features}, Qubits: {n_qubits}, Layers: {n_layers}")
    print(f"Epochs: {epochs}, LR: {lr}, Batch size: {batch_size}")
    print()

    model = QuantumClassifier(n_features, n_qubits, n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Training loop
    train_start = time.time()
    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        y_pred_all, y_true_all = [], []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(y_batch)
            y_pred_all.extend((y_pred.detach() > 0.5).int().numpy())
            y_true_all.extend(y_batch.int().numpy())

        train_loss = epoch_loss / len(y_pred_all)
        train_acc = accuracy_score(y_true_all, y_pred_all)

        # Evaluate on test set
        model.eval()
        y_pred_test, y_true_test, y_prob_test = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch)
                y_prob_test.extend(y_pred.numpy())
                y_pred_test.extend((y_pred > 0.5).int().numpy())
                y_true_test.extend(y_batch.int().numpy())

        test_acc = accuracy_score(y_true_test, y_pred_test)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    train_time = time.time() - train_start
    print(f"\nTraining time: {train_time:.1f}s")

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
        "model": "Quantum (PennyLane VQC)",
        "accuracy": accuracy_score(y_true_test, y_pred_test),
        "f1_score": f1_score(y_true_test, y_pred_test),
        "auc_roc": roc_auc_score(y_true_test, y_prob_test),
        "train_time_seconds": train_time,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "epochs": epochs,
    }

    print(f"\n{'='*60}")
    print("FINAL RESULTS - QUANTUM CLASSIFIER")
    print(f"{'='*60}")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  F1 Score:  {results['f1_score']:.4f}")
    print(f"  AUC-ROC:   {results['auc_roc']:.4f}")
    print(f"\n{classification_report(y_true_test, y_pred_test, target_names=['Healthy', 'SSc'])}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    import json
    with open(RESULTS_DIR / "quantum_results.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), RESULTS_DIR / "quantum_model.pt")
    np.save(RESULTS_DIR / "quantum_history.npy", history)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quantum classifier for SSc")
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    train_model(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
