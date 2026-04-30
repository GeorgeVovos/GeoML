"""
Classical Neural Network Classifier for SSc vs Healthy using PyTorch.

Architecture: Multi-layer perceptron (MLP) with:
- Input layer matching feature count
- Hidden layers with BatchNorm, ReLU, Dropout
- Binary output with sigmoid
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

DATA_DIR = Path("data/GSE76809/processed")
RESULTS_DIR = Path("results")


class ClassicalClassifier(nn.Module):
    """Multi-layer perceptron for binary classification."""

    def __init__(self, n_features: int, hidden_sizes=(128, 64, 32), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_size = n_features

        for h_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = h_size

        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


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
    hidden_sizes=(128, 64, 32),
    dropout: float = 0.3,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 32,
):
    """Train the classical neural network."""
    print("=" * 60)
    print("CLASSICAL NEURAL NETWORK (PyTorch MLP)")
    print("=" * 60)

    train_loader, test_loader, n_features = load_data(batch_size)
    print(f"Features: {n_features}, Hidden layers: {hidden_sizes}")
    print(f"Dropout: {dropout}, Epochs: {epochs}, LR: {lr}, Batch size: {batch_size}")
    print()

    model = ClassicalClassifier(n_features, hidden_sizes, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print()

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

        if (epoch + 1) % 10 == 0 or epoch == 0:
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
        "model": "Classical (PyTorch MLP)",
        "accuracy": accuracy_score(y_true_test, y_pred_test),
        "f1_score": f1_score(y_true_test, y_pred_test),
        "auc_roc": roc_auc_score(y_true_test, y_prob_test),
        "train_time_seconds": train_time,
        "hidden_sizes": list(hidden_sizes),
        "n_params": n_params,
        "epochs": epochs,
    }

    print(f"\n{'='*60}")
    print("FINAL RESULTS - CLASSICAL NEURAL NETWORK")
    print(f"{'='*60}")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  F1 Score:  {results['f1_score']:.4f}")
    print(f"  AUC-ROC:   {results['auc_roc']:.4f}")
    print(f"  Parameters: {n_params:,}")
    print(f"\n{classification_report(y_true_test, y_pred_test, target_names=['Healthy', 'SSc'])}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    import json
    with open(RESULTS_DIR / "classical_results.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), RESULTS_DIR / "classical_model.pt")
    np.save(RESULTS_DIR / "classical_history.npy", history)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classical NN for SSc classification")
    parser.add_argument("--hidden", nargs="+", type=int, default=[128, 64, 32])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    train_model(
        hidden_sizes=tuple(args.hidden),
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
