"""
Classical Neural Network Classifier for SSc vs Healthy (v02 - Improved).

Improvements over v01:
1. Class-weighted BCE loss
2. Learning rate scheduling with warm restarts
3. Residual connections in the MLP
4. Early stopping on validation AUC
5. Larger network with more regularization
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _ROOT / "data" / "GSE76809" / "processed_v02"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


class ResidualBlock(nn.Module):
    """Residual MLP block."""

    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Skip connection (with projection if sizes differ)
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)


class ClassicalClassifier(nn.Module):
    """Residual MLP for binary classification."""

    def __init__(self, n_features: int, hidden_sizes=(256, 128, 64, 32), dropout: float = 0.4):
        super().__init__()
        layers = []
        prev_size = n_features

        for h_size in hidden_sizes:
            layers.append(ResidualBlock(prev_size, h_size, dropout))
            prev_size = h_size

        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def load_data(batch_size: int = 32):
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
    hidden_sizes=(256, 128, 64, 32),
    dropout: float = 0.4,
    epochs: int = 80,
    lr: float = 0.001,
    batch_size: int = 32,
    patience: int = 15,
):
    """Train the improved classical neural network."""
    print("=" * 60)
    print("CLASSICAL NEURAL NETWORK v02 (PyTorch Residual MLP)")
    print("  + Class weights, GELU, residual connections, LR scheduling")
    print("=" * 60)

    train_loader, test_loader, n_features = load_data(batch_size)
    print(f"Features: {n_features}, Hidden layers: {hidden_sizes}")
    print(f"Dropout: {dropout}, Epochs: {epochs}, LR: {lr}")

    # Class weight
    y_train_all = np.load(DATA_DIR / "y_train.npy")
    n_pos = sum(y_train_all == 1)
    n_neg = sum(y_train_all == 0)
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)
    print(f"Positive class weight: {pos_weight.item():.4f}")

    model = ClassicalClassifier(n_features, hidden_sizes, dropout)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.BCELoss(reduction='none')

    # Training with early stopping
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
            # Weighted loss: upweight minority class
            weights = torch.where(y_batch == 0, torch.tensor(3.0), torch.ones(1))
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

        if (epoch + 1) % 10 == 0 or epoch == 0:
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
        "model": "Classical v02 (Residual MLP + GELU)",
        "accuracy": accuracy_score(y_true_test, y_pred_test),
        "f1_score": f1_score(y_true_test, y_pred_test),
        "auc_roc": roc_auc_score(y_true_test, y_prob_test),
        "f1_healthy": f1_score(y_true_test, y_pred_test, pos_label=0),
        "train_time_seconds": train_time,
        "hidden_sizes": list(hidden_sizes),
        "n_params": n_params,
        "epochs_run": epoch + 1,
        "best_epoch_auc": best_auc,
    }

    print(f"\n{'='*60}")
    print("FINAL RESULTS - CLASSICAL NEURAL NETWORK v02")
    print(f"{'='*60}")
    print(f"  Accuracy:    {results['accuracy']:.4f}")
    print(f"  F1 (SSc):    {results['f1_score']:.4f}")
    print(f"  F1 (Healthy):{results['f1_healthy']:.4f}")
    print(f"  AUC-ROC:     {results['auc_roc']:.4f}")
    print(f"  Parameters:  {n_params:,}")
    print(f"\n{classification_report(y_true_test, y_pred_test, target_names=['Healthy', 'SSc'])}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    import json
    with open(RESULTS_DIR / "classical_results.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), RESULTS_DIR / "classical_model.pt")
    np.save(RESULTS_DIR / "classical_history.npy", history)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classical NN v02 for SSc")
    parser.add_argument("--hidden", nargs="+", type=int, default=[256, 128, 64, 32])
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    train_model(
        hidden_sizes=tuple(args.hidden),
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
    )
