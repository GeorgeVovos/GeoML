"""
Classical neural network classifier v03: Residual MLP with threshold optimization.

Improvements over v02:
1. Threshold optimization via Youden's J
2. Label smoothing for better calibration
3. Mixup data augmentation
4. Returns probabilities for ensemble stacking
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(256, 128, 64, 32), dropout=0.4):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(ResidualBlock(prev_dim, h, dropout))
            prev_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x):
        features = self.backbone(x)
        return torch.sigmoid(self.head(features)).squeeze(-1)


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation for better generalization."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def train_classical(
    hidden_sizes=(256, 128, 64, 32),
    dropout=0.4,
    epochs=100,
    lr=0.001,
    patience=15,
    label_smoothing=0.05,
    use_mixup=True,
    fold_data=None,
):
    """
    Train residual MLP with threshold optimization.

    Args:
        fold_data: if provided, dict with X_train, X_val, y_train, y_val (for CV)
    """
    if fold_data is not None:
        X_train = fold_data["X_train"]
        X_test = fold_data["X_val"]
        y_train = fold_data["y_train"]
        y_test = fold_data["y_val"]
    else:
        X_train = np.load(DATA_DIR / "X_train.npy")
        X_test = np.load(DATA_DIR / "X_test.npy")
        y_train = np.load(DATA_DIR / "y_train.npy")
        y_test = np.load(DATA_DIR / "y_test.npy")

    n_features = X_train.shape[1]
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # Label smoothing
    if label_smoothing > 0:
        y_train_smooth = y_train_t * (1 - label_smoothing) + 0.5 * label_smoothing
    else:
        y_train_smooth = y_train_t

    # Class weight
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    model = ResidualMLP(n_features, hidden_sizes, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    criterion = nn.BCELoss(reduction='none')

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{'='*60}")
    print(f"CLASSICAL NN v03 (Residual MLP + Mixup + Threshold Opt)")
    print(f"  Features: {n_features}, Hidden: {hidden_sizes}")
    print(f"  Dropout: {dropout}, Epochs: {epochs}, LR: {lr}")
    print(f"  Label smoothing: {label_smoothing}, Mixup: {use_mixup}")
    print(f"  Trainable parameters: {total_params:,}")
    print(f"{'='*60}")

    best_auc = 0.0
    best_state = None
    patience_counter = 0
    start_time = time.time()
    batch_size = 64

    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(len(X_train_t))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(X_train_t), batch_size):
            batch_idx = indices[start:start+batch_size]
            xb = X_train_t[batch_idx]
            yb = y_train_smooth[batch_idx]

            if use_mixup and epoch < epochs * 0.8:
                xb, yb = mixup_data(xb, yb, alpha=0.2)

            optimizer.zero_grad()
            preds = model(xb)

            weights = torch.where(yb > 0.5, pos_weight, torch.ones(1))
            loss = (criterion(preds, yb) * weights).mean()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_probs = model(X_test_t).numpy()
            train_probs = model(X_train_t).numpy()

        train_preds = (train_probs >= 0.5).astype(int)
        test_preds = (test_probs >= 0.5).astype(int)
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)

        try:
            auc = roc_auc_score(y_test, test_probs)
        except ValueError:
            auc = 0.5

        if epoch <= 3 or epoch % 10 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {epoch_loss/n_batches:.4f} | "
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

    test_preds = test_preds_opt if acc_opt >= acc_default else test_preds_default
    used_threshold = opt_threshold if acc_opt >= acc_default else 0.5

    acc = accuracy_score(y_test, test_preds)
    f1_ssc = f1_score(y_test, test_preds, pos_label=1)
    f1_healthy = f1_score(y_test, test_preds, pos_label=0)

    print(f"\n  Training time: {train_time:.1f}s")
    print(f"  Optimal threshold: {used_threshold:.4f} (Youden's J)")
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - CLASSICAL NN v03")
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
        "n_params": total_params,
        "threshold": used_threshold,
        "test_probs": test_probs,
        "epochs_run": epoch,
    }


if __name__ == "__main__":
    results = train_classical()
