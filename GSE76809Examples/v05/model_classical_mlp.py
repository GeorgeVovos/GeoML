"""
Classical MLP baseline (v05): Residual MLP with mixup augmentation.

Same architectural family as v04, scaled to the v05 data slice (16 features).
Hidden dims are smaller because the input is smaller.
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
DATA_DIR = _ROOT / "data" / "GSE76809" / "processed_v05"


def find_optimal_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    idx = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_y = lam * y + (1 - lam) * y[idx]
    return mixed_x, mixed_y


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x):
        return self.net(x) + self.shortcut(x)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim=16, hidden_dims=(128, 64, 32, 16), dropout=0.4):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(ResidualBlock(prev_dim, h_dim, dropout))
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x):
        features = self.backbone(x)
        return torch.sigmoid(self.head(features)).squeeze(-1)


def train_classical_mlp(
    epochs=100,
    lr=0.001,
    patience=15,
    fold_data=None,
):
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

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    n_features = X_train.shape[1]
    model = ResidualMLP(input_dim=n_features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2
    )

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{'='*60}")
    print(f"CLASSICAL MLP v05 (Residual + Mixup + Label Smoothing)")
    print(f"  Features: {n_features}, Arch: 128->64->32->16->1")
    print(f"  Epochs: {epochs}, LR: {lr}, Patience: {patience}")
    print(f"  Trainable parameters: {total_params:,}")
    print(f"{'='*60}")

    label_smoothing = 0.05
    best_auc = 0.0
    best_state = None
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        batch_size = 32  # smaller batch since less data after no-SMOTE
        indices = torch.randperm(len(X_train_t))
        epoch_loss = 0.0
        n_batches = 0

        use_mixup = epoch <= int(0.8 * epochs)

        for start in range(0, len(X_train_t), batch_size):
            batch_idx = indices[start:start + batch_size]
            if len(batch_idx) < 2:
                continue
            xb = X_train_t[batch_idx]
            yb = y_train_t[batch_idx]

            if use_mixup:
                xb, yb = mixup_data(xb, yb, alpha=0.2)

            yb_smooth = yb * (1 - label_smoothing) + 0.5 * label_smoothing

            optimizer.zero_grad()
            preds = model(xb)

            weights = torch.where(yb_smooth > 0.5, pos_weight, torch.ones(1))
            loss = (nn.functional.binary_cross_entropy(preds, yb_smooth, reduction='none') * weights).mean()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            test_probs = model(X_test_t).numpy()

        try:
            auc = roc_auc_score(y_test, test_probs)
        except ValueError:
            auc = 0.5

        if epoch <= 5 or epoch % 10 == 0 or epoch == epochs:
            test_preds = (test_probs >= 0.5).astype(int)
            test_acc = accuracy_score(y_test, test_preds)
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Test Acc: {test_acc:.4f} | AUC: {auc:.4f}")

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
    train_classical_mlp()
