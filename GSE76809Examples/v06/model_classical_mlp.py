"""
Parameter-Matched Classical MLP for v06.

Key design principle: SAME number of parameters as the quantum VQC (~1,500).
This isolates whether the quantum circuit's STRUCTURE provides benefit,
vs simply having more trainable parameters.

Architecture: 16 → 48 → 16 → 1 (simple feedforward, no residual blocks)
- Same optimizer (Adam), same epochs, same early stopping as VQC
- No mixup, no label smoothing (keep it minimal and fair)
- Class-weighted BCE loss (same as VQC)
- ~1,600 trainable parameters

This is deliberately simpler than v04's 120K-param ResidualMLP.
If VQC outperforms this, it suggests quantum structure helps.
If this matches VQC, classical is sufficient at same parameter budget.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from preprocess_gse76809 import apply_smote_to_fold


class MatchedMLP(nn.Module):
    """
    MLP with ~1,600 parameters — matched to VQC parameter count.
    
    Architecture: 16 → 48 → 16 → 1
    Params: 16*48+48 + 48*16+16 + 16*1+1 = 816 + 784 + 17 = 1,617
    """

    def __init__(self, n_features=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 48),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(48, 16),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_classical_mlp(fold_data, epochs=80, batch_size=24, lr=0.005,
                        patience=15, use_smote=True, random_state=2026):
    """
    Train the parameter-matched MLP on a single fold.
    
    Same training procedure as VQC for fair comparison:
    - Same epochs, batch size, learning rate, patience
    - Same per-fold SMOTE
    - Same class-weighted loss
    """
    X_train = fold_data["X_train"]
    X_val = fold_data["X_val"]
    y_train = fold_data["y_train"]
    y_val = fold_data["y_val"]

    # Per-fold SMOTE (same as VQC)
    if use_smote:
        X_train, y_train = apply_smote_to_fold(X_train, y_train, random_state=random_state)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)

    # Initialize model
    model = MatchedMLP(n_features=X_train.shape[1])
    print(f"    MLP parameters: {model.count_parameters()}")

    # Class balance handled by per-fold SMOTE above; use plain BCE for parity
    # with the VQC (mixing SMOTE with pos_weight double-corrects).
    criterion = nn.BCELoss()

    # Same optimizer as VQC
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop (identical structure to VQC)
    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        
        perm = torch.randperm(len(X_train_t))
        X_shuffled = X_train_t[perm]
        y_shuffled = y_train_t[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(X_shuffled), batch_size):
            end = min(start + batch_size, len(X_shuffled))
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation (same schedule as VQC)
        if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_t).numpy()

            try:
                val_auc = roc_auc_score(y_val, val_preds)
            except ValueError:
                val_auc = 0.5

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 15 == 0:
                print(f"    Epoch {epoch+1}/{epochs} — Loss: {epoch_loss/n_batches:.4f}, "
                      f"Val AUC: {val_auc:.4f} (best: {best_val_auc:.4f})")

            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    # Load best and evaluate
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_probs = model(X_val_t).numpy()

    val_preds_binary = (val_probs >= 0.5).astype(int)
    
    try:
        auc = roc_auc_score(y_val, val_probs)
    except ValueError:
        auc = 0.5
    
    acc = accuracy_score(y_val, val_preds_binary)
    f1 = f1_score(y_val, val_preds_binary, zero_division=0)

    print(f"    → AUC: {auc:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")

    return {
        "auc_roc": auc,
        "accuracy": acc,
        "f1_score": f1,
        "predictions": val_probs,
        "pred_binary": val_preds_binary,
        "y_true": y_val,
    }
