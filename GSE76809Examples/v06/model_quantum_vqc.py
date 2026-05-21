"""
Data Reuploading Quantum VQC for v06.

Key improvement over v04/v05's amplitude encoding:
- Re-encodes input data at EVERY layer (Pérez-Salinas et al., 2020)
- Each qubit receives a trainable linear combination of ALL 16 features per layer
- Creates more expressive decision boundaries with fewer qubits
- Theoretically: universal function approximation with single qubit + enough layers

Architecture:
- 4 qubits, 8 variational layers
- Each layer: data encoding (RY with learned feature map) + variational (RZ, RX) + CNOT ring
- Multi-basis measurement (X, Y, Z) → 12 quantum features
- Post-net: 12 → 64 → 1
- Total: ~1,500 trainable parameters (matches classical MLP for fair comparison)
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from preprocess_gse76809 import apply_smote_to_fold

# Circuit configuration
N_QUBITS = 4
N_LAYERS = 8
N_FEATURES = 16


def create_quantum_circuit():
    """Create the data-reuploading quantum circuit."""
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights_enc, weights_var):
        """
        Data reuploading circuit.
        
        Args:
            inputs: (16,) input features (scaled)
            weights_enc: (n_layers, n_qubits, n_features) encoding weights
            weights_var: (n_layers, n_qubits, 2) variational parameters [RZ, RX angles]
        """
        for layer in range(N_LAYERS):
            # Data encoding: each qubit gets a learned linear combination of all features
            for qubit in range(N_QUBITS):
                # RY(sum_i w[l,q,i] * x[i]) — data-dependent rotation
                angle = torch.dot(weights_enc[layer, qubit], inputs)
                qml.RY(angle, wires=qubit)

            # Variational rotations (data-independent, learnable)
            for qubit in range(N_QUBITS):
                qml.RZ(weights_var[layer, qubit, 0], wires=qubit)
                qml.RX(weights_var[layer, qubit, 1], wires=qubit)

            # Entanglement: ring CNOT topology
            for qubit in range(N_QUBITS):
                qml.CNOT(wires=[qubit, (qubit + 1) % N_QUBITS])

        # Multi-basis measurement
        measurements = []
        for qubit in range(N_QUBITS):
            measurements.append(qml.expval(qml.PauliZ(qubit)))
        for qubit in range(N_QUBITS):
            measurements.append(qml.expval(qml.PauliX(qubit)))
        for qubit in range(N_QUBITS):
            measurements.append(qml.expval(qml.PauliY(qubit)))

        return measurements

    return circuit


class DataReuploadingVQC(nn.Module):
    """
    Data Reuploading Variational Quantum Classifier.
    
    Key insight: By re-encoding data at every layer with DIFFERENT learned projections,
    the circuit can approximate any function (universal approximation theorem for
    quantum circuits). This is fundamentally more expressive than single-encoding VQCs.
    """

    def __init__(self, n_features=N_FEATURES):
        super().__init__()
        self.n_features = n_features
        self.circuit = create_quantum_circuit()

        # Encoding weights: each layer, each qubit learns how to project all features
        # Shape: (n_layers, n_qubits, n_features)
        self.weights_enc = nn.Parameter(
            torch.randn(N_LAYERS, N_QUBITS, n_features) * 0.1
        )

        # Variational weights: RZ and RX per qubit per layer
        # Shape: (n_layers, n_qubits, 2)
        self.weights_var = nn.Parameter(
            torch.randn(N_LAYERS, N_QUBITS, 2) * 0.3
        )

        # Post-processing network (12 quantum features → 1 output)
        n_quantum_out = 3 * N_QUBITS  # 12
        self.post_net = nn.Sequential(
            nn.Linear(n_quantum_out, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward pass with data reuploading."""
        batch_size = x.shape[0]
        outputs = []

        for i in range(batch_size):
            q_out = self.circuit(x[i], self.weights_enc, self.weights_var)
            q_tensor = torch.stack(q_out)
            outputs.append(q_tensor)

        q_batch = torch.stack(outputs).float()  # Ensure float32
        return self.post_net(q_batch).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_quantum_vqc(fold_data, epochs=80, batch_size=24, lr=0.005,
                      patience=15, use_smote=True, random_state=2026):
    """
    Train the data reuploading VQC on a single fold.
    
    Per-fold SMOTE is applied here (correct methodology).
    """
    X_train = fold_data["X_train"]
    X_val = fold_data["X_val"]
    y_train = fold_data["y_train"]
    y_val = fold_data["y_val"]

    # Per-fold SMOTE (correct: augment training data INSIDE the fold)
    if use_smote:
        X_train, y_train = apply_smote_to_fold(X_train, y_train, random_state=random_state)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)

    # Initialize model
    model = DataReuploadingVQC(n_features=N_FEATURES)
    print(f"    VQC parameters: {model.count_parameters()}")

    # Class balance handled by per-fold SMOTE above, so plain (unweighted) BCE
    # is the consistent choice; mixing SMOTE with pos_weight double-corrects.
    criterion = nn.BCELoss()

    # Optimizer with warmup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        
        # Shuffle
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

        # Validation
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

    # Load best model and evaluate
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
