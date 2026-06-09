"""Per-layer reuploading + upstream-encoding VQC for v09C.

Shares the v06/v09 setup in EVERY respect (preprocessing, the
4-qubit / 8-layer variational stack, post-net, optimiser, training schedule,
CV folds and random seed). The defining architectural choice of v09C:

  the upstream encoding gate block is applied at EVERY layer, interleaved
  with the learned data-reuploading and the variational gates.

This is the literal "learned per-layer data-reuploading module applied in
combination with each encoding" reading of the template, and matches the
canonical Pérez-Salinas data-reuploading architecture (re-inject data every
layer).

Circuit per layer l:
  ① if upstream gate exists:  upstream_gate_fn(inputs)   ← EVERY layer in v09C
  ② for q in 0..3:  RY(dot(weights_enc[l, q], inputs), wire=q)  ← learned, always
  ③ for q in 0..3:  RZ(weights_var[l,q,0]), RX(weights_var[l,q,1])
  ④ CNOT ring
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent / "v06"))   # preprocess_gse76809
sys.path.insert(0, str(_THIS.parent))           # shared
sys.path.insert(0, str(_THIS))                  # local v09C imports

from preprocess_gse76809 import apply_smote_to_fold   # noqa: E402
from quantum_encodings import UPSTREAM_GATES, N_QUBITS  # noqa: E402

N_LAYERS = 8
N_FEATURES = 16


def _build_qnode(upstream_gate_fn):
    """Build a PennyLane qnode for the per-layer upstream+reuploading circuit."""
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights_enc, weights_var):
        for layer in range(N_LAYERS):
            # ① Upstream encoding — applied at EVERY layer in v09C
            if upstream_gate_fn is not None:
                upstream_gate_fn(inputs)
            # ② Learned data-reuploading — every layer, all experiments
            for q in range(N_QUBITS):
                angle = torch.dot(weights_enc[layer, q], inputs)
                qml.RY(angle, wires=q)
            # ③ Variational block
            for q in range(N_QUBITS):
                qml.RZ(weights_var[layer, q, 0], wires=q)
                qml.RX(weights_var[layer, q, 1], wires=q)
            # ④ Entanglement ring
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
        # Multi-basis measurement: PauliZ, PauliX, PauliY — 12 outputs (same as v06/v09)
        out = []
        for q in range(N_QUBITS):
            out.append(qml.expval(qml.PauliZ(q)))
        for q in range(N_QUBITS):
            out.append(qml.expval(qml.PauliX(q)))
        for q in range(N_QUBITS):
            out.append(qml.expval(qml.PauliY(q)))
        return out

    return circuit


class CombinedEncodingVQC(nn.Module):
    """v09C VQC: upstream encoding (every layer) combined with learned reuploading (every layer)."""

    def __init__(self, encoding: str = "data_reuploading"):
        super().__init__()
        if encoding not in UPSTREAM_GATES:
            raise KeyError(
                f"unknown encoding '{encoding}'; available: {list(UPSTREAM_GATES)}"
            )
        self.encoding_name = encoding
        upstream_gate_fn = UPSTREAM_GATES[encoding]
        self.circuit = _build_qnode(upstream_gate_fn)

        # Learned reuploading weights — always a trainable parameter for all experiments.
        # Shape: (N_LAYERS, N_QUBITS, N_FEATURES) = (8, 4, 16).
        self.weights_enc = nn.Parameter(
            torch.randn(N_LAYERS, N_QUBITS, N_FEATURES) * 0.1
        )
        self.weights_var = nn.Parameter(
            torch.randn(N_LAYERS, N_QUBITS, 2) * 0.3
        )
        # Post-net: 12 inputs (3 * N_QUBITS), same as v06/v09
        self.post_net = nn.Sequential(
            nn.Linear(3 * N_QUBITS, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(x.shape[0]):
            q = self.circuit(x[i], self.weights_enc, self.weights_var)
            outs.append(torch.stack(q))
        return self.post_net(torch.stack(outs).float()).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_quantum_vqc(
    fold_data: dict,
    encoding: str = "data_reuploading",
    epochs: int = 80,
    batch_size: int = 24,
    lr: float = 0.005,
    patience: int = 15,
    use_smote: bool = True,
    random_state: int = 2026,
    init_seed=None,
) -> dict:
    """Train one fold.  Identical procedure to v09 — only the encoding-site changes."""
    X_train = fold_data["X_train"]
    X_val   = fold_data["X_val"]
    y_train = fold_data["y_train"]
    y_val   = fold_data["y_val"]

    if use_smote:
        X_train, y_train = apply_smote_to_fold(
            X_train, y_train, random_state=random_state
        )

    # Amplitude encoding needs unit-norm inputs
    if encoding == "amplitude_combined":
        norms = np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-10
        X_train = X_train / norms
        X_val_use = X_val / (np.linalg.norm(X_val, axis=1, keepdims=True) + 1e-10)
    else:
        X_val_use = X_val

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t   = torch.FloatTensor(X_val_use)

    if init_seed is not None:
        torch.manual_seed(int(init_seed))
    model = CombinedEncodingVQC(encoding=encoding)
    print(f"    [{encoding}] params={model.count_parameters()}")

    criterion = nn.BCELoss()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_auc, best_state, bad = 0.0, None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))
        Xs, ys = X_train_t[perm], y_train_t[perm]
        for start in range(0, len(Xs), batch_size):
            xb = Xs[start : start + batch_size]
            yb = ys[start : start + batch_size]
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                vp = model(X_val_t).numpy()
            try:
                vauc = roc_auc_score(y_val, vp)
            except ValueError:
                vauc = 0.5
            if vauc > best_auc:
                best_auc, best_state, bad = vauc, model.state_dict().copy(), 0
            else:
                bad += 1
            if bad >= patience:
                print(f"    [{encoding}] early stop at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = model(X_val_t).numpy()
    preds = (probs >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_val, probs)
    except ValueError:
        auc = 0.5
    return {
        "encoding":    encoding,
        "auc_roc":     auc,
        "accuracy":    accuracy_score(y_val, preds),
        "f1_score":    f1_score(y_val, preds, zero_division=0),
        "predictions": probs,
        "pred_binary": preds,
        "y_true":      y_val,
    }
