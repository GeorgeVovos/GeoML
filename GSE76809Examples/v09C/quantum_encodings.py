"""Upstream encoding gate fragments for v09C's per-layer reuploading combination ablation.

In v09C the upstream gates run at EVERY layer, in combination with the learned
reuploading at every layer (the literal "applied in combination ... per-layer"
reading of the original template, and the canonical Pérez-Salinas data-reuploading
architecture).

The learned data-reuploading module (weights_enc of shape
(N_LAYERS, N_QUBITS, N_FEATURES)) runs every layer for all experiments.

Experiments:
  data_reuploading      — no upstream gates; pure learned reuploading (canonical ref)
  angle_combined        — RY(x_i * π) per qubit (first 4 features) + reuploading
  dense_angle_combined  — RY(x_{2q}*π) + RZ(x_{2q+1}*π) per qubit (first 8) + reuploading
  amplitude_combined    — AmplitudeEmbedding (unit-norm, 16 amplitudes) + reuploading
  iqp_combined          — IQP feature map (H, RZ, ZZ ring) + reuploading

Note: amplitude_combined applies AmplitudeEmbedding at EVERY layer in v09C.
Because AmplitudeEmbedding re-prepares the full statevector, applying it
mid-circuit *overwrites* the accumulated state each layer. This is a real,
intentional architectural consequence of the every-layer design.
"""

from __future__ import annotations

import pennylane as qml
import torch

N_QUBITS = 4


def _apply_angle_gates(inputs: torch.Tensor) -> None:
    """RY(x_i * π) on each qubit — first 4 features."""
    for q in range(N_QUBITS):
        qml.RY(inputs[q] * torch.pi, wires=q)


def _apply_dense_angle_gates(inputs: torch.Tensor) -> None:
    """RY(x_{2q} * π) + RZ(x_{2q+1} * π) per qubit — first 8 features."""
    for q in range(N_QUBITS):
        qml.RY(inputs[2 * q] * torch.pi, wires=q)
        qml.RZ(inputs[2 * q + 1] * torch.pi, wires=q)


def _apply_amplitude_gates(inputs: torch.Tensor) -> None:
    """AmplitudeEmbedding over all 16 features.

    Caller MUST normalise inputs to unit length before the circuit is called.
    """
    qml.AmplitudeEmbedding(
        inputs, wires=range(N_QUBITS), normalize=False, pad_with=0.0
    )


def _apply_iqp_gates(inputs: torch.Tensor) -> None:
    """IQP-style feature map: H, RZ(x_i), then ring ZZ(x_i * x_j)."""
    n = N_QUBITS
    for q in range(n):
        qml.Hadamard(wires=q)
    for q in range(n):
        qml.RZ(inputs[q], wires=q)
    for i in range(n):
        j = (i + 1) % n
        qml.CNOT(wires=[i, j])
        qml.RZ(inputs[i] * inputs[j], wires=j)
        qml.CNOT(wires=[i, j])


# Map encoding name → upstream gate function (applied at EVERY layer in v09C).
# None means no upstream gates (pure learned reuploading = canonical reference).
UPSTREAM_GATES: dict[str, object] = {
    "data_reuploading":     None,
    "angle_combined":       _apply_angle_gates,
    "dense_angle_combined": _apply_dense_angle_gates,
    "amplitude_combined":   _apply_amplitude_gates,
    "iqp_combined":         _apply_iqp_gates,
}
