"""Combined data-loading circuit fragments for v09B's encoding ablation.

Where v09 compared five *standalone* encodings (one of which was
``data_reuploading``), v09B asks a different question: **does prepending a
structured base encoding to data reuploading help, hurt, or do nothing?**

Every encoding here (except the standalone ``data_reuploading`` reference)
is a *combination*: the base encoding is applied **once at layer 0**, then
Pérez-Salinas-style data reuploading runs at **layers 1..N-1**. Because the
reuploading stage learns a linear combination of all features, every
encoding in this file requires learnable encoding weights
(``needs_enc_weights=True``).

Each function takes:
	inputs      : torch.Tensor of shape (n_features,)
	layer_idx   : int — which variational layer is about to run
	weights_enc : torch.Tensor of shape (n_layers, n_qubits, n_features)

and applies quantum gates to the *current* device wires. They are designed
to be called inside a PennyLane qnode that already has the device set.

Notes:
- All encodings target N_QUBITS=4 and N_FEATURES=16 (matching v06/v09).
- ``amplitude_reup`` requires the input to already be unit-norm at layer 0;
  the driver (model_quantum_vqc.train_quantum_vqc) does that. The later
  reuploading layers dot the (unit-norm) inputs with the learned weights.
- The standalone ``data_reuploading`` reference is byte-for-byte identical
  to v09's so the two experiments share a common baseline.
"""

from __future__ import annotations

import pennylane as qml
import torch

N_QUBITS = 4


def _reupload(inputs, layer_idx, weights_enc):
	"""Pérez-Salinas-style reuploading: each qubit gets a learned linear
	combination of ALL features at this layer."""
	for q in range(N_QUBITS):
		angle = torch.dot(weights_enc[layer_idx, q], inputs)
		qml.RY(angle, wires=q)


def data_reuploading(inputs, layer_idx, weights_enc):
	"""Reference encoding (identical to v09): reuploading at EVERY layer."""
	_reupload(inputs, layer_idx, weights_enc)


def angle_reup(inputs, layer_idx, weights_enc):
	"""angle + data_reuploading.

	Layer 0: RY(x_i) on each qubit (1 feature/qubit, the v01-style base).
	Layers 1..N-1: learned data reuploading of all features.
	"""
	if layer_idx == 0:
		for q in range(N_QUBITS):
			qml.RY(inputs[q] * torch.pi, wires=q)
	else:
		_reupload(inputs, layer_idx, weights_enc)


def dense_angle_reup(inputs, layer_idx, weights_enc):
	"""dense_angle + data_reuploading.

	Layer 0: RY(x_{2q}) then RZ(x_{2q+1}) on each qubit (2 features/qubit,
	loading the first 8 of 16 features). Layers 1..N-1: learned reuploading.
	"""
	if layer_idx == 0:
		for q in range(N_QUBITS):
			qml.RY(inputs[2 * q] * torch.pi, wires=q)
			qml.RZ(inputs[2 * q + 1] * torch.pi, wires=q)
	else:
		_reupload(inputs, layer_idx, weights_enc)


def amplitude_reup(inputs, layer_idx, weights_enc):
	"""amplitude + data_reuploading.

	Layer 0: AmplitudeEmbedding loads all 16 features as amplitudes of the
	4-qubit state (inputs MUST be unit-norm — the driver guarantees this).
	Layers 1..N-1: learned reuploading of the (unit-norm) features.
	"""
	if layer_idx == 0:
		qml.AmplitudeEmbedding(inputs, wires=range(N_QUBITS), normalize=False,
							pad_with=0.0)
	else:
		_reupload(inputs, layer_idx, weights_enc)


def iqp_reup(inputs, layer_idx, weights_enc):
	"""iqp + data_reuploading.

	Layer 0: IQP-style feature map (Havlicek 2019) — H, RZ(x_i), then ring
	ZZ(x_i*x_j). Layers 1..N-1: learned reuploading of all features.
	"""
	if layer_idx == 0:
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
	else:
		_reupload(inputs, layer_idx, weights_enc)


# All combined encodings need learnable encoding weights (True).
# data_reuploading is the standalone reference baseline.
ENCODINGS = {
	"data_reuploading": (data_reuploading, True),   # reference baseline
	"angle_reup": (angle_reup, True),               # angle + reuploading
	"dense_angle_reup": (dense_angle_reup, True),   # dense_angle + reuploading
	"amplitude_reup": (amplitude_reup, True),       # amplitude + reuploading
	"iqp_reup": (iqp_reup, True),                   # iqp + reuploading
}

# Encodings whose layer-0 base requires unit-norm inputs.
AMPLITUDE_ENCODINGS = {"amplitude_reup"}
