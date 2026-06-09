"""Five data-loading circuit fragments for v09's encoding ablation.

Each function takes:
	inputs      : torch.Tensor of shape (n_features,)
	layer_idx   : int — which variational layer is about to run
	weights_enc : Optional[torch.Tensor] — encoding weights (only used by
			  data_reuploading)

and applies quantum gates to the *current* device wires. They are designed
to be called inside a PennyLane qnode that already has the device set.

Notes:
- All encodings target N_QUBITS=4 and N_FEATURES=16 (matching v06).
- amplitude encoding requires the input to already be unit-norm; the
  driver does that.
- iqp / amplitude / angle are *single-shot* — they ignore layer_idx (the
  encoding runs at layer 0 only). data_reuploading uses every layer.
- dense_angle is also single-shot but packs 2 features per qubit (RY+RZ),
  loading 8 of the 16 features vs 4 for plain angle.
"""

from __future__ import annotations

import pennylane as qml
import torch

N_QUBITS = 4

def angle_encoding(inputs, layer_idx, weights_enc=None):
	"""RY(x_i) on each qubit — one feature per qubit, at layer 0 only.

	Because we have 16 features but only 4 qubits, the first 4 features
	are encoded; the rest are unused. This deliberately matches v01's
	information bottleneck.
	"""
	if layer_idx != 0:
		return
	for q in range(N_QUBITS):
		qml.RY(inputs[q] * torch.pi, wires=q)

def dense_angle_encoding(inputs, layer_idx, weights_enc=None):
	"""Dense angle encoding — TWO features per qubit, at layer 0 only.

	Each qubit carries two features on orthogonal rotation axes:
	RY(x_{2q}) followed by RZ(x_{2q+1}). With 4 qubits this loads the
	first 8 of the 16 features (vs only 4 for plain angle encoding),
	doubling the information packed into the same register without
	adding qubits or trainable encoding weights.
	"""
	if layer_idx != 0:
		return
	for q in range(N_QUBITS):
		qml.RY(inputs[2 * q] * torch.pi, wires=q)
		qml.RZ(inputs[2 * q + 1] * torch.pi, wires=q)

def amplitude_encoding(inputs, layer_idx, weights_enc=None):
	"""Amplitude embedding — 16 amplitudes in 4 qubits, at layer 0 only.

	Caller MUST normalise inputs to unit length before calling. The
	embedding template raises if it isn't.
	"""
	if layer_idx != 0:
		return
	qml.AmplitudeEmbedding(inputs, wires=range(N_QUBITS), normalize=False,
						pad_with=0.0)

def iqp_encoding(inputs, layer_idx, weights_enc=None):
	"""IQP-style feature map (Havlicek 2019): H, RZ(x_i), then ring ZZ(x_i*x_j).

	A single repetition at layer 0; the variational layers downstream
	provide depth.
	"""
	if layer_idx != 0:
		return
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

def data_reuploading(inputs, layer_idx, weights_enc):
	"""Pérez-Salinas-style data reuploading: each qubit gets a learned
	linear combination of ALL features at every layer.

	weights_enc shape: (n_layers, n_qubits, n_features)
	"""
	for q in range(N_QUBITS):
		angle = torch.dot(weights_enc[layer_idx, q], inputs)
		qml.RY(angle, wires=q)

ENCODINGS = {
	"angle": (angle_encoding, False),               # (fn, needs_enc_weights)
	"dense_angle": (dense_angle_encoding, False),   # 2 features / qubit
	"amplitude": (amplitude_encoding, False),
	"iqp": (iqp_encoding, False),
	"data_reuploading": (data_reuploading, True),
}
