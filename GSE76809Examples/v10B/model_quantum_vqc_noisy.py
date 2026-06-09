"""Noisy-simulation port of v09's encoding-ablation VQC for v10B.

Same variational structure, post-net, optimiser, and training schedule as
v09.model_quantum_vqc.EncodingVQC, with the encoding stage still selected
by the ``encoding`` argument (see v09's ``quantum_encodings.ENCODINGS``).
The only difference from v09 is the device: this runs on
``default.mixed`` and injects depolarising noise after every gate.

A single-qubit ``DepolarizingChannel`` with probability ``noise_p`` is
applied after every single-qubit rotation, and a two-qubit depolarising
channel (product-of-single-qubit approximation at probability
``min(10*noise_p, 0.75)``) is applied after every CNOT — including the
CNOTs that some encodings (e.g. ``iqp``) emit inside the data-loading
stage.

For ``noise_p == 0`` each encoding's circuit is mathematically identical
to v09 (modulo density-matrix vs statevector representation), so the
``noise_p = 0`` column reproduces v09's noiseless encoding ranking.
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
sys.path.insert(0, str(_THIS.parent / "v06"))
sys.path.insert(0, str(_THIS.parent / "v09"))
from preprocess_gse76809 import apply_smote_to_fold  # noqa: E402
from quantum_encodings import ENCODINGS, N_QUBITS  # noqa: E402

N_LAYERS = 8
N_FEATURES = 16


def _build_noisy_qnode(encoding_fn, needs_enc_weights, noise_p: float):
	"""Density-matrix qnode with a swappable encoding and depolarising noise.

	A depolarising channel is applied after each single-qubit rotation and
	after each CNOT. To keep the channel insertion encoding-agnostic we wrap
	``qml.RY/RZ/RX`` and ``qml.CNOT`` for the duration of the circuit build
	so the noise also follows the gates emitted *inside* the encoding
	fragment (angle/dense_angle/iqp all emit single-qubit rotations, iqp
	also emits CNOTs).
	"""
	dev = qml.device("default.mixed", wires=N_QUBITS)
	two_qubit_p = min(10.0 * noise_p, 0.75)

	@qml.qnode(dev, interface="torch", diff_method="backprop")
	def circuit(inputs, weights_enc, weights_var):
		# Wrap the rotation/entangling primitives so every gate (including
		# those emitted by the encoding fragment) is followed by noise.
		_RY, _RZ, _RX, _CNOT = qml.RY, qml.RZ, qml.RX, qml.CNOT

		def ry(theta, wires):
			_RY(theta, wires=wires)
			if noise_p > 0:
				qml.DepolarizingChannel(noise_p, wires=wires)

		def rz(theta, wires):
			_RZ(theta, wires=wires)
			if noise_p > 0:
				qml.DepolarizingChannel(noise_p, wires=wires)

		def rx(theta, wires):
			_RX(theta, wires=wires)
			if noise_p > 0:
				qml.DepolarizingChannel(noise_p, wires=wires)

		def cnot(wires):
			_CNOT(wires=wires)
			if two_qubit_p > 0:
				qml.DepolarizingChannel(two_qubit_p, wires=wires[0])
				qml.DepolarizingChannel(two_qubit_p, wires=wires[1])

		qml.RY, qml.RZ, qml.RX, qml.CNOT = (
			lambda theta, wires: ry(theta, wires),
			lambda theta, wires: rz(theta, wires),
			lambda theta, wires: rx(theta, wires),
			lambda wires: cnot(wires),
		)
		try:
			for layer in range(N_LAYERS):
				encoding_fn(inputs, layer, weights_enc if needs_enc_weights else None)
				for q in range(N_QUBITS):
					qml.RZ(weights_var[layer, q, 0], wires=q)
					qml.RX(weights_var[layer, q, 1], wires=q)
				for q in range(N_QUBITS):
					qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
		finally:
			# Restore the originals so we never leak the monkeypatch.
			qml.RY, qml.RZ, qml.RX, qml.CNOT = _RY, _RZ, _RX, _CNOT

		out = []
		for q in range(N_QUBITS):
			out.append(qml.expval(qml.PauliZ(q)))
		for q in range(N_QUBITS):
			out.append(qml.expval(qml.PauliX(q)))
		for q in range(N_QUBITS):
			out.append(qml.expval(qml.PauliY(q)))
		return out

	return circuit


class NoisyEncodingVQC(nn.Module):
	"""v09 EncodingVQC on default.mixed with depolarising noise."""

	def __init__(self, encoding: str = "data_reuploading", noise_p: float = 0.0):
		super().__init__()
		if encoding not in ENCODINGS:
			raise KeyError(f"unknown encoding '{encoding}'; available: {list(ENCODINGS)}")
		self.encoding_name = encoding
		self.noise_p = noise_p
		encoding_fn, needs_enc = ENCODINGS[encoding]
		self.needs_enc = needs_enc
		self.circuit = _build_noisy_qnode(encoding_fn, needs_enc, noise_p)

		if needs_enc:
			self.weights_enc = nn.Parameter(torch.randn(N_LAYERS, N_QUBITS, N_FEATURES) * 0.1)
		else:
			self.register_buffer("weights_enc", torch.zeros(N_LAYERS, N_QUBITS, N_FEATURES))
		self.weights_var = nn.Parameter(torch.randn(N_LAYERS, N_QUBITS, 2) * 0.3)

		self.post_net = nn.Sequential(
			nn.Linear(3 * N_QUBITS, 64),
			nn.GELU(), nn.Dropout(0.1),
			nn.Linear(64, 1), nn.Sigmoid(),
		)

	def forward(self, x):
		outs = []
		for i in range(x.shape[0]):
			outs.append(torch.stack(self.circuit(x[i], self.weights_enc, self.weights_var)))
		return self.post_net(torch.stack(outs).float()).squeeze(-1)

	def count_parameters(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_noisy_vqc(fold_data, encoding: str = "data_reuploading", noise_p: float = 0.0,
					epochs: int = 40, batch_size: int = 24, lr: float = 0.005,
					patience: int = 10, use_smote: bool = True,
					random_state: int = 2026, init_seed=None):
	"""Same training procedure as v09 — encoding *and* noise level vary."""
	X_train, X_val = fold_data["X_train"], fold_data["X_val"]
	y_train, y_val = fold_data["y_train"], fold_data["y_val"]

	if use_smote:
		X_train, y_train = apply_smote_to_fold(X_train, y_train, random_state=random_state)

	# amplitude encoding needs unit-norm inputs (same as v09).
	if encoding == "amplitude":
		X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-10)
		X_val_use = X_val / (np.linalg.norm(X_val, axis=1, keepdims=True) + 1e-10)
	else:
		X_val_use = X_val

	X_train_t = torch.FloatTensor(X_train)
	y_train_t = torch.FloatTensor(y_train)
	X_val_t = torch.FloatTensor(X_val_use)

	if init_seed is not None:
		torch.manual_seed(int(init_seed))
	model = NoisyEncodingVQC(encoding=encoding, noise_p=noise_p)

	crit = nn.BCELoss()
	opt = torch.optim.Adam(model.parameters(), lr=lr)
	sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

	best_auc, best_state, bad = 0.0, None, 0
	for epoch in range(epochs):
		model.train()
		perm = torch.randperm(len(X_train_t))
		Xs, ys = X_train_t[perm], y_train_t[perm]
		for s in range(0, len(Xs), batch_size):
			xb, yb = Xs[s:s + batch_size], ys[s:s + batch_size]
			opt.zero_grad()
			loss = crit(model(xb), yb)
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
		"encoding": encoding,
		"noise_p": noise_p,
		"auc_roc": auc,
		"accuracy": accuracy_score(y_val, preds),
		"f1_score": f1_score(y_val, preds, zero_division=0),
	}
