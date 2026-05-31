"""Noisy-simulation port of v06's data-reuploading VQC for v10A.

This is the **same data-reuploading ansatz v08 used** (v06's 4-qubit,
8-layer DataReuploadingVQC), re-hosted on PennyLane's ``default.mixed``
density-matrix simulator so we can inject realistic NISQ gate error.

A single-qubit ``DepolarizingChannel`` with probability ``noise_p`` is
applied after every single-qubit rotation, and a two-qubit depolarising
channel (approximated by a product of single-qubit channels at
probability ``min(10*noise_p, 0.75)``) is applied after every CNOT.

For ``noise_p == 0`` the circuit is mathematically identical to the v08
VQC (modulo the density-matrix vs statevector representation), so the
``noise_p = 0`` column reproduces v08's noiseless small-data curve.

The public ``train_noisy_vqc`` keeps v08's calling convention
(``fold_data`` dict + ``use_smote`` + ``init_seed``) so the v10A driver
can reuse v08's fold-building / subsampling harness unchanged.
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
from preprocess_gse76809 import apply_smote_to_fold  # noqa: E402

N_QUBITS = 4
N_LAYERS = 8
N_FEATURES = 16


def _build_noisy_qnode(noise_p: float):
	"""Build a density-matrix qnode with depolarising noise after every gate."""
	dev = qml.device("default.mixed", wires=N_QUBITS)
	two_qubit_p = min(10.0 * noise_p, 0.75)  # cap to the physical range

	def _depol_single(q):
		if noise_p > 0:
			qml.DepolarizingChannel(noise_p, wires=q)

	def _depol_two(q1, q2):
		# Product-of-single-qubit approximation of a 2-qubit depolarising
		# channel (the standard cheap version; a fully general 2-qubit
		# channel exists but is much slower).
		if two_qubit_p > 0:
			qml.DepolarizingChannel(two_qubit_p, wires=q1)
			qml.DepolarizingChannel(two_qubit_p, wires=q2)

	@qml.qnode(dev, interface="torch", diff_method="backprop")
	def circuit(inputs, weights_enc, weights_var):
		for layer in range(N_LAYERS):
			# Data reuploading: each qubit gets a learned linear combination
			# of ALL features at every layer.
			for q in range(N_QUBITS):
				angle = torch.dot(weights_enc[layer, q], inputs)
				qml.RY(angle, wires=q)
				_depol_single(q)
			for q in range(N_QUBITS):
				qml.RZ(weights_var[layer, q, 0], wires=q)
				_depol_single(q)
				qml.RX(weights_var[layer, q, 1], wires=q)
				_depol_single(q)
			for q in range(N_QUBITS):
				qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
				_depol_two(q, (q + 1) % N_QUBITS)
		out = []
		for q in range(N_QUBITS):
			out.append(qml.expval(qml.PauliZ(q)))
		for q in range(N_QUBITS):
			out.append(qml.expval(qml.PauliX(q)))
		for q in range(N_QUBITS):
			out.append(qml.expval(qml.PauliY(q)))
		return out

	return circuit


class NoisyVQC(nn.Module):
	"""v06/v08 data-reuploading VQC on default.mixed with depolarising noise."""

	def __init__(self, noise_p: float):
		super().__init__()
		self.noise_p = noise_p
		self.circuit = _build_noisy_qnode(noise_p)
		self.weights_enc = nn.Parameter(torch.randn(N_LAYERS, N_QUBITS, N_FEATURES) * 0.1)
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


def train_noisy_vqc(fold_data, noise_p: float = 0.0, epochs: int = 40,
					batch_size: int = 24, lr: float = 0.005, patience: int = 10,
					use_smote: bool = True, random_state: int = 2026,
					init_seed=None):
	"""Train the noisy VQC on one (subsample) fold; evaluate on the fixed test set.

	Signature mirrors v08's ``train_quantum_vqc`` so the v10A driver can
	reuse v08's ``make_fold`` / ``apply_smote_if_needed`` harness. The
	extra ``noise_p`` argument selects the depolarising strength.
	"""
	X_train, X_val = fold_data["X_train"], fold_data["X_val"]
	y_train, y_val = fold_data["y_train"], fold_data["y_val"]

	if use_smote:
		X_train, y_train = apply_smote_to_fold(X_train, y_train, random_state=random_state)

	X_train_t = torch.FloatTensor(X_train)
	y_train_t = torch.FloatTensor(y_train)
	X_val_t = torch.FloatTensor(X_val)

	if init_seed is not None:
		torch.manual_seed(int(init_seed))
	model = NoisyVQC(noise_p=noise_p)

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
		"noise_p": noise_p,
		"auc_roc": auc,
		"accuracy": accuracy_score(y_val, preds),
		"f1_score": f1_score(y_val, preds, zero_division=0),
	}
