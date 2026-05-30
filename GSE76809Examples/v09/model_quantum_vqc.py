"""Encoding-ablation VQC for v09.

Same variational structure, post-net, optimiser, and training schedule as
v06.model_quantum_vqc.DataReuploadingVQC. The ONLY thing that varies is
how the input is loaded into the quantum register, selected by the
``encoding`` argument (see encodings.py).
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

from quantum_encodings import ENCODINGS, N_QUBITS  # noqa: E402

N_LAYERS = 8
N_FEATURES = 16


def _build_qnode(encoding_fn, needs_enc_weights):
	dev = qml.device("default.qubit", wires=N_QUBITS)

	@qml.qnode(dev, interface="torch", diff_method="backprop")
	def circuit(inputs, weights_enc, weights_var):
		for layer in range(N_LAYERS):
			encoding_fn(inputs, layer, weights_enc if needs_enc_weights else None)
			for q in range(N_QUBITS):
				qml.RZ(weights_var[layer, q, 0], wires=q)
				qml.RX(weights_var[layer, q, 1], wires=q)
			for q in range(N_QUBITS):
				qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
		# Multi-basis measurement, 3 * N_QUBITS = 12 outputs (same as v06)
		out = []
		for q in range(N_QUBITS):
			out.append(qml.expval(qml.PauliZ(q)))
		for q in range(N_QUBITS):
			out.append(qml.expval(qml.PauliX(q)))
		for q in range(N_QUBITS):
			out.append(qml.expval(qml.PauliY(q)))
		return out

	return circuit


class EncodingVQC(nn.Module):
	"""v06-equivalent VQC with a swappable encoding stage."""

	def __init__(self, encoding: str = "data_reuploading"):
		super().__init__()
		if encoding not in ENCODINGS:
			raise KeyError(f"unknown encoding '{encoding}'; "
						   f"available: {list(ENCODINGS)}")
		self.encoding_name = encoding
		encoding_fn, needs_enc = ENCODINGS[encoding]
		self.needs_enc = needs_enc
		self.circuit = _build_qnode(encoding_fn, needs_enc)

		# Encoding weights only used by data_reuploading
		if needs_enc:
			self.weights_enc = nn.Parameter(
				torch.randn(N_LAYERS, N_QUBITS, N_FEATURES) * 0.1)
		else:
			# Dummy buffer so the qnode signature is stable across encodings
			self.register_buffer(
				"weights_enc",
				torch.zeros(N_LAYERS, N_QUBITS, N_FEATURES),
			)
		self.weights_var = nn.Parameter(
			torch.randn(N_LAYERS, N_QUBITS, 2) * 0.3)

		self.post_net = nn.Sequential(
			nn.Linear(3 * N_QUBITS, 64),
			nn.GELU(),
			nn.Dropout(0.1),
			nn.Linear(64, 1),
			nn.Sigmoid(),
		)

	def forward(self, x):
		outs = []
		for i in range(x.shape[0]):
			q = self.circuit(x[i], self.weights_enc, self.weights_var)
			outs.append(torch.stack(q))
		return self.post_net(torch.stack(outs).float()).squeeze(-1)

	def count_parameters(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_quantum_vqc(fold_data, encoding: str = "data_reuploading",
					   epochs: int = 80, batch_size: int = 24, lr: float = 0.005,
					   patience: int = 15, use_smote: bool = True,
					   random_state: int = 2026, init_seed=None):
	"""Same training procedure as v06 — only the encoding changes."""
	X_train = fold_data["X_train"]
	X_val = fold_data["X_val"]
	y_train = fold_data["y_train"]
	y_val = fold_data["y_val"]

	if use_smote:
		X_train, y_train = apply_smote_to_fold(X_train, y_train, random_state=random_state)

	# amplitude encoding needs unit-norm inputs
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
	model = EncodingVQC(encoding=encoding)
	print(f"    [{encoding}] params={model.count_parameters()}")

	criterion = nn.BCELoss()
	opt = torch.optim.Adam(model.parameters(), lr=lr)
	sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

	best_auc, best_state, bad = 0.0, None, 0
	for epoch in range(epochs):
		model.train()
		perm = torch.randperm(len(X_train_t))
		Xs, ys = X_train_t[perm], y_train_t[perm]
		for start in range(0, len(Xs), batch_size):
			xb = Xs[start:start + batch_size]
			yb = ys[start:start + batch_size]
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
				print(f"    [{encoding}] early stop epoch {epoch+1}")
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
		"auc_roc": auc,
		"accuracy": accuracy_score(y_val, preds),
		"f1_score": f1_score(y_val, preds, zero_division=0),
		"predictions": probs,
		"pred_binary": preds,
		"y_true": y_val,
	}
