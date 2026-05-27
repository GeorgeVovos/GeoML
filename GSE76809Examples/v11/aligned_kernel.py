"""Quantum Kernel Alignment (Hubregtsen et al. 2021).

Adds a small set of trainable rotation parameters to the ZZ feature map
and gradient-trains them to maximise kernel-target alignment with the
labels before fitting the SVM.

Kernel-target alignment (Cristianini et al. 2002):
	A(K, K*) = <K, K*>_F / sqrt(<K,K>_F * <K*,K*>_F)
where K is the (Gram) kernel matrix and K* = y y^T (ideal class kernel).

Higher alignment => kernel "agrees" with the labels => SVM trained on K
will generalise better. We maximise the centered version (Cortes 2012)
which is more robust to label imbalance.

Compute note: per gradient step we materialise the full n x n Gram
matrix, so this is O(n^2) per step. For n_train ~ 170, with 100 steps,
that is ~3 M circuit evaluations per fold. Slow but tractable.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC

N_QUBITS = 6
DEPTH = 2


def _build_aligned_qnode():
	dev = qml.device("default.qubit", wires=N_QUBITS)

	@qml.qnode(dev, interface="torch", diff_method="backprop")
	def overlap(x1, x2, theta):
		"""Compute |<φ(x1)|φ(x2)>|² for a trainable feature map.

		``theta`` shape: (DEPTH, N_QUBITS) — per-qubit per-layer rotation
		angle that scales the data-dependent ZZ couplings.
		"""
		def apply(x):
			for d in range(DEPTH):
				for q in range(N_QUBITS):
					qml.Hadamard(wires=q)
				for q in range(N_QUBITS):
					qml.RZ(x[q] * theta[d, q], wires=q)
				for i in range(N_QUBITS):
					for j in range(i + 1, N_QUBITS):
						qml.CNOT(wires=[i, j])
						qml.RZ(x[i] * x[j] * theta[d, q % N_QUBITS], wires=j)
						qml.CNOT(wires=[i, j])

		apply(x1)
		qml.adjoint(apply)(x2)
		return qml.probs(wires=range(N_QUBITS))

	return overlap


def _gram(X1, X2, theta, qnode):
	n1, n2 = len(X1), len(X2)
	K = torch.zeros((n1, n2), dtype=torch.float32)
	for i in range(n1):
		for j in range(n2):
			K[i, j] = qnode(X1[i], X2[j], theta)[0]
	return K


def _centered_alignment(K: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	"""Cortes 2012 centered KTA. Returns scalar in [-1, 1]; higher = better."""
	n = K.shape[0]
	one = torch.ones((n, n), dtype=K.dtype) / n
	Kc = K - one @ K - K @ one + one @ K @ one
	yy = (y.unsqueeze(0) * y.unsqueeze(1)).float()
	yyc = yy - one @ yy - yy @ one + one @ yy @ one
	num = (Kc * yyc).sum()
	den = torch.sqrt((Kc * Kc).sum() * (yyc * yyc).sum() + 1e-12)
	return num / den


def train_aligned_kernel(fold_data, n_align_steps: int = 50, lr: float = 0.05,
						  C: float = 10.0):
	Xtr_np = fold_data["X_train_pca"]
	Xva_np = fold_data["X_val_pca"]
	ytr = fold_data["y_train"]
	yva = fold_data["y_val"]

	Xtr = torch.tensor(Xtr_np, dtype=torch.float32)
	Xva = torch.tensor(Xva_np, dtype=torch.float32)
	# Map {0,1} -> {-1,+1} for KTA
	y_pm = torch.tensor(np.where(ytr == 1, 1.0, -1.0), dtype=torch.float32)

	theta = torch.nn.Parameter(torch.ones((DEPTH, N_QUBITS), dtype=torch.float32))
	qnode = _build_aligned_qnode()
	opt = torch.optim.Adam([theta], lr=lr)

	print(f"    QKA: optimising alignment, {n_align_steps} steps "
		  f"({len(Xtr)}^2 = {len(Xtr)**2} kernel calls / step)")
	for step in range(n_align_steps):
		opt.zero_grad()
		K = _gram(Xtr, Xtr, theta, qnode)
		loss = -_centered_alignment(K, y_pm)
		loss.backward()
		opt.step()
		if (step + 1) % 10 == 0:
			print(f"      step {step+1}/{n_align_steps}  -KTA={loss.item():.4f}")

	# Fit SVM on the trained kernel
	with torch.no_grad():
		K_tr_final = _gram(Xtr, Xtr, theta, qnode).numpy()
		K_va_final = _gram(Xva, Xtr, theta, qnode).numpy()
	K_tr_final = (K_tr_final + K_tr_final.T) / 2

	svm = SVC(kernel="precomputed", C=C, class_weight="balanced",
			  probability=False, random_state=2026)
	svm.fit(K_tr_final, ytr)
	scores = svm.decision_function(K_va_final)
	preds = svm.predict(K_va_final)

	try:
		auc = roc_auc_score(yva, scores)
	except ValueError:
		auc = 0.5
	print(f"    → QKA AUC: {auc:.4f}")
	return {
		"auc_roc": auc,
		"accuracy": accuracy_score(yva, preds),
		"f1_score": f1_score(yva, preds, zero_division=0),
		"predictions": scores, "pred_binary": preds, "y_true": yva,
	}
