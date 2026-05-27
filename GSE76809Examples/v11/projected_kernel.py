"""Projected Quantum Kernel (Huang et al. 2021).

For each input x:
  1. Build the same ZZ feature-map state |φ(x)⟩ as v06.
  2. Measure single-qubit Pauli expectations (X, Y, Z) on each qubit.
  3. Stack them into a classical feature vector phi(x) of length 3 * n_qubits.
Then train a classical RBF SVM on {phi(x_i)}.

The key insight: instead of an inner-product kernel
``|⟨φ(x)|φ(y)⟩|²`` (which exponentially concentrates on small data),
PQK extracts a low-dimensional summary of |φ(x)⟩ and applies a
well-behaved classical kernel on top. This is the construction that
Huang et al. show has provable separations from classical kernels.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

N_QUBITS = 6
DEPTH = 3


def _build_feature_extractor():
	dev = qml.device("default.qubit", wires=N_QUBITS)

	@qml.qnode(dev, interface="numpy")
	def extract(x):
		# Same ZZ feature map as v06.model_quantum_kernel
		for _ in range(DEPTH):
			for q in range(N_QUBITS):
				qml.Hadamard(wires=q)
			for q in range(N_QUBITS):
				qml.RZ(x[q], wires=q)
			for i in range(N_QUBITS):
				for j in range(i + 1, N_QUBITS):
					qml.CNOT(wires=[i, j])
					qml.RZ(x[i] * x[j], wires=j)
					qml.CNOT(wires=[i, j])
		out = []
		for q in range(N_QUBITS):
			out.append(qml.expval(qml.PauliX(q)))
		for q in range(N_QUBITS):
			out.append(qml.expval(qml.PauliY(q)))
		for q in range(N_QUBITS):
			out.append(qml.expval(qml.PauliZ(q)))
		return out

	return extract


def _features(X, extractor):
	return np.asarray([extractor(x) for x in X])


def train_projected_kernel(fold_data, C_values=None):
	Xtr, Xva = fold_data["X_train_pca"], fold_data["X_val_pca"]
	ytr, yva = fold_data["y_train"], fold_data["y_val"]

	print(f"    PQK: extracting features ({len(Xtr)} train + {len(Xva)} val)...")
	extractor = _build_feature_extractor()
	phi_tr = _features(Xtr, extractor)
	phi_va = _features(Xva, extractor)

	# Classical RBF on the projected features (this is the "Q" part of PQK —
	# the feature extractor used a quantum circuit; the kernel itself is
	# classical and so behaves well on small data).
	if C_values is None:
		C_values = [0.1, 1.0, 10.0, 100.0]
	grid = GridSearchCV(
		SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=2026),
		{"C": C_values, "gamma": ["scale", "auto", 0.1, 1.0]},
		cv=3, scoring="roc_auc", n_jobs=-1,
	)
	grid.fit(phi_tr, ytr)
	clf = grid.best_estimator_
	probs = clf.predict_proba(phi_va)[:, 1]
	preds = clf.predict(phi_va)

	try:
		auc = roc_auc_score(yva, probs)
	except ValueError:
		auc = 0.5
	print(f"    → PQK AUC: {auc:.4f}")
	return {
		"auc_roc": auc,
		"accuracy": accuracy_score(yva, preds),
		"f1_score": f1_score(yva, preds, zero_division=0),
		"predictions": probs, "pred_binary": preds, "y_true": yva,
	}
