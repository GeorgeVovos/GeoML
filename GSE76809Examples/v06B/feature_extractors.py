"""Feature-extraction / embedding methods for v06B.

Compares three ways of turning the ~4,500 variance-filtered, quantile-
normalised genes into a 16-dimensional feature vector that the downstream
quantum / classical classifiers consume:

	"anova_pca"     : ANOVA F-test top-k -> StandardScaler -> PCA(16)
					  (the v06 baseline, here producing 16 PCA comps)
	"mlp_encoder"   : Supervised MLP classifier whose penultimate layer is
					  a 16-unit bottleneck; the bottleneck activations are
					  the embedding. Learns label-correlated features.
	"autoencoder"   : Unsupervised autoencoder with a 16-unit latent code;
					  the encoder output is the embedding. Learns
					  reconstruction-preserving correlated features.

CRITICAL: every extractor is FIT ON THE FOLD-TRAIN ROWS ONLY and then
applied to the validation rows, so there is no leakage. The supervised
MLP encoder uses y_train only (never y_val).

All extractors return (X_train_emb, X_val_emb) as float32 arrays scaled
so the embedding columns are comparable to the v06 ANOVA features.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler

EMB_DIM = 16


# --------------------------------------------------------------------------- #
# Baseline: ANOVA + PCA
# --------------------------------------------------------------------------- #

def anova_pca_extract(X_train_raw, y_train, X_val_raw, n_select=64,
					  emb_dim=EMB_DIM, random_state=2026):
	"""ANOVA top-`n_select` genes -> StandardScaler -> PCA(emb_dim).

	Mirrors the v06 selection+projection idea but outputs `emb_dim` PCA
	components so it is dimensionally comparable to the learned encoders.
	"""
	f_scores, _ = f_classif(X_train_raw, y_train)
	f_scores = np.nan_to_num(f_scores, nan=0.0)
	n_select = min(n_select, X_train_raw.shape[1])
	top_idx = np.argsort(f_scores)[-n_select:]

	X_tr = X_train_raw[:, top_idx]
	X_va = X_val_raw[:, top_idx]

	scaler = StandardScaler()
	X_tr = scaler.fit_transform(X_tr)
	X_va = scaler.transform(X_va)

	n_comp = min(emb_dim, X_tr.shape[1], X_tr.shape[0])
	pca = PCA(n_components=n_comp, random_state=random_state)
	X_tr_emb = pca.fit_transform(X_tr)
	X_va_emb = pca.transform(X_va)

	# Pad to emb_dim if PCA produced fewer comps (small folds)
	if X_tr_emb.shape[1] < emb_dim:
		pad = emb_dim - X_tr_emb.shape[1]
		X_tr_emb = np.hstack([X_tr_emb, np.zeros((X_tr_emb.shape[0], pad))])
		X_va_emb = np.hstack([X_va_emb, np.zeros((X_va_emb.shape[0], pad))])

	return _standardize(X_tr_emb, X_va_emb)


# --------------------------------------------------------------------------- #
# Supervised MLP encoder
# --------------------------------------------------------------------------- #

class _MLPEncoderNet(nn.Module):
	"""Classifier with a 16-unit bottleneck used as the embedding layer."""

	def __init__(self, n_in, emb_dim=EMB_DIM):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Linear(n_in, 128),
			nn.GELU(),
			nn.Dropout(0.3),
			nn.Linear(128, 64),
			nn.GELU(),
			nn.Dropout(0.3),
			nn.Linear(64, emb_dim),
			nn.Tanh(),            # bounded embedding
		)
		self.head = nn.Sequential(
			nn.Linear(emb_dim, 1),
			nn.Sigmoid(),
		)

	def forward(self, x):
		z = self.encoder(x)
		return self.head(z).squeeze(-1), z


def mlp_encoder_extract(X_train_raw, y_train, X_val_raw, emb_dim=EMB_DIM,
						epochs=120, lr=1e-3, batch_size=24,
						random_state=2026):
	"""Train a supervised classifier; return its bottleneck activations.

	Pre-reduces the wide gene matrix with a train-only PCA(256) for speed
	and stability, then learns a label-correlated 16-d bottleneck.
	"""
	torch.manual_seed(int(random_state))
	np.random.seed(int(random_state))

	# Train-only PCA pre-reduction to tame the ~4,500-dim input
	pre = StandardScaler()
	X_tr = pre.fit_transform(X_train_raw)
	X_va = pre.transform(X_val_raw)
	n_pre = min(256, X_tr.shape[1], X_tr.shape[0])
	pca = PCA(n_components=n_pre, random_state=random_state)
	X_tr = pca.fit_transform(X_tr)
	X_va = pca.transform(X_va)

	X_tr_t = torch.FloatTensor(X_tr)
	y_tr_t = torch.FloatTensor(y_train.astype(float))
	X_va_t = torch.FloatTensor(X_va)

	model = _MLPEncoderNet(n_in=X_tr.shape[1], emb_dim=emb_dim)
	opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
	sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
	criterion = nn.BCELoss()

	for _ in range(epochs):
		model.train()
		perm = torch.randperm(len(X_tr_t))
		Xs, ys = X_tr_t[perm], y_tr_t[perm]
		for start in range(0, len(Xs), batch_size):
			xb = Xs[start:start + batch_size]
			yb = ys[start:start + batch_size]
			opt.zero_grad()
			preds, _ = model(xb)
			loss = criterion(preds, yb)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			opt.step()
		sched.step()

	model.eval()
	with torch.no_grad():
		_, z_tr = model(X_tr_t)
		_, z_va = model(X_va_t)

	return _standardize(z_tr.numpy(), z_va.numpy())


# --------------------------------------------------------------------------- #
# Unsupervised autoencoder
# --------------------------------------------------------------------------- #

class _AutoEncoderNet(nn.Module):
	"""Symmetric autoencoder with a 16-unit latent code."""

	def __init__(self, n_in, emb_dim=EMB_DIM):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Linear(n_in, 128),
			nn.GELU(),
			nn.Linear(128, 64),
			nn.GELU(),
			nn.Linear(64, emb_dim),
			nn.Tanh(),
		)
		self.decoder = nn.Sequential(
			nn.Linear(emb_dim, 64),
			nn.GELU(),
			nn.Linear(64, 128),
			nn.GELU(),
			nn.Linear(128, n_in),
		)

	def forward(self, x):
		z = self.encoder(x)
		return self.decoder(z), z


def autoencoder_extract(X_train_raw, y_train, X_val_raw, emb_dim=EMB_DIM,
						epochs=150, lr=1e-3, batch_size=24,
						random_state=2026):
	"""Train an unsupervised AE; return its latent code as the embedding.

	y_train is accepted for a uniform signature but NOT used (unsupervised).
	A train-only PCA(256) pre-reduction matches the MLP encoder for a fair
	comparison.
	"""
	torch.manual_seed(int(random_state))
	np.random.seed(int(random_state))

	pre = StandardScaler()
	X_tr = pre.fit_transform(X_train_raw)
	X_va = pre.transform(X_val_raw)
	n_pre = min(256, X_tr.shape[1], X_tr.shape[0])
	pca = PCA(n_components=n_pre, random_state=random_state)
	X_tr = pca.fit_transform(X_tr)
	X_va = pca.transform(X_va)

	X_tr_t = torch.FloatTensor(X_tr)
	X_va_t = torch.FloatTensor(X_va)

	model = _AutoEncoderNet(n_in=X_tr.shape[1], emb_dim=emb_dim)
	opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
	sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
	criterion = nn.MSELoss()

	for _ in range(epochs):
		model.train()
		perm = torch.randperm(len(X_tr_t))
		Xs = X_tr_t[perm]
		for start in range(0, len(Xs), batch_size):
			xb = Xs[start:start + batch_size]
			opt.zero_grad()
			recon, _ = model(xb)
			loss = criterion(recon, xb)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			opt.step()
		sched.step()

	model.eval()
	with torch.no_grad():
		_, z_tr = model(X_tr_t)
		_, z_va = model(X_va_t)

	return _standardize(z_tr.numpy(), z_va.numpy())


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _standardize(X_tr_emb, X_va_emb):
	"""Standardize embedding columns (fit on train) so downstream models
	and the VQC's data-reuploading encoding see comparable scales."""
	scaler = StandardScaler()
	X_tr = scaler.fit_transform(X_tr_emb).astype(np.float32)
	X_va = scaler.transform(X_va_emb).astype(np.float32)
	return X_tr, X_va


EXTRACTORS = {
	"anova_pca": anova_pca_extract,
	"mlp_encoder": mlp_encoder_extract,
	"autoencoder": autoencoder_extract,
}
