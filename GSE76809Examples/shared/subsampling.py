"""Subsampling helpers shared across v06+ examples.

Provides:
- stratified_subsample : pick exactly n_per_class samples from each class
- fixed_ratio_subsample : pick a total of n samples with a forced pos:neg ratio

These let small-data experiments decouple sample size from class imbalance —
the classic confound when subsampling a 6:1 imbalanced dataset.
"""

from __future__ import annotations

import numpy as np


def stratified_subsample(y, n_per_class: int, rng: np.random.RandomState):
	"""Return indices selecting exactly ``n_per_class`` samples from each class.

	Raises ValueError if any class has fewer than n_per_class samples.
	"""
	y = np.asarray(y)
	classes = np.unique(y)
	chosen = []
	for c in classes:
		idx_c = np.where(y == c)[0]
		if len(idx_c) < n_per_class:
			raise ValueError(
				f"class {c} has only {len(idx_c)} samples but n_per_class={n_per_class}"
			)
		chosen.append(rng.choice(idx_c, n_per_class, replace=False))
	out = np.concatenate(chosen)
	rng.shuffle(out)
	return out


def fixed_ratio_subsample(y, n_total: int, pos_ratio: float,
						   rng: np.random.RandomState, pos_label: int = 1):
	"""Return indices selecting ``n_total`` samples with a forced pos:neg split.

	pos_ratio = fraction of positives in the returned set (e.g. 0.5 = balanced).
	Truncates if a class doesn't have enough samples and warns by raising.
	"""
	y = np.asarray(y)
	n_pos = int(round(n_total * pos_ratio))
	n_neg = n_total - n_pos
	pos_idx = np.where(y == pos_label)[0]
	neg_idx = np.where(y != pos_label)[0]
	if len(pos_idx) < n_pos:
		raise ValueError(f"need {n_pos} positives but only {len(pos_idx)} available")
	if len(neg_idx) < n_neg:
		raise ValueError(f"need {n_neg} negatives but only {len(neg_idx)} available")
	sel_pos = rng.choice(pos_idx, n_pos, replace=False)
	sel_neg = rng.choice(neg_idx, n_neg, replace=False)
	out = np.concatenate([sel_pos, sel_neg])
	rng.shuffle(out)
	return out
