"""Decision-curve analysis (Vickers & Elkin 2006).

Net benefit at threshold t:
	NB(t) = TP/N - (FP/N) * (t / (1 - t))

Returned as arrays over a threshold grid. Comparisons:
- "treat all" baseline: NB(t) = prevalence - (1 - prevalence) * t/(1-t)
- "treat none" baseline: NB(t) = 0
A model is clinically useful at threshold t iff its NB(t) exceeds both.
"""

from __future__ import annotations

import numpy as np


def net_benefit(y_true, y_score, thresholds=None):
	if thresholds is None:
		thresholds = np.linspace(0.01, 0.99, 99)
	y_true = np.asarray(y_true).astype(int)
	y_score = np.asarray(y_score, dtype=float)
	N = len(y_true)
	prev = y_true.mean()
	nb = np.zeros_like(thresholds, dtype=float)
	for i, t in enumerate(thresholds):
		pred_pos = y_score >= t
		tp = np.sum(pred_pos & (y_true == 1))
		fp = np.sum(pred_pos & (y_true == 0))
		nb[i] = (tp / N) - (fp / N) * (t / max(1 - t, 1e-9))
	nb_all = prev - (1 - prev) * thresholds / np.clip(1 - thresholds, 1e-9, None)
	return {
		"thresholds": thresholds,
		"net_benefit": nb,
		"net_benefit_treat_all": nb_all,
		"prevalence": float(prev),
	}
