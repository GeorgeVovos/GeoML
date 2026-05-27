"""Statistical-test utilities shared across v06+ examples.

Includes:
- cohens_d                  — paired effect size
- mcnemars_test             — McNemar with exact-binomial fallback
- wilcoxon_signed_rank      — non-parametric paired test (low-power-safe)
- holm_bonferroni           — multiple-comparison correction (Holm's step-down)
- corrected_resampled_ttest — Nadeau-Bengio corrected t-test for k-fold CV

These give a more honest small-n analysis than the bare paired t-test that
v04-v06 used (which has near-zero power on 5 CV folds).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Effect sizes
# --------------------------------------------------------------------------- #

def cohens_d(group1, group2) -> float:
	"""Cohen's d (pooled). Suitable for paired or independent samples here."""
	g1 = np.asarray(group1, dtype=float)
	g2 = np.asarray(group2, dtype=float)
	n1, n2 = len(g1), len(g2)
	if n1 < 2 or n2 < 2:
		return 0.0
	var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
	pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
	if pooled == 0:
		return 0.0
	return float((g1.mean() - g2.mean()) / pooled)


def effect_size_label(d: float) -> str:
	a = abs(d)
	if a < 0.2:
		return "negligible"
	if a < 0.5:
		return "small"
	if a < 0.8:
		return "medium"
	return "large"


# --------------------------------------------------------------------------- #
# Paired tests
# --------------------------------------------------------------------------- #

def mcnemars_test(y_true, preds_a, preds_b) -> float:
	"""Two-sided McNemar test; exact binomial when (b+c) < 25."""
	y_true = np.asarray(y_true)
	preds_a = np.asarray(preds_a)
	preds_b = np.asarray(preds_b)
	correct_a = (preds_a == y_true).astype(int)
	correct_b = (preds_b == y_true).astype(int)
	b = int(np.sum((correct_a == 1) & (correct_b == 0)))
	c = int(np.sum((correct_a == 0) & (correct_b == 1)))
	n = b + c
	if n == 0:
		return 1.0
	if n < 25:
		try:
			return float(stats.binomtest(min(b, c), n=n, p=0.5,
										 alternative="two-sided").pvalue)
		except AttributeError:  # scipy < 1.7
			return float(stats.binom_test(min(b, c), n=n, p=0.5))
	statistic = (abs(b - c) - 1) ** 2 / n
	return float(1 - stats.chi2.cdf(statistic, df=1))


def wilcoxon_signed_rank(a, b) -> tuple[float, float]:
	"""Wilcoxon signed-rank test on paired CV scores.

	Use this *instead of* paired t-test when the per-fold differences are not
	plausibly Gaussian — which is essentially always with 5-10 folds.

	Returns (statistic, p_value). If all differences are zero (degenerate),
	returns (0.0, 1.0) instead of raising.
	"""
	a = np.asarray(a, dtype=float)
	b = np.asarray(b, dtype=float)
	if np.allclose(a, b):
		return 0.0, 1.0
	try:
		result = stats.wilcoxon(a, b, zero_method="wilcox",
								alternative="two-sided", mode="auto")
	except ValueError:
		return 0.0, 1.0
	return float(result.statistic), float(result.pvalue)


def corrected_resampled_ttest(a, b, n_train, n_test) -> tuple[float, float]:
	"""Nadeau-Bengio corrected resampled t-test for k-fold CV scores.

	The vanilla paired t-test on k CV folds underestimates variance because
	the folds share most of their training data. Nadeau & Bengio (2003)
	proposed the correction below; it is the standard for comparing learners
	via k-fold CV. With k=5 it gives genuinely conservative p-values, unlike
	the raw paired t-test.
	"""
	a = np.asarray(a, dtype=float)
	b = np.asarray(b, dtype=float)
	diffs = a - b
	k = len(diffs)
	if k < 2:
		return 0.0, 1.0
	mean = diffs.mean()
	var = diffs.var(ddof=1)
	if var == 0:
		return 0.0, 1.0
	correction = (1.0 / k) + (n_test / max(n_train, 1))
	se = np.sqrt(correction * var)
	if se == 0:
		return 0.0, 1.0
	t = mean / se
	p = 2.0 * (1.0 - stats.t.cdf(abs(t), df=k - 1))
	return float(t), float(p)


# --------------------------------------------------------------------------- #
# Multiple-comparison correction
# --------------------------------------------------------------------------- #

def holm_bonferroni(p_values):
	"""Holm's step-down correction. Returns a list of adjusted p-values
	(clipped to [0, 1]) in the original input order.

	More powerful than plain Bonferroni; controls family-wise error rate.
	"""
	p = np.asarray(p_values, dtype=float)
	n = len(p)
	if n == 0:
		return []
	order = np.argsort(p)
	adjusted = np.empty(n, dtype=float)
	running_max = 0.0
	for rank, idx in enumerate(order):
		scaled = (n - rank) * p[idx]
		running_max = max(running_max, scaled)
		adjusted[idx] = min(running_max, 1.0)
	return adjusted.tolist()
