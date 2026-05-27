"""Plot v08 sample-efficiency curves with 95% CI bands.

Reads results/v08_sample_efficiency.json and writes
results/v08_learning_curves.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

_THIS = Path(__file__).resolve().parent
RESULTS = _THIS / "results" / "v08_sample_efficiency.json"
OUT = _THIS / "results" / "v08_learning_curves.png"


def main():
	if not RESULTS.exists():
		raise SystemExit(f"{RESULTS} not found. Run sample_efficiency.py first.")
	with open(RESULTS) as f:
		data = json.load(f)

	summary = data["summary"]
	fig, ax = plt.subplots(figsize=(9, 6))
	colors = plt.cm.tab10.colors

	for i, (model, cells) in enumerate(sorted(summary.items())):
		ns = sorted(int(k) for k, v in cells.items() if v.get("n_samples", 0) > 0)
		if not ns:
			continue
		means = [cells[str(n)]["mean"] for n in ns]
		lows = [cells[str(n)]["ci_low"] for n in ns]
		highs = [cells[str(n)]["ci_high"] for n in ns]
		xs = [2 * n for n in ns]  # total N (both classes)
		ax.plot(xs, means, marker="o", label=model, color=colors[i % len(colors)])
		ax.fill_between(xs, lows, highs, alpha=0.15, color=colors[i % len(colors)])

	ax.set_xscale("log")
	ax.set_xlabel("Training samples (1:1 stratified)")
	ax.set_ylabel("Test AUC on fixed v06 holdout (54 samples)")
	ax.set_title("v08 — Sample-Efficiency Curves with 95% CI")
	ax.axhline(0.5, color="grey", linestyle=":", label="chance")
	ax.set_ylim(0.3, 1.02)
	ax.grid(alpha=0.3)
	ax.legend(loc="lower right", fontsize=9)
	fig.tight_layout()
	fig.savefig(OUT, dpi=150)
	print(f"Saved: {OUT}")


if __name__ == "__main__":
	main()
