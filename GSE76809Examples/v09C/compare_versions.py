"""Compare v09B (layer-0 encoding) vs v09C (every-layer encoding).

Loads both versions' results JSONs and reports, per encoding:
  - mean AUC for each version
  - the difference Δ = v09C − v09B
  - Cohen's d and a Wilcoxon signed-rank paired test across the 5 CV folds

This answers the practical question: does applying the upstream encoding at
EVERY layer (v09C) actually produce different results from applying it at
layer 0 only (v09B)?

Run both ablations first:
    python v09B/compare_encodings.py
    python v09C/compare_encodings.py
then:
    python v09C/compare_versions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))  # shared

from shared.stats_utils import (  # noqa: E402
    wilcoxon_signed_rank, cohens_d, effect_size_label,
)

B_PATH = _THIS.parent / "v09B" / "results" / "v09B_encoding_ablation.json"
C_PATH = _THIS / "results" / "v09C_encoding_ablation.json"


def _load(path: Path, label: str) -> dict:
    if not path.exists():
        raise SystemExit(
            f"{label} results not found at {path}\n"
            f"Run the {label} ablation first (python {label}/compare_encodings.py)."
        )
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    b = _load(B_PATH, "v09B")
    c = _load(C_PATH, "v09C")

    print("=" * 78)
    print("v09B (encoding @ layer 0)  vs  v09C (encoding @ every layer)")
    print("=" * 78)
    header = (
        f"{'encoding':<22}{'v09B':>8}{'v09C':>8}"
        f"{'Δ(C-B)':>9}{'d':>7}  paired-test"
    )
    print(header)
    print("-" * 78)

    rows = []
    for enc in c["cv"]:
        if enc not in b["cv"]:
            continue
        ba = np.asarray(b["cv"][enc]["aucs"])
        ca = np.asarray(c["cv"][enc]["aucs"])
        if len(ba) == 0 or len(ca) == 0 or len(ba) != len(ca):
            print(f"{enc:<22}  (incomplete folds — skipped)")
            continue
        _, p = wilcoxon_signed_rank(ca, ba)
        d = cohens_d(ca, ba)
        delta = ca.mean() - ba.mean()
        print(
            f"{enc:<22}{ba.mean():>8.3f}{ca.mean():>8.3f}"
            f"{delta:>+9.3f}{d:>+7.2f}  p={p:.3f} ({effect_size_label(d)})"
        )
        rows.append({
            "encoding":      enc,
            "v09B_mean_auc": float(ba.mean()),
            "v09C_mean_auc": float(ca.mean()),
            "delta_c_minus_b": float(delta),
            "cohens_d":      float(d),
            "effect_size":   effect_size_label(d),
            "wilcoxon_p":    float(p),
        })

    print("-" * 78)
    print("Δ > 0  →  every-layer encoding (v09C) helps that encoding")
    print("Δ < 0  →  layer-0-only encoding (v09B) is better for that encoding")
    print("Watch amplitude_combined: every-layer AmplitudeEmbedding overwrites")
    print("the statevector each layer, so a large negative Δ is expected there.")

    out_path = _THIS / "results" / "v09B_vs_v09C_comparison.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"comparison": rows}, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
