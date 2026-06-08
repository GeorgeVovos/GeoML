"""v09C per-layer reuploading combination ablation driver.

Runs 5-fold CV for each of five encoding experiments:
  data_reuploading      — pure learned reuploading, no upstream gates (canonical ref)
  angle_combined        — angle encoding every layer + learned reuploading every layer
  dense_angle_combined  — dense-angle encoding every layer + learned reuploading
  amplitude_combined    — amplitude encoding every layer + learned reuploading
  iqp_combined          — IQP feature map every layer + learned reuploading

Identical to v09B in preprocessing, variational stack, post-net, optimiser,
schedule, CV folds and seed. The ONLY difference is that the upstream encoding
is applied at EVERY layer (v09C) rather than at layer 0 only (v09B).

Resumable: after every encoding/fold a partial checkpoint is written. On
re-launch the checkpoint is loaded and completed fold/encoding combos are
skipped automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent / "v06"))   # preprocess_gse76809
sys.path.insert(0, str(_THIS.parent))           # shared
sys.path.insert(0, str(_THIS))                  # v09C local imports

from shared.stats_utils import (                # noqa: E402
    cohens_d, effect_size_label, wilcoxon_signed_rank,
    corrected_resampled_ttest, holm_bonferroni,
)
from preprocess_gse76809 import preprocess      # noqa: E402
from model_quantum_vqc import train_quantum_vqc  # noqa: E402
from quantum_encodings import UPSTREAM_GATES    # noqa: E402

# Canonical order: reference first, then the four hybrid combinations
DEFAULT_ENCODINGS = list(UPSTREAM_GATES.keys())
REFERENCE_ENCODING = "data_reuploading"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="v09C: per-layer upstream encoding + learned reuploading ablation"
    )
    ap.add_argument(
        "--encodings", nargs="+", default=DEFAULT_ENCODINGS,
        help="subset of encodings to run (default: all five)",
    )
    args = ap.parse_args()

    bad = [e for e in args.encodings if e not in UPSTREAM_GATES]
    if bad:
        raise SystemExit(f"unknown encodings {bad}; known: {list(UPSTREAM_GATES)}")

    start_time = time.time()
    data = preprocess()

    out_dir = _THIS / "results"
    out_dir.mkdir(exist_ok=True)
    checkpoint_path = out_dir / "v09C_encoding_ablation_partial.json"

    # ── Resume: load partial checkpoint if present ────────────────────────────
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            cv = json.load(f)
        print(f"Resuming from checkpoint: {checkpoint_path}")
        for enc in args.encodings:
            if enc not in cv:
                cv[enc] = {"aucs": [], "accs": [], "f1s": []}
    else:
        cv = {enc: {"aucs": [], "accs": [], "f1s": []} for enc in args.encodings}

    # ── Main CV loop ──────────────────────────────────────────────────────────
    for fold_idx, fold in enumerate(data["folds"]):
        print(f"\n--- Fold {fold_idx + 1}/{len(data['folds'])} ---")
        for enc in args.encodings:
            if len(cv[enc]["aucs"]) > fold_idx:
                print(f"  [resume] Skipping {enc} fold {fold_idx + 1}")
                continue
            res = train_quantum_vqc(fold_data=fold, encoding=enc)
            cv[enc]["aucs"].append(res["auc_roc"])
            cv[enc]["accs"].append(res["accuracy"])
            cv[enc]["f1s"].append(res["f1_score"])
            with open(checkpoint_path, "w") as f:
                json.dump(cv, f, indent=2, cls=NumpyEncoder)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\nPer-encoding 5-fold AUC:")
    for enc, r in sorted(
        cv.items(), key=lambda kv: np.mean(kv[1]["aucs"]), reverse=True
    ):
        print(
            f"  {enc:<25}  mean={np.mean(r['aucs']):.4f}"
            f"  +/- {np.std(r['aucs']):.4f}"
        )

    # ── Paired tests vs canonical reference (data_reuploading) ───────────────
    stats_rows: list[dict] = []
    if REFERENCE_ENCODING in cv:
        ref_aucs = np.asarray(cv[REFERENCE_ENCODING]["aucs"])
        raw_ps: list[float] = []
        for enc in args.encodings:
            if enc == REFERENCE_ENCODING:
                continue
            a = np.asarray(cv[enc]["aucs"])
            _, wp  = wilcoxon_signed_rank(a, ref_aucs)
            _, nbp = corrected_resampled_ttest(
                a, ref_aucs,
                n_train=len(data["X_train"]) * 4 // 5,
                n_test=len(data["X_train"]) // 5,
            )
            d = cohens_d(a, ref_aucs)
            stats_rows.append({
                "encoding":      enc,
                "vs":            REFERENCE_ENCODING,
                "mean_diff":     float(a.mean() - ref_aucs.mean()),
                "wilcoxon_p":    wp,
                "nb_corrected_p": nbp,
                "cohens_d":      d,
                "effect_size":   effect_size_label(d),
            })
            raw_ps.append(nbp)

        for row, adj in zip(stats_rows, holm_bonferroni(raw_ps)):
            row["holm_adjusted_p"]  = adj
            row["holm_significant"] = adj < 0.05

        print(f"\nPaired tests vs {REFERENCE_ENCODING} (Holm-adjusted):")
        for row in stats_rows:
            flag = " ***" if row["holm_significant"] else ""
            print(
                f"  {row['encoding']:<25}  d={row['cohens_d']:+.2f}"
                f"  p_adj={row['holm_adjusted_p']:.4f}{flag}"
            )

    # ── Write final results JSON ──────────────────────────────────────────────
    elapsed_min = (time.time() - start_time) / 60
    out = {
        "metadata": {
            "version":   "v09C",
            "purpose":   (
                "ablation: upstream encoding applied at EVERY layer combined with "
                "learned per-layer data-reuploading vs pure data_reuploading "
                "(v09C counterpart of v09B's layer-0-only encoding)"
            ),
            "encodings":        args.encodings,
            "seed":             2026,
            "runtime_minutes":  elapsed_min,
        },
        "cv": {
            enc: {
                "aucs":     r["aucs"],
                "mean_auc": float(np.mean(r["aucs"])),
                "std_auc":  float(np.std(r["aucs"])),
            }
            for enc, r in cv.items()
        },
        "paired_stats_vs_data_reuploading": stats_rows,
    }
    out_path = out_dir / "v09C_encoding_ablation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved {out_path}  (runtime {elapsed_min:.1f} min)")


if __name__ == "__main__":
    main()
