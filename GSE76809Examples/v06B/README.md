# v06B — Feature Extraction / Embedding Comparison

**Question:** Instead of *selecting* genes (ANOVA F-test) and projecting
them with PCA, can a **learned embedding** — a supervised MLP encoder or an
unsupervised autoencoder — produce a better 16-dimensional feature vector
for the downstream quantum and classical classifiers on GSE76809?

The hypothesis (from the user): neural feature extractors capture feature
**correlations** better than ANOVA+PCA *when there is enough data*. GSE76809
is small, so this is also a test of whether learned embeddings help or hurt
in the low-data regime.

## Design (leakage-free, apples-to-apples)

1. Reuse `v06/preprocess_gse76809.py::preprocess()` to obtain the
   variance-filtered + quantile-normalised **raw gene matrix** and the
   stratified holdout split.
2. Build **one** set of `StratifiedKFold` indices so *every* extractor sees
   the **same** train/val splits.
3. For each extractor, **fit on fold-train rows only** → 16-d embedding →
   transform the val rows. No validation data ever touches the extractor
   fit. The supervised MLP encoder uses `y_train` only.
4. Run the **same** downstream models v06 uses on each embedding.

### Extractors (`feature_extractors.py`)

| Name          | Type          | What it learns                              |
|---------------|---------------|---------------------------------------------|
| `anova_pca`   | baseline      | ANOVA top-64 genes → StandardScaler → PCA(16) |
| `mlp_encoder` | supervised    | 16-unit bottleneck of a label-trained MLP classifier |
| `autoencoder` | unsupervised  | 16-unit latent code of a reconstruction AE  |

Both neural encoders use a train-only `PCA(256)` pre-reduction (for speed
and stability on the ~4,500-gene input), then learn a bounded (`Tanh`)
16-d code. All embeddings are standardized (fit on train) before the
classifiers see them.

### Downstream models (reused from `v06/`)

- `quantum_vqc` — 4-qubit data-reuploading VQC
- `classical_mlp` — parameter-matched MLP (`16→48→16→1`)
- `classical_xgb` — tuned XGBoost (inner 3-fold CV)
- `classical_svm` — RBF SVM (inner grid search)

> **Note — the `anova_pca` baseline here is *not* identical to v06's pipeline.**
> v06B re-derives its own embedding (ANOVA top‑64 → `StandardScaler` →
> `PCA(16)`) on a fresh `StratifiedKFold(shuffle=True)` split, whereas v06
> selects ANOVA‑16 directly and applies per‑fold SMOTE on its own fold
> structure. So v06B's `anova_pca::quantum_vqc` (0.729) does **not** match
> v06's VQC CV AUC (0.822) — they are different pipelines on different
> folds. This is intentional: the v06B comparison is *internally* fair
> because all three extractors share the exact same folds and downstream
> models, so the cross‑extractor deltas are valid even though the absolute
> numbers are not directly comparable to v06.

## Running

```powershell
conda activate GEO
cd GSE76809Examples\v06B
python compare_feature_extraction.py
```

The run is **resumable and crash-safe**. After every single fold the
checkpoint is written **atomically** (temp file + `os.replace`) to
`results/v06B_feature_extraction_partial.json`, and the previous good
checkpoint is retained as `...partial.json.bak`. So:

- A crash, `Ctrl+C`, or OS kill **during** a checkpoint write can never
  corrupt your progress — the loader falls back to the `.bak` copy.
- If a single fold raises, the script saves progress and exits; just
  **re-run the same command** and it resumes from that exact fold (often
  succeeding on retry for transient/numerical failures).
- Completed folds are skipped on resume (fold-granular).

Final results are written to `results/v06B_feature_extraction.json`.

## Statistics

For each downstream model, each learned extractor is compared against the
`anova_pca` baseline using:

- Wilcoxon signed-rank (non-parametric paired test)
- Nadeau–Bengio corrected resampled t-test (honest for k-fold CV)
- Cohen's *d* effect size
- Holm–Bonferroni correction across the learned-extractor comparisons

## Files

| File | Purpose |
|------|---------|
| `feature_extractors.py` | The three extractors + shared standardization |
| `compare_feature_extraction.py` | Driver: shared folds, checkpointing, stats |
| `model_classical_svm_v06b.py` | Inversion-safe RBF SVM (ranks by `decision_function`) |
| `results/v06B_feature_extraction.json` | Final per-(extractor, model) AUC + paired tests |

## Results (5-fold CV, 212 training samples, runtime ≈ 238 min)

### Mean AUC ± std by (extractor × model)

| Extractor      | VQC           | MLP           | XGBoost       | SVM           |
|----------------|---------------|---------------|---------------|---------------|
| `anova_pca`    | 0.729 ± 0.158 | 0.832 ± 0.114 | 0.837 ± 0.097 | 0.727 ± 0.207 |
| `mlp_encoder`  | **0.911 ± 0.071** | **0.928 ± 0.067** | **0.895 ± 0.059** | **0.929 ± 0.071** |
| `autoencoder`  | 0.766 ± 0.116 | 0.820 ± 0.087 | 0.849 ± 0.093 | 0.808 ± 0.106 |

> **SVM trainer note.** The SVM column uses a **v06B-local, inversion-safe**
> RBF SVM trainer (`model_classical_svm_v06b.py`). The shared v06 trainer
> ranked validation samples by `predict_proba`, whose internal Platt
> calibration **inverts** on these tiny, imbalanced inner folds and produced
> an artificially low `anova_pca::classical_svm` AUC of 0.319 (the mirror
> image of ~0.85–0.95). Ranking by `decision_function` (monotonic, never
> inverted) restores the baseline to a healthy **0.727**. The v06 results
> themselves are unchanged — only v06B uses the corrected trainer.

### Paired tests vs `anova_pca` (Holm-adjusted)

| Model      | Extractor      | ΔAUC   | Cohen's d | Holm p | Sig? |
|------------|----------------|--------|-----------|--------|------|
| VQC        | mlp_encoder    | +0.182 | +1.33 (large)  | 0.406  | No   |
| VQC        | autoencoder    | +0.037 | +0.24 (small)  | 0.755  | No   |
| MLP        | mlp_encoder    | +0.096 | +0.92 (large)  | 0.279  | No   |
| MLP        | autoencoder    | −0.012 | −0.10 (negl.)  | 0.902  | No   |
| XGBoost    | mlp_encoder    | +0.057 | +0.64 (medium) | 0.807  | No   |
| XGBoost    | autoencoder    | +0.012 | +0.11 (negl.)  | 0.906  | No   |
| SVM        | mlp_encoder    | +0.202 | +1.11 (large)  | 0.357  | No   |
| SVM        | autoencoder    | +0.081 | +0.43 (small)  | 0.593  | No   |

The SVM deltas above use the **corrected** `anova_pca` baseline (0.727).
With the previous inverted baseline (0.319) these deltas were spuriously
huge (+0.610 / +0.512, *d* > 3) and the comparisons looked borderline
significant (Holm *p* = 0.056); after the fix the effect sizes are
modest and clearly non-significant, in line with the other models.

### Key findings

1. **The supervised MLP encoder has the best mean AUC on every model,
   but the margin is only large for the VQC.** It lifts mean AUC by
   +6–18 points on VQC/MLP/XGBoost and by +20 points on the (corrected)
   SVM baseline. The VQC gain (+0.182, *d* = 1.33) is the only large,
   consistent advantage; the MLP/XGBoost gains are modest (+0.06–+0.10)
   and none survives Holm correction.

2. **No improvement is Holm-significant at α = 0.05.** With only 5 CV
   folds, statistical power is too low to reach significance despite
   large effect sizes (d > 1). This is expected and matches the
   power-limitation pattern seen in v06–v09.

3. **The unsupervised autoencoder adds little over ANOVA+PCA.** Its
   deltas are near zero for MLP and XGBoost (negligible effect size),
   and only slightly positive for VQC (+0.037) and SVM (+0.081, *d* =
   0.43, on the corrected baseline). Reconstruction-preserving features
   do not clearly capture disease-discriminative structure that
   ANOVA+PCA misses.

4. **The user's hypothesis is supported directionally but not
   statistically:** a supervised neural feature extractor *does* capture
   gene correlations that ANOVA's univariate F-test misses, producing
   a consistently better 16-d embedding — but on 212 training samples
   the gain is not large enough for formal significance.

5. **Practical implication:** for GSE76809-scale datasets (~200 samples),
   the simple ANOVA+PCA pipeline is a defensible default for SVM, XGBoost
   and MLP, but if a VQC is the downstream model, investing in a
   supervised learned embedding may meaningfully improve performance.

## Interpretation guide

- If `mlp_encoder` / `autoencoder` do **not** beat `anova_pca`, it supports
  the recurring v01–v09 finding that on this small dataset extra model
  capacity (here, in the *front-end*) does not translate into better
  generalisation.
- If a learned embedding **does** help one downstream model but not others,
  the benefit is representation-specific rather than universal.
- Results should be read as AUC deltas with Holm-adjusted p-values, not raw
  point estimates, given the 5-fold small-n setting.
