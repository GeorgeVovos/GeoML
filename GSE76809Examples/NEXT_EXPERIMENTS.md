# Suggested Next Experiments — v06, v06B, v07, v08, v09

## Cross-cutting fairness issue: the classical post-net

The VQC is not a "pure" quantum classifier — it uses a **classical post-net
(`12 → 64 → 1`, ~830 params)** on top of 12 measured expectations. Any
"quantum vs classical" comparison is partly "quantum features + MLP head vs
standalone classical model."

**Suggested control (reusable across v06/v09):**
- Feed 12 **classical random-projection features** (or PCA-12) into the
  *identical* `12→64→1` head.
- If the quantum-feature head beats the random-feature head at equal
  capacity, *that* is a defensible quantum signal.

---

## v06 — Tuned baselines

### Fairness fixes
1. **Match the classifier head, not just parameter count.** Give the MLP
   baseline the *same* `12→64→1` head fed by 12 classical features (e.g.
   PCA-12 or a random projection of the 16 inputs). Currently
   "parameter-matched" matches count but not architecture/role.
2. **Report holdout AUC with a bootstrap CI.** With n=54 holdout, VQC
   0.905 vs MLP 0.902 is within noise — a 1000× bootstrap CI on the AUC
   difference would make "essentially tied" quantitative.
3. **Nested CV for classical tuning.** XGBoost/SVM grid search should use
   a separate inner loop so their tuning advantage isn't on the same folds
   the VQC is evaluated on.

### Quantum-advantage angle
- **Lowest-variance is your real story.** VQC std=0.069 vs XGB std=0.101.
  Test this formally (Levene or Brown–Forsythe test on per-fold AUCs, or
  bootstrap the std) and frame stability as the quantum niche.

---

## v06B — Feature extraction / embeddings

### Fairness fixes
1. **Add a learned *quantum* embedding** to the extractor set. Currently
   all front-ends are classical. Add a VQC-derived 16-d embedding (the 12
   measurements + a small projection) so "learned embedding" includes a
   quantum option.
2. **Decouple encoder capacity from supervision.** The supervised
   MLP-encoder wins largely because it's *supervised*. Add a **supervised
   linear extractor** (e.g. LDA or a linear bottleneck) to isolate
   "non-linearity" from "supervision."
3. **Add a `anova_pca_v06` variant** that exactly replicates v06's pipeline
   (ANOVA-16, per-fold SMOTE) so absolute numbers are directly comparable
   to v06.

### Quantum-advantage angle
- **Quantum kernel embedding fairness:** if a quantum feature map is added,
  compare it against a classical **RBF random-feature map** of equal
  dimension on the same downstream model — a clean "does the quantum
  Hilbert space help?" test.

---

## v07 — Cross-dataset replication

### Fairness fixes
1. **Actually run the other three datasets** (currently placeholders).
   A single-dataset "replication" cannot support generalization. This is
   the single biggest credibility upgrade in the entire suite.
2. **Harmonize preprocessing across datasets** (same gene-count selection,
   same fold seeds) so cross-dataset deltas aren't preprocessing artifacts.

### Quantum-advantage angle
- **Look for a consistency signal across datasets**, not a per-dataset win:
  "VQC is top-2 on N/4 datasets with the lowest variance" is a more
  believable quantum narrative than one big win on one dataset.

---

## v08 — Sample-efficiency sweep

### Fairness fixes
1. **Include the quantum kernel** (skipped for runtime). Excluding it
   biases the small-data comparison — either include it or explicitly
   scope the claim to "VQC vs classical."
2. **Push to ≥20 seeds** at N=10 and N=20 with the same fixed test set
   and report **per-seed paired** differences so variance is captured
   properly.

### Quantum-advantage angle — **best candidate for a genuine edge**
- v06 already showed VQC holds AUC 0.683 at 10% data while XGB collapses
  to 0.500. Make *that* the headline experiment:
  - Report **learning-curve crossover points** with bootstrap CIs.
  - Try a **feature-to-sample ratio stress test**: hold N small, increase
	feature count, where the VQC's implicit regularisation may help.
  - Consider adding the quantum kernel at these tiny sizes (runtime is
	manageable when N is very small).

---

## v09 — Encoding ablation

### Fairness fixes
1. **Equalize encoding parameter budgets.** `data_reuploading` has
   *learned* encoding weights at every layer (`Σ w[l,q,i]·x[i]`), while
   `angle`/`amplitude`/`iqp` have **no trainable encoding params**. You are
   partly measuring "more trainable parameters win." Add:
   - A **single-layer re-uploading** variant (learned weights, but only
	 at layer 0 — same budget as angle).
   - A **fixed-weight re-uploading** variant (re-upload at every layer, but
	 with random frozen projections — same layer-wise structure, no learned
	 encoding weights).
2. **Add a classical-projection control:** replace the quantum block with a
   random/learned `16→12` linear map into the same `12→64→1` post-net.
   If `data_reuploading` doesn't beat that, the "advantage" is the
   post-net, not the quantum circuit.

### Quantum-advantage angle
- **Increase folds / repeats.** The only significant result (amplitude,
  p=0.049) is borderline on 5 folds. **10× repeated 5-fold CV** (50 total
  evaluations) would give the Holm correction real power and could turn
  the large effect sizes (d>1.3) into defensible significance —
  strengthening the "encoding matters" claim that underpins the suite.

---

## Priority ranking (if only doing a few)

| Priority | Experiment | Why |
|----------|-----------|-----|
| 1 | **v09 + cross-cutting:** classical-projection-into-same-post-net control + fixed-weight re-uploading variant | Directly tests whether the advantage is quantum or just the head/extra params. Highest scientific payoff. |
| 2 | **v08:** rigorous small-data robustness (≥20 seeds, include kernel, crossover CIs) | Best shot at a *real*, defensible quantum edge. |
| 3 | **v07:** finish the other three datasets | Turns an open question into evidence; high credibility impact. |
| 4 | **v06:** bootstrap CIs + formal variance test | Low effort, immediately strengthens the "VQC = stable" claim. |
| 5 | **v06B:** add supervised-linear + quantum embedding extractors | Disambiguates the MLP-encoder win and adds a quantum front-end option. |
