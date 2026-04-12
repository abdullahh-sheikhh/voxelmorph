# Dataset Constraints and Limitations

All the ways this dataset constrains what we can claim. These are limits of the data, not of any algorithm. Read before designing experiments.

---

## 1. Scale and Structure Limitations

### 1.1 Only two sequences
- The entire dataset consists of 2 sequences. Everything we measure is conditional on these 2 recordings.
- There is no independent third sequence to serve as a held-out generalization test.
- Any train/val/test split drawn from these 2 sequences only tests generalization across *frames* (and only within the same imaging conditions, cell line, microscope, and field of view), not across experiments.

### 1.2 Small number of cells per frame
- Per-frame cell count ranges from **3 to 8**.
- Per-sequence unique cell count: 8 (seq01), 12 (seq02).
- Cell-level statistics averaged over a single pair have extreme variance — the mean of 3–8 values.
- Pooling cells across pairs raises N but correlates observations, because the same cell appears in many consecutive pairs.

### 1.3 Short temporal span
- 115 frames per sequence × 15 minutes = ~28.75 hours of imaging.
- Many cell lifetimes are shorter (e.g. seq02 ID 5 lives only 4 frames; ID 7 lives 7 frames).
- Short-lived cells contribute few pair observations and their behaviour dominates if weighted equally with long-lived cells.

### 1.4 Only 228 total pairs
- This is the absolute ceiling on any consecutive-pair evaluation.
- 212 of them are "stable" (same cell set on both sides), 16 are "transition" pairs.
- On GT/SEG this collapses to 5 pairs — not a sample size.

---

## 2. Statistical Weaknesses

### 2.1 No independent held-out data
- Any model trained on these frames and evaluated on these frames shares the same underlying imaging conditions. Claims about "accuracy" apply only to these conditions.
- Even a sequence-held-out protocol (train on seq01, evaluate on seq02) only gives one test point.

### 2.2 Correlated samples
- Consecutive pairs share cells and share slow-moving background. Observation `pair(t, t+1)` and `pair(t+1, t+2)` are not independent.
- Confidence intervals computed under an IID assumption are optimistic. True effective sample size is lower than the count of pairs.

### 2.3 Small cell count per pair inflates per-pair variance
- A per-pair mean over 3–8 cells has high variance by construction. Do not over-interpret per-pair fluctuations.
- Robust summaries: pool all matched cells across all pairs (report N_cells, not just N_pairs), and report dispersion.

### 2.4 No ground-truth displacement field
- There is **no per-pixel ground truth** for the displacement. Only per-cell annotations exist.
- Evaluation is therefore *structural* (does the warp move cell pixels onto cell pixels correctly?) rather than pointwise (does the warp reproduce the true (u, v) at every pixel?).
- Sub-pixel accuracy cannot be verified from this dataset alone.

---

## 3. Annotation Imperfections

### 3.1 GT/SEG sparsity
- 15/115 frames (seq01), 19/115 frames (seq02). Not enough for any dense temporal analysis.
- Exists only at irregular intervals. Most consecutive pairs have GT/SEG on zero or one side.

### 3.2 GT/SEG seq02 is not a tracking scheme
- Seq02 labels are per-frame spatial indices. The same integer refers to different physical cells at different frames.
- Any assumption of the form "label *k* at frame *s* = label *k* at frame *t*" is silently wrong on seq02 GT/SEG.
- Correct use requires spatial re-matching against ST or TRA.

### 3.3 GT/SEG seq01 label divergence
- From frame 059 onwards, GT/SEG uses label `7` for the cell that TRA/ST call `8`. This looks like consistency but is not — any code cross-referencing GT/SEG labels to TRA/ST labels after frame 059 in seq01 is silently wrong too.

### 3.4 Silver Truth is automatic
- ST is produced by an automatic curator, not a human. It is the best dense annotation available but not pixel-perfect.
- Mean binary IoU against GT/SEG where both exist:
  - Seq01: 0.956 (range 0.94–0.96) — very close agreement.
  - Seq02: 0.878 (range 0.59–0.95) — several frames drop below 0.80.
- Weakest frames in seq02: **026** (IoU 0.77), **051** (0.76), **085** (0.59), **106** (0.64). These are frames where cells touch or merge and the automatic boundary guess diverges from the human annotator.
- Any ST-based result near these frames should be flagged or excluded from sanity-check subsets.

### 3.5 TRA markers are approximate centroids, not exact
- TRA markers are small blobs (~13 × 13 px) placed near cell centres. The exact centroid of a TRA marker is not guaranteed to equal the geometric centroid of the corresponding cell body.
- For coarse motion evaluation this is adequate. For precise centroid-displacement claims this is a source of systematic error on the order of a few pixels.

### 3.6 ERR_SEG is not a fourth annotation
- `NN_ERR_SEG/` contains ST shapes with permuted labels (label identities shuffled). Labels in ERR_SEG do not correspond to labels in ST/TRA.
- This is not an "error-corrected" or "enhanced" segmentation — it is a deliberately mislabelled dataset, unusable for matching-based evaluation.

---

## 4. Imaging-Related Issues

### 4.1 Phase-contrast halos violate brightness constancy
- Phase-contrast microscopy produces bright halos around refractive objects (cells). Halo intensity depends on cell position relative to the optical axis and substrate topography, not on cell identity.
- The standard motion-estimation assumption `I(x, t) = I(x + u, t+1)` is violated in phase-contrast: the same physical point on a cell does not have a stable intensity across frames.
- Consequence: any similarity term that directly compares intensities (SSD, MSE, cross-correlation without normalization) is fighting a signal it cannot fully explain. This is a modelling limit, not a bug.

### 4.2 Background is not static
- The image background contains substrate texture, debris, and possibly fluid-driven drift.
- Frame-to-frame background differences are not all due to cell motion. A perfect motion estimator would still see residual intensity differences.

### 4.3 Low contrast between cell interior and background
- U373 cells on polyacrylamide substrate have low native contrast. Intensity gradients inside cells are weak.
- Gradient-based methods (Horn–Schunck and its descendants) have poor local information inside cell bodies and rely heavily on smoothness regularization to propagate flow from cell edges inward.

### 4.4 Cell-boundary ambiguity where cells touch
- Multiple frames (especially in seq02) have cells in direct contact. At the contact interface the "correct" displacement is ambiguous because the boundary itself moves.
- ST disagrees with GT/SEG most strongly exactly in these cases. Any method will have its largest errors here.

### 4.5 Rare fast motion and large deformation events
- The 15-minute interval is long relative to small displacements but very short relative to cell-scale morphological rearrangement. Most frame-to-frame motion is small, but occasional rapid shape changes occur (cells rounding up during division, cells extending lamellipodia).
- Average-case metrics will be dominated by the common small-motion case and understate method behaviour on the rarer but more interesting large-deformation events. Report tail statistics.

### 4.6 Natural biological area variation is real, not noise
- Cell areas change by factors of 2×–5× across a cell's lifetime (e.g. seq01 cell 6: 3985 px → 13 319 px). This is genuine biological deformation.
- Treating area changes as annotation noise and smoothing them away destroys the signal we are trying to measure.

---

## 5. Practical Consequences for Future Work

1. **Use ST as the primary evaluation source.** Nothing else has enough coverage.
2. **Always report no-registration and identity baselines alongside any result.** The dataset baseline is already fairly high because cell motion between consecutive frames is small.
3. **Report per-sequence numbers.** Seq01 and seq02 differ systematically.
4. **Report N explicitly.** N_pairs and N_cells are different quantities and both are relevant.
5. **Report variance and CIs, not just means.** With this few samples, means alone are misleading.
6. **Do not treat frame 026, 051, 085, 106 in seq02 as ground truth.** ST disagrees with GT there.
7. **Do not evaluate deformation near cell contact events without caveats.** These are the worst-annotated regions.
8. **Do not draw conclusions about cross-experiment generalization.** Only 2 sequences from the same source exist.
9. **Use seq-held-out evaluation as a cross-sequence cross-check, not as a generalization claim.** It is the strongest split available but is still only one test point.
10. **Expect intensity-based losses to underperform their nominal assumptions** because of the phase-contrast artefacts. This is not an implementation issue.
