# Evaluation Problem — Formal Definition

This document defines what "evaluating cell motion and deformation" actually means on this dataset, independent of any specific algorithm. It answers: *what is a valid pair? what is a valid cell correspondence? how are births and deaths handled? what can we measure?*

---

## 1. The Inference Problem (Restated)

Given two consecutive images `I_t` and `I_{t+1}`, produce a dense displacement field `φ : Ω → R²` such that warping `I_t` by `φ` approximates `I_{t+1}`. From `φ`, we derive two classes of observation:

1. **Motion** — where a cell goes.
2. **Deformation** — how a cell's shape changes.

Evaluation checks how well `φ` describes the true inter-frame correspondence of the cells.

---

## 2. Valid Frame Pair

A pair `(t, t+1)` is **valid for evaluation** if all of the following hold:

1. Both frames `t` and `t+1` exist as images.
2. Both frames have annotations of the required type.
3. At least one cell ID is present in both frames.

### Count per annotation type

| Annotation | Valid pairs | Stable pairs (ID set unchanged) | Transition pairs (birth/death) |
|---|---|---|---|
| GT/SEG | **5** | 4 | 1 |
| GT/TRA | **228** | 212 | 16 |
| ST/SEG | **228** | 212 | 16 |

### The 5 GT/SEG pairs (global dataset index, sequence, source→target frame)

```
idx   5   seq01   005 → 006
idx   6   seq01   006 → 007
idx 133   seq02   019 → 020
idx 149   seq02   035 → 036     (transition)
idx 165   seq02   051 → 052
```

### ST/TRA transition pairs

**Seq01 (3 transitions)**
```
008 → 009   +ID 7   (birth)
022 → 023   −ID 7   (death)
053 → 054   +ID 8   (birth)
```

**Seq02 (13 transitions)**
```
003 → 004  −5    014 → 015  +6    021 → 022  −6
023 → 024  +7    029 → 030  +18   030 → 031  −7
032 → 033  +9    058 → 059  −11   067 → 068  +12
072 → 073  −14   079 → 080  −1    088 → 089  +2
105 → 106  −18
```

These are not "bad pairs" — they are pairs where the cell population changes. They are evaluated only on the cells present in both frames.

---

## 3. Valid Cell Correspondence

A **cell correspondence** between source frame `s` and target frame `t` is a label pair `(k_s, k_t)` identifying the same physical cell on both sides.

### Rules

1. **Primary rule (ST/TRA).** Because TRA and ST share the same tracking IDs, a cell correspondence is simply `(k, k)` for every label `k` present in both frames — no spatial matching required.
2. **Cross-annotation rule (GT/SEG ↔ ST/TRA).** Because GT/SEG labels are not tracking IDs (seq01 partially, seq02 entirely), correspondence between a GT/SEG label and a TRA/ST label must be established by **maximum pixel overlap**: for each `k_seg`, the matched ST label is `argmax over k_st of |SEG == k_seg AND ST == k_st|`.
3. **Rule for transitions.** Only labels present in both source and target are matched. Births (label in target only) and deaths (label in source only) are excluded from cell-level metrics for that pair.

### Label set to iterate

Use `np.unique(mask)` minus `{0}` — never `range(1, N+1)`. Seq02 ST labels are non-contiguous (e.g. `{1, 3, 4, 11, 14}` at frame 010).

---

## 4. Handling Births and Deaths

### Definition

Let `L_s` = set of labels in source frame, `L_t` = set of labels in target frame.
- **Stable pair:** `L_s == L_t`.
- **Transition pair:** `L_s ≠ L_t`.
- **Births at this pair:** `L_t \ L_s`.
- **Deaths at this pair:** `L_s \ L_t`.

### Policy

1. **Per-cell metrics** (e.g. per-cell deformation) are computed only for `L_s ∩ L_t`.
2. **Global-per-pair metrics** (e.g. mean deformation over all visible cells in the pair) average over `L_s ∩ L_t` — the denominator is not constant across pairs.
3. **Reporting.** Always report the number of cells and the number of pairs actually used, not just the pair index. A mean over 212 stable pairs and a mean over 228 total pairs are different quantities.
4. **Do not use births/deaths as "prediction errors".** A displacement field cannot invent or destroy cells; these transitions are ground-truth events outside the scope of what the field represents.

---

## 5. What Metrics Are Possible (Conceptually)

This section lists the *types* of measurement the dataset supports. Formulas are deferred to future design work.

### 5.1 Motion metrics (possible on TRA, ST, and the 5 GT/SEG pairs)
- **Centroid displacement error.** For each matched cell, compare the ground-truth centroid-to-centroid vector against the predicted centroid displacement computed from `φ`.
- **Trajectory consistency.** Chain predicted displacements across multiple frames and compare against the TRA trajectory.
- **Binary motion agreement.** How well the warped source-foreground aligns with target-foreground.

### 5.2 Deformation metrics (possible on ST, and the 5 GT/SEG pairs)
- **Warped-mask overlap.** Warp the source cell mask by `φ` and measure its overlap with the target cell mask (per cell and binary). This is *structural* not *intensity* — it tests whether the predicted correspondence moves pixels consistently with the cell identity.
- **Per-cell area change.** Compare `|warp(mask_k^s)|` against `|mask_k^t|`. Also report ground-truth area change `|mask_k^t|/|mask_k^s|`.
- **Local strain / Jacobian determinant of `φ`, restricted to cell pixels.** Measures expansion, contraction, and folding *inside the cell*. Must be masked to cell regions — background Jacobian is meaningless because there is nothing moving there.
- **Smoothness of `φ` on cells.** A separate property of the field, not a comparison against ground truth.

### 5.3 Sanity checks (possible only on the 5 GT/SEG pairs)
- Verify that any ST-based metric gives consistent results on the same 5 pairs when run against GT/SEG. This is the only way to calibrate ST-based numbers against human truth.

### 5.4 Baselines that must exist before any model metric is meaningful
- **No-registration baseline.** Compare the raw source mask against the raw target mask without warping. This is the lower bound on any structural overlap metric and establishes what "doing nothing" already achieves. Any method's numbers are meaningless without this reference line.
- **Identity field baseline.** Equivalent to no-registration for mask overlap, but should also be reported for any intensity-based similarity metric.

---

## 6. Correct Evaluation Setup

A complete and defensible evaluation on this dataset must include all of the following:

1. **Primary evaluation set.** 228 ST pairs across both sequences, reported with the stable/transition split visible.
2. **Sanity-check set.** 5 GT/SEG pairs, reported separately. If the ST-based numbers on these 5 pairs do not agree with the GT-based numbers, ST is not trustworthy for this metric.
3. **Per-cell metrics only over `L_s ∩ L_t`.** Birth and death cells excluded.
4. **Baseline row in every table.** No-registration and/or identity-field baseline, computed the same way as every other method.
5. **Per-sequence breakdown.** Seq01 and seq02 reported separately, because seq02 has smaller and more numerous cells and more transitions. Aggregating obscures systematic differences.
6. **Explicit N.** Number of pairs used, and number of cell observations pooled, stated next to each statistic.
7. **Variance reporting.** Mean is not enough; report standard deviation or confidence interval. With only 228 pairs and 3–8 cells each, confidence intervals will be wide and that is the honest picture.
8. **No use of GT/SEG seq02 labels as tracking IDs.** If GT/SEG seq02 is used at all, labels must be re-matched to ST via spatial overlap first.

---

## 7. Non-evaluation Uses of Each Annotation

Not every use of a mask is "evaluation". Summary of what each annotation is good for:

| Use | GT/SEG | TRA | ST |
|---|---|---|---|
| Dense structural evaluation | No (too sparse) | No (no body) | **Yes** |
| Centroid trajectory evaluation | 5 pairs | **Yes** | Yes |
| Per-cell area/deformation analysis | 5 pairs | No | **Yes** |
| Cross-check for ST quality | **Yes** | No (different role) | — |
| Training supervision target | Possible but redundant | No | Possible |
| Qualitative visualization | Yes | Yes (as points) | Yes |
