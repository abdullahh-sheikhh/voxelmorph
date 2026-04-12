# Implementation Plan — Mask-Supervised Training Redesign

**Goal.** Make training learn cell motion and deformation by supervising on Silver Truth (ST)
masks instead of raw pixel intensities alone.

**Scope.** `scripts/cell_tracking/` only. No VoxelMorph library files touched.

---

## Execution Order

Steps must run in this order. Each step has a clear completion criterion.

```
Step 1  dataset.py   — expose ST masks through the data loader
Step 2  train.py     — add loss helpers + rewire training loop
Step 3  verify       — static read-back checks (no training run needed)
Step 4  sanity_check.py — 1-batch overfit + visualization hook
```

Steps 1 and 2 are independent of each other at the code level
(train.py only reads from the batch dict it receives),
but Step 2 is meaningless until Step 1 produces masks in batches.

---

## Step 1 — `dataset.py`

| # | Location | Change type | Detail |
|---|---|---|---|
| D1 | `__init__` signature (line 41) | Add parameter | `use_masks: bool = False` |
| D2 | `__init__` body (line 48) | Store attribute | `self.use_masks = use_masks` |
| D3 | After `_load_and_preprocess` (line 121) | New method | `_load_mask(image_path) -> Tensor` |
| D4 | `__getitem__`, consecutive branch (line 87) | Extend | Load source/target masks when `use_masks` |
| D5 | `__getitem__`, return statement (line 99) | Extend | Add `source_mask`, `target_mask` to return dict |

**Key design constraints for D3:**
- Derive ST path from image path: `{seq}_ST/SEG/man_seg{frame:03d}.tif`
- Return `(1, H, W)` float32, padded to `self.pad_to` (same as images)
- Return zeros silently if file missing (defensive; should not happen with this dataset)
- Labels stored as float32 (small integers ≤ 18, no precision loss; required for `> 0` ops)

**Not changed:**
- Random pairing path — non-consecutive pairs make temporal mask correspondence undefined
- `_load_and_preprocess` — unchanged
- Anything in evaluate.py, register.py — they construct the dataset with default `use_masks=False`

---

## Step 2 — `train.py`

### 2a — Two module-level helper functions (insert before `train_epoch`)

| Function | Purpose |
|---|---|
| `soft_dice_loss(prediction, target, smooth)` | Differentiable 1 − Dice; `prediction` is soft (bilinear warp output), `target` is hard binary |
| `dilate_mask(binary_mask, radius)` | Morphological dilation via `max_pool2d`; defines cell neighbourhood for intensity loss |

### 2b — `train_epoch` signature changes

| Parameter added | Type | Default | Purpose |
|---|---|---|---|
| `mask_warper` | `nn.Module \| None` | `None` | `SpatialTransformer(interpolation_mode='linear')` for bilinear mask warp |
| `mask_weight` | `float` | `1.0` | α — scales Dice mask loss |
| `int_mask_weight` | `float` | `0.1` | β — scales cell-restricted intensity loss |

Return type changes from `tuple[float, float, float]` to `tuple[float, float, float, float]`
(adds `avg_mask`).

### 2c — `train_epoch` loop body changes

Replace single-loss computation with three-term loss:

```
Step A  Extract source_binary, target_binary from batch (when has_masks)
Step B  Compute union_binary + dilated cell_region
Step C  Forward pass — displacement, warped_source (unchanged)
Step D  img_loss  on (target * cell_region, warped_source * cell_region) when masks available
         else img_loss on full image (fallback, preserves backward compat)
Step E  warped_mask = mask_warper(source_binary, displacement)  [bilinear]
        mask_loss  = soft_dice_loss(warped_mask, target_binary)
Step F  grad_loss  (unchanged)
Step G  total = mask_weight * mask_loss
              + int_mask_weight * img_loss
              + loss_weights[1] * grad_loss
```

Note: `loss_weights[0]` (always 1.0 in old code) is replaced by `int_mask_weight`.
`loss_weights[1]` (λ) is unchanged.

### 2d — `main()` changes

| Location | Change |
|---|---|
| After model instantiation | Add `mask_warper = vxm.nn.modules.SpatialTransformer(interpolation_mode='linear').to(device)` |
| Dataset construction | Pass `use_masks=(args.mask_weight > 0.0 or args.int_weight > 0.0)` |
| argparse | Add `--mask-weight` (default 1.0) and `--int-weight` (default 0.1) |
| `loss_history` | Add `'mask_dice': []` key |
| `train_epoch` call | Pass `mask_warper`, `mask_weight`, `int_mask_weight`; unpack 4-tuple |
| Epoch log line | Print `mask_dice` alongside existing metrics |
| `config` JSON | Save `mask_weight` and `int_weight` |

---

## Step 3 — Verification (no training run)

Read back the modified files and confirm:

| Check | What to look for |
|---|---|
| SpatialTransformer call | `mask_warper(source_binary, displacement)` — same interface as model's internal transformer |
| No target leakage | `target_binary` appears only as the Dice target, never as model input or loss multiplier for the prediction side |
| Birth/death penalty | `soft_dice_loss(warped_mask, target_binary)` — births score 0 (model can't create cells), deaths partially penalised; accept as training noise (~7% of pairs) |
| Return type matches | `train_epoch` returns 4-tuple; `main()` unpacks 4 values |
| Backward compat | `evaluate.py` constructs `CellTrackingDataset` with default `use_masks=False` — no mask loading, no change in eval behavior |

---

## Step 4 — `sanity_check.py`

New file: `scripts/cell_tracking/sanity_check.py`

### What it does

1. **1-batch overfit test.**
   Load a single batch (pair index 5, the first GT/SEG-eligible pair).
   Train for 200 steps on that one batch with mask loss enabled.
   Assert that `mask_loss` falls below 0.05 (the model must be able to overfit a single pair).
   This confirms: masks are flowing through, the Dice gradient is working, the warp is correct.

2. **Visualization hook.**
   After overfit, run a forward pass and save a 3-panel figure:
   - Panel 1: source image + source cell outlines
   - Panel 2: target image + target cell outlines
   - Panel 3: warped source image + source cell outlines warped by φ overlaid on target outlines
   Save to `output/sanity/overfit_pair005.png`.

### Arguments

```
--data-dir     default: dataset/train
--output-dir   default: output/sanity
--steps        default: 200   (overfit steps)
--threshold    default: 0.05  (mask_loss convergence threshold)
```

---

## What is NOT in this plan

- `evaluate.py` fixes (axis convention, GT→ST switch, no-registration baseline) —
  separate plan, separate diff
- Validation split logic — train.py currently has no val loop;
  adding one is a follow-on step after training is confirmed to work with masks
- Architecture changes (mask as input channel) — reserved for after binary Dice is working
- Per-label (non-binary) Dice in training loss — reserved for after binary Dice is working

---

## Completion criterion

The plan is done when:

1. `python -m scripts.cell_tracking.sanity_check` exits 0 with "PASSED" printed.
2. `python -m scripts.cell_tracking.train --data-dir dataset/train --epochs 1` runs without error
   and prints a non-zero `mask_dice` value in the epoch log.
3. `python -m scripts.cell_tracking.evaluate --model output/best.pt --data-dir dataset/train --gt-dir dataset/train`
   still runs without error (backward compat check).
