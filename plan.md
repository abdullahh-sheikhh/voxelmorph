# Project Plan — Cell Deformation Estimation with VoxelMorph

## Goal

Use VoxelMorph to learn displacement fields between consecutive cell microscopy frames, then use those displacement fields to study how cells deform over time. Compare against Horn & Schunck classical optical flow as baseline.

## Pipeline

```
TIF frame pairs (PhC-C2DH-U373)
    │
    ├── Train VxmPairwise (MSE/NCC + SpatialGradient)
    │       → displacement field per pair
    │
    ├── Horn & Schunck baseline (Lip6 multi-scale)
    │       → displacement field per pair
    │
    ├── Evaluate accuracy
    │       → Dice score (cell mask overlap)
    │       → Jacobian determinant within cell regions (deformation quality)
    │       → Runtime comparison
    │
    └── Study cell deformations
            → Per-cell Jacobian statistics (expansion vs compression)
            → Masked deformation maps (color-coded per cell)
            → VoxelMorph vs Horn & Schunck deformation comparison
```

## Completed Steps

### 1. Data loading (`dataset.py`)
- Loads TIF pairs from sequences 01 and 02
- Pads to 704x544, normalizes to 0-1
- Consecutive frame pairing

### 2. Training (`train.py`)
- VxmPairwise with ndim=2, nb_features=[16, 32, 32, 32]
- MSE or NCC training loss + SpatialGradient regularization
- Saves best.pt, final.pt, config.json, loss curves

### 3. Horn & Schunck baseline (`horn_schunck.py`)
- Replaced simple version with Lip6 lab's multi-scale implementation
- API: `hs_optical_flow(reference, moving, alpha)` and `warp(image, flow)`
- Axis convention: flow[0]=y, flow[1]=x (differs from VoxelMorph)

### 4. Evaluation rewrite (`evaluate.py`)
- Dice as primary metric (not MSE — MSE is training loss only, per paper Table I)
- Jacobian determinant (currently full image — needs masking to cell regions)
- Horn & Schunck axis swap handled: `hs_disp[[1, 0]]` before VoxelMorph-convention functions
- Runtime comparison

### 5. Notebook update (`train_colab.ipynb`)
- Comparison table reads new JSON format (Dice, Jacobian, runtime)
- Corrupted Unicode fixed
- Markdown descriptions updated

## Current Step — Cell Deformation Analysis

### What needs to happen in evaluate.py:

1. **Mask Jacobian to cell regions** — compute only where segmentation mask > 0
   - Background is ~95% of image, trivially Jacobian ~1.0
   - Masking gives meaningful per-cell deformation numbers

2. **Per-cell deformation statistics** — for each cell label:
   - Mean Jacobian (overall expansion or compression?)
   - Percentage of cell pixels expanding (Jacobian > 1)
   - Percentage of cell pixels compressing (Jacobian < 1)

3. **Remove contours from visualization** — they do not represent anything we compute

4. **Masked deformation visualization** — show Jacobian color map only within cell regions
   - Red = compression (<1), white = no change (=1), blue = expansion (>1)

5. **Update JSON output** — include per-cell deformation stats for both methods

## Future Steps

### Run experiments on Colab
- Run full notebook with all 4 variants (VM-1 through VM-4)
- Fill results table in README.md

### Results table format

| Method | Dice | |Jφ| ≤ 0 in cells (%) | Mean Jacobian in cells | Runtime (s/pair) |
|--------|------|---------------------|----------------------|------------------|
| Horn & Schunck | ? | ? | ? | ? |
| VM-1 (MSE, direct) | ? | ? | ? | ? |
| VM-2 (NCC, direct) | ? | ? | ? | ? |
| VM-3 (MSE, diffeomorphic) | ? | ? | ? | ? |
| VM-4 (NCC, diffeomorphic) | ? | ? | ? | ? |
