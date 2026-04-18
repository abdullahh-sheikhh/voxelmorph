# Project Plan — Cell Deformation Estimation with VoxelMorph

## Goal

Adapt the VoxelMorph registration family to the PhC-C2DH-U373 cell dataset and compare it against Horn & Schunck optical flow.

The goal is pragmatic:

- get strong registration results on the cell dataset
- keep the code simple
- compare learned registration against Horn & Schunck fairly
- use a simple train/test split by sequence instead of training and evaluating on the same pairs

## Current Decisions

- **Baseline**: Horn & Schunck
- **Main learned model**: `VxmPairwise` with `ndim=2`
- **Comparison structure**: 2x2 grid
  - MSE + direct displacement
  - NCC + direct displacement
  - MSE + diffeomorphic
  - NCC + diffeomorphic
- **Training modes**:
  - mask-guided training as the main mode
  - optional unsupervised training as a lightweight comparison
- **Data split**:
  - keep it simple
  - use sequence-based splitting, e.g. train on `01`, evaluate on `02`
- **Validation**: skipped for now
- **Epoch budget**: 150 as the new standard run length

## Minimal Pipeline

```
PhC-C2DH-U373 frame pairs
    │
    ├── Train VoxelMorph on selected training sequences
    │       ├── mask-guided mode
    │       └── optional unsupervised mode
    │
    ├── Evaluate on held-out sequences
    │       ├── Dice
    │       ├── Masked MSE
    │       └── Runtime
    │
    └── Compare against Horn & Schunck
```

## Current Implementation Direction

### 1. Dataset handling
- Keep the current consecutive-pair loader
- Make sequence selection explicit and easy to use for train/test splitting

### 2. Training
- Keep current mask-guided loss as the default
- Add an optional unsupervised mode with minimal branching
- Update defaults/examples to 150 epochs
- Keep logging simple and avoid misleading `Dice` wording when the logged value is actually `1 - Dice`

### 3. Evaluation
- Evaluate on selected sequences only
- Use the same held-out split for both VoxelMorph and Horn & Schunck
- Keep metrics limited to what the current code really computes:
  - Dice
  - MaskedMSE
  - runtime

## Not In Scope Right Now

- validation pipeline
- early stopping
- large training framework changes
- Jacobian/per-cell deformation analysis as the main priority

Those can be added later if the core split-and-compare pipeline works well.

## Expected Experiment Set

### Main 2x2 runs

| Variant | Similarity | Transform |
|--------|------------|-----------|
| VM-1 | MSE | direct |
| VM-2 | NCC | direct |
| VM-3 | MSE | diffeomorphic |
| VM-4 | NCC | diffeomorphic |

### Optional extra comparison

- unsupervised MSE
- unsupervised NCC

## Success Criteria

- the code supports a simple sequence-based split
- the current 2x2 experiments are easy to rerun
- Horn & Schunck and VoxelMorph are evaluated on the same held-out sequence(s)
- documentation reflects the actual code and experiment setup
