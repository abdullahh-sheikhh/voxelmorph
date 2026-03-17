# Cell Tracking with VoxelMorph

2D deformable registration for cell tracking using [VoxelMorph](https://github.com/voxelmorph/voxelmorph).

We build on top of the VoxelMorph library (no original files modified) to register consecutive phase-contrast microscopy frames, capturing cell motion between timepoints.

## Dataset: PhC-C2DH-U373

Glioblastoma-astrocytoma U373 cells on polyacrylamide substrate ([Cell Tracking Challenge](http://celltrackingchallenge.net/)).

| Property | Value |
|----------|-------|
| Cell type | Glioblastoma-astrocytoma U373 |
| Source | Dr. S. Kumar, UC Berkeley Bioengineering |
| Microscope | Nikon, Plan Fluor DLL 20x/0.5 |
| Image format | 2D TIF, 696x520, 8-bit grayscale |
| Pixel size | 0.65 x 0.65 um |
| Time step | 15 min between frames |
| Sequences | 2 x 115 frames (train + test) |

Download:
- Train: https://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip
- Test: https://data.celltrackingchallenge.net/test-datasets/PhC-C2DH-U373.zip

Extract into `dataset/` at the project root:
```
dataset/
├── train/
│   ├── 01/          (t000.tif - t114.tif)
│   ├── 01_GT/SEG/   (segmentation masks)
│   ├── 01_GT/TRA/   (tracking annotations)
│   ├── 02/ ...
└── test/
    ├── 01/
    └── 02/
```

## Setup

```bash
pip install -e .
pip install imagecodecs  # Required for LZW-compressed TIF files
```

## How It Works

We use `VxmPairwise` from VoxelMorph with `ndim=2`:

```
Source frame (t_n) ──┐
                     ├── concat → UNet → Flow Layer → Displacement Field
Target frame (t_n+1)─┘                                      │
                                                    Spatial Transformer
                                                             │
                                                      Warped Source
```

**Loss**: `MSE(target, warped_source) + λ * SpatialGradient(displacement)`

- Model: 83,544 parameters
- Images padded to 704x544 (divisible by 32 for UNet)
- Hyperparameters from paper: ADAM lr=1e-4, λ=0.01, batch_size=1

## Usage

All commands run from the project root.

### Train

```bash
python -m scripts.cell_tracking.train --data-dir dataset/train --epochs 500
```

Options:
- `--epochs N` — number of epochs (default: 500)
- `--batch-size N` — batch size (default: 1)
- `--lr F` — learning rate (default: 1e-4)
- `--lambda F` — regularization weight (default: 0.01)
- `--pairing consecutive|random` — frame pairing (default: consecutive)
- `--int-steps N` — 0=direct displacement, >0=diffeomorphic (default: 0)
- `--nb-features 16 32 32 32` — UNet feature counts per level
- `--output-dir DIR` — where to save models (default: output/)
- `--save-every N` — checkpoint interval (default: 50)

### Evaluate

```bash
python -m scripts.cell_tracking.evaluate \
    --model output/best.pt \
    --data-dir dataset/test \
    --output-dir output/eval
```

Generates per-pair visualizations:
- Source / Target / Warped Source
- Difference map (residual error)
- Displacement field (color-coded)
- Jacobian determinant (deformation regularity)

### Register

```bash
python -m scripts.cell_tracking.register \
    --moving dataset/test/01/t000.tif \
    --fixed dataset/test/01/t001.tif \
    --model output/best.pt \
    --moved result.tif \
    --warp warp.npy
```

## Files

```
scripts/cell_tracking/
├── README.md         # This file
├── __init__.py
├── dataset.py        # CellTrackingDataset — TIF loader, padding, normalization
├── train.py          # Training loop
├── evaluate.py       # Visualization + metrics
└── register.py       # Pairwise inference
```

## References

- VoxelMorph: Balakrishnan et al., IEEE TMI 2019. [arXiv:1809.05231](https://arxiv.org/abs/1809.05231)
- Cell Tracking Challenge: http://celltrackingchallenge.net/
