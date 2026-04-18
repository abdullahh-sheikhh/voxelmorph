# Cell Deformation Estimation with VoxelMorph

2D deformable registration for estimating cell deformations using [VoxelMorph](https://github.com/voxelmorph/voxelmorph), compared against Horn & Schunck optical flow as the classical baseline.

We build on top of the VoxelMorph library (no original files modified) to learn displacement fields between consecutive phase-contrast microscopy frames.

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

**Training loss**:

`alpha * DiceLoss(warped_source_mask, target_mask) + beta * CellRestrictedImageLoss(target, warped_source) + lambda * SpatialGradient(displacement)`

- Image term: MSE or NCC, applied only in a dilated cell region built from the union of source and target masks
- Default weights: `alpha = 1.0`, `beta = 0.1`
- Smoothness: SpatialGradient with `lambda = 0.01` for MSE and `lambda = 1.0` for NCC unless overridden
- Model: 83,544 parameters
- Images padded to 704x544 (divisible by 32 for UNet)
- Optimizer: ADAM, `lr=1e-4`

## Baselines

We compare VoxelMorph against the classical Horn & Schunck optical flow (multi-scale pyramid implementation from Sorbonne Lip6 lab, based on Meinhardt-Llopis & Sanchez 2013).

## Experimental Direction

The current experiment design is intentionally simple:

- use a sequence-based split instead of training and evaluating on the same pairs
- keep the 2x2 comparison:
  - MSE + direct displacement
  - NCC + direct displacement
  - MSE + diffeomorphic
  - NCC + diffeomorphic
- keep mask-guided training as the default mode
- add optional unsupervised training for direct comparison
  - in unsupervised mode, the mask Dice term is removed and the model trains with image loss + smoothness only
- use 150 epochs as the standard run length

## Usage

All commands run from the project root.

### Train

```bash
python -m scripts.cell_tracking.train --data-dir dataset/train --sequences 01 --epochs 150
```

Options:
- `--loss mse|ncc` — image similarity loss (default: mse)
- `--epochs N` — number of epochs (default: 150)
- `--batch-size N` — batch size (default: 1)
- `--lr F` — learning rate (default: 1e-4)
- `--lambda F` — regularization weight (default: 0.01 for MSE, 1.0 for NCC)
- `--mask-weight F` — weight on mask Dice loss (default: 1.0)
- `--int-weight F` — weight on cell-restricted image loss (default: 0.1)
- `--sequences 01 02 ...` — which sequences to train on
- `--unsupervised` — disable mask Dice loss and train with image loss + smoothness only
- `--int-steps N` — 0=direct displacement, >0=diffeomorphic (default: 0)
- `--output-dir DIR` — where to save models (default: output/)

Outputs: `best.pt`, `final.pt`, `checkpoint_epoch*.pt`, `loss_curve.png`

### Evaluate

```bash
python -m scripts.cell_tracking.evaluate \
    --model output/best.pt \
    --data-dir dataset/train \
    --gt-dir dataset/train \
    --sequences 02 \
    --output-dir output/eval \
    --max-pairs 0
```

Runs VoxelMorph + Horn & Schunck baseline and outputs:
- Per-pair visualizations for the first 5 pairs: source, target, warped source, Dice overlay
- `metrics.json` with Dice, MaskedMSE, and runtime results

Use `--no-baselines` to skip classical methods.

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
├── README.md             # This file
├── __init__.py
├── dataset.py            # CellTrackingDataset — TIF loader, padding, normalization
├── train.py              # Training loop with loss curves (MSE or NCC, direct or diffeomorphic)
├── evaluate.py           # Evaluation with baselines + merged visualizations
├── horn_schunck.py          # Horn & Schunck optical flow
├── register.py           # Pairwise inference
├── track.py              # Cell tracking via mask propagation
├── train_colab.ipynb     # Full pipeline notebook for Google Colab (GPU)
└── requirements.txt      # Extra dependencies (imagecodecs)
```

## Results

Current Colab notebook results from `train_colab.ipynb`:

- 100 epochs on Colab T4
- Batch size 4
- Evaluation on `dataset/train` over 228 consecutive pairs
- Horn and Schunck reported once alongside VM-1, then VM-2 to VM-4 evaluated separately

| Method | Dice | Masked MSE | Runtime (s/pair) |
|--------|------|------------|------------------|
| Horn & Schunck | 0.8933 ± 0.0507 | 0.004787 ± 0.001288 | 6.4264 |
| VM-1 (MSE, direct) | 0.7848 ± 0.0548 | 0.006702 ± 0.001314 | 0.0105 |
| VM-2 (NCC, direct) | 0.8495 ± 0.0517 | 0.004000 ± 0.001051 | 0.0103 |
| VM-3 (MSE, diffeomorphic) | 0.8422 ± 0.0542 | 0.006480 ± 0.001181 | 0.0128 |
| VM-4 (NCC, diffeomorphic) | 0.8327 ± 0.0515 | 0.003794 ± 0.001024 | 0.0128 |

Interpretation of the current run:

- Horn and Schunck is strongest on Dice in this notebook run.
- NCC variants outperform MSE variants on both Dice and MaskedMSE.
- VoxelMorph is orders of magnitude faster at inference.
- The diffeomorphic NCC model gives the best MaskedMSE, while the direct NCC model gives the best Dice among VoxelMorph variants.

These results were produced before introducing the simple sequence-based train/test split. They should be treated as same-split reference numbers rather than final held-out results.

## Recommended Commands

Train on sequence `01`, evaluate on sequence `02`:

```bash
python -m scripts.cell_tracking.train --data-dir dataset/train --sequences 01 --epochs 150 --loss mse --int-steps 0 --output-dir output/vm1
python -m scripts.cell_tracking.evaluate --model output/vm1/best.pt --data-dir dataset/train --gt-dir dataset/train --sequences 02 --output-dir output/eval_vm1 --max-pairs 0
```

Unsupervised comparison:

```bash
python -m scripts.cell_tracking.train --data-dir dataset/train --sequences 01 --epochs 150 --loss mse --int-steps 0 --unsupervised --output-dir output/vm1_unsup
```

2x2 held-out runs:

```bash
python -m scripts.cell_tracking.train --data-dir dataset/train --sequences 01 --epochs 150 --loss mse --int-steps 0 --output-dir output/vm1
python -m scripts.cell_tracking.train --data-dir dataset/train --sequences 01 --epochs 150 --loss ncc --int-steps 0 --output-dir output/vm2
python -m scripts.cell_tracking.train --data-dir dataset/train --sequences 01 --epochs 150 --loss mse --int-steps 7 --output-dir output/vm3
python -m scripts.cell_tracking.train --data-dir dataset/train --sequences 01 --epochs 150 --loss ncc --int-steps 7 --output-dir output/vm4
```

## References

- VoxelMorph: Balakrishnan et al., IEEE TMI 2019. [arXiv:1809.05231](https://arxiv.org/abs/1809.05231)
- Horn & Schunck: Determining Optical Flow, Artificial Intelligence 1981.
- Cell Tracking Challenge: http://celltrackingchallenge.net/
