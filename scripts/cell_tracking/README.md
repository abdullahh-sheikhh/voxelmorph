# Cell Deformation Estimation with VoxelMorph

2D deformable registration for estimating cell deformations using [VoxelMorph](https://github.com/voxelmorph/voxelmorph), compared against Horn & Schunck, Farneback, and TV-L1 optical flow baselines.

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
│   ├── 01_ST/SEG/   (silver truth masks — used for training)
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

**Training loss** (mask-guided mode):

`alpha * SoftDice(warped_source_mask, target_mask) + beta * CellRestrictedImageLoss(target, warped_source) + lambda * SpatialGradient(displacement)`

- Mask term: soft Dice loss between bilinearly warped source mask and binary target mask
- Image term: MSE, NCC, SSIM, or combinations thereof — applied only inside a 10-pixel dilated union of source and target masks
- Default weights: `alpha = 1.0`, `beta = 0.1`
- Smoothness: SpatialGradient L2, `lambda = 0.01` for MSE-only, `lambda = 1.0` for NCC/SSIM variants
- Model: 83,544 parameters, direct displacement (`integration_steps=0`)
- Images padded to 704x544 (divisible by 32 for 5-level UNet)
- Optimizer: Adam, `lr=1e-4`

**Training masks**: Silver Truth (`{seq}_ST/SEG/man_seg{n:03d}.tif`) — full cell body on all 115 frames. GT/TRA marks nucleus dots only (0.3% of image area vs 7% for ST/SEG) and causes a prolonged training plateau.

## Baselines

| Method | Description |
|--------|-------------|
| Horn & Schunck | Multi-scale pyramid, SOR solver, iterative warping (Lip6/Sorbonne implementation based on Meinhardt-Llopis & Sanchez 2013) |
| Farneback | Polynomial expansion optical flow (OpenCV implementation) |
| TV-L1 | Total variation L1 optical flow (OpenCV DualTVL1, `lambda=0.10`, `theta=0.20`) |

## Usage

All commands run from the project root.

### Train

```bash
python -m scripts.cell_tracking.train \
    --data-dir dataset/train \
    --sequences 01 \
    --epochs 200 \
    --batch-size 4 \
    --loss ncc \
    --output-dir output/ncc
```

Options:
- `--loss mse|ncc|ssim|ncc+ssim|mse+ncc+ssim` — image similarity loss or combination (default: mse)
- `--epochs N` — number of epochs (default: 150)
- `--batch-size N` — batch size (default: 1)
- `--lr F` — learning rate (default: 1e-4)
- `--lambda F` — regularization weight (default: 0.01 for MSE, 1.0 for NCC/SSIM)
- `--mask-weight F` — weight alpha on mask Dice loss (default: 1.0)
- `--int-weight F` — weight beta on cell-restricted image loss (default: 0.1)
- `--mse-weight F`, `--ncc-weight F`, `--ssim-weight F` — per-component weights for combined losses
- `--sequences 01 02 ...` — which sequences to train on
- `--unsupervised` — disable mask Dice loss; train with image loss + smoothness only
- `--output-dir DIR` — where to save models (default: output/)

Outputs: `best.pt`, `final.pt`, `checkpoint_epoch*.pt`, `loss_curve.png`

### Evaluate

```bash
python -m scripts.cell_tracking.evaluate \
    --model output/ncc/best.pt \
    --data-dir dataset/train \
    --gt-dir dataset/train \
    --sequences 02 \
    --output-dir output/eval_ncc \
    --max-pairs 0
```

Runs VoxelMorph + Horn & Schunck, Farneback, and TV-L1 baselines and outputs:
- Per-pair visualizations for the first 5 pairs: source, target, warped source, Dice overlay
- `metrics.json` with Dice, Masked MSE, and runtime results

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
├── train.py              # Training loop with loss curves
├── evaluate.py           # Evaluation with baselines + per-pair visualizations
├── horn_schunck.py       # Lip6 Horn & Schunck optical flow
├── register.py           # Pairwise inference
├── track.py              # Cell tracking via mask propagation
├── train_colab.ipynb     # Full pipeline notebook for Google Colab (GPU)
└── requirements.txt      # Extra dependencies (imagecodecs)
```

## Results

**Setup**: trained on sequence 01 (114 pairs), evaluated on held-out sequence 02 (114 pairs).
200 epochs, batch size 4, lr=1e-4, mask-guided training (`mask_weight=1.0`, `int_weight=0.1`).
Baselines evaluated on the same 114 pairs.

| Method | Dice | Masked MSE | Runtime (s/pair) |
|--------|------|------------|------------------|
| Horn & Schunck | 0.8806 ± 0.0629 | 0.004457 ± 0.001300 | 6.138 |
| Farneback | 0.8162 ± 0.0657 | 0.008819 ± 0.002019 | 0.119 |
| TV-L1 | 0.8392 ± 0.0630 | 0.008762 ± 0.002019 | 12.244 |
| VoxelMorph — Masked MSE | 0.8190 ± 0.0634 | 0.008686 ± 0.001764 | 0.013 |
| VoxelMorph — NCC | 0.8720 ± 0.0645 | **0.003581 ± 0.001062** | 0.012 |
| VoxelMorph — SSIM | 0.8645 ± 0.0639 | 0.003935 ± 0.001017 | 0.012 |
| **VoxelMorph — NCC+SSIM** | **0.8737 ± 0.0641** | 0.003839 ± 0.001085 | **0.011** |
| VoxelMorph — MSE+NCC+SSIM | 0.8709 ± 0.0641 | 0.004047 ± 0.001055 | 0.012 |

**Key observations**:

- **NCC+SSIM is the best VoxelMorph variant** on Dice (0.8737), 0.0069 below Horn & Schunck (0.8806) at 540× the inference speed.
- **NCC achieves the best Masked MSE of all methods** (0.003581), including Horn & Schunck — phase-contrast microscopy cells are intensity-shift invariant, which is exactly what NCC models.
- **Masked MSE underperforms badly** (Dice 0.8190, comparable to Farneback 0.8162). MSE optimizes global pixel error dominated by background; it is a poor proxy for cell boundary alignment.
- **Adding SSIM to NCC gives a small but consistent Dice improvement** (+0.0017 over pure NCC). Adding MSE on top (MSE+NCC+SSIM) slightly degrades both Dice and Masked MSE.
- All NCC-family VoxelMorph variants outperform Farneback and TV-L1 on Dice, at 10× and 1000× their respective speeds.

## References

- VoxelMorph: Balakrishnan et al., IEEE TMI 2019. [arXiv:1809.05231](https://arxiv.org/abs/1809.05231)
- Horn & Schunck: Determining Optical Flow, Artificial Intelligence 1981.
- Meinhardt-Llopis & Sanchez: Horn-Schunck Optical Flow with a Multi-Scale Strategy, IPOL 2013.
- Cell Tracking Challenge: http://celltrackingchallenge.net/