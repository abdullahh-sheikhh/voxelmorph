# VoxelMorph Project — Full Summary

A personal reference covering the research paper, the codebase, and the dataset.

---

## Current Project Direction

For the cell-tracking work in this repository, the active direction is:

- adapt the VoxelMorph family to the PhC-C2DH-U373 dataset
- compare against Horn & Schunck as the classical baseline
- keep the implementation simple
- use a sequence-based train/test split instead of evaluating on the same data used for training
- keep the 2x2 comparison:
  - MSE + direct
  - NCC + direct
  - MSE + diffeomorphic
  - NCC + diffeomorphic
- keep mask-guided training as the main mode
- add optional unsupervised training as a lightweight comparison
- use 150 epochs as the standard run length for the current experiments

The current priority is not a full validation framework or a heavy refactor. The priority is a clean and fair comparison pipeline on the cell dataset.

---

## 1. The Paper: VoxelMorph (Balakrishnan et al., IEEE TMI 2019)

### The Problem

Medical image registration = aligning two images by finding a spatial transformation (deformation) that maps one onto the other. Traditional methods (ANTs, NiftyReg, Demons) solve an **optimization problem for every single image pair** — this takes minutes to hours per pair on CPU. Not practical for large-scale analysis.

### The Key Idea

Instead of optimizing per pair, **train a neural network once** that learns to predict the deformation field for any new pair in a single forward pass. Training is **unsupervised** — no ground truth deformations needed.

### Architecture

```
Source image (moving) ──┐
                        ├── concat ──→ UNet ──→ Flow Layer ──→ Velocity/Displacement Field
Target image (fixed) ───┘                                            │
                                                                     ▼
                                                        Spatial Transformer
                                                        (warps source using field)
                                                                     │
                                                                     ▼
                                                              Warped Source
```

- **UNet backbone**: encoder-decoder with skip connections. Encoder extracts multi-scale features, decoder generates the deformation field at full resolution
- **Flow layer**: final convolutional layer initialized with very small weights (~1e-5) so initial deformation is near-identity
- **Spatial transformer**: differentiable module that warps the source image according to the predicted displacement field using bilinear interpolation

### Loss Function

**Unsupervised loss** (no labels needed):

```
L = L_similarity(target, warped_source) + λ × L_regularization(displacement)
```

- **Similarity term** — how well does the warped source match the target?
  - **MSE** (mean squared error): simple, works for same-modality images
  - **NCC** (normalized cross-correlation): more robust, works across intensity variations
- **Regularization term** — is the deformation smooth and physically plausible?
  - **Diffusion regularizer**: penalizes spatial gradients of the displacement field
  - λ controls trade-off: small λ → more accurate but potentially folding deformations, large λ → smoother but less precise

**Semi-supervised loss** (when segmentation labels available):

```
L = L_similarity + λ × L_regularization + γ × L_dice(target_seg, warped_source_seg)
```

Adds Dice overlap between warped source segmentation and target segmentation. This provides anatomical guidance.

### Diffeomorphic Variant

Instead of directly predicting a displacement field, predict a **stationary velocity field (SVF)**. Then integrate it using **scaling-and-squaring** to get a diffeomorphic (invertible, topology-preserving) displacement:

1. Scale velocity: v' = v / 2^T
2. Squaring steps: φ = v', then φ ← φ ∘ φ (repeat T times)

This guarantees the deformation is smooth, invertible, and doesn't create folding artifacts.

### Key Results (from the paper)

| Method | Dice Score | Runtime (CPU) | Runtime (GPU) |
|--------|-----------|---------------|---------------|
| ANTs SyN | 0.749 | ~9100s | — |
| NiftyReg | 0.740 | ~1170s | — |
| VoxelMorph (MSE) | 0.750 | ~57s | ~0.45s |
| VoxelMorph (CC) | 0.753 | ~57s | ~0.45s |
| VoxelMorph + Dice (semi-sup) | 0.765 | ~57s | ~0.45s |

- **150x faster than ANTs** on CPU, thousands of times faster on GPU
- Comparable or better accuracy
- Semi-supervised variant is the best when labels are available
- Tested on 3,731 T1-weighted brain MRI scans, 30 anatomical structures

### Important Hyperparameters

- **λ** (regularization weight): 1.0 for NCC loss, 0.01 for MSE loss — robust across wide range (Fig. 7)
- **Integration steps**: 7 for diffeomorphic variant (more = smoother, slower)
- **UNet features**: [16, 32, 32, 32] encoder, [32, 32, 32, 32, 16, 16] decoder in the paper
- **Optimizer**: ADAM, lr=1e-4, batch_size=1, 150k iterations
- **Activations**: LeakyReLU(0.2) after each conv, kernel size 3, stride 2 in encoder
- **Training set size**: 100 images already near-optimal (Fig. 8) — diminishing returns beyond that
- **Instance-specific optimization**: Fine-tuning displacement u for each test pair (100 iterations) adds ~1 Dice point

### Key Insights for Our Cell Tracking Use Case

1. **Optical flow connection** (Section III-C): VoxelMorph is closely related to optical flow estimation — both predict 2D displacement fields between image pairs. Our consecutive frame registration is exactly this.
2. **Sparse labels** (Section V-G): Even with a single labeled structure, training with auxiliary Dice loss improves results without hurting unobserved structures. Our ~10 sparse segmentation masks per sequence can be leveraged.
3. **Amortized optimization** (Section IV-D): The shared network parameters act as implicit regularization. Even with λ=0, results improve over affine — global function learning naturally regularizes.
4. **Subject-to-subject** (Section V-F): When registering pairs with more variability, doubling feature counts helps. If random pairing produces poor results, increase features.

For this repo, that translates into a pragmatic working setup:

- use Horn & Schunck as the external baseline
- keep MSE and NCC both in the comparison
- keep direct and diffeomorphic variants both in the comparison
- compare on a held-out sequence rather than on the same sequence used for fitting
- optionally compare mask-guided and unsupervised training with the same backbone

---

## 2. The Codebase

### Module Structure

```
voxelmorph/                 # Main Python package (v0.2)
├── nn/                     # Neural network components (PyTorch)
│   ├── models.py           # VxmPairwise — the main registration model
│   ├── modules.py          # SpatialTransformer, IntegrateVelocityField, ResizeDisplacementField
│   ├── losses.py           # DEPRECATED — raises NotImplementedError, use neurite instead
│   └── functional.py       # Thin wrappers that handle image format → tensor conversions
│
├── py/                     # Pure Python utilities (no GPU)
│   ├── generators.py       # Data generators for various training paradigms
│   └── utils.py            # I/O (load_volfile, save_volfile), padding, resizing, seg utils
│
├── functional.py           # Core differentiable operations
│                           #   spatial_transform() — warp images with displacement fields
│                           #   integrate_disp() — scaling-and-squaring integration
│                           #   compose() — compose displacement fields
│                           #   random_disp() — generate random displacements (fractal noise)
│                           #   random_transform() — random affine + nonlinear warps
│                           #   disp_to_coords() / coords_to_disp() — coordinate conversions
└── __init__.py             # Exposes vxm.nn and vxm.py

scripts/
├── train.py                # Training entry point
└── register.py             # Inference entry point

tests/                      # pytest suite
├── test_models.py          # VxmPairwise forward pass tests
├── test_modules.py         # Module-level tests
├── test_functional.py      # Core functional tests
└── test_imports.py         # Import validation
```

### How VxmPairwise Works (step by step)

1. **Input**: source image `(B, 1, H, W)` and target image `(B, 1, H, W)` — for 2D
2. **Concatenate**: `(B, 2, H, W)` — source and target stacked along channel dim
3. **UNet** (`ne.nn.models.BasicUNet`): processes concatenated input through encoder-decoder with skip connections → outputs features `(B, ndim, H, W)`
4. **Flow layer** (`ne.nn.modules.ConvBlock`): refines features into velocity field `(B, 2, H, W)`. Initialized with tiny weights so initial prediction ≈ zero (no deformation)
5. **[Optional] Velocity integration**: if `integration_steps > 0`, integrates velocity → diffeomorphic displacement via scaling-and-squaring
6. **Spatial transformer**: warps source image using the displacement field with bilinear interpolation
7. **Output**: displacement field `(B, 2, H, W)`, optionally warped source/target

### Training Pipeline (`scripts/train.py`)

```python
# Current flow (OASIS-specific, needs adaptation):
dataset = VxmIterableDataset()          # Loads NIfTI from remote cluster
dataloader = DataLoader(dataset)
model = VxmPairwise(ndim=3, ...)        # 3D model — change to ndim=2

for epoch in range(epochs):
    for batch in dataloader:
        displacement, warped_source = model(source, target, ...)
        loss = MSE(target, warped_source) + λ * SpatialGradient(displacement)
        loss.backward()
        optimizer.step()
```

### Key Dependencies

| Package | Role |
|---------|------|
| `torch` | Deep learning framework |
| `neurite` (≥0.2) | UNet backbone, loss functions (MSE, NCC, Dice, SpatialGradient), ConvBlock |
| `nibabel` | NIfTI/MGZ medical image I/O |
| `scikit-image` | Image processing utilities |
| `numpy` / `scipy` | Numerical operations |
| `h5py` | HDF5 file support |

### Recent Development History

The codebase has been actively refactored:
- Renamed `method` → `fractal_mode` for clarity
- Adopted neurite's public API (`ne.parse_non_spatial_dims()`)
- Tried then removed neurite-sandbox early stopping (too coupled)
- Added spatial transform tests, documented modules
- Added this dataset and research paper

---

## 3. The Dataset: PhC-C2DH-U373

### What It Is

**Phase-contrast microscopy** time-lapse sequences of **Glioblastoma-astrocytoma U373 cells** (brain cancer cells) migrating on a polyacrylamide substrate. Part of the [Cell Tracking Challenge](http://celltrackingchallenge.net/) (CTC) benchmark.

### Acquisition Details

| Property | Value |
|----------|-------|
| Cell type | Glioblastoma-astrocytoma U373 |
| Substrate | Polyacrylamide |
| Lab | Dr. S. Kumar, UC Berkeley Bioengineering |
| Microscope | Nikon |
| Objective | Plan Fluor DLL 20x / 0.5 NA |
| Pixel size | 0.65 × 0.65 μm |
| Time step | 15 minutes between frames |
| Image size | 696 × 520 pixels |
| Bit depth | 8-bit grayscale |
| Compression | LZW (lossless) |
| Format | TIF |

### Directory Structure

```
dataset/
├── train/
│   ├── 01/                  # Sequence 1: 115 frames (t000.tif → t114.tif)
│   ├── 01_GT/               # Gold Truth (manually annotated)
│   │   ├── SEG/             # Segmentation masks: man_seg001.tif, man_seg005.tif, ...
│   │   │                    #   (sparse! only ~10 annotated frames)
│   │   │                    #   Pixel values: 0=background, N=cell ID
│   │   └── TRA/             # Tracking annotations
│   │       ├── man_track000.tif ... man_track114.tif  (cell instance masks per frame)
│   │       └── man_track.txt                          (lineage/tracking graph)
│   ├── 01_ST/SEG/           # Silver Truth — algorithmically generated segmentation
│   ├── 01_ERR_SEG/          # Error segmentation (for benchmarking)
│   ├── 02/                  # Sequence 2: 115 frames
│   ├── 02_GT/ 02_ST/ 02_ERR_SEG/
│
└── test/
    ├── 01/                  # Test sequence 1: 115 frames (no ground truth)
    └── 02/                  # Test sequence 2: 115 frames
```

### Ground Truth Details

**Segmentation (SEG/)**: Sparse — only ~10 frames per sequence have manual segmentation masks. Each mask is a TIF where pixel value = cell instance ID (0 = background). Useful for semi-supervised training with Dice loss.

**Tracking (TRA/)**: Dense — every frame has a tracking mask (`man_track000.tif` etc.) where pixel value = cell ID consistent across time. The `man_track.txt` file encodes cell lineage:

```
cell_id  start_frame  end_frame  parent_id
1        0            114        0          ← cell 1 exists frames 0-114, no parent
7        9            22         0          ← cell 7 exists frames 9-22, no parent
8        54           114        7          ← cell 8 is daughter of cell 7
```

Parent_id = 0 means the cell appeared independently (not from division).

### How This Relates to VoxelMorph

VoxelMorph registers **pairs of images** by finding the deformation that maps one onto the other. For cell tracking:

- **Source/target pairs**: consecutive frames (t_n, t_{n+1}) — the deformation field captures cell movement over 15 minutes
- **Registration**: finding how cells moved/deformed between frames
- **Segmentation labels**: can be used for semi-supervised training (Dice loss on cell masks)
- **Key difference from brain MRI**: these are 2D images (not 3D volumes), much smaller, and the deformations are local cell movements rather than whole-organ alignment

### What Needs to Be Adapted

1. **Dataset loader**: read TIF files instead of NIfTI, create consecutive frame pairs
2. **Model dimension**: `ndim=2` instead of `ndim=3`
3. **Normalization**: scale 8-bit [0, 255] to [0.0, 1.0]
4. **Image size**: 696×520 — may need padding to be divisible by 2^(UNet depth) = 32
5. **Hyperparameters**: λ, learning rate, UNet depth may need tuning for this smaller-scale 2D task
