"""
Sanity check for mask-supervised VoxelMorph training.

Runs a 1-batch overfit test on a single consecutive frame pair with mask
loss enabled. Confirms that:
  - Silver Truth masks load correctly through CellTrackingDataset
  - The bilinear SpatialTransformer warps the mask differentiably
  - The soft Dice gradient propagates back into the displacement field
  - The model can overfit a single pair (mask_loss < threshold)

Also saves a 3-panel visualization showing source / target / warped source
with cell outlines overlaid, to visually confirm the warp quality.

Usage:
    python -m scripts.cell_tracking.sanity_check
    python -m scripts.cell_tracking.sanity_check --steps 300 --threshold 0.1
    python -m scripts.cell_tracking.sanity_check --pair-index 0 --data-dir dataset/train
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import voxelmorph as vxm

from scripts.cell_tracking.dataset import CellTrackingDataset
from scripts.cell_tracking.train import soft_dice_loss


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Sanity check: 1-batch overfit test for mask-supervised VoxelMorph'
    )
    parser.add_argument('--data-dir', type=str, default='dataset/train',
                        help='Path to training data directory (default: dataset/train)')
    parser.add_argument('--output-dir', type=str, default='output/sanity',
                        help='Directory to save output figures (default: output/sanity)')
    parser.add_argument('--steps', type=int, default=200,
                        help='Number of overfit gradient steps (default: 200)')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='mask_loss convergence threshold (default: 0.05)')
    parser.add_argument('--pair-index', type=int, default=5,
                        help='Dataset pair index to overfit on (default: 5, first GT/SEG-eligible pair)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Dataset — consecutive pairs with Silver Truth masks enabled          #
    # ------------------------------------------------------------------ #
    dataset = CellTrackingDataset(
        data_dir=args.data_dir,
        sequences=['01', '02'],
        pairing='consecutive',
        use_masks=True,
    )

    print(f'Dataset: {len(dataset)} consecutive pairs')

    if args.pair_index >= len(dataset):
        print(
            f'ERROR: --pair-index {args.pair_index} is out of range '
            f'(dataset has {len(dataset)} pairs, valid indices: 0–{len(dataset) - 1})'
        )
        sys.exit(1)

    print(f'Using pair index {args.pair_index} for overfit test')

    # ------------------------------------------------------------------ #
    # Load single batch and move to device                                 #
    # ------------------------------------------------------------------ #
    batch = dataset[args.pair_index]

    # Images: (1, H, W) → add batch dim → (1, 1, H, W)
    source = batch['source'].unsqueeze(0).to(device)
    target = batch['target'].unsqueeze(0).to(device)

    # Binary masks: binarise label IDs (any non-zero = cell), add batch dim
    source_binary = (batch['source_mask'].unsqueeze(0).to(device) > 0).float()
    target_binary = (batch['target_mask'].unsqueeze(0).to(device) > 0).float()

    source_coverage = source_binary.mean().item()
    target_coverage = target_binary.mean().item()
    print(f'Source mask coverage: {source_coverage:.4f}  ({source_coverage * 100:.2f}% of pixels)')
    print(f'Target mask coverage: {target_coverage:.4f}  ({target_coverage * 100:.2f}% of pixels)')

    if source_coverage < 1e-4:
        print(
            'WARNING: Source mask is nearly empty — this pair may lack a Silver Truth annotation.\n'
            'Consider choosing a different --pair-index. Continuing anyway.'
        )

    # ------------------------------------------------------------------ #
    # Model and spatial transformer                                        #
    # ------------------------------------------------------------------ #
    model = vxm.nn.models.VxmPairwise(
        ndim=2,
        source_channels=1,
        target_channels=1,
        nb_features=[16, 32, 32, 32],
        integration_steps=0,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: VxmPairwise(ndim=2, features=[16,32,32,32], int_steps=0), {param_count:,} parameters')

    # SpatialTransformer with bilinear interpolation — same as in train.py.
    # interpolation_mode='linear' maps to bilinear in 2D, producing continuous
    # outputs in [0, 1] so gradients flow back through the warped mask.
    mask_warper = vxm.nn.modules.SpatialTransformer(
        interpolation_mode='linear'
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ------------------------------------------------------------------ #
    # Overfit loop — pure mask Dice loss, no intensity term                #
    # ------------------------------------------------------------------ #
    print(f'\nOverfit test: {args.steps} steps, target mask_loss < {args.threshold}')

    final_mask_loss = float('inf')
    converged_at: int | None = None

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad()

        displacement, warped_source = model(
            source,
            target,
            return_warped_source=True,
            return_field_type='displacement',
        )

        # Warp source binary mask with predicted displacement (bilinear, differentiable)
        warped_mask = mask_warper(source_binary, displacement)

        # Soft Dice loss — prediction is soft (bilinear), target is hard binary
        mask_loss = soft_dice_loss(warped_mask, target_binary)
        mask_loss.backward()
        optimizer.step()

        final_mask_loss = mask_loss.item()

        if step % 20 == 0 or step == 1:
            print(f'  Step {step:4d}/{args.steps} — mask_loss: {final_mask_loss:.6f}')

        if final_mask_loss < args.threshold and converged_at is None:
            converged_at = step
            print(f'\n  Converged at step {step}: mask_loss = {final_mask_loss:.6f} < {args.threshold}')
            break

    passed = converged_at is not None or final_mask_loss < args.threshold

    # ------------------------------------------------------------------ #
    # Forward pass for visualization — eval mode, no grad                  #
    # ------------------------------------------------------------------ #
    model.eval()
    with torch.no_grad():
        displacement_vis, warped_source_vis = model(
            source,
            target,
            return_warped_source=True,
            return_field_type='displacement',
        )
        warped_mask_vis = mask_warper(source_binary, displacement_vis)

    # Convert to numpy — remove batch and channel dims → (H, W)
    source_np = source[0, 0].cpu().numpy()
    target_np = target[0, 0].cpu().numpy()
    warped_np = warped_source_vis[0, 0].cpu().numpy()
    source_mask_np = batch['source_mask'][0].numpy()   # (H, W), integer label IDs
    target_mask_np = batch['target_mask'][0].numpy()   # (H, W), integer label IDs
    warped_mask_np = warped_mask_vis[0, 0].cpu().numpy()   # (H, W), soft values in [0, 1]

    # Binary masks for contour plotting
    source_binary_np = (source_mask_np > 0).astype(float)
    target_binary_np = (target_mask_np > 0).astype(float)
    warped_binary_np = (warped_mask_np > 0.5).astype(float)

    # Crop padded region — original images are 520×696; padding adds zeros to 544×704
    orig_h, orig_w = 520, 696
    source_np        = source_np[:orig_h, :orig_w]
    target_np        = target_np[:orig_h, :orig_w]
    warped_np        = warped_np[:orig_h, :orig_w]
    source_binary_np = source_binary_np[:orig_h, :orig_w]
    target_binary_np = target_binary_np[:orig_h, :orig_w]
    warped_binary_np = warped_binary_np[:orig_h, :orig_w]

    # ------------------------------------------------------------------ #
    # 3-panel figure                                                       #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(source_np, cmap='gray', vmin=0, vmax=1)
    axes[0].contour(source_binary_np, levels=[0.5], colors=['cyan'], linewidths=1.0)
    axes[0].set_title('Source (t_n)\nCyan: source cell boundaries', fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(target_np, cmap='gray', vmin=0, vmax=1)
    axes[1].contour(target_binary_np, levels=[0.5], colors=['yellow'], linewidths=1.0)
    axes[1].set_title('Target (t_{n+1})\nYellow: target cell boundaries', fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(warped_np, cmap='gray', vmin=0, vmax=1)
    axes[2].contour(warped_binary_np, levels=[0.5], colors=['cyan'], linewidths=1.0)
    axes[2].contour(target_binary_np, levels=[0.5], colors=['yellow'], linewidths=1.0)
    axes[2].set_title(
        'Warped source\nCyan: warped source boundaries | Yellow: target boundaries',
        fontsize=10,
    )
    axes[2].axis('off')

    status_str = 'PASSED' if passed else 'FAILED'
    fig.suptitle(
        f'Sanity check — pair {args.pair_index} — {args.steps} steps — '
        f'final mask_loss = {final_mask_loss:.4f} — {status_str}',
        fontsize=12,
    )
    plt.tight_layout()

    fig_path = output_dir / f'overfit_pair{args.pair_index:03d}.png'
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nVisualization saved to {fig_path}')

    # ------------------------------------------------------------------ #
    # Final verdict                                                        #
    # ------------------------------------------------------------------ #
    if passed:
        print(
            f'\nPASSED — mask_loss = {final_mask_loss:.6f} < {args.threshold}\n'
            'Gradient flow through bilinear warp and soft Dice loss confirmed.'
        )
        sys.exit(0)
    else:
        print(
            f'\nFAILED — mask_loss = {final_mask_loss:.6f} did not reach {args.threshold} '
            f'within {args.steps} steps.\n'
            'Diagnostic checklist:\n'
            '  1. Check that Silver Truth masks exist: dataset/train/01_ST/SEG/man_seg*.tif\n'
            '  2. Check source mask coverage > 0 (pair may fall in a gap in ST annotation)\n'
            '  3. Check that mask_warper receives displacement, not velocity\n'
            '  4. Try --steps 500 or --threshold 0.15 if coverage is low for this pair'
        )
        sys.exit(1)


if __name__ == '__main__':
    main()
