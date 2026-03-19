"""
Evaluate a trained VoxelMorph model on cell tracking data.

Metrics:
- MSE: image similarity between target and warped source
- Dice: segmentation overlap (requires --gt-dir with TRA masks)
- Jacobian determinant: deformation regularity (% folding pixels)
- Runtime: seconds per registration pair

Usage:
    python -m scripts.cell_tracking.evaluate \
        --model output/best.pt \
        --data-dir dataset/train \
        --gt-dir dataset/train \
        --output-dir output/eval
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import io

import voxelmorph as vxm
from scripts.cell_tracking.dataset import CellTrackingDataset
from scripts.cell_tracking.track import warp_mask


def compute_jacobian_determinant(displacement: np.ndarray) -> np.ndarray:
    """Jacobian determinant of a 2D displacement field (2, H, W) -> (H, W)."""
    dudx = np.gradient(displacement[0], axis=1)
    dudy = np.gradient(displacement[0], axis=0)
    dvdx = np.gradient(displacement[1], axis=1)
    dvdy = np.gradient(displacement[1], axis=0)

    jac_det = (1 + dudx) * (1 + dvdy) - dudy * dvdx
    return jac_det


def compute_dice_scores(warped_mask: np.ndarray, target_mask: np.ndarray) -> dict[int, float]:
    """Per-cell Dice between warped source mask and target mask.
    Returns {cell_id: dice} for each non-background label."""
    labels = set(np.unique(warped_mask)) | set(np.unique(target_mask))
    labels.discard(0)

    scores = {}
    for label in sorted(labels):
        a = (warped_mask == label)
        b = (target_mask == label)
        intersection = np.sum(a & b)
        total = np.sum(a) + np.sum(b)
        if total == 0:
            continue
        scores[int(label)] = 2.0 * intersection / total
    return scores


def load_gt_mask(gt_dir: Path, frame_idx: int) -> np.ndarray | None:
    """Load GT tracking mask for given frame. Returns None if file missing."""
    mask_path = gt_dir / f'man_track{frame_idx:03d}.tif'
    if not mask_path.exists():
        return None
    return io.imread(str(mask_path)).astype(np.int32)


def visualize_pair(source, target, warped, displacement, save_path, pair_idx):
    """Save a 2x3 visualization grid for one registration pair."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(source, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Source (moving)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(target, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Target (fixed)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(warped, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Warped Source')
    axes[0, 2].axis('off')

    diff = np.abs(target - warped)
    im_diff = axes[1, 0].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
    axes[1, 0].set_title(f'|Target - Warped| (MSE={np.mean(diff**2):.6f})')
    axes[1, 0].axis('off')
    fig.colorbar(im_diff, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # displacement magnitude
    dx, dy = displacement[0], displacement[1]
    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    im_mag = axes[1, 1].imshow(magnitude, cmap='viridis')
    axes[1, 1].set_title(f'Displacement Magnitude (max={magnitude.max():.2f}px)')
    axes[1, 1].axis('off')
    fig.colorbar(im_mag, ax=axes[1, 1], fraction=0.046, pad=0.04, label='pixels')

    jac_det = compute_jacobian_determinant(displacement)
    folding_count = np.sum(jac_det <= 0)
    pct_folding = 100.0 * folding_count / jac_det.size

    im_jac = axes[1, 2].imshow(jac_det, cmap='RdBu', vmin=0, vmax=2)
    axes[1, 2].set_title(f'Jacobian Det (folding: {pct_folding:.2f}%)')
    axes[1, 2].axis('off')
    fig.colorbar(im_jac, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.suptitle(f'Pair {pair_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / f'pair_{pair_idx:04d}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate VoxelMorph on cell tracking data'
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pt)')
    parser.add_argument('--data-dir', type=str, default='dataset/train',
                        help='Path to data directory')
    parser.add_argument('--gt-dir', type=str, default=None,
                        help='Path to GT directory (enables Dice evaluation). '
                        'Should contain {seq}_GT/TRA/ folders.')
    parser.add_argument('--sequences', nargs='+', default=['01', '02'],
                        help='Sequence folders to evaluate')
    parser.add_argument('--output-dir', type=str, default='output/eval',
                        help='Directory to save evaluation results')
    parser.add_argument('--nb-features', nargs='+', type=int,
                        default=[16, 32, 32, 32],
                        help='UNet feature counts (must match training)')
    parser.add_argument('--int-steps', type=int, default=0,
                        help='Integration steps (must match training)')
    parser.add_argument('--max-pairs', type=int, default=20,
                        help='Max pairs to evaluate (0 = all)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Load model
    model = vxm.nn.models.VxmPairwise(
        ndim=2,
        source_channels=1,
        target_channels=1,
        nb_features=args.nb_features,
        integration_steps=args.int_steps,
    ).to(device)

    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()
    print(f'Loaded model from {args.model}')

    # Load dataset
    dataset = CellTrackingDataset(
        data_dir=args.data_dir,
        sequences=args.sequences,
        pairing='consecutive',
    )

    n_total = len(dataset)
    n_pairs = n_total if args.max_pairs == 0 else min(n_total, args.max_pairs)
    print(f'Dataset: {n_total} consecutive pairs, evaluating {n_pairs}')

    use_dice = args.gt_dir is not None
    if use_dice:
        print(f'Dice evaluation enabled (GT from {args.gt_dir})')

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate
    all_mse = []
    all_folding_pct = []
    all_dice = []
    all_runtimes = []

    print(f'\nEvaluating {n_pairs} pairs...\n')

    with torch.no_grad():
        for i in range(n_pairs):
            batch = dataset[i]
            source = batch['source'].unsqueeze(0).to(device)
            target = batch['target'].unsqueeze(0).to(device)

            # Timed inference
            start = time.time()
            displacement, warped_source = model(
                source, target,
                return_warped_source=True,
                return_field_type='displacement',
            )
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - start
            all_runtimes.append(elapsed)

            # Convert to numpy
            src_np = source[0, 0].cpu().numpy()
            tgt_np = target[0, 0].cpu().numpy()
            warp_np = warped_source[0, 0].cpu().numpy()
            disp_np = displacement[0].cpu().numpy()  # (2, H, W)

            # MSE
            mse = np.mean((tgt_np - warp_np) ** 2)
            all_mse.append(mse)

            # Jacobian
            jac_det = compute_jacobian_determinant(disp_np)
            folding_pct = 100 * np.sum(jac_det <= 0) / jac_det.size
            all_folding_pct.append(folding_pct)

            # Dice (if GT available)
            pair_dice = None
            if use_dice:
                source_path, target_path = dataset.pairs[i]
                seq = source_path.parent.name
                src_frame = int(source_path.stem[1:])
                tgt_frame = int(target_path.stem[1:])

                gt_dir = Path(args.gt_dir) / f'{seq}_GT' / 'TRA'
                src_mask = load_gt_mask(gt_dir, src_frame)
                tgt_mask = load_gt_mask(gt_dir, tgt_frame)

                if src_mask is not None and tgt_mask is not None:
                    warped_mask = warp_mask(src_mask, disp_np)
                    dice_scores = compute_dice_scores(warped_mask, tgt_mask)
                    if dice_scores:
                        pair_dice = np.mean(list(dice_scores.values()))
                        all_dice.append(pair_dice)

            # Visualize (only first 20 to avoid too many files)
            if i < 20:
                visualize_pair(src_np, tgt_np, warp_np, disp_np, output_dir, i)

            # Log
            log = f'  Pair {i}: MSE={mse:.6f}, Folding={folding_pct:.2f}%'
            if pair_dice is not None:
                log += f', Dice={pair_dice:.4f}'
            log += f', {elapsed:.3f}s'
            print(log)

    # Summary
    print(f'\n--- Summary ({n_pairs} pairs) ---')
    print(f'MSE:     {np.mean(all_mse):.6f} +/- {np.std(all_mse):.6f}')
    print(f'Folding: {np.mean(all_folding_pct):.2f}% +/- {np.std(all_folding_pct):.2f}%')
    print(f'Runtime: {np.mean(all_runtimes):.4f} +/- {np.std(all_runtimes):.4f} s/pair ({device})')
    if all_dice:
        print(f'Dice:    {np.mean(all_dice):.4f} +/- {np.std(all_dice):.4f}')

    # Save metrics to JSON
    results = {
        'n_pairs': n_pairs,
        'device': device,
        'mse_mean': float(np.mean(all_mse)),
        'mse_std': float(np.std(all_mse)),
        'folding_mean': float(np.mean(all_folding_pct)),
        'folding_std': float(np.std(all_folding_pct)),
        'runtime_mean': float(np.mean(all_runtimes)),
        'runtime_std': float(np.std(all_runtimes)),
    }
    if all_dice:
        results['dice_mean'] = float(np.mean(all_dice))
        results['dice_std'] = float(np.std(all_dice))

    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nMetrics saved to {metrics_path}')
    print(f'Visualizations saved to {output_dir}/')


if __name__ == '__main__':
    main()
