"""
Evaluate a trained VoxelMorph model on cell tracking test data.

Generates visualizations:
- Source / Target / Warped Source side-by-side
- Difference map: |target - warped_source|
- Displacement field: color-coded flow (RGB per dimension, like paper Fig. 6)
- Metrics: MSE, Jacobian determinant regularity

Usage:
    python -m scripts.cell_tracking.evaluate \
        --model output/best.pt \
        --data-dir dataset/test \
        --output-dir output/eval
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import voxelmorph as vxm
from scripts.cell_tracking.dataset import CellTrackingDataset


def compute_jacobian_determinant(displacement: np.ndarray) -> np.ndarray:
    """
    Compute the Jacobian determinant of a 2D displacement field.

    Parameters
    ----------
    displacement : np.ndarray
        Displacement field of shape (2, H, W).

    Returns
    -------
    np.ndarray
        Jacobian determinant at each pixel, shape (H, W).
    """
    # displacement[0] = dx, displacement[1] = dy
    # Jacobian of phi = Id + u:
    # J = [[1 + du_x/dx, du_x/dy],
    #      [du_y/dx, 1 + du_y/dy]]
    dudx = np.gradient(displacement[0], axis=1)  # du_x / dx
    dudy = np.gradient(displacement[0], axis=0)  # du_x / dy
    dvdx = np.gradient(displacement[1], axis=1)  # du_y / dx
    dvdy = np.gradient(displacement[1], axis=0)  # du_y / dy

    jac_det = (1 + dudx) * (1 + dvdy) - dudy * dvdx
    return jac_det


def visualize_pair(
    source: np.ndarray,
    target: np.ndarray,
    warped: np.ndarray,
    displacement: np.ndarray,
    save_path: Path,
    pair_idx: int,
):
    """
    Create a visualization figure for one registration pair.

    Parameters
    ----------
    source : np.ndarray
        Source image (H, W).
    target : np.ndarray
        Target image (H, W).
    warped : np.ndarray
        Warped source image (H, W).
    displacement : np.ndarray
        Displacement field (2, H, W).
    save_path : Path
        Directory to save the figure.
    pair_idx : int
        Pair index for filename.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Source, Target, Warped Source
    axes[0, 0].imshow(source, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Source (moving)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(target, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Target (fixed)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(warped, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Warped Source')
    axes[0, 2].axis('off')

    # Row 2: Difference map, Displacement field (color), Jacobian determinant
    diff = np.abs(target - warped)
    axes[1, 0].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
    axes[1, 0].set_title(f'|Target - Warped| (MSE={np.mean(diff**2):.6f})')
    axes[1, 0].axis('off')

    # Displacement as color-coded flow (like paper Fig. 6)
    # Map dx to red, dy to green, magnitude to blue
    dx = displacement[0]
    dy = displacement[1]
    magnitude = np.sqrt(dx ** 2 + dy ** 2)

    # Normalize for visualization
    max_mag = max(magnitude.max(), 1e-8)
    flow_rgb = np.stack([
        np.clip(np.abs(dx) / max_mag, 0, 1),
        np.clip(np.abs(dy) / max_mag, 0, 1),
        np.clip(magnitude / max_mag, 0, 1),
    ], axis=-1)

    axes[1, 1].imshow(flow_rgb)
    axes[1, 1].set_title(f'Displacement Field (max={max_mag:.2f}px)')
    axes[1, 1].axis('off')

    # Jacobian determinant
    jac_det = compute_jacobian_determinant(displacement)
    n_folding = np.sum(jac_det <= 0)
    pct_folding = 100 * n_folding / jac_det.size

    axes[1, 2].imshow(jac_det, cmap='RdBu', vmin=0, vmax=2)
    axes[1, 2].set_title(f'Jacobian Det (folding: {pct_folding:.2f}%)')
    axes[1, 2].axis('off')

    plt.suptitle(f'Pair {pair_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / f'pair_{pair_idx:04d}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate VoxelMorph on cell tracking test data'
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pt)')
    parser.add_argument('--data-dir', type=str, default='dataset/test',
                        help='Path to test data directory')
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
                        help='Max pairs to visualize (default: 20)')
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
    print(f'Test set: {len(dataset)} consecutive pairs')

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate
    all_mse = []
    all_folding_pct = []
    n_pairs = min(len(dataset), args.max_pairs)

    print(f'\nEvaluating {n_pairs} pairs...\n')

    with torch.no_grad():
        for i in range(n_pairs):
            batch = dataset[i]
            source = batch['source'].unsqueeze(0).to(device)
            target = batch['target'].unsqueeze(0).to(device)

            displacement, warped_source = model(
                source, target,
                return_warped_source=True,
                return_field_type='displacement',
            )

            # Convert to numpy (remove batch and channel dims)
            src_np = source[0, 0].cpu().numpy()
            tgt_np = target[0, 0].cpu().numpy()
            warp_np = warped_source[0, 0].cpu().numpy()
            disp_np = displacement[0].cpu().numpy()  # (2, H, W)

            # Metrics
            mse = np.mean((tgt_np - warp_np) ** 2)
            jac_det = compute_jacobian_determinant(disp_np)
            folding_pct = 100 * np.sum(jac_det <= 0) / jac_det.size

            all_mse.append(mse)
            all_folding_pct.append(folding_pct)

            # Visualize
            visualize_pair(src_np, tgt_np, warp_np, disp_np, output_dir, i)
            print(f'  Pair {i}: MSE={mse:.6f}, Folding={folding_pct:.2f}%')

    # Summary
    print(f'\n--- Summary ({n_pairs} pairs) ---')
    print(f'MSE:     {np.mean(all_mse):.6f} +/- {np.std(all_mse):.6f}')
    print(f'Folding: {np.mean(all_folding_pct):.2f}% +/- {np.std(all_folding_pct):.2f}%')
    print(f'\nVisualizations saved to {output_dir}/')


if __name__ == '__main__':
    main()
