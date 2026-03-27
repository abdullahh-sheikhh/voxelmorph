"""
Evaluate registration quality using metrics from the VoxelMorph paper
(Balakrishnan et al., IEEE TMI 2019, Table I).

Metrics:
    - Dice score (primary) — segmentation overlap after warping source mask
    - Jacobian determinant — percentage of pixels where |Jphi| <= 0
    - Runtime — seconds per pair

Compares a trained VoxelMorph model against the Horn & Schunck baseline.

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
from scripts.cell_tracking.horn_schunck import hs_optical_flow, warp


def warp_mask(
    mask: np.ndarray,
    displacement: np.ndarray,
    pad_to: tuple[int, int] = (544, 704),
) -> np.ndarray:
    """
    Warp a segmentation mask using a displacement field.

    Uses nearest-neighbor interpolation to keep integer cell IDs intact.
    Assumes VoxelMorph axis convention: displacement[0] = x (horizontal),
    displacement[1] = y (vertical).

    Parameters
    ----------
    mask : np.ndarray
        Segmentation mask (H, W) with integer cell labels.
    displacement : np.ndarray
        Displacement field (2, H_pad, W_pad) in VoxelMorph convention.
    pad_to : tuple
        Padded size matching model input.

    Returns
    -------
    np.ndarray
        Warped mask (H, W) with preserved cell IDs.
    """
    orig_h, orig_w = mask.shape
    pad_h, pad_w = pad_to

    padded_mask = np.zeros((pad_h, pad_w), dtype=np.float32)
    padded_mask[:orig_h, :orig_w] = mask.astype(np.float32)

    # displacement[0] = dx (horizontal), displacement[1] = dy (vertical)
    grid_y, grid_x = np.mgrid[0:pad_h, 0:pad_w].astype(np.float32)
    sample_x = grid_x + displacement[0]
    sample_y = grid_y + displacement[1]

    sample_x = np.clip(np.round(sample_x).astype(int), 0, pad_w - 1)
    sample_y = np.clip(np.round(sample_y).astype(int), 0, pad_h - 1)

    warped = padded_mask[sample_y, sample_x]

    return warped[:orig_h, :orig_w].astype(np.int32)


def compute_dice(warped_mask: np.ndarray, target_mask: np.ndarray) -> dict[int, float]:
    """
    Compute per-cell Dice scores between warped and target masks.

    Parameters
    ----------
    warped_mask : np.ndarray
        Warped source segmentation mask (H, W).
    target_mask : np.ndarray
        Target ground truth segmentation mask (H, W).

    Returns
    -------
    dict
        {cell_label: dice_score} for each non-background label.
    """
    labels = set(np.unique(warped_mask)) | set(np.unique(target_mask))
    labels.discard(0)

    scores = {}
    for label in sorted(labels):
        region_warped = (warped_mask == label)
        region_target = (target_mask == label)
        intersection = np.sum(region_warped & region_target)
        total = np.sum(region_warped) + np.sum(region_target)
        if total == 0:
            continue
        scores[int(label)] = 2.0 * intersection / total
    return scores


def compute_jacobian_determinant(displacement: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute the Jacobian determinant map and percentage of non-positive pixels.

    The Jacobian matrix of the deformation field phi = identity + displacement
    captures local deformation properties. |Jphi| <= 0 means the deformation
    folds space (not diffeomorphic).

    For 2D with phi = (x + u, y + v)::

        Jphi = [[1 + du/dx, du/dy],
                [dv/dx, 1 + dv/dy]]

        det(Jphi) = (1 + du/dx)(1 + dv/dy) - (du/dy)(dv/dx)

    Parameters
    ----------
    displacement : np.ndarray
        Displacement field (2, H, W) in VoxelMorph convention:
        displacement[0] = x component (horizontal),
        displacement[1] = y component (vertical).

    Returns
    -------
    jacobian_map : np.ndarray
        Jacobian determinant at each pixel (H, W).
    negative_percentage : float
        Percentage of pixels where det(Jphi) <= 0.
    """
    # u = displacement[0] (x component), v = displacement[1] (y component)
    # np.gradient returns [gradient_along_axis0, gradient_along_axis1]
    # axis 0 = rows = y direction, axis 1 = columns = x direction
    du_dy, du_dx = np.gradient(displacement[0])
    dv_dy, dv_dx = np.gradient(displacement[1])

    jacobian_map = (1.0 + du_dx) * (1.0 + dv_dy) - du_dy * dv_dx

    total_pixels = jacobian_map.size
    negative_count = np.sum(jacobian_map <= 0)
    negative_percentage = 100.0 * negative_count / total_pixels

    return jacobian_map, negative_percentage


def load_ground_truth_mask(
    ground_truth_directory: Path, frame_index: int
) -> np.ndarray | None:
    """
    Load a ground truth segmentation mask (man_seg*.tif).

    Parameters
    ----------
    ground_truth_directory : Path
        Path to the SEG directory containing man_seg*.tif files.
    frame_index : int
        Frame number to load.

    Returns
    -------
    np.ndarray or None
        Segmentation mask (H, W) with integer cell IDs, or None if not found.
    """
    mask_path = ground_truth_directory / f'man_seg{frame_index:03d}.tif'
    if not mask_path.exists():
        return None
    return io.imread(str(mask_path)).astype(np.int32)


def evaluate_pair(
    displacement: np.ndarray,
    source_mask: np.ndarray | None,
    target_mask: np.ndarray | None,
) -> dict[str, float | None]:
    """
    Compute Dice and Jacobian determinant for a displacement field.

    Parameters
    ----------
    displacement : np.ndarray
        Displacement field (2, H, W) in VoxelMorph convention.
    source_mask : np.ndarray or None
        Source frame ground truth segmentation mask.
    target_mask : np.ndarray or None
        Target frame ground truth segmentation mask.

    Returns
    -------
    dict
        {'dice': float or None, 'jacobian_negative_pct': float}.
    """
    _, jacobian_negative_pct = compute_jacobian_determinant(displacement)

    dice = None
    if source_mask is not None and target_mask is not None:
        warped_mask_result = warp_mask(source_mask, displacement)
        scores = compute_dice(warped_mask_result, target_mask)
        if scores:
            dice = float(np.mean(list(scores.values())))

    return {'dice': dice, 'jacobian_negative_pct': jacobian_negative_pct}


def visualize_registration(
    source: np.ndarray,
    target: np.ndarray,
    voxelmorph_displacement: np.ndarray | None,
    voxelmorph_warped: np.ndarray | None,
    horn_schunck_warped: np.ndarray | None,
    save_path: Path,
    pair_index: int,
    source_mask: np.ndarray | None = None,
    target_mask: np.ndarray | None = None,
    jacobian_map: np.ndarray | None = None,
) -> None:
    """
    Save a 2x3 visualization grid for a registration pair.

    Top row: Source with contours, Target with contours, Warped source with both contours.
    Bottom row: Jacobian determinant map, Displacement magnitude, Dice alignment overlay.

    Parameters
    ----------
    source : np.ndarray
        Source image (H, W), normalized [0, 1].
    target : np.ndarray
        Target image (H, W), normalized [0, 1].
    voxelmorph_displacement : np.ndarray or None
        VoxelMorph displacement field (2, H, W).
    voxelmorph_warped : np.ndarray or None
        VoxelMorph warped source image (H, W).
    horn_schunck_warped : np.ndarray or None
        Horn & Schunck warped source image (H, W).
    save_path : Path
        Directory to save the figure.
    pair_index : int
        Pair number for the filename.
    source_mask : np.ndarray or None
        Source segmentation mask for contour overlay.
    target_mask : np.ndarray or None
        Target segmentation mask for contour overlay.
    jacobian_map : np.ndarray or None
        Jacobian determinant map (H, W) for visualization.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)

    # Top row: source, target, warped
    warped_display = voxelmorph_warped if voxelmorph_warped is not None else source
    for ax, img, title in [
        (axes[0, 0], source, 'Source (moving)'),
        (axes[0, 1], target, 'Target (fixed)'),
        (axes[0, 2], warped_display, 'Warped Source (VoxelMorph)'),
    ]:
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')

    if source_mask is not None:
        axes[0, 0].contour(source_mask > 0, colors='cyan', linewidths=2, alpha=0.8)
        axes[0, 2].contour(source_mask > 0, colors='cyan', linewidths=1.5, alpha=0.6)
    if target_mask is not None:
        axes[0, 1].contour(target_mask > 0, colors='lime', linewidths=2, alpha=0.8)
        axes[0, 2].contour(target_mask > 0, colors='lime', linewidths=2, alpha=0.8)

    # Bottom left: Jacobian determinant map
    if jacobian_map is not None:
        jacobian_image = axes[1, 0].imshow(
            jacobian_map, cmap='RdBu', vmin=-1, vmax=3,
        )
        negative_mask = jacobian_map <= 0
        negative_count = np.sum(negative_mask)
        if negative_count > 0:
            negative_overlay = np.ma.masked_where(~negative_mask, jacobian_map)
            axes[1, 0].imshow(negative_overlay, cmap='Reds_r', vmin=-1, vmax=0, alpha=0.7)
        axes[1, 0].set_title(
            f'Jacobian determinant ({negative_count} pixels <= 0)'
        )
        axes[1, 0].axis('off')
        fig.colorbar(jacobian_image, ax=axes[1, 0], fraction=0.046, pad=0.04)
    else:
        axes[1, 0].axis('off')

    # Bottom center: displacement magnitude
    if voxelmorph_displacement is not None:
        magnitude = np.sqrt(
            voxelmorph_displacement[0] ** 2 + voxelmorph_displacement[1] ** 2
        )
        magnitude_image = axes[1, 1].imshow(magnitude, cmap='viridis')
        axes[1, 1].set_title(f'Displacement magnitude (max={magnitude.max():.2f}px)')
        axes[1, 1].axis('off')
        fig.colorbar(
            magnitude_image, ax=axes[1, 1], fraction=0.046, pad=0.04, label='pixels'
        )
    else:
        axes[1, 1].axis('off')

    # Bottom right: Dice alignment overlay (warped mask vs target mask)
    if (
        source_mask is not None
        and target_mask is not None
        and voxelmorph_displacement is not None
    ):
        warped_mask_result = warp_mask(source_mask, voxelmorph_displacement)

        # Green = target mask, Magenta = warped source mask, White = overlap
        overlay = np.zeros((*target_mask.shape, 3), dtype=np.float32)
        target_binary = target_mask > 0
        warped_binary = warped_mask_result > 0
        overlap_binary = target_binary & warped_binary

        # Magenta for warped-only regions
        overlay[warped_binary & ~overlap_binary] = [1.0, 0.0, 1.0]
        # Green for target-only regions
        overlay[target_binary & ~overlap_binary] = [0.0, 1.0, 0.0]
        # White for overlap
        overlay[overlap_binary] = [1.0, 1.0, 1.0]

        axes[1, 2].imshow(source, cmap='gray', vmin=0, vmax=1, alpha=0.4)
        axes[1, 2].imshow(overlay, alpha=0.6)

        dice_scores = compute_dice(warped_mask_result, target_mask)
        mean_dice = np.mean(list(dice_scores.values())) if dice_scores else 0.0
        axes[1, 2].set_title(f'Dice overlay (mean={mean_dice:.4f})')
        axes[1, 2].axis('off')
    else:
        axes[1, 2].axis('off')

    plt.suptitle(f'Pair {pair_index}', fontsize=14, fontweight='bold')
    plt.savefig(save_path / f'pair_{pair_index:04d}.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate registration with Dice, Jacobian, and runtime'
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pt)')
    parser.add_argument('--data-dir', type=str, default='dataset/train',
                        help='Path to image data directory')
    parser.add_argument('--gt-dir', type=str, required=True,
                        help='Path to ground truth directory (required for Dice)')
    parser.add_argument('--sequences', nargs='+', default=['01', '02'],
                        help='Sequence IDs to evaluate')
    parser.add_argument('--output-dir', type=str, default='output/eval',
                        help='Directory to save evaluation results')
    parser.add_argument('--nb-features', nargs='+', type=int,
                        default=[16, 32, 32, 32],
                        help='UNet features (must match training)')
    parser.add_argument('--int-steps', type=int, default=0,
                        help='Integration steps (must match training)')
    parser.add_argument('--max-pairs', type=int, default=20,
                        help='Maximum pairs to evaluate (0 = all)')
    parser.add_argument('--no-baselines', action='store_true',
                        help='Skip Horn & Schunck baseline')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    model = vxm.nn.models.VxmPairwise(
        ndim=2, source_channels=1, target_channels=1,
        nb_features=args.nb_features, integration_steps=args.int_steps,
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    dataset = CellTrackingDataset(
        data_dir=args.data_dir, sequences=args.sequences, pairing='consecutive',
    )
    number_of_pairs = (
        len(dataset) if args.max_pairs == 0
        else min(len(dataset), args.max_pairs)
    )

    use_baselines = not args.no_baselines
    print(f'Pairs: {number_of_pairs}, Baselines: {use_baselines}')

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    method_names = ['horn_schunck', 'vxm'] if use_baselines else ['vxm']
    metrics: dict[str, dict[str, list[float]]] = {
        method: {
            'dice': [],
            'jacobian_negative_pct': [],
            'runtime': [],
        }
        for method in method_names
    }
    display_names = {'horn_schunck': 'Horn & Schunck', 'vxm': 'VoxelMorph'}

    print(f'\nEvaluating {number_of_pairs} pairs...\n')

    with torch.no_grad():
        for i in range(number_of_pairs):
            batch = dataset[i]
            source = batch['source'].unsqueeze(0).to(device)
            target = batch['target'].unsqueeze(0).to(device)
            source_numpy = source[0, 0].cpu().numpy()
            target_numpy = target[0, 0].cpu().numpy()

            # Load ground truth masks
            source_path, target_path = dataset.pairs[i]
            sequence = source_path.parent.name
            ground_truth_directory = (
                Path(args.gt_dir) / f'{sequence}_GT' / 'SEG'
            )
            source_mask = load_ground_truth_mask(
                ground_truth_directory, int(source_path.stem[1:])
            )
            target_mask = load_ground_truth_mask(
                ground_truth_directory, int(target_path.stem[1:])
            )

            # --- VoxelMorph ---
            time_start = time.time()
            displacement, warped_source = model(
                source, target,
                return_warped_source=True,
                return_field_type='displacement',
            )
            if device == 'cuda':
                torch.cuda.synchronize()
            voxelmorph_runtime = time.time() - time_start

            displacement_numpy = displacement[0].cpu().numpy()
            warped_numpy = warped_source[0, 0].cpu().numpy()

            voxelmorph_result = evaluate_pair(
                displacement_numpy, source_mask, target_mask,
            )
            metrics['vxm']['runtime'].append(voxelmorph_runtime)
            metrics['vxm']['jacobian_negative_pct'].append(
                voxelmorph_result['jacobian_negative_pct']
            )
            if voxelmorph_result['dice'] is not None:
                metrics['vxm']['dice'].append(voxelmorph_result['dice'])

            # Jacobian map for visualization
            jacobian_map, _ = compute_jacobian_determinant(displacement_numpy)

            # --- Horn & Schunck baseline ---
            horn_schunck_warped = None
            if use_baselines:
                time_start = time.time()
                horn_schunck_displacement = hs_optical_flow(
                    reference=target_numpy, moving=source_numpy, alpha=1.0,
                )
                horn_schunck_runtime = time.time() - time_start

                # Warp image using Lip6 warp (handles its own axis convention)
                horn_schunck_warped = warp(source_numpy, horn_schunck_displacement)

                # For Dice: swap axes from Horn & Schunck convention to VoxelMorph
                # Horn & Schunck: flow[0] = y (row), flow[1] = x (column)
                # VoxelMorph: displacement[0] = x (horizontal), displacement[1] = y
                horn_schunck_displacement_vxm = horn_schunck_displacement[[1, 0]]

                horn_schunck_result = evaluate_pair(
                    horn_schunck_displacement_vxm, source_mask, target_mask,
                )
                metrics['horn_schunck']['runtime'].append(horn_schunck_runtime)
                metrics['horn_schunck']['jacobian_negative_pct'].append(
                    horn_schunck_result['jacobian_negative_pct']
                )
                if horn_schunck_result['dice'] is not None:
                    metrics['horn_schunck']['dice'].append(
                        horn_schunck_result['dice']
                    )

            # Visualize first 5 pairs
            if i < 5:
                visualize_registration(
                    source_numpy, target_numpy,
                    displacement_numpy, warped_numpy,
                    horn_schunck_warped,
                    output_directory, i,
                    source_mask=source_mask,
                    target_mask=target_mask,
                    jacobian_map=jacobian_map,
                )

            # Per-pair log
            log_line = f'  Pair {i}:'
            if voxelmorph_result['dice'] is not None:
                log_line += f' Dice={voxelmorph_result["dice"]:.4f}'
            log_line += (
                f', |Jphi|<=0={voxelmorph_result["jacobian_negative_pct"]:.2f}%'
            )
            print(log_line)

    # Summary
    print(f'\n--- Results ({number_of_pairs} pairs) ---')
    for method in method_names:
        name = display_names[method]
        dice_string = 'N/A'
        if metrics[method]['dice']:
            dice_mean = np.mean(metrics[method]['dice'])
            dice_std = np.std(metrics[method]['dice'])
            dice_string = f'{dice_mean:.4f} +/- {dice_std:.4f}'
        jacobian_mean = np.mean(metrics[method]['jacobian_negative_pct'])
        runtime_mean = np.mean(metrics[method]['runtime'])
        print(
            f'  {name}: Dice={dice_string},'
            f' |Jphi|<=0={jacobian_mean:.2f}%,'
            f' Runtime={runtime_mean:.4f}s'
        )

    # Save JSON results
    results: dict = {'n_pairs': number_of_pairs, 'device': device}
    for method in method_names:
        results[method] = {}
        if metrics[method]['dice']:
            results[method]['dice_mean'] = float(np.mean(metrics[method]['dice']))
            results[method]['dice_std'] = float(np.std(metrics[method]['dice']))
        results[method]['jacobian_negative_pct_mean'] = float(
            np.mean(metrics[method]['jacobian_negative_pct'])
        )
        results[method]['runtime_mean'] = float(
            np.mean(metrics[method]['runtime'])
        )

    with open(output_directory / 'metrics.json', 'w') as json_file:
        json.dump(results, json_file, indent=2)
    print(f'\nSaved to {output_directory}/')


if __name__ == '__main__':
    main()
