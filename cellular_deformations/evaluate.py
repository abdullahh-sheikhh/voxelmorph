"""
Evaluate registration quality for trained VoxelMorph models.

Metrics:
    - Dice score (primary) — segmentation overlap after warping source mask to target
    - Masked MSE — pixel intensity error inside cell regions only
    - Runtime — seconds per pair

Compares a trained VoxelMorph model against the Horn & Schunck baseline.
Masks are loaded from Silver Truth (ST) annotations, which cover all 115 frames
per sequence (228 consecutive pairs total).

Usage:
    python -m cellular_deformations.evaluate \
        --model output/best.pt \
        --data-dir dataset/train \
        --gt-dir dataset/train \
        --sequences 02 \
        --output-dir output/eval
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage.registration import optical_flow_tvl1

import voxelmorph as vxm
from cellular_deformations.dataset import CellTrackingDataset
from cellular_deformations.horn_schunck import hs_optical_flow, warp

@dataclass
class PairMetrics:
    """Evaluation metrics for a single registration pair."""

    dice: float | None
    masked_mse: float | None
    runtime: float


def warp_mask(
    mask: np.ndarray,
    displacement: np.ndarray,
    pad_to: tuple[int, int] = (544, 704),
) -> np.ndarray:
    """
    Warp a segmentation mask using a displacement field.

    Uses nearest-neighbor interpolation to keep integer cell IDs intact.
    VoxelMorph axis convention: displacement[0] = x (horizontal),
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

    grid_y, grid_x = np.mgrid[0:pad_h, 0:pad_w].astype(np.float32)
    sample_x = np.clip(np.round(grid_x + displacement[0]).astype(int), 0, pad_w - 1)
    sample_y = np.clip(np.round(grid_y + displacement[1]).astype(int), 0, pad_h - 1)

    return padded_mask[sample_y, sample_x][:orig_h, :orig_w].astype(np.int32)


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
        region_warped = warped_mask == label
        region_target = target_mask == label
        intersection = np.sum(region_warped & region_target)
        total = np.sum(region_warped) + np.sum(region_target)
        if total == 0:
            continue
        scores[int(label)] = 2.0 * intersection / total
    return scores


def compute_masked_mse(
    warped: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    """
    Compute MSE between warped source and target, restricted to cell pixels.

    Only pixels where mask > 0 contribute to the error. Background pixels
    are excluded, making the comparison purely about cell-region quality.

    Parameters
    ----------
    warped : np.ndarray
        Warped source image (H, W), values in [0, 1].
    target : np.ndarray
        Target image (H, W), values in [0, 1].
    mask : np.ndarray
        Union of source and target masks (H, W). Non-zero = cell pixel.

    Returns
    -------
    float or None
        Mean squared error over cell pixels, or None if mask is empty.
    """
    cell_pixels = mask > 0
    if not np.any(cell_pixels):
        return None
    return float(np.mean((warped[cell_pixels] - target[cell_pixels]) ** 2))


def load_st_mask(st_seg_directory: Path, frame_index: int) -> np.ndarray | None:
    """
    Load a Silver Truth segmentation mask (man_seg*.tif).

    Parameters
    ----------
    st_seg_directory : Path
        Path to the {seq}_ST/SEG directory containing man_seg*.tif files.
    frame_index : int
        Frame number to load.

    Returns
    -------
    np.ndarray or None
        Segmentation mask (H, W) with integer cell IDs, or None if not found.
    """
    mask_path = st_seg_directory / f'man_seg{frame_index:03d}.tif'
    if not mask_path.exists():
        return None
    return io.imread(str(mask_path)).astype(np.int32)


def evaluate_pair(
    displacement: np.ndarray,
    source_mask: np.ndarray | None,
    target_mask: np.ndarray | None,
    warped_source: np.ndarray | None,
    target_image: np.ndarray | None,
    runtime: float,
) -> PairMetrics:
    """
    Compute Dice and masked MSE for a single registration pair.

    Parameters
    ----------
    displacement : np.ndarray
        Displacement field (2, H_pad, W_pad) in VoxelMorph convention.
    source_mask : np.ndarray or None
        Source frame Silver Truth mask (H_orig, W_orig).
    target_mask : np.ndarray or None
        Target frame Silver Truth mask (H_orig, W_orig).
    warped_source : np.ndarray or None
        Warped source image (H_pad, W_pad), values in [0, 1].
    target_image : np.ndarray or None
        Target image (H_pad, W_pad), values in [0, 1].
    runtime : float
        Wall-clock seconds for the registration forward pass.

    Returns
    -------
    PairMetrics
    """
    dice = None
    masked_mse = None

    if source_mask is not None and target_mask is not None:
        warped_mask_result = warp_mask(source_mask, displacement)
        scores = compute_dice(warped_mask_result, target_mask)
        if scores:
            dice = float(np.mean(list(scores.values())))

        if warped_source is not None and target_image is not None:
            orig_h, orig_w = source_mask.shape
            union_mask = (source_mask > 0) | (target_mask > 0)
            masked_mse = compute_masked_mse(
                warped_source[:orig_h, :orig_w],
                target_image[:orig_h, :orig_w],
                union_mask,
            )

    return PairMetrics(dice=dice, masked_mse=masked_mse, runtime=runtime)


def visualize_registration(
    source: np.ndarray,
    target: np.ndarray,
    voxelmorph_warped: np.ndarray | None,
    save_path: Path,
    pair_index: int,
    source_mask: np.ndarray | None = None,
    target_mask: np.ndarray | None = None,
    voxelmorph_displacement: np.ndarray | None = None,
) -> None:
    """
    Save a 2×2 visualization grid for a registration pair.

    Top row:    Source image  |  Target image
    Bottom row: Warped source |  Dice overlay

    Parameters
    ----------
    source : np.ndarray
        Source image (H, W), normalized [0, 1].
    target : np.ndarray
        Target image (H, W), normalized [0, 1].
    voxelmorph_warped : np.ndarray or None
        VoxelMorph warped source image (H, W).
    save_path : Path
        Directory to save the figure.
    pair_index : int
        Pair number for the filename.
    source_mask : np.ndarray or None
        Source segmentation mask (H_orig, W_orig).
    target_mask : np.ndarray or None
        Target segmentation mask (H_orig, W_orig).
    voxelmorph_displacement : np.ndarray or None
        Displacement field (2, H_pad, W_pad) for Dice overlay.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    warped_display = voxelmorph_warped if voxelmorph_warped is not None else source

    axes[0, 0].imshow(source, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Source (moving)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(target, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Target (fixed)')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(warped_display, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Warped Source (VoxelMorph)')
    axes[1, 0].axis('off')

    if (
        source_mask is not None
        and target_mask is not None
        and voxelmorph_displacement is not None
    ):
        warped_mask_result = warp_mask(source_mask, voxelmorph_displacement)
        target_binary = target_mask > 0
        warped_binary = warped_mask_result > 0
        overlap_binary = target_binary & warped_binary

        overlay = np.zeros((*target_mask.shape, 3), dtype=np.float32)
        overlay[warped_binary & ~overlap_binary] = [1.0, 0.0, 1.0]   # magenta = warped only
        overlay[target_binary & ~overlap_binary] = [0.0, 1.0, 0.0]   # green   = target only
        overlay[overlap_binary]                  = [1.0, 1.0, 1.0]   # white   = overlap

        axes[1, 1].imshow(source, cmap='gray', vmin=0, vmax=1, alpha=0.4)
        axes[1, 1].imshow(overlay, alpha=0.6)

        dice_scores = compute_dice(warped_mask_result, target_mask)
        mean_dice = np.mean(list(dice_scores.values())) if dice_scores else 0.0
        axes[1, 1].set_title(
            f'Dice overlay  (mean={mean_dice:.4f})\n'
            'Magenta=warped  |  Green=target  |  White=overlap'
        )
    else:
        axes[1, 1].set_title('Dice overlay (no mask)')

    axes[1, 1].axis('off')

    plt.suptitle(f'Pair {pair_index}', fontsize=13, fontweight='bold')
    plt.savefig(save_path / f'pair_{pair_index:04d}.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


def _warp_image_cv2(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp image (H, W) using OpenCV dense flow (H, W, 2). Returns (H, W)."""
    h, w = image.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    return cv2.remap(
        image,
        grid_x + flow[..., 0],
        grid_y + flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _append_metrics(metrics: dict, method: str, result: PairMetrics) -> None:
    metrics[method]['runtime'].append(result.runtime)
    if result.dice is not None:
        metrics[method]['dice'].append(result.dice)
    if result.masked_mse is not None:
        metrics[method]['masked_mse'].append(result.masked_mse)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate VoxelMorph registration with Dice, masked MSE, and runtime'
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pt)')
    parser.add_argument('--data-dir', type=str, default='dataset/train',
                        help='Path to image data directory')
    parser.add_argument('--gt-dir', type=str, required=True,
                        help='Path to dataset root containing {seq}_ST/SEG/ directories')
    parser.add_argument('--sequences', nargs='+', default=['01', '02'],
                        help='Sequence IDs to evaluate, e.g. 02 for held-out testing')
    parser.add_argument('--output-dir', type=str, default='output/eval',
                        help='Directory to save evaluation results')
    parser.add_argument('--max-pairs', type=int, default=20,
                        help='Maximum pairs to evaluate (0 = all)')
    parser.add_argument('--no-baselines', action='store_true',
                        help='Skip Horn & Schunck baseline')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    model = vxm.nn.models.VxmPairwise(
        ndim=2, source_channels=1, target_channels=1,
        nb_features=[16, 32, 32, 32], integration_steps=0,
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    dataset = CellTrackingDataset(
        data_dir=args.data_dir, sequences=args.sequences,
    )
    number_of_pairs = (
        len(dataset) if args.max_pairs == 0
        else min(len(dataset), args.max_pairs)
    )

    use_baselines = not args.no_baselines
    print(f'Sequences: {args.sequences}')
    print(f'Pairs: {number_of_pairs}, Baselines: {use_baselines}')

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    methods: list[tuple[str, str]] = (
        [
            ('horn_schunck', 'Horn & Schunck'),
            ('farneback', 'Farneback'),
            ('tvl1', 'TV-L1'),
            ('vxm', 'VoxelMorph'),
        ]
        if use_baselines
        else [('vxm', 'VoxelMorph')]
    )
    method_names = [key for key, _ in methods]
    metrics: dict[str, dict[str, list[float]]] = {
        method: {'dice': [], 'masked_mse': [], 'runtime': []}
        for method in method_names
    }
    def _tvl1_flow(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Run scikit-image TV-L1 and return flow in OpenCV layout (H, W, 2) [x, y].

        OpenCV's `cv2.optflow.DualTVL1OpticalFlow_create` was removed in
        opencv-contrib 4.6; scikit-image's implementation is the same algorithm.
        """
        flow_yx = optical_flow_tvl1(
            reference_image=target,
            moving_image=source,
            attachment=15.0,
            tightness=0.10,   # ~ OpenCV's lambda
            num_warp=7,       # ~ OpenCV's WarpingsNumber
            num_iter=10,
            tol=0.005,        # ~ OpenCV's Epsilon
        )
        # (2, H, W) [y, x] -> (H, W, 2) [x, y]
        return np.stack([flow_yx[1], flow_yx[0]], axis=-1)

    print(f'\nEvaluating {number_of_pairs} pairs...\n')

    with torch.no_grad():
        for i in range(number_of_pairs):
            batch = dataset[i]
            source = batch['source'].unsqueeze(0).to(device)
            target = batch['target'].unsqueeze(0).to(device)
            source_numpy = source[0, 0].cpu().numpy()
            target_numpy = target[0, 0].cpu().numpy()

            source_path, target_path = dataset.pairs[i]
            sequence = source_path.parent.name
            st_seg_directory = Path(args.gt_dir) / f'{sequence}_ST' / 'SEG'
            source_mask = load_st_mask(st_seg_directory, int(source_path.stem[1:]))
            target_mask = load_st_mask(st_seg_directory, int(target_path.stem[1:]))

            # VoxelMorph
            time_start = time.time()
            displacement, warped_source = model(
                source, target,
                return_warped_source=True,
                return_field_type='displacement',
            )
            if device == 'cuda':
                torch.cuda.synchronize()
            vxm_runtime = time.time() - time_start

            displacement_numpy = displacement[0].cpu().numpy()
            warped_numpy = warped_source[0, 0].cpu().numpy()

            vxm_result = evaluate_pair(
                displacement_numpy, source_mask, target_mask,
                warped_numpy, target_numpy, vxm_runtime,
            )
            _append_metrics(metrics, 'vxm', vxm_result)

            # Horn & Schunck baseline
            if use_baselines:
                time_start = time.time()
                hs_displacement = hs_optical_flow(
                    reference=target_numpy, moving=source_numpy, alpha=1.0,
                )
                hs_runtime = time.time() - time_start

                hs_warped = warp(source_numpy, hs_displacement)
                # Swap axes: H&S [y, x] → VoxelMorph [x, y]
                hs_displacement_vxm = hs_displacement[[1, 0]]

                hs_result = evaluate_pair(
                    hs_displacement_vxm, source_mask, target_mask,
                    hs_warped, target_numpy, hs_runtime,
                )
                _append_metrics(metrics, 'horn_schunck', hs_result)

                src_uint8 = (source_numpy * 255).astype(np.uint8)
                tgt_uint8 = (target_numpy * 255).astype(np.uint8)

                # Farneback
                time_start = time.time()
                flow_fb = cv2.calcOpticalFlowFarneback(
                    src_uint8, tgt_uint8, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                fb_runtime = time.time() - time_start
                fb_warped = _warp_image_cv2(source_numpy, flow_fb)
                fb_disp = flow_fb.transpose(2, 0, 1)
                fb_result = evaluate_pair(
                    fb_disp, source_mask, target_mask,
                    fb_warped, target_numpy, fb_runtime,
                )
                _append_metrics(metrics, 'farneback', fb_result)

                # TV-L1 (scikit-image — OpenCV's DualTVL1 was removed in opencv-contrib 4.6)
                time_start = time.time()
                flow_tvl1 = _tvl1_flow(source_numpy, target_numpy)
                tvl1_runtime = time.time() - time_start
                tvl1_warped = _warp_image_cv2(source_numpy, flow_tvl1)
                tvl1_disp = flow_tvl1.transpose(2, 0, 1)
                tvl1_result = evaluate_pair(
                    tvl1_disp, source_mask, target_mask,
                    tvl1_warped, target_numpy, tvl1_runtime,
                )
                _append_metrics(metrics, 'tvl1', tvl1_result)

            if i < 5:
                visualize_registration(
                    source_numpy, target_numpy, warped_numpy,
                    output_directory, i,
                    source_mask=source_mask,
                    target_mask=target_mask,
                    voxelmorph_displacement=displacement_numpy,
                )

            log_line = f'  Pair {i}:'
            if vxm_result.dice is not None:
                log_line += f' Dice={vxm_result.dice:.4f}'
            if vxm_result.masked_mse is not None:
                log_line += f', MaskedMSE={vxm_result.masked_mse:.6f}'
            print(log_line)

    print(f'\n--- Results ({number_of_pairs} pairs) ---')
    for method, name in methods:
        if not metrics[method]['runtime']:
            print(f'  {name}: [skipped]')
            continue
        dice_string = 'N/A'
        mse_string = 'N/A'
        if metrics[method]['dice']:
            dice_mean = np.mean(metrics[method]['dice'])
            dice_std = np.std(metrics[method]['dice'])
            dice_string = f'{dice_mean:.4f} ± {dice_std:.4f}'
        if metrics[method]['masked_mse']:
            mse_mean = np.mean(metrics[method]['masked_mse'])
            mse_std = np.std(metrics[method]['masked_mse'])
            mse_string = f'{mse_mean:.6f} ± {mse_std:.6f}'
        runtime_mean = np.mean(metrics[method]['runtime'])
        print(
            f'  {name}: Dice={dice_string},'
            f' MaskedMSE={mse_string},'
            f' Runtime={runtime_mean:.4f}s'
        )

    results: dict = {'n_pairs': number_of_pairs, 'device': device}
    for method in method_names:
        if not metrics[method]['runtime']:
            continue
        results[method] = {'runtime_mean': float(np.mean(metrics[method]['runtime']))}
        if metrics[method]['dice']:
            results[method]['dice_mean'] = float(np.mean(metrics[method]['dice']))
            results[method]['dice_std'] = float(np.std(metrics[method]['dice']))
        if metrics[method]['masked_mse']:
            results[method]['masked_mse_mean'] = float(np.mean(metrics[method]['masked_mse']))
            results[method]['masked_mse_std'] = float(np.std(metrics[method]['masked_mse']))

    with open(output_directory / 'metrics.json', 'w') as json_file:
        json.dump(results, json_file, indent=2)
    print(f'\nSaved to {output_directory}/')


if __name__ == '__main__':
    main()
