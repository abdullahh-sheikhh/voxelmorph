"""
Evaluate classical optical flow baselines (Horn & Schunck, Farneback, TV-L1)
on the per-cell segmented dataset.

Loads NPZ crops produced by SegmentedCellBuilder, runs each baseline on every
crop in the validation split, and writes per-baseline Dice / runtime into
metrics.json in the same format as evaluate.py so the segmented Results table
can include these rows.

Usage:
    python -m cellular_deformations.evaluate_segmented_cells_baselines \
        --data-dir dataset/segmented_cells \
        --val-sequence 02 \
        --output-dir output/segmented_cells_baselines
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.registration import optical_flow_tvl1
from tqdm import tqdm

from cellular_deformations.cell_segmentation import SegmentedCellDataset
from cellular_deformations.horn_schunck import hs_optical_flow, warp


def _warp_image_cv2(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp a grayscale image with a Farneback/TV-L1 flow field (H, W, 2)."""
    h, w = image.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)


def _binary_dice(pred_mask: np.ndarray, target_mask: np.ndarray) -> float | None:
    """Binary Dice between two boolean / binary float masks."""
    pred_bin = pred_mask > 0.5
    tgt_bin = target_mask > 0.5
    intersection = np.sum(pred_bin & tgt_bin)
    total = np.sum(pred_bin) + np.sum(tgt_bin)
    if total == 0:
        return None
    return 2.0 * intersection / total


def _tvl1_flow(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Run scikit-image TV-L1 and return flow in OpenCV layout (H, W, 2) [x, y].

    OpenCV's `cv2.optflow.DualTVL1OpticalFlow_create` was removed in
    opencv-contrib 4.6; scikit-image keeps the same algorithm. `optical_flow_tvl1`
    returns (2, H, W) with [0]=row(y) flow and [1]=col(x) flow.
    """
    flow_yx = optical_flow_tvl1(
        reference_image=target,
        moving_image=source,
        attachment=15.0,
        tightness=0.10,
        num_warp=7,
        num_iter=10,
        tol=0.005,
    )
    # (2, H, W) [y, x] -> (H, W, 2) [x, y]
    return np.stack([flow_yx[1], flow_yx[0]], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate H&S / Farneback / TV-L1 baselines on segmented cell crops'
    )
    parser.add_argument('--data-dir', type=str, default='dataset/segmented_cells')
    parser.add_argument('--val-sequence', type=str, default='02')
    parser.add_argument('--output-dir', type=str, default='output/segmented_cells_baselines')
    args = parser.parse_args()

    val_dataset = SegmentedCellDataset(
        root_dir=args.data_dir,
        split='val',
        val_sequence=args.val_sequence,
    )
    print(f'Validation crops: {len(val_dataset)}')

    metrics = {
        'horn_schunck': {'dice': [], 'runtime': []},
        'farneback':    {'dice': [], 'runtime': []},
        'tvl1':         {'dice': [], 'runtime': []},
    }

    for i in tqdm(range(len(val_dataset)), desc='Baselines'):
        sample = val_dataset[i]
        source = sample['source'][0].numpy()
        target = sample['target'][0].numpy()
        source_mask = sample['source_mask'][0].numpy()
        target_mask = sample['target_mask'][0].numpy()

        # Horn & Schunck — Lip6 multi-scale
        t0 = time.perf_counter()
        hs_flow = hs_optical_flow(reference=target, moving=source, alpha=1.0)
        hs_runtime = time.perf_counter() - t0
        hs_warped_mask = warp(source_mask, hs_flow)
        hs_dice = _binary_dice(hs_warped_mask, target_mask)
        if hs_dice is not None:
            metrics['horn_schunck']['dice'].append(hs_dice)
        metrics['horn_schunck']['runtime'].append(hs_runtime)

        # OpenCV needs uint8
        src_uint8 = (source * 255).astype(np.uint8)
        tgt_uint8 = (target * 255).astype(np.uint8)

        # Farneback
        t0 = time.perf_counter()
        flow_fb = cv2.calcOpticalFlowFarneback(
            src_uint8, tgt_uint8, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        fb_runtime = time.perf_counter() - t0
        fb_warped_mask = _warp_image_cv2(source_mask, flow_fb)
        fb_dice = _binary_dice(fb_warped_mask, target_mask)
        if fb_dice is not None:
            metrics['farneback']['dice'].append(fb_dice)
        metrics['farneback']['runtime'].append(fb_runtime)

        # TV-L1 (scikit-image — OpenCV's DualTVL1 was removed in opencv-contrib 4.6)
        t0 = time.perf_counter()
        flow_tvl1 = _tvl1_flow(source, target)
        tvl1_runtime = time.perf_counter() - t0
        tvl1_warped_mask = _warp_image_cv2(source_mask, flow_tvl1)
        tvl1_dice = _binary_dice(tvl1_warped_mask, target_mask)
        if tvl1_dice is not None:
            metrics['tvl1']['dice'].append(tvl1_dice)
        metrics['tvl1']['runtime'].append(tvl1_runtime)

    results = {}
    for method in ('horn_schunck', 'farneback', 'tvl1'):
        dice_scores = metrics[method]['dice']
        runtimes = metrics[method]['runtime']
        if not runtimes:
            continue
        results[method] = {
            'dice_mean':    float(np.mean(dice_scores)) if dice_scores else 0.0,
            'dice_std':     float(np.std(dice_scores))  if dice_scores else 0.0,
            'runtime_mean': float(np.mean(runtimes)),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'metrics.json').write_text(json.dumps(results, indent=2))

    print('\nDone. Results:')
    for method, label in [
        ('horn_schunck', 'Horn & Schunck'),
        ('farneback',    'Farneback'),
        ('tvl1',         'TV-L1'),
    ]:
        r = results.get(method)
        if r is None:
            print(f'  {label:14s}  [skipped]')
            continue
        print(f'  {label:14s}  Dice: {r["dice_mean"]:.4f} +/- {r["dice_std"]:.4f}  '
              f'Runtime: {r["runtime_mean"]:.4f} s/crop')
    print(f'\nMetrics saved to {output_dir / "metrics.json"}')


if __name__ == '__main__':
    main()
