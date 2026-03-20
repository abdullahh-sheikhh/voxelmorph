"""
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
from scripts.cell_tracking.horn_schunck import horn_schunck, warp_image


def warp_mask(mask, displacement, pad_to=(544, 704)):
    """Warp segmentation mask using nearest-neighbor to preserve cell IDs."""
    orig_h, orig_w = mask.shape
    pad_h, pad_w = pad_to

    padded = np.zeros((pad_h, pad_w), dtype=np.float32)
    padded[:orig_h, :orig_w] = mask.astype(np.float32)

    grid_y, grid_x = np.mgrid[0:pad_h, 0:pad_w].astype(np.float32)
    sample_x = np.clip(np.round(grid_x + displacement[0]).astype(int), 0, pad_w - 1)
    sample_y = np.clip(np.round(grid_y + displacement[1]).astype(int), 0, pad_h - 1)

    return padded[sample_y, sample_x][:orig_h, :orig_w].astype(np.int32)


def compute_dice(warped_mask, target_mask):
    """Per-cell Dice scores between warped and target masks."""
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


def load_gt_mask(gt_dir, frame_idx):
    """Load GT segmentation mask (man_seg*.tif) for cell shape tracking."""
    mask_path = gt_dir / f'man_seg{frame_idx:03d}.tif'
    if not mask_path.exists():
        return None
    return io.imread(str(mask_path)).astype(np.int32)


def eval_displacement(src_np, tgt_np, displacement, warped, src_mask, tgt_mask):
    """Compute MSE and Dice for a displacement field."""
    mse = float(np.mean((tgt_np - warped) ** 2))

    dice = None
    if src_mask is not None and tgt_mask is not None:
        warped_m = warp_mask(src_mask, displacement)
        scores = compute_dice(warped_m, tgt_mask)
        if scores:
            dice = float(np.mean(list(scores.values())))

    return {'mse': mse, 'dice': dice}


def visualize_registration(source, target, vxm_disp, method_warps,
                           save_path, pair_idx, src_mask=None, tgt_mask=None):
    """
    2x3 grid:
    Top: Source + contours, Target + contours, Warped + both contours
    Bottom: H&S SE, VxM SE, Displacement magnitude
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)

    vxm_warped = method_warps.get('VoxelMorph')
    for ax, img, title in [
        (axes[0, 0], source, 'Source (moving)'),
        (axes[0, 1], target, 'Target (fixed)'),
        (axes[0, 2], vxm_warped if vxm_warped is not None else source, 'Warped Source'),
    ]:
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')

    if src_mask is not None:
        axes[0, 0].contour(src_mask > 0, colors='cyan', linewidths=2, alpha=0.8)
        axes[0, 2].contour(src_mask > 0, colors='cyan', linewidths=1.5, alpha=0.6)
    if tgt_mask is not None:
        axes[0, 1].contour(tgt_mask > 0, colors='lime', linewidths=2, alpha=0.8)
        axes[0, 2].contour(tgt_mask > 0, colors='lime', linewidths=2, alpha=0.8)

    # SE maps with shared scale
    se_maps = {}
    for name, warped in method_warps.items():
        se_maps[name] = (target - warped) ** 2

    vmax = max(se.max() for se in se_maps.values())
    vmax = max(vmax, 0.01)

    for j, name in enumerate(['Horn & Schunck', 'VoxelMorph']):
        if name not in se_maps:
            axes[1, j].axis('off')
            continue
        se = se_maps[name]
        im = axes[1, j].imshow(se, cmap='inferno', vmin=0, vmax=vmax)
        axes[1, j].set_title(f'{name} SE (MSE={np.mean(se):.6f})')
        axes[1, j].axis('off')
        fig.colorbar(im, ax=axes[1, j], fraction=0.046, pad=0.04)

    # Displacement magnitude
    if vxm_disp is not None:
        mag = np.sqrt(vxm_disp[0] ** 2 + vxm_disp[1] ** 2)
        im_mag = axes[1, 2].imshow(mag, cmap='viridis')
        axes[1, 2].set_title(f'Displacement (max={mag.max():.2f}px)')
        axes[1, 2].axis('off')
        fig.colorbar(im_mag, ax=axes[1, 2], fraction=0.046, pad=0.04, label='pixels')
    else:
        axes[1, 2].axis('off')

    plt.suptitle(f'Pair {pair_idx}', fontsize=14, fontweight='bold')
    plt.savefig(save_path / f'pair_{pair_idx:04d}.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='dataset/train')
    parser.add_argument('--gt-dir', type=str, default=None)
    parser.add_argument('--sequences', nargs='+', default=['01', '02'])
    parser.add_argument('--output-dir', type=str, default='output/eval')
    parser.add_argument('--nb-features', nargs='+', type=int, default=[16, 32, 32, 32])
    parser.add_argument('--int-steps', type=int, default=0)
    parser.add_argument('--max-pairs', type=int, default=20)
    parser.add_argument('--no-baselines', action='store_true')
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
    n_pairs = len(dataset) if args.max_pairs == 0 else min(len(dataset), args.max_pairs)

    use_dice = args.gt_dir is not None
    use_baselines = not args.no_baselines
    print(f'Pairs: {n_pairs}, Dice: {use_dice}, Baselines: {use_baselines}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_names = ['horn_schunck', 'vxm'] if use_baselines else ['vxm']
    metrics = {m: {'mse': [], 'dice': [], 'runtime': []} for m in method_names}
    display_names = {'horn_schunck': 'Horn & Schunck', 'vxm': 'VoxelMorph'}

    print(f'\nEvaluating {n_pairs} pairs...\n')

    with torch.no_grad():
        for i in range(n_pairs):
            batch = dataset[i]
            source = batch['source'].unsqueeze(0).to(device)
            target = batch['target'].unsqueeze(0).to(device)
            src_np = source[0, 0].cpu().numpy()
            tgt_np = target[0, 0].cpu().numpy()

            src_mask, tgt_mask = None, None
            if use_dice:
                sp, tp = dataset.pairs[i]
                seq = sp.parent.name
                gt_dir = Path(args.gt_dir) / f'{seq}_GT' / 'SEG'
                src_mask = load_gt_mask(gt_dir, int(sp.stem[1:]))
                tgt_mask = load_gt_mask(gt_dir, int(tp.stem[1:]))

            # VoxelMorph
            t0 = time.time()
            displacement, warped_source = model(
                source, target, return_warped_source=True, return_field_type='displacement',
            )
            if device == 'cuda':
                torch.cuda.synchronize()
            vxm_time = time.time() - t0

            disp_np = displacement[0].cpu().numpy()
            warp_np = warped_source[0, 0].cpu().numpy()

            vxm_res = eval_displacement(src_np, tgt_np, disp_np, warp_np, src_mask, tgt_mask)
            metrics['vxm']['mse'].append(vxm_res['mse'])
            metrics['vxm']['runtime'].append(vxm_time)
            if vxm_res['dice'] is not None:
                metrics['vxm']['dice'].append(vxm_res['dice'])

            # Horn & Schunck
            warps = {'VoxelMorph': warp_np}
            if use_baselines:
                t0 = time.time()
                hs_disp = horn_schunck(src_np, tgt_np)
                hs_time = time.time() - t0

                hs_warped = warp_image(src_np, hs_disp)
                hs_res = eval_displacement(src_np, tgt_np, hs_disp, hs_warped, src_mask, tgt_mask)

                metrics['horn_schunck']['mse'].append(hs_res['mse'])
                metrics['horn_schunck']['runtime'].append(hs_time)
                if hs_res['dice'] is not None:
                    metrics['horn_schunck']['dice'].append(hs_res['dice'])

                warps['Horn & Schunck'] = hs_warped

            if i < 5:
                visualize_registration(
                    src_np, tgt_np, disp_np, warps, output_dir, i,
                    src_mask=src_mask, tgt_mask=tgt_mask,
                )

            log = f'  Pair {i}: VxM MSE={vxm_res["mse"]:.6f}'
            if vxm_res['dice'] is not None:
                log += f', Dice={vxm_res["dice"]:.4f}'
            print(log)

    # Summary
    print(f'\n--- Results ({n_pairs} pairs) ---')
    for m in method_names:
        name = display_names[m]
        mse_str = f'{np.mean(metrics[m]["mse"]):.6f} +/- {np.std(metrics[m]["mse"]):.6f}'
        dice_str = 'N/A'
        if metrics[m]['dice']:
            dice_str = f'{np.mean(metrics[m]["dice"]):.4f} +/- {np.std(metrics[m]["dice"]):.4f}'
        rt_str = f'{np.mean(metrics[m]["runtime"]):.4f}s'
        print(f'  {name}: MSE={mse_str}, Dice={dice_str}, Runtime={rt_str}')

    results = {'n_pairs': n_pairs, 'device': device}
    for m in method_names:
        results[m] = {
            'mse_mean': float(np.mean(metrics[m]['mse'])),
            'mse_std': float(np.std(metrics[m]['mse'])),
        }
        if metrics[m]['runtime']:
            results[m]['runtime_mean'] = float(np.mean(metrics[m]['runtime']))
            results[m]['runtime_std'] = float(np.std(metrics[m]['runtime']))
        if metrics[m]['dice']:
            results[m]['dice_mean'] = float(np.mean(metrics[m]['dice']))
            results[m]['dice_std'] = float(np.std(metrics[m]['dice']))

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {output_dir}/')


if __name__ == '__main__':
    main()
