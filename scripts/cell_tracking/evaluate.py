"""
Evaluate deformation estimation: VoxelMorph vs classical baselines.

Metrics: MSE, Dice (with GT masks), Jacobian folding %, runtime.
Baseline: Horn & Schunck (1981).

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
from scripts.cell_tracking.baselines import horn_schunck, warp_image
from scripts.cell_tracking.track import warp_mask


def compute_jacobian_determinant(displacement):
    """Jacobian determinant of a 2D displacement field (2, H, W) -> (H, W)."""
    dudx = np.gradient(displacement[0], axis=1)
    dudy = np.gradient(displacement[0], axis=0)
    dvdx = np.gradient(displacement[1], axis=1)
    dvdy = np.gradient(displacement[1], axis=0)
    return (1 + dudx) * (1 + dvdy) - dudy * dvdx


def compute_dice_scores(warped_mask, target_mask):
    """Per-cell Dice scores. Returns {cell_id: dice}."""
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
    """Load GT tracking mask. Returns None if missing."""
    mask_path = gt_dir / f'man_track{frame_idx:03d}.tif'
    if not mask_path.exists():
        return None
    return io.imread(str(mask_path)).astype(np.int32)


def eval_displacement(src_np, tgt_np, displacement, warped, src_mask, tgt_mask):
    """Compute all metrics for a single displacement field."""
    mse = float(np.mean((tgt_np - warped) ** 2))
    jac = compute_jacobian_determinant(displacement)
    folding = 100.0 * np.sum(jac <= 0) / jac.size

    dice = None
    if src_mask is not None and tgt_mask is not None:
        warped_m = warp_mask(src_mask, displacement)
        scores = compute_dice_scores(warped_m, tgt_mask)
        if scores:
            dice = float(np.mean(list(scores.values())))

    return {'mse': mse, 'folding': folding, 'dice': dice}


# -- Visualization --

def visualize_registration(source, target, vxm_disp, method_warps,
                           save_path, pair_idx, src_mask=None, tgt_mask=None):
    """
    Single 2x3 figure comparing registration methods.

    Top row: Source, Target, Raw Difference (with quiver arrows)
    Bottom row: Horn & Schunck Squared Error, TV-L1 Squared Error, VoxelMorph Squared Error
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # -- Top row: the problem --

    # Source with cell contours
    axes[0, 0].imshow(source, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Source (moving)')
    axes[0, 0].axis('off')
    if src_mask is not None:
        axes[0, 0].contour(src_mask > 0, colors='cyan', linewidths=0.8, alpha=0.7)

    # Target with cell contours
    axes[0, 1].imshow(target, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Target (fixed)')
    axes[0, 1].axis('off')
    if tgt_mask is not None:
        axes[0, 1].contour(tgt_mask > 0, colors='lime', linewidths=0.8, alpha=0.7)

    # Raw difference with quiver arrows showing VoxelMorph displacement
    raw_diff = np.abs(target - source)
    im_diff = axes[0, 2].imshow(raw_diff, cmap='inferno', vmin=0, vmax=max(raw_diff.max(), 0.01))
    axes[0, 2].set_title(f'Raw Difference (MSE={np.mean((target - source)**2):.6f})')
    axes[0, 2].axis('off')
    fig.colorbar(im_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Quiver arrows for VoxelMorph displacement field
    if vxm_disp is not None:
        step = 20
        H, W = source.shape
        Y, X = np.mgrid[0:H:step, 0:W:step]
        dx, dy = vxm_disp[0], vxm_disp[1]
        axes[0, 2].quiver(X, Y, dx[::step, ::step], dy[::step, ::step],
                          color='white', alpha=0.6, scale=50, width=0.003)

    # -- Bottom row: the solutions (shared colorbar) --

    # Compute all squared error maps
    se_maps = {}
    for name, warped in method_warps.items():
        se_maps[name] = (target - warped) ** 2

    # Shared scale across all methods
    vmax = max(se.max() for se in se_maps.values())
    vmax = max(vmax, 0.01)

    method_order = ['Horn & Schunck Squared Error', 'VoxelMorph Squared Error']
    for j, name in enumerate(method_order):
        if name not in se_maps:
            axes[1, j].axis('off')
            continue
        se = se_maps[name]
        im = axes[1, j].imshow(se, cmap='inferno', vmin=0, vmax=vmax)
        axes[1, j].set_title(f'{name}\nMSE={np.mean(se):.6f}')
        axes[1, j].axis('off')

    # Hide unused third cell in bottom row
    axes[1, 2].axis('off')

    # Single shared colorbar for the bottom row
    fig.colorbar(im, ax=axes[1, :].tolist(), fraction=0.02, pad=0.04)

    plt.suptitle(f'Pair {pair_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / f'pair_{pair_idx:04d}.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


# -- Main --

def main():
    parser = argparse.ArgumentParser(description='Evaluate deformation estimation')
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

    # Load model
    model = vxm.nn.models.VxmPairwise(
        ndim=2, source_channels=1, target_channels=1,
        nb_features=args.nb_features, integration_steps=args.int_steps,
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    # Dataset
    dataset = CellTrackingDataset(
        data_dir=args.data_dir, sequences=args.sequences, pairing='consecutive',
    )
    n_pairs = len(dataset) if args.max_pairs == 0 else min(len(dataset), args.max_pairs)

    use_dice = args.gt_dir is not None
    use_baselines = not args.no_baselines
    print(f'Pairs: {n_pairs}, Dice: {use_dice}, Baselines: {use_baselines}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Methods to evaluate
    method_names = ['horn_schunck', 'vxm'] if use_baselines else ['vxm']
    metrics = {m: {'mse': [], 'dice': [], 'runtime': [], 'folding': []} for m in method_names}

    baseline_fns = {'horn_schunck': horn_schunck}
    display_names = {
        'horn_schunck': 'Horn & Schunck',
        'vxm': 'VoxelMorph',
    }

    print(f'\nEvaluating {n_pairs} pairs...\n')

    with torch.no_grad():
        for i in range(n_pairs):
            batch = dataset[i]
            source = batch['source'].unsqueeze(0).to(device)
            target = batch['target'].unsqueeze(0).to(device)
            src_np = source[0, 0].cpu().numpy()
            tgt_np = target[0, 0].cpu().numpy()

            # GT masks
            src_mask, tgt_mask = None, None
            if use_dice:
                sp, tp = dataset.pairs[i]
                seq = sp.parent.name
                gt_dir = Path(args.gt_dir) / f'{seq}_GT' / 'TRA'
                src_mask = load_gt_mask(gt_dir, int(sp.stem[1:]))
                tgt_mask = load_gt_mask(gt_dir, int(tp.stem[1:]))

            # VoxelMorph inference
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
            metrics['vxm']['folding'].append(vxm_res['folding'])
            metrics['vxm']['runtime'].append(vxm_time)
            if vxm_res['dice'] is not None:
                metrics['vxm']['dice'].append(vxm_res['dice'])

            # Baselines
            warps = {'VoxelMorph Squared Error': warp_np}
            if use_baselines:
                for key, fn in baseline_fns.items():
                    t0 = time.time()
                    bl_disp = fn(src_np, tgt_np)
                    bl_time = time.time() - t0

                    bl_warped = warp_image(src_np, bl_disp)
                    bl_res = eval_displacement(src_np, tgt_np, bl_disp, bl_warped, src_mask, tgt_mask)

                    metrics[key]['mse'].append(bl_res['mse'])
                    metrics[key]['folding'].append(bl_res['folding'])
                    metrics[key]['runtime'].append(bl_time)
                    if bl_res['dice'] is not None:
                        metrics[key]['dice'].append(bl_res['dice'])

                    warps[f'{display_names[key]} Squared Error'] = bl_warped

            # Visualize first 5 pairs
            if i < 5:
                visualize_registration(
                    src_np, tgt_np, disp_np, warps, output_dir, i,
                    src_mask=src_mask, tgt_mask=tgt_mask,
                )

            # Log
            log = f'  Pair {i}: VoxelMorph MSE={vxm_res["mse"]:.6f}'
            if vxm_res['dice'] is not None:
                log += f', Dice={vxm_res["dice"]:.4f}'
            print(log)

    # Summary table
    print(f'\n--- Results ({n_pairs} pairs) ---')
    for m in method_names:
        name = display_names[m]
        mse_str = f'{np.mean(metrics[m]["mse"]):.6f} +/- {np.std(metrics[m]["mse"]):.6f}'
        dice_str = 'N/A'
        if metrics[m]['dice']:
            dice_str = f'{np.mean(metrics[m]["dice"]):.4f} +/- {np.std(metrics[m]["dice"]):.4f}'
        fold_str = f'{np.mean(metrics[m]["folding"]):.2f}%'
        rt_str = f'{np.mean(metrics[m]["runtime"]):.4f}s' if metrics[m]['runtime'] else 'N/A'
        print(f'  {name}: MSE={mse_str}, Dice={dice_str}, Folding={fold_str}, Runtime={rt_str}')

    # Save to JSON
    results = {'n_pairs': n_pairs, 'device': device}
    for m in method_names:
        results[m] = {
            'mse_mean': float(np.mean(metrics[m]['mse'])),
            'mse_std': float(np.std(metrics[m]['mse'])),
            'folding_mean': float(np.mean(metrics[m]['folding'])),
            'folding_std': float(np.std(metrics[m]['folding'])),
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
