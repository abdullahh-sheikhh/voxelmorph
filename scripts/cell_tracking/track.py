"""
Track cells across a full TIF sequence using a trained VoxelMorph model.

Chains displacement fields from consecutive frame pairs to propagate
segmentation masks forward in time. Each cell keeps its ID across frames.

Pipeline:
    1. Load trained model + image sequence (t000–t114)
    2. Run pairwise registration on all consecutive pairs → displacement fields
    3. Load initial segmentation mask (frame 0 ground truth)
    4. Warp mask forward frame-by-frame using displacement fields
    5. Output: tracked masks per frame + cell centroid trajectories

Usage:
    python -m scripts.cell_tracking.track \
        --model output/best.pt \
        --data-dir dataset/train \
        --sequence 01 \
        --output-dir output/tracking
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from skimage import io
import matplotlib.pyplot as plt

import voxelmorph as vxm


def load_initial_mask(data_dir: Path, sequence: str) -> np.ndarray:
    """
    Load the first ground truth tracking mask for a sequence.

    Parameters
    ----------
    data_dir : Path
        Root data directory (e.g., dataset/train).
    sequence : str
        Sequence ID (e.g., '01').

    Returns
    -------
    np.ndarray
        Segmentation mask (H, W) with integer cell IDs (0 = background).
    """
    tra_dir = data_dir / f'{sequence}_GT' / 'TRA'
    mask_path = sorted(tra_dir.glob('man_track*.tif'))[0]
    mask = io.imread(str(mask_path))
    return mask.astype(np.int32)


def warp_mask(
    mask: np.ndarray,
    displacement: np.ndarray,
    pad_to: tuple[int, int] = (544, 704),
) -> np.ndarray:
    """
    Warp a segmentation mask using a displacement field.

    Uses nearest-neighbor interpolation to keep integer cell IDs intact.

    Parameters
    ----------
    mask : np.ndarray
        Segmentation mask (H, W) with integer cell labels.
    displacement : np.ndarray
        Displacement field (2, H_pad, W_pad) from model output.
    pad_to : tuple
        Padded size matching model input.

    Returns
    -------
    np.ndarray
        Warped mask (H, W) with preserved cell IDs.
    """
    orig_h, orig_w = mask.shape
    pad_h, pad_w = pad_to

    # Pad mask to model input size
    padded_mask = np.zeros((pad_h, pad_w), dtype=np.float32)
    padded_mask[:orig_h, :orig_w] = mask.astype(np.float32)

    # Build sampling grid: identity + displacement
    # displacement[0] = dx (horizontal), displacement[1] = dy (vertical)
    grid_y, grid_x = np.mgrid[0:pad_h, 0:pad_w].astype(np.float32)
    sample_x = grid_x + displacement[0]
    sample_y = grid_y + displacement[1]

    # Nearest-neighbor sampling to preserve integer labels
    sample_x = np.clip(np.round(sample_x).astype(int), 0, pad_w - 1)
    sample_y = np.clip(np.round(sample_y).astype(int), 0, pad_h - 1)

    warped = padded_mask[sample_y, sample_x]

    # Crop back to original size
    return warped[:orig_h, :orig_w].astype(np.int32)


def compute_centroids(mask: np.ndarray) -> dict[int, tuple[float, float]]:
    """
    Compute centroid (y, x) for each cell label in a mask.

    Parameters
    ----------
    mask : np.ndarray
        Segmentation mask with integer cell IDs.

    Returns
    -------
    dict
        {cell_id: (centroid_y, centroid_x)} for each non-zero label.
    """
    centroids = {}
    for label in np.unique(mask):
        if label == 0:
            continue
        ys, xs = np.where(mask == label)
        centroids[label] = (float(ys.mean()), float(xs.mean()))
    return centroids


def visualize_tracking(
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    trajectories: dict[int, list[tuple[float, float] | None]],
    output_dir: Path,
    every_n: int = 10,
):
    """
    Save tracking visualizations: overlay masks on frames + trajectory plot.

    Parameters
    ----------
    frames : list of np.ndarray
        Raw image frames.
    masks : list of np.ndarray
        Tracked segmentation masks (one per frame).
    trajectories : dict
        {cell_id: [(y, x) or None per frame]}.
    output_dir : Path
        Directory to save figures.
    every_n : int
        Visualize every N-th frame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Overlay masks on selected frames
    for i in range(0, len(frames), every_n):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(frames[i], cmap='gray')
        axes[0].set_title(f'Frame {i}')
        axes[0].axis('off')

        # Mask overlay
        axes[1].imshow(frames[i], cmap='gray')
        masked = np.ma.masked_where(masks[i] == 0, masks[i])
        axes[1].imshow(masked, cmap='tab10', alpha=0.5, vmin=1, vmax=10)
        axes[1].set_title(f'Tracked Mask (frame {i})')
        axes[1].axis('off')

        # Trajectories up to this frame
        axes[2].imshow(frames[i], cmap='gray')
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for cell_id, traj in trajectories.items():
            pts = [(t[1], t[0]) for t in traj[:i + 1] if t is not None]
            if len(pts) > 1:
                xs, ys = zip(*pts)
                c = colors[(cell_id - 1) % 10]
                axes[2].plot(xs, ys, '-', color=c, linewidth=1.5, label=f'Cell {cell_id}')
                axes[2].plot(xs[-1], ys[-1], 'o', color=c, markersize=5)
        axes[2].set_title(f'Trajectories (up to frame {i})')
        axes[2].axis('off')
        if i == 0:
            axes[2].legend(fontsize=7, loc='upper right')

        plt.tight_layout()
        plt.savefig(output_dir / f'track_frame_{i:04d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Full trajectory plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for cell_id, traj in trajectories.items():
        pts = [(t[1], t[0]) for t in traj if t is not None]
        if len(pts) > 1:
            xs, ys = zip(*pts)
            c = colors[(cell_id - 1) % 10]
            ax.plot(xs, ys, '-', color=c, linewidth=1.5, label=f'Cell {cell_id}')
            ax.plot(xs[0], ys[0], 's', color=c, markersize=6)
            ax.plot(xs[-1], ys[-1], 'o', color=c, markersize=6)
    ax.set_title('Cell Trajectories (full sequence)')
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectories.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Track cells across a TIF sequence using trained VoxelMorph'
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pt)')
    parser.add_argument('--data-dir', type=str, default='dataset/train',
                        help='Path to data directory')
    parser.add_argument('--sequence', type=str, default='01',
                        help='Sequence to track (default: 01)')
    parser.add_argument('--output-dir', type=str, default='output/tracking',
                        help='Directory to save tracking results')
    parser.add_argument('--nb-features', nargs='+', type=int,
                        default=[16, 32, 32, 32],
                        help='UNet features (must match training)')
    parser.add_argument('--int-steps', type=int, default=0,
                        help='Integration steps (must match training)')
    parser.add_argument('--viz-every', type=int, default=10,
                        help='Visualize every N frames (default: 10)')
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

    # Load all frames for this sequence
    data_dir = Path(args.data_dir)
    seq_dir = data_dir / args.sequence
    frame_paths = sorted(seq_dir.glob('t*.tif'))
    print(f'Sequence {args.sequence}: {len(frame_paths)} frames')

    # Load frames as raw images (for visualization) and as tensors (for model)
    pad_to = (544, 704)
    raw_frames = []
    tensors = []

    for path in frame_paths:
        img = io.imread(str(path)).astype(np.float32) / 255.0
        raw_frames.append(img)

        # Pad for model
        padded = np.zeros(pad_to, dtype=np.float32)
        padded[:img.shape[0], :img.shape[1]] = img
        tensors.append(torch.from_numpy(padded).unsqueeze(0).unsqueeze(0))

    # Load initial segmentation mask
    initial_mask = load_initial_mask(data_dir, args.sequence)
    cell_ids = [c for c in np.unique(initial_mask) if c != 0]
    print(f'Initial mask: {len(cell_ids)} cells (IDs: {cell_ids})')

    # Run pairwise registration on all consecutive pairs
    print(f'\nComputing displacement fields for {len(frame_paths) - 1} pairs...')
    displacements = []

    with torch.no_grad():
        for i in range(len(tensors) - 1):
            source = tensors[i].to(device)
            target = tensors[i + 1].to(device)

            displacement, _ = model(
                source, target,
                return_warped_source=True,
                return_field_type='displacement',
            )
            displacements.append(displacement[0].cpu().numpy())

            if (i + 1) % 20 == 0:
                print(f'  {i + 1}/{len(tensors) - 1} pairs done')

    print(f'  All {len(displacements)} displacement fields computed')

    # Propagate mask through the sequence
    print('\nPropagating segmentation masks...')
    tracked_masks = [initial_mask]
    trajectories: dict[int, list[tuple[float, float] | None]] = {
        cid: [] for cid in cell_ids
    }

    # Frame 0 centroids
    centroids_0 = compute_centroids(initial_mask)
    for cid in cell_ids:
        trajectories[cid].append(centroids_0.get(cid))

    current_mask = initial_mask
    for i, disp in enumerate(displacements):
        current_mask = warp_mask(current_mask, disp, pad_to=pad_to)
        tracked_masks.append(current_mask)

        centroids = compute_centroids(current_mask)
        for cid in cell_ids:
            trajectories[cid].append(centroids.get(cid))

    print(f'Tracked {len(cell_ids)} cells across {len(tracked_masks)} frames')

    # Summary: cells present in last frame
    final_centroids = compute_centroids(tracked_masks[-1])
    print(f'\nCells remaining in final frame: {list(final_centroids.keys())}')
    for cid in cell_ids:
        traj = trajectories[cid]
        n_present = sum(1 for t in traj if t is not None)
        print(f'  Cell {cid}: present in {n_present}/{len(traj)} frames')

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save tracked masks
    masks_dir = output_dir / 'masks'
    masks_dir.mkdir(exist_ok=True)
    for i, mask in enumerate(tracked_masks):
        io.imsave(str(masks_dir / f'mask_{i:04d}.tif'), mask.astype(np.uint16))

    # Save trajectories as numpy
    np.save(str(output_dir / 'trajectories.npy'), trajectories)

    # Visualize
    print('\nGenerating visualizations...')
    visualize_tracking(
        raw_frames, tracked_masks, trajectories, output_dir,
        every_n=args.viz_every,
    )

    print(f'\nResults saved to {output_dir}/')
    print(f'  masks/       — tracked segmentation masks per frame')
    print(f'  trajectories.npy — cell centroid positions over time')
    print(f'  *.png        — visualization figures')


if __name__ == '__main__':
    main()
