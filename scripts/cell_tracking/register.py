"""
Register two 2D TIF images using a trained VoxelMorph model.

Usage:
    python -m scripts.cell_tracking.register \
        --moving dataset/test/01/t000.tif \
        --fixed dataset/test/01/t001.tif \
        --model output/best.pt \
        --moved result.tif \
        --warp warp.npy
"""

import argparse

import numpy as np
import torch
from skimage import io

import voxelmorph as vxm


def load_and_pad(path: str, pad_to: tuple[int, int] = (544, 704)) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Load a TIF image, normalize, and pad for model input.

    Returns
    -------
    tensor : torch.Tensor
        Image tensor (1, 1, H, W) ready for model.
    original_shape : tuple
        Original (H, W) before padding.
    """
    img = io.imread(path).astype(np.float32) / 255.0
    original_shape = img.shape

    if pad_to is not None:
        target_h, target_w = pad_to
        h, w = img.shape
        if h < target_h or w < target_w:
            padded = np.zeros((target_h, target_w), dtype=np.float32)
            padded[:h, :w] = img
            img = padded

    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0), original_shape


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Register two 2D TIF images with a trained VoxelMorph model'
    )
    parser.add_argument('--moving', required=True, help='Moving (source) TIF image')
    parser.add_argument('--fixed', required=True, help='Fixed (target) TIF image')
    parser.add_argument('--model', required=True, help='Trained model path (.pt)')
    parser.add_argument('--moved', required=True, help='Output warped image path (.tif)')
    parser.add_argument('--warp', default=None, help='Output displacement field path (.npy)')
    parser.add_argument('--nb-features', nargs='+', type=int,
                        default=[16, 32, 32, 32],
                        help='UNet features (must match training)')
    parser.add_argument('--int-steps', type=int, default=0,
                        help='Integration steps (must match training)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    # Load images
    moving, orig_shape = load_and_pad(args.moving)
    fixed, _ = load_and_pad(args.fixed)

    moving = moving.to(device)
    fixed = fixed.to(device)

    # Register
    with torch.no_grad():
        displacement, warped = model(
            moving, fixed,
            return_warped_source=True,
            return_field_type='displacement',
        )

    # Crop back to original size and save
    h, w = orig_shape
    moved_np = warped[0, 0, :h, :w].cpu().numpy()
    moved_uint8 = (np.clip(moved_np, 0, 1) * 255).astype(np.uint8)
    io.imsave(args.moved, moved_uint8)
    print(f'Saved warped image to {args.moved}')

    # Save displacement field
    if args.warp:
        disp_np = displacement[0, :, :h, :w].cpu().numpy()
        np.save(args.warp, disp_np)
        print(f'Saved displacement field to {args.warp}')


if __name__ == '__main__':
    main()
