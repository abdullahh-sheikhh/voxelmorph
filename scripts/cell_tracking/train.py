"""
Train VoxelMorph for 2D cell tracking registration.

Uses VxmPairwise from the VoxelMorph library with ndim=2 to learn deformable
registration between consecutive phase-contrast microscopy frames of
Glioblastoma-astrocytoma U373 cells (PhC-C2DH-U373, Cell Tracking Challenge).

This script builds *over* the existing VoxelMorph codebase — no original files
are modified. We import VxmPairwise and loss functions from the library.

Paper reference:
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca.
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019.

Usage:
    python -m scripts.cell_tracking.train --data-dir dataset/train --epochs 500
    python -m scripts.cell_tracking.train --data-dir dataset/train --loss ncc --lambda 1.0 --epochs 500
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import neurite as ne

import voxelmorph as vxm
from scripts.cell_tracking.dataset import CellTrackingDataset


def soft_dice_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    Differentiable soft Dice loss for binary mask alignment.

    Parameters
    ----------
    prediction : torch.Tensor
        Soft warped mask (B, 1, H, W), values in [0, 1] from bilinear warp.
    target : torch.Tensor
        Hard binary target mask (B, 1, H, W), values in {0, 1}.
    smooth : float
        Laplace smoothing to avoid division by zero on empty masks.

    Returns
    -------
    torch.Tensor
        Scalar: 1 - mean Dice across batch.
    """
    pred_flat = prediction.reshape(prediction.size(0), -1)
    tgt_flat = target.reshape(target.size(0), -1)
    intersection = (pred_flat * tgt_flat).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        pred_flat.sum(dim=1) + tgt_flat.sum(dim=1) + smooth
    )
    return 1.0 - dice.mean()


def dilate_mask(binary_mask: torch.Tensor, radius: int = 10) -> torch.Tensor:
    """
    Morphological dilation of a binary mask via max pooling.

    Used to extend the cell region slightly so the cell-restricted intensity
    loss covers cells and their immediate neighbourhood (halos, edges).

    Parameters
    ----------
    binary_mask : torch.Tensor
        Binary mask (B, 1, H, W), values in {0, 1}.
    radius : int
        Dilation radius in pixels.

    Returns
    -------
    torch.Tensor
        Dilated binary mask (B, 1, H, W), clamped to [0, 1].
    """
    kernel = 2 * radius + 1
    return torch.nn.functional.max_pool2d(
        binary_mask, kernel_size=kernel, stride=1, padding=radius,
    ).clamp(0.0, 1.0)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    image_loss_fn: nn.Module,
    grad_loss_fn: nn.Module,
    lambda_smooth: float,
    device: str = 'cuda',
    negate_image_loss: bool = False,
    mask_warper: nn.Module | None = None,
    mask_weight: float = 1.0,
    int_mask_weight: float = 0.1,
) -> float:
    """
    Run one training epoch, return mean mask_dice loss across batches.

    Total loss per batch:
        loss = mask_weight * soft_dice + int_mask_weight * img_loss + lambda_smooth * grad_loss

    Only mask_dice is returned for logging — the other terms are small and
    dominated by mask_dice in practice.
    """
    model.train()
    total_mask = 0.0
    n_batches = 0

    for batch in dataloader:
        source = batch['source'].to(device)
        target = batch['target'].to(device)

        source_binary = (batch['source_mask'].to(device) > 0).float()
        target_binary = (batch['target_mask'].to(device) > 0).float()
        union_binary = (source_binary + target_binary).clamp(0.0, 1.0)
        cell_region = dilate_mask(union_binary, radius=10)

        optimizer.zero_grad()

        displacement, warped_source = model(
            source, target,
            return_warped_source=True,
            return_field_type='displacement',
        )

        img_loss = image_loss_fn(target * cell_region, warped_source * cell_region)
        if negate_image_loss:
            img_loss = -img_loss

        warped_mask = mask_warper(source_binary, displacement)
        mask_loss = soft_dice_loss(warped_mask, target_binary)

        grad_loss = grad_loss_fn(displacement)

        loss = mask_weight * mask_loss + int_mask_weight * img_loss + lambda_smooth * grad_loss
        loss.backward()
        optimizer.step()

        total_mask += mask_loss.item()
        n_batches += 1

    return total_mask / max(n_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train VoxelMorph for 2D cell tracking registration'
    )

    parser.add_argument('--data-dir', type=str, default='dataset/train',
                        help='Path to training data directory')
    parser.add_argument('--int-steps', type=int, default=0,
                        help='Integration steps (0=direct displacement, >0=diffeomorphic)')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'ncc'],
                        help='Image similarity loss (default: mse)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (paper default: 1)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (paper default: 1e-4)')
    parser.add_argument('--lambda', type=float, dest='lambda_param', default=None,
                        help='Smoothness weight (default: 0.01 for MSE, 1.0 for NCC)')
    parser.add_argument('--mask-weight', type=float, default=1.0,
                        help='Weight alpha for binary mask Dice loss')
    parser.add_argument('--int-weight', type=float, default=0.1,
                        help='Weight beta for cell-restricted intensity loss')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save model checkpoints')
    parser.add_argument('--save-every', type=int, default=50,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = CellTrackingDataset(data_dir=args.data_dir, use_masks=True)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )

    model = vxm.nn.models.VxmPairwise(
        ndim=2, source_channels=1, target_channels=1,
        nb_features=[16, 32, 32, 32], integration_steps=args.int_steps,
    ).to(device)

    # Bilinear SpatialTransformer for differentiable mask warping during training.
    mask_warper = vxm.nn.modules.SpatialTransformer(interpolation_mode='linear').to(device)

    # NCC returns positive similarity (1.0 = perfect) — negate for minimization.
    if args.loss == 'ncc':
        image_loss_fn = ne.nn.modules.NCC()
        negate_image_loss = True
        lambda_default = 1.0
    else:
        image_loss_fn = ne.nn.modules.MSE()
        negate_image_loss = False
        lambda_default = 0.01

    lambda_smooth = args.lambda_param if args.lambda_param is not None else lambda_default
    grad_loss_fn = ne.nn.modules.SpatialGradient('l2')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Device: {device}  |  Dataset: {len(dataset)} pairs  |  Loss: {args.loss.upper()}  '
          f'|  int_steps: {args.int_steps}  |  lambda: {lambda_smooth}\n')

    best_mask_dice = float('inf')
    mask_dice_history: list[float] = []

    for epoch in tqdm(range(1, args.epochs + 1), desc='Epochs'):
        avg_mask = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            image_loss_fn=image_loss_fn,
            grad_loss_fn=grad_loss_fn,
            lambda_smooth=lambda_smooth,
            device=device,
            negate_image_loss=negate_image_loss,
            mask_warper=mask_warper,
            mask_weight=args.mask_weight,
            int_mask_weight=args.int_weight,
        )
        mask_dice_history.append(avg_mask)

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}/{args.epochs} — mask_dice: {avg_mask:.6f}')

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), output_dir / f'checkpoint_epoch{epoch}.pt')

        if avg_mask < best_mask_dice:
            best_mask_dice = avg_mask
            torch.save(model.state_dict(), output_dir / 'best.pt')

    torch.save(model.state_dict(), output_dir / 'final.pt')

    # Loss curve — single line, the only signal we care about
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, args.epochs + 1), mask_dice_history, linewidth=2, color='#c0392b')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mask_dice (1 − Dice)')
    ax.set_title(f'Training Loss Curve  ({args.loss.upper()}, int_steps={args.int_steps})')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'\nDone. Best mask_dice: {best_mask_dice:.6f}  →  Dice ≈ {1 - best_mask_dice:.4f}')
    print(f'Models and loss curve saved to {output_dir}/')


if __name__ == '__main__':
    main()
