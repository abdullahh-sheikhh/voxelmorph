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
import json
from pathlib import Path
from typing import Sequence

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
    loss_weights: Sequence[float],
    device: str = 'cuda',
    negate_image_loss: bool = False,
    mask_warper: nn.Module | None = None,
    mask_weight: float = 1.0,
    int_mask_weight: float = 0.1,
) -> tuple[float, float, float, float]:
    """
    model : nn.Module
        VxmPairwise model.
    dataloader : DataLoader
        DataLoader over CellTrackingDataset.
    optimizer : torch.optim.Optimizer
        Optimizer (ADAM recommended).
    image_loss_fn : nn.Module
        Image similarity loss (MSE or NCC).
    grad_loss_fn : nn.Module
        Displacement field regularization loss.
    loss_weights : Sequence[float]
        Weights [unused_slot, grad_loss_weight]. Index 1 is lambda (smoothness).
        Index 0 is kept for backward compatibility but superseded by int_mask_weight.
    device : str
        Device to train on.
    negate_image_loss : bool
        If True, negate the image loss (for NCC which returns similarity).
    mask_warper : nn.Module or None
        SpatialTransformer(interpolation_mode='linear') for differentiable mask warping.
        When None (or when batch has no masks), falls back to full-image intensity loss.
    mask_weight : float
        Weight alpha for the binary mask Dice loss term.
    int_mask_weight : float
        Weight beta for the cell-restricted intensity similarity loss term.

    Returns
    -------
    tuple[float, float, float, float]
        Average total loss, average similarity loss, average regularization loss,
        average mask Dice loss (0.0 when masks not available).
    """
    model.train()
    total_loss = 0.0
    total_sim = 0.0
    total_reg = 0.0
    total_mask = 0.0
    n_batches = 0

    for batch in dataloader:
        source = batch['source'].to(device)
        target = batch['target'].to(device)

        # Pre-compute binary masks once per batch when available
        has_masks = 'source_mask' in batch and mask_warper is not None
        if has_masks:
            source_binary = (batch['source_mask'].to(device) > 0).float()
            target_binary = (batch['target_mask'].to(device) > 0).float()
            # Union of both frames, dilated: covers cells + halos + small motion margin
            union_binary = (source_binary + target_binary).clamp(0.0, 1.0)
            cell_region = dilate_mask(union_binary, radius=10)

        optimizer.zero_grad()

        displacement, warped_source = model(
            source,
            target,
            return_warped_source=True,
            return_field_type='displacement',
        )

        # Image similarity — restricted to cell neighbourhood when masks available,
        # full-image fallback preserves backward compatibility (use_masks=False).
        if has_masks and int_mask_weight > 0.0:
            img_loss = image_loss_fn(target * cell_region, warped_source * cell_region)
        else:
            img_loss = image_loss_fn(target, warped_source)
        if negate_image_loss:
            img_loss = -img_loss

        # Mask Dice loss — primary signal.
        # Warp source binary mask with the predicted displacement (bilinear, differentiable).
        # Compare against the target binary mask.
        # Births: target has pixels the model can never produce -> accepted miss.
        # Deaths: model may warp dead-cell pixels elsewhere -> model learns to suppress.
        mask_loss = torch.tensor(0.0, device=device)
        if has_masks and mask_weight > 0.0:
            warped_mask = mask_warper(source_binary, displacement)
            mask_loss = soft_dice_loss(warped_mask, target_binary)

        # Smoothness regularization (unchanged)
        grad_loss = grad_loss_fn(displacement)

        loss = (
            mask_weight * mask_loss
            + int_mask_weight * img_loss
            + loss_weights[1] * grad_loss
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_sim += img_loss.item()
        total_reg += grad_loss.item()
        total_mask += mask_loss.item()
        n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, total_sim / n, total_reg / n, total_mask / n


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train VoxelMorph for 2D cell tracking registration'
    )

    # Data
    parser.add_argument('--data-dir', type=str, default='dataset/train',
                        help='Path to training data directory')
    parser.add_argument('--sequences', nargs='+', default=['01', '02'],
                        help='Sequence folders to use (default: 01 02)')
    parser.add_argument('--pairing', type=str, default='consecutive',
                        choices=['consecutive', 'random'],
                        help='Frame pairing strategy (default: consecutive)')

    # Model
    parser.add_argument('--nb-features', nargs='+', type=int,
                        default=[16, 32, 32, 32],
                        help='UNet feature counts per level (default: 16 32 32 32)')
    parser.add_argument('--int-steps', type=int, default=0,
                        help='Integration steps (0=direct displacement, >0=diffeomorphic)')

    # Training
    parser.add_argument('--loss', type=str, default='mse',
                        choices=['mse', 'ncc'],
                        help='Image similarity loss (default: mse)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (paper default: 1)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (paper default: 1e-4)')
    parser.add_argument('--lambda', type=float, dest='lambda_param', default=None,
                        help='Regularization weight (default: 0.01 for MSE, 1.0 for NCC)')
    parser.add_argument('--workers', type=int, default=0,
                        help='DataLoader workers')
    parser.add_argument('--mask-weight', type=float, default=1.0,
                        help='Weight alpha for binary mask Dice loss (default: 1.0, 0 to disable)')
    parser.add_argument('--int-weight', type=float, default=0.1,
                        help='Weight beta for cell-restricted intensity loss (default: 0.1)')

    # Output
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save model checkpoints')
    parser.add_argument('--save-every', type=int, default=50,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Dataset
    use_masks = args.mask_weight > 0.0 or args.int_weight > 0.0
    dataset = CellTrackingDataset(
        data_dir=args.data_dir,
        sequences=args.sequences,
        pairing=args.pairing,
        use_masks=use_masks,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    print(f'Dataset: {len(dataset)} pairs from sequences {args.sequences}')

    # Model — 2D VxmPairwise
    model = vxm.nn.models.VxmPairwise(
        ndim=2,
        source_channels=1,
        target_channels=1,
        nb_features=args.nb_features,
        integration_steps=args.int_steps,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: VxmPairwise(ndim=2, features={args.nb_features}, '
          f'int_steps={args.int_steps}), {param_count:,} parameters')

    # Spatial transformer for differentiable bilinear mask warping during training.
    # Uses the same interface as the model's internal SpatialTransformer.
    # interpolation_mode='linear' maps to bilinear in 2D, giving soft outputs in [0,1]
    # so that gradients flow back through the warped mask into the displacement field.
    mask_warper = vxm.nn.modules.SpatialTransformer(
        interpolation_mode='linear'
    ).to(device)

    # Loss functions (from neurite, as per codebase conventions)
    # NCC returns positive similarity (1.0 = perfect) — negate for minimization
    if args.loss == 'ncc':
        image_loss_fn = ne.nn.modules.NCC()
        negate_image_loss = True
        lambda_default = 1.0
    else:
        image_loss_fn = ne.nn.modules.MSE()
        negate_image_loss = False
        lambda_default = 0.01

    lambda_param = args.lambda_param if args.lambda_param is not None else lambda_default
    grad_loss_fn = ne.nn.modules.SpatialGradient('l2')
    loss_weights = [1.0, lambda_param]
    loss_name = args.loss.upper()
    print(f'Loss: {loss_name} + {lambda_param} * SpatialGradient(L2)')

    # Optimizer (ADAM, paper default)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_loss = float('inf')
    loss_history = {'total': [], 'similarity': [], 'regularization': [], 'mask_dice': []}
    print(f'\nTraining for {args.epochs} epochs...')
    print(f'Loss: {loss_name} (alpha={args.mask_weight} mask_dice, '
          f'beta={args.int_weight} intensity, lambda={lambda_param} smooth)\n')

    for epoch in tqdm(range(1, args.epochs + 1), desc='Epochs'):
        avg_loss, avg_sim, avg_reg, avg_mask = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            image_loss_fn=image_loss_fn,
            grad_loss_fn=grad_loss_fn,
            loss_weights=loss_weights,
            device=device,
            negate_image_loss=negate_image_loss,
            mask_warper=mask_warper,
            mask_weight=args.mask_weight,
            int_mask_weight=args.int_weight,
        )

        loss_history['total'].append(avg_loss)
        loss_history['similarity'].append(avg_sim)
        loss_history['regularization'].append(avg_reg)
        loss_history['mask_dice'].append(avg_mask)

        # Log
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}/{args.epochs} — Loss: {avg_loss:.6f} '
                  f'(mask_dice: {avg_mask:.6f}, similarity: {avg_sim:.6f}, '
                  f'regularization: {avg_reg:.6f})')

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = output_dir / f'checkpoint_epoch{epoch}.pt'
            torch.save(model.state_dict(), ckpt_path)
            print(f'  Checkpoint: {ckpt_path}')

        # Best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_dir / 'best.pt')

    # Final model
    torch.save(model.state_dict(), output_dir / 'final.pt')

    # Save loss history
    with open(output_dir / 'loss_history.json', 'w') as f:
        json.dump(loss_history, f)

    # Plot loss curves
    epochs = range(1, args.epochs + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, loss_history['total'], label='Total Loss', linewidth=2)
    ax.plot(epochs, loss_history['similarity'], label=f'Similarity ({loss_name})', linewidth=1.5, alpha=0.8)
    ax.plot(epochs, loss_history['regularization'], label='Regularization (Spatial Gradient)', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training Loss Curve ({loss_name} + {lambda_param} * Spatial Gradient)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save training config for reproducibility
    config = {
        'loss': args.loss,
        'lambda': lambda_param,
        'mask_weight': args.mask_weight,
        'int_weight': args.int_weight,
        'int_steps': args.int_steps,
        'nb_features': args.nb_features,
        'lr': args.lr,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'best_loss': best_loss,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f'\nDone. Best loss: {best_loss:.6f}')
    print(f'Models saved to {output_dir}/')
    print(f'Loss curve saved to {output_dir / "loss_curve.png"}')


if __name__ == '__main__':
    main()
