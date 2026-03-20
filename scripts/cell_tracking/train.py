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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    image_loss_fn: nn.Module,
    grad_loss_fn: nn.Module,
    loss_weights: Sequence[float],
    device: str = 'cuda',
    negate_image_loss: bool = False,
) -> float:
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
        Weights [image_loss_weight, grad_loss_weight].
    device : str
        Device to train on.
    negate_image_loss : bool
        If True, negate the image loss (for NCC which returns similarity).
    """
    model.train()
    total_loss = 0.0
    total_sim = 0.0
    total_reg = 0.0
    n_batches = 0

    for batch in dataloader:
        source = batch['source'].to(device)
        target = batch['target'].to(device)

        optimizer.zero_grad()

        displacement, warped_source = model(
            source,
            target,
            return_warped_source=True,
            return_field_type='displacement',
        )

        img_loss = image_loss_fn(target, warped_source)
        if negate_image_loss:
            img_loss = -img_loss
        grad_loss = grad_loss_fn(displacement)
        loss = loss_weights[0] * img_loss + loss_weights[1] * grad_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_sim += img_loss.item()
        total_reg += grad_loss.item()
        n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, total_sim / n, total_reg / n


def main():
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
    dataset = CellTrackingDataset(
        data_dir=args.data_dir,
        sequences=args.sequences,
        pairing=args.pairing,
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
    loss_history = {'total': [], 'similarity': [], 'regularization': []}
    print(f'\nTraining for {args.epochs} epochs...\n')

    for epoch in tqdm(range(1, args.epochs + 1), desc='Epochs'):
        avg_loss, avg_sim, avg_reg = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            image_loss_fn=image_loss_fn,
            grad_loss_fn=grad_loss_fn,
            loss_weights=loss_weights,
            device=device,
            negate_image_loss=negate_image_loss,
        )

        loss_history['total'].append(avg_loss)
        loss_history['similarity'].append(avg_sim)
        loss_history['regularization'].append(avg_reg)

        # Log
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}/{args.epochs} — Loss: {avg_loss:.6f} '
                  f'(similarity: {avg_sim:.6f}, regularization: {avg_reg:.6f})')

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
