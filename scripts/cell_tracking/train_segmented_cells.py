"""
Train VoxelMorph on per-cell segmented crop dataset with NCC+SSIM loss.

Loads NPZ crops from CellCropDataset (built by CellCropBuilder), trains
VxmPairwise with NCC+SSIM + spatial gradient regularization, validates on
held-out sequence 02 crops each epoch, and saves metrics.json in the same
format as evaluate.py so the Results table can include this experiment.

Usage:
    python -m scripts.cell_tracking.train_segmented_cells \
        --data-dir dataset/cell_crops \
        --epochs 200 \
        --output-dir output/cell_crops_ncc_ssim
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import neurite as ne

import voxelmorph as vxm
from pytorch_msssim import SSIM

from scripts.cell_tracking.cell_segmentation import CellCropDataset
from scripts.cell_tracking.train import (
    _BorderSpatialTransformer,
    _Negated,
    CombinedImageLoss,
    soft_dice_loss,
    dilate_mask,
)


def _build_loss() -> nn.Module:
    return CombinedImageLoss(
        loss_functions=[
            _Negated(ne.nn.modules.NCC()),
            _Negated(SSIM(data_range=1.0, size_average=True, channel=1)),
        ],
        weights=[1.0, 1.0],
    )


def _train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    image_loss_fn: nn.Module,
    grad_loss_fn: nn.Module,
    mask_warper: nn.Module,
    lambda_smooth: float,
    mask_weight: float,
    int_mask_weight: float,
    device: str,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in dataloader:
        source = batch['source'].to(device)
        target = batch['target'].to(device)
        source_binary = (batch['source_mask'].to(device) > 0).float()
        target_binary = (batch['target_mask'].to(device) > 0).float()

        optimizer.zero_grad()
        displacement, warped_source = model(
            source, target,
            return_warped_source=True,
            return_field_type='displacement',
        )

        union_binary = (source_binary + target_binary).clamp(0.0, 1.0)
        cell_region = dilate_mask(union_binary, radius=10)
        img_loss = image_loss_fn(target * cell_region, warped_source * cell_region)
        warped_mask = mask_warper(source_binary, displacement)
        mask_loss = soft_dice_loss(warped_mask, target_binary)
        grad_loss = grad_loss_fn(displacement)

        loss = mask_weight * mask_loss + int_mask_weight * img_loss + lambda_smooth * grad_loss
        loss.backward()
        optimizer.step()

        total += loss.item()
        n += 1
    return total / max(n, 1)


def _val_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    image_loss_fn: nn.Module,
    grad_loss_fn: nn.Module,
    mask_warper: nn.Module,
    lambda_smooth: float,
    mask_weight: float,
    int_mask_weight: float,
    device: str,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in dataloader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            source_binary = (batch['source_mask'].to(device) > 0).float()
            target_binary = (batch['target_mask'].to(device) > 0).float()

            displacement, warped_source = model(
                source, target,
                return_warped_source=True,
                return_field_type='displacement',
            )

            union_binary = (source_binary + target_binary).clamp(0.0, 1.0)
            cell_region = dilate_mask(union_binary, radius=10)
            img_loss = image_loss_fn(target * cell_region, warped_source * cell_region)
            warped_mask = mask_warper(source_binary, displacement)
            mask_loss = soft_dice_loss(warped_mask, target_binary)
            grad_loss = grad_loss_fn(displacement)

            loss = mask_weight * mask_loss + int_mask_weight * img_loss + lambda_smooth * grad_loss
            total += loss.item()
            n += 1
    return total / max(n, 1)


def _evaluate_dice(
    model: nn.Module,
    dataloader: DataLoader,
    nn_warper: nn.Module,
    device: str,
) -> dict:
    """
    Compute per-crop Dice and runtime on the validation set.
    Returns a dict matching the 'vxm' key format used by evaluate.py:
        {'dice_mean': float, 'dice_std': float, 'runtime_mean': float}
    """
    model.eval()
    dice_scores: list[float] = []
    runtimes: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            source_binary = (batch['source_mask'].to(device) > 0).float()
            target_binary = (batch['target_mask'].to(device) > 0).float()

            t0 = time.perf_counter()
            displacement, _ = model(
                source, target,
                return_warped_source=True,
                return_field_type='displacement',
            )
            runtimes.append((time.perf_counter() - t0) / source.shape[0])

            warped_mask = nn_warper(source_binary, displacement)
            warped_bin = (warped_mask > 0.5).float()

            for b in range(source.shape[0]):
                pred = warped_bin[b].reshape(-1)
                tgt = target_binary[b].reshape(-1)
                intersection = (pred * tgt).sum().item()
                denom = pred.sum().item() + tgt.sum().item()
                if denom > 0:
                    dice_scores.append(2.0 * intersection / denom)

    return {
        'dice_mean':    float(np.mean(dice_scores)),
        'dice_std':     float(np.std(dice_scores)),
        'runtime_mean': float(np.mean(runtimes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train VoxelMorph on segmented per-cell crop dataset (NCC+SSIM)'
    )
    parser.add_argument('--data-dir', type=str, default='dataset/cell_crops')
    parser.add_argument('--val-sequence', type=str, default='02')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda-reg', type=float, default=1.0)
    parser.add_argument('--mask-weight', type=float, default=1.0)
    parser.add_argument('--int-weight', type=float, default=0.1)
    parser.add_argument('--output-dir', type=str, default='output/cell_crops_ncc_ssim')
    parser.add_argument('--save-every', type=int, default=50)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = CellCropDataset(args.data_dir, split='train', val_sequence=args.val_sequence)
    val_dataset   = CellCropDataset(args.data_dir, split='val',   val_sequence=args.val_sequence)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = vxm.nn.models.VxmPairwise(
        ndim=2, source_channels=1, target_channels=1,
        nb_features=[16, 32, 32, 32], integration_steps=0,
    ).to(device)

    mask_warper_bilinear = _BorderSpatialTransformer(interpolation_mode='linear').to(device)
    mask_warper_nn       = _BorderSpatialTransformer(interpolation_mode='nearest').to(device)

    image_loss_fn = _build_loss()
    grad_loss_fn  = ne.nn.modules.SpatialGradient('l2')
    optimizer     = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Device: {device}  |  Train: {len(train_dataset)} crops  |  Val: {len(val_dataset)} crops')
    print(f'Loss: NCC+SSIM  |  lambda_reg: {args.lambda_reg}\n')

    best_val = float('inf')
    train_history: list[float] = []
    val_history:   list[float] = []

    for epoch in tqdm(range(1, args.epochs + 1), desc='Epochs'):
        train_loss = _train_epoch(
            model, train_loader, optimizer,
            image_loss_fn, grad_loss_fn, mask_warper_bilinear,
            args.lambda_reg, args.mask_weight, args.int_weight, device,
        )
        val_loss = _val_epoch(
            model, val_loader,
            image_loss_fn, grad_loss_fn, mask_warper_bilinear,
            args.lambda_reg, args.mask_weight, args.int_weight, device,
        )
        train_history.append(train_loss)
        val_history.append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}/{args.epochs} — train: {train_loss:.4f}  val: {val_loss:.4f}')

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), output_dir / f'checkpoint_epoch{epoch}.pt')

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), output_dir / 'best.pt')

    torch.save(model.state_dict(), output_dir / 'final.pt')

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs_range = range(1, args.epochs + 1)
    ax.plot(epochs_range, train_history, linewidth=2, color='#c0392b', label='Train')
    ax.plot(epochs_range, val_history,   linewidth=2, color='#2980b9', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (mask Dice + NCC+SSIM + grad)')
    ax.set_title('Training Curve — Segmented Cells NCC+SSIM')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    model.load_state_dict(torch.load(output_dir / 'best.pt', map_location=device))
    dice_metrics = _evaluate_dice(model, val_loader, mask_warper_nn, device)

    metrics = {'vxm': dice_metrics}
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    print(f'\nDone. Best val loss: {best_val:.4f}')
    print(f'Val Dice: {dice_metrics["dice_mean"]:.4f} +/- {dice_metrics["dice_std"]:.4f}')
    print(f'Metrics and model saved to {output_dir}/')


if __name__ == '__main__':
    main()
