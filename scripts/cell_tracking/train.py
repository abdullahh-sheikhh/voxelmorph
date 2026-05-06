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
    python -m scripts.cell_tracking.train --data-dir dataset/train --sequences 01 --epochs 150
    python -m scripts.cell_tracking.train --data-dir dataset/train --sequences 01 --loss ncc --lambda 1.0 --epochs 150
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
from pytorch_msssim import SSIM
from scripts.cell_tracking.dataset import CellTrackingDataset


class _BorderSpatialTransformer(vxm.nn.modules.SpatialTransformer):
    """SpatialTransformer that uses border (edge-replication) padding instead of zeros.

    Used for mask warping only, so that cell labels near the image boundary are
    replicated rather than set to zero, which would create artificial black halos
    that bias the masked intensity loss.
    """

    def forward(
        self,
        moving_image: torch.Tensor,
        deformation_field: torch.Tensor,
    ) -> torch.Tensor:
        spatial_shape = moving_image.shape[2:]
        if (
            not hasattr(self, 'meshgrid')
            or self.meshgrid.shape[1:] != spatial_shape
            or self.meshgrid.device != moving_image.device
            or self.meshgrid.dtype != moving_image.dtype
        ):
            self.meshgrid = ne.volshape_to_ndgrid(
                size=spatial_shape,
                device=moving_image.device,
                dtype=moving_image.dtype,
                stack=True,
            )

        return vxm.spatial_transform(
            image=moving_image,
            trf=deformation_field,
            mode=self.interpolation_mode,
            isdisp=True,
            meshgrid=self.meshgrid,
            non_spatial_dims=(0, 1),
            align_corners=self.align_corners,
            padding_mode='border',
        )


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


class CombinedImageLoss(nn.Module):
    """Weighted sum of image similarity losses.

    Handles per-component sign: similarity losses (NCC, SSIM) that return
    positive-is-good values are negated internally so the combined loss is
    always minimised by gradient descent.

    Parameters
    ----------
    loss_functions : list[nn.Module]
        Individual loss or similarity callables, each taking (target, warped).
    weights : list[float]
        Per-component scalar weights.
    negations : list[bool]
        True for similarity losses (NCC, SSIM); False for error losses (MSE).
    """

    def __init__(
        self,
        loss_functions: list[nn.Module],
        weights: list[float],
        negations: list[bool],
    ) -> None:
        super().__init__()
        self.loss_fns = nn.ModuleList(loss_functions)
        self.weights = weights
        self.negations = negations

    def forward(self, target: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
        total = target.new_zeros(())
        for loss_fn, weight, negate in zip(self.loss_fns, self.weights, self.negations):
            value = loss_fn(target, warped)
            total = total + weight * (-value if negate else value)
        return total


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
    unsupervised: bool = False,
) -> float:
    """
    Run one training epoch, return mean optimization metric across batches.

    Total loss per batch:
        mask-guided:
            loss = mask_weight * soft_dice + int_mask_weight * img_loss + lambda_smooth * grad_loss

        unsupervised:
            loss = img_loss + lambda_smooth * grad_loss

    The returned scalar is the mean objective used for model selection/logging.
    """
    model.train()
    total_metric = 0.0
    n_batches = 0

    for batch in dataloader:
        source = batch['source'].to(device)
        target = batch['target'].to(device)

        optimizer.zero_grad()

        displacement, warped_source = model(
            source, target,
            return_warped_source=True,
            return_field_type='displacement',
        )

        if unsupervised:
            img_loss = image_loss_fn(target, warped_source)
        else:
            source_binary = (batch['source_mask'].to(device) > 0).float()
            target_binary = (batch['target_mask'].to(device) > 0).float()
            union_binary = (source_binary + target_binary).clamp(0.0, 1.0)
            cell_region = dilate_mask(union_binary, radius=10)
            img_loss = image_loss_fn(target * cell_region, warped_source * cell_region)

        if negate_image_loss:
            img_loss = -img_loss

        grad_loss = grad_loss_fn(displacement)

        if unsupervised:
            loss = img_loss + lambda_smooth * grad_loss
            metric = loss
        else:
            warped_mask = mask_warper(source_binary, displacement)
            mask_loss = soft_dice_loss(warped_mask, target_binary)
            loss = mask_weight * mask_loss + int_mask_weight * img_loss + lambda_smooth * grad_loss
            metric = mask_loss

        loss.backward()
        optimizer.step()

        total_metric += metric.item()
        n_batches += 1

    return total_metric / max(n_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train VoxelMorph for 2D cell tracking registration'
    )

    parser.add_argument('--data-dir', type=str, default='dataset/train',
                        help='Path to training data directory')
    parser.add_argument('--sequences', nargs='+', default=['01', '02'],
                        help='Sequence IDs to train on')
    parser.add_argument('--loss', type=str, default='mse',
                        choices=['mse', 'ncc', 'ssim', 'ncc+ssim', 'mse+ncc+ssim'],
                        help='Image similarity loss or combination (default: mse)')
    parser.add_argument('--mse-weight', type=float, default=1.0,
                        help='MSE component weight for combined losses')
    parser.add_argument('--ncc-weight', type=float, default=1.0,
                        help='NCC component weight for combined losses')
    parser.add_argument('--ssim-weight', type=float, default=1.0,
                        help='SSIM component weight for combined losses')
    parser.add_argument('--epochs', type=int, default=150,
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
    parser.add_argument('--unsupervised', action='store_true',
                        help='Disable mask Dice supervision and train with image loss + smoothness only')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save model checkpoints')
    parser.add_argument('--save-every', type=int, default=50,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = CellTrackingDataset(
        data_dir=args.data_dir,
        sequences=args.sequences,
        use_masks=not args.unsupervised,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )

    model = vxm.nn.models.VxmPairwise(
        ndim=2, source_channels=1, target_channels=1,
        nb_features=[16, 32, 32, 32], integration_steps=0,
    ).to(device)

    mask_warper = None
    if not args.unsupervised:
        # Bilinear SpatialTransformer for differentiable mask warping during training.
        mask_warper = _BorderSpatialTransformer(interpolation_mode='linear').to(device)

    # NCC and SSIM return positive similarity (1.0 = perfect) — negate for minimization.
    # CombinedImageLoss handles per-component negation internally, so negate_image_loss=False.
    if args.loss == 'ncc':
        image_loss_fn = ne.nn.modules.NCC()
        negate_image_loss = True
        lambda_default = 1.0
    elif args.loss == 'ssim':
        image_loss_fn = SSIM(data_range=1.0, size_average=True, channel=1)
        negate_image_loss = True
        lambda_default = 1.0
    elif args.loss == 'ncc+ssim':
        image_loss_fn = CombinedImageLoss(
            loss_functions=[
                ne.nn.modules.NCC(),
                SSIM(data_range=1.0, size_average=True, channel=1),
            ],
            weights=[args.ncc_weight, args.ssim_weight],
            negations=[True, True],
        )
        negate_image_loss = False
        lambda_default = 1.0
    elif args.loss == 'mse+ncc+ssim':
        image_loss_fn = CombinedImageLoss(
            loss_functions=[
                ne.nn.modules.MSE(),
                ne.nn.modules.NCC(),
                SSIM(data_range=1.0, size_average=True, channel=1),
            ],
            weights=[args.mse_weight, args.ncc_weight, args.ssim_weight],
            negations=[False, True, True],
        )
        negate_image_loss = False
        lambda_default = 1.0
    else:  # mse
        image_loss_fn = ne.nn.modules.MSE()
        negate_image_loss = False
        lambda_default = 0.01

    lambda_smooth = args.lambda_param if args.lambda_param is not None else lambda_default
    grad_loss_fn = ne.nn.modules.SpatialGradient('l2')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_mode = 'unsupervised' if args.unsupervised else 'mask-guided'
    print(f'Device: {device}  |  Dataset: {len(dataset)} pairs  |  Sequences: {args.sequences}  '
          f'|  Loss: {args.loss.upper()}  |  Mode: {training_mode}  '
          f'|  lambda: {lambda_smooth}\n')

    best_metric = float('inf')
    metric_history: list[float] = []
    metric_name = 'train_loss' if args.unsupervised else 'mask_dice_loss'

    for epoch in tqdm(range(1, args.epochs + 1), desc='Epochs'):
        avg_metric = train_epoch(
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
            unsupervised=args.unsupervised,
        )
        metric_history.append(avg_metric)

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}/{args.epochs} — {metric_name}: {avg_metric:.6f}')

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), output_dir / f'checkpoint_epoch{epoch}.pt')

        if avg_metric < best_metric:
            best_metric = avg_metric
            torch.save(model.state_dict(), output_dir / 'best.pt')

    torch.save(model.state_dict(), output_dir / 'final.pt')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, args.epochs + 1), metric_history, linewidth=2, color='#c0392b')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric_name)
    ax.set_title(
        f'Training Curve ({args.loss.upper()}, {training_mode})'
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'\nDone. Best {metric_name}: {best_metric:.6f}')
    print(f'Models and loss curve saved to {output_dir}/')


if __name__ == '__main__':
    main()
