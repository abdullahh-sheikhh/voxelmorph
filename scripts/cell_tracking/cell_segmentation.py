"""
Per-cell segmentation dataset builder and loader for VoxelMorph cell registration.

Extracts individual cell crops from PhC-C2DH-U373 consecutive frame pairs.
Each crop is a 224x224 region centered on the average centroid of the cell
in source and target frames, with all neighboring cells zeroed out.

Saves one NPZ file per valid cell pair:
    {seq}_f{t:03d}_f{t+1:03d}_c{label}.npz
    Keys: source, target, source_mask, target_mask — all float32 (224, 224)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io


class SegmentedCellBuilder:
    """
    Preprocesses the Cell Tracking Challenge dataset into per-cell NPZ crops.

    Parameters
    ----------
    data_dir : str or Path
        Root of the training dataset (e.g., dataset/train/).
        Expected layout: data_dir/{seq}/t*.tif and data_dir/{seq}_ST/SEG/man_seg*.tif
    output_dir : str or Path
        Directory where NPZ crop files will be written.
    crop_size : int
        Side length of the square crop window in pixels. Must be divisible by 32.
    """

    def __init__(
        self,
        data_dir: str | Path,
        output_dir: str | Path,
        crop_size: int = 224,
    ) -> None:
        assert crop_size % 32 == 0, f'crop_size must be divisible by 32 for the 5-level UNet, got {crop_size}'
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.crop_size = crop_size

    def build(self, sequences: list[str] | None = None) -> None:
        """
        Extract all valid per-cell crops and save them as NPZ files.

        For each consecutive frame pair in each sequence, finds cell labels
        present in both source and target ST masks, computes average centroids,
        and saves one NPZ file per valid cell.

        Parameters
        ----------
        sequences : list of str, optional
            Sequence IDs to process. Defaults to ['01', '02'].
        """
        if sequences is None:
            sequences = ['01', '02']

        self.output_dir.mkdir(parents=True, exist_ok=True)

        total_saved = 0
        total_skipped = 0
        total_pairs = 0

        for seq in sequences:
            seq_dir = self.data_dir / seq
            frames = sorted(seq_dir.glob('t*.tif'))

            for i in range(len(frames) - 1):
                source_path = frames[i]
                target_path = frames[i + 1]
                t_source = int(source_path.stem[1:])
                t_target = int(target_path.stem[1:])
                total_pairs += 1

                source_mask = self._load_mask(seq, t_source)
                target_mask = self._load_mask(seq, t_target)

                source_labels = set(np.unique(source_mask).astype(int)) - {0}
                target_labels = set(np.unique(target_mask).astype(int)) - {0}
                common_labels = source_labels & target_labels

                if not common_labels:
                    continue

                source_img = self._load_image(source_path)
                target_img = self._load_image(target_path)
                img_h, img_w = source_img.shape

                for label in sorted(common_labels):
                    src_centroid = self._compute_centroid(source_mask, label)
                    tgt_centroid = self._compute_centroid(target_mask, label)

                    if src_centroid is None or tgt_centroid is None:
                        total_skipped += 1
                        continue

                    avg_cy = (src_centroid[0] + tgt_centroid[0]) / 2.0
                    avg_cx = (src_centroid[1] + tgt_centroid[1]) / 2.0

                    left, top, right, bottom = self._crop_window(
                        center_x=avg_cx,
                        center_y=avg_cy,
                        img_w=img_w,
                        img_h=img_h,
                    )

                    src_isolated = self._isolate_cell(source_img, source_mask, label)
                    tgt_isolated = self._isolate_cell(target_img, target_mask, label)
                    src_mask_bin = (source_mask == label).astype(np.float32)
                    tgt_mask_bin = (target_mask == label).astype(np.float32)

                    crop = {
                        'source':      src_isolated[top:bottom, left:right].astype(np.float32),
                        'target':      tgt_isolated[top:bottom, left:right].astype(np.float32),
                        'source_mask': src_mask_bin[top:bottom, left:right],
                        'target_mask': tgt_mask_bin[top:bottom, left:right],
                    }

                    out_path = (
                        self.output_dir
                        / f'{seq}_f{t_source:03d}_f{t_target:03d}_c{label}.npz'
                    )
                    np.savez_compressed(str(out_path), **crop)
                    total_saved += 1

        print(f'Done — {total_pairs} pairs processed')
        print(f'       {total_saved} crops saved to {self.output_dir}/')
        print(f'       {total_skipped} cells skipped (label missing in one frame)')

    def _load_image(self, path: Path) -> np.ndarray:
        return io.imread(str(path)).astype(np.float32) / 255.0

    def _load_mask(self, seq: str, frame_index: int) -> np.ndarray:
        mask_path = (
            self.data_dir
            / f'{seq}_ST'
            / 'SEG'
            / f'man_seg{frame_index:03d}.tif'
        )
        if not mask_path.exists():
            print(f'Warning: mask not found, skipping pairs for {seq} frame {frame_index}: {mask_path}')
            return np.zeros((520, 696), dtype=np.float32)
        return io.imread(str(mask_path)).astype(np.float32)

    def _compute_centroid(
        self, mask: np.ndarray, label: int
    ) -> Optional[tuple[float, float]]:
        """Return (cy, cx) centroid of the given label in the mask, or None if absent."""
        ys, xs = np.where(mask == label)
        if len(ys) == 0:
            return None
        return float(ys.mean()), float(xs.mean())

    def _crop_window(
        self,
        center_x: float,
        center_y: float,
        img_w: int,
        img_h: int,
    ) -> tuple[int, int, int, int]:
        """
        Return (left, top, right, bottom) for a crop_size x crop_size window
        centered as close to (center_x, center_y) as possible while staying
        entirely inside the image bounds.
        """
        half = self.crop_size // 2
        left = int(round(center_x)) - half
        top = int(round(center_y)) - half
        left = max(0, min(left, img_w - self.crop_size))
        top = max(0, min(top, img_h - self.crop_size))
        return left, top, left + self.crop_size, top + self.crop_size

    def _isolate_cell(
        self, image: np.ndarray, mask: np.ndarray, label: int
    ) -> np.ndarray:
        """Zero out all pixels that do not belong to the target cell label."""
        isolated = image.copy()
        isolated[mask != label] = 0.0
        return isolated


class SegmentedCellDataset(Dataset):
    """
    PyTorch Dataset that loads pre-built per-cell NPZ crops.

    Each NPZ file contains: source, target, source_mask, target_mask —
    all float32 arrays of shape (224, 224). Files are split by sequence
    prefix in the filename: val_sequence goes to the val split, all others
    to train.

    Parameters
    ----------
    root_dir : str or Path
        Directory containing the NPZ crop files (output of SegmentedCellBuilder).
    split : str
        'train' or 'val'.
    val_sequence : str
        The sequence ID reserved for validation (e.g., '02').
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = 'train',
        val_sequence: str = '02',
    ) -> None:
        root = Path(root_dir)
        all_files = sorted(root.glob('*.npz'))
        val_prefix = f'{val_sequence}_'
        if split == 'train':
            self.files = [f for f in all_files if not f.name.startswith(val_prefix)]
        else:
            self.files = [f for f in all_files if f.name.startswith(val_prefix)]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = np.load(str(self.files[idx]))
        return {
            'source':      torch.from_numpy(data['source']).unsqueeze(0),
            'target':      torch.from_numpy(data['target']).unsqueeze(0),
            'source_mask': torch.from_numpy(data['source_mask']).unsqueeze(0),
            'target_mask': torch.from_numpy(data['target_mask']).unsqueeze(0),
        }
