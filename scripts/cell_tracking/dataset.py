"""
Dataset loader for Cell Tracking Challenge data (PhC-C2DH-U373).

Loads 2D phase-contrast microscopy TIF frames and creates source/target pairs
for VoxelMorph registration training.

Dataset: Glioblastoma-astrocytoma U373 cells on polyacrylamide substrate.
Source: Dr. S. Kumar, Dept. of Bioengineering, UC Berkeley.
Microscope: Nikon, Plan Fluor DLL 20x/0.5, pixel size 0.65x0.65 um, 15 min/frame.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io


class CellTrackingDataset(Dataset):
    """
    PyTorch Dataset for Cell Tracking Challenge TIF sequences.
    Loads 2D grayscale TIF frames from one or more sequences and creates
    source/target pairs for VoxelMorph registration.

    data_dir : str or Path
        Root directory containing sequence folders (e.g., dataset/train/).
        Expected structure: data_dir/{01,02,...}/t000.tif, t001.tif, ...
    sequences : list of str, optional
        Which sequence folders to use. Default is ['01', '02'].
    pairing : {'consecutive', 'random'}
        How to create source/target pairs:
        - 'consecutive': pair frame t_n with t_{n+1} (default)
        - 'random': pair any two frames from the same sequence
    pad_to : tuple of int, optional
        Pad images to this (H, W) size. Default is (544, 704) which is
        divisible by 32 (required for 5-level UNet).
    """

    def __init__(
        self,
        data_dir: str | Path,
        sequences: list[str] | None = None,
        pairing: Literal['consecutive', 'random'] = 'consecutive',
        pad_to: tuple[int, int] = (544, 704),
        use_masks: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.sequences = sequences or ['01', '02']
        self.pairing = pairing
        self.pad_to = pad_to
        self.use_masks = use_masks

        # Collect all frame paths per sequence
        self.sequence_frames: dict[str, list[Path]] = {}
        for seq in self.sequences:
            seq_dir = self.data_dir / seq
            if not seq_dir.exists():
                raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")
            frames = sorted(seq_dir.glob('t*.tif'))
            if len(frames) < 2:
                raise ValueError(f"Need at least 2 frames in {seq_dir}, found {len(frames)}")
            self.sequence_frames[seq] = frames

        # Build index of all pairs
        self.pairs: list[tuple[Path, Path]] = []
        if self.pairing == 'consecutive':
            for seq, frames in self.sequence_frames.items():
                for i in range(len(frames) - 1):
                    self.pairs.append((frames[i], frames[i + 1]))
        elif self.pairing == 'random':
            # For random pairing, store flat list and sample in __getitem__
            self.all_frames_by_seq = [
                (seq, frames) for seq, frames in self.sequence_frames.items()
            ]
        else:
            raise ValueError(f"pairing must be 'consecutive' or 'random', got '{pairing}'")

    def __len__(self) -> int:
        if self.pairing == 'consecutive':
            return len(self.pairs)
        else:
            # For random pairing, return total frames across sequences
            return sum(len(f) for f in self.sequence_frames.values())

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.pairing == 'consecutive':
            source_path, target_path = self.pairs[idx]
            source = self._load_and_preprocess(source_path)
            target = self._load_and_preprocess(target_path)
            if self.use_masks:
                source_mask = self._load_mask(source_path)
                target_mask = self._load_mask(target_path)
        else:
            # Random: pick a random sequence, then two random frames
            seq, frames = self.all_frames_by_seq[
                np.random.randint(len(self.all_frames_by_seq))
            ]
            idx1, idx2 = np.random.choice(len(frames), size=2, replace=False)
            source = self._load_and_preprocess(frames[idx1])
            target = self._load_and_preprocess(frames[idx2])

        result: dict[str, torch.Tensor] = {'source': source, 'target': target}
        if self.use_masks and self.pairing == 'consecutive':
            result['source_mask'] = source_mask
            result['target_mask'] = target_mask
        return result

    def _load_mask(self, image_path: Path) -> torch.Tensor:
        """
        Load the Silver Truth segmentation mask for a given image frame.

        Derives the ST mask path from the image path:
            dataset/train/01/t005.tif  ->  dataset/train/01_ST/SEG/man_seg005.tif

        Parameters
        ----------
        image_path : Path
            Path to the source or target image TIF.

        Returns
        -------
        torch.Tensor
            Shape (1, H, W), dtype float32, containing integer label IDs.
            Padded to self.pad_to with zeros. Returns all-zeros if mask not found.
        """
        sequence = image_path.parent.name
        frame_index = int(image_path.stem[1:])
        mask_path = (
            image_path.parent.parent
            / f'{sequence}_ST'
            / 'SEG'
            / f'man_seg{frame_index:03d}.tif'
        )

        pad_h, pad_w = self.pad_to if self.pad_to else (520, 696)

        if not mask_path.exists():
            return torch.zeros(1, pad_h, pad_w, dtype=torch.float32)

        mask = io.imread(str(mask_path)).astype(np.float32)

        if self.pad_to is not None:
            target_h, target_w = self.pad_to
            h, w = mask.shape
            if h < target_h or w < target_w:
                padded = np.zeros((target_h, target_w), dtype=np.float32)
                padded[:h, :w] = mask
                mask = padded

        return torch.from_numpy(mask).unsqueeze(0)

    def _load_and_preprocess(self, path: Path) -> torch.Tensor:
        """
        Load a TIF image, normalize to [0, 1], pad, and return as tensor.
        """
        # Load as numpy array (uint8, 520x696)
        img = io.imread(str(path)).astype(np.float32)

        # Normalize to [0, 1]
        img = img / 255.0

        # Pad to target size (zero-padding on right and bottom)
        if self.pad_to is not None:
            target_h, target_w = self.pad_to
            h, w = img.shape
            if h < target_h or w < target_w:
                padded = np.zeros((target_h, target_w), dtype=np.float32)
                padded[:h, :w] = img
                img = padded

        # Convert to tensor with channel dimension: (1, H, W)
        return torch.from_numpy(img).unsqueeze(0)
