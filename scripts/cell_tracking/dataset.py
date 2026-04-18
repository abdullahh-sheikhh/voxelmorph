"""
Dataset loader for Cell Tracking Challenge data (PhC-C2DH-U373).

Loads 2D phase-contrast microscopy TIF frames and creates consecutive
source/target pairs for VoxelMorph registration training.

Dataset: Glioblastoma-astrocytoma U373 cells on polyacrylamide substrate.
Source: Dr. S. Kumar, Dept. of Bioengineering, UC Berkeley.
Microscope: Nikon, Plan Fluor DLL 20x/0.5, pixel size 0.65x0.65 um, 15 min/frame.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io


class CellTrackingDataset(Dataset):
    """
    PyTorch Dataset for Cell Tracking Challenge TIF sequences.

    Loads 2D grayscale TIF frames from one or more sequences and creates
    consecutive source/target pairs (frame t_n → t_{n+1}) for VoxelMorph
    registration.

    Parameters
    ----------
    data_dir : str or Path
        Root directory containing sequence folders (e.g., dataset/train/).
        Expected structure: data_dir/{01,02,...}/t000.tif, t001.tif, ...
    sequences : list of str, optional
        Which sequence folders to use. This is the main mechanism for simple
        train/test splitting, e.g. train on ['01'] and evaluate on ['02'].
        Default is ['01', '02'].
    pad_to : tuple of int, optional
        Pad images to this (H, W) size. Default is (544, 704) which is
        divisible by 32 (required for 5-level UNet).
    use_masks : bool, optional
        If True, load Silver Truth segmentation masks alongside images.
    """

    def __init__(
        self,
        data_dir: str | Path,
        sequences: list[str] | None = None,
        pad_to: tuple[int, int] = (544, 704),
        use_masks: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.sequences = sequences or ['01', '02']
        self.pad_to = pad_to
        self.use_masks = use_masks

        # Collect all frame paths per sequence and build consecutive pairs
        self.pairs: list[tuple[Path, Path]] = []
        for seq in self.sequences:
            seq_dir = self.data_dir / seq
            if not seq_dir.exists():
                raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")
            frames = sorted(seq_dir.glob('t*.tif'))
            if len(frames) < 2:
                raise ValueError(f"Need at least 2 frames in {seq_dir}, found {len(frames)}")
            for i in range(len(frames) - 1):
                self.pairs.append((frames[i], frames[i + 1]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        source_path, target_path = self.pairs[idx]
        source = self._load_and_preprocess(source_path)
        target = self._load_and_preprocess(target_path)

        result: dict[str, torch.Tensor] = {'source': source, 'target': target}
        if self.use_masks:
            result['source_mask'] = self._load_mask(source_path)
            result['target_mask'] = self._load_mask(target_path)
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

        pad_h, pad_w = self.pad_to

        if not mask_path.exists():
            return torch.zeros(1, pad_h, pad_w, dtype=torch.float32)

        mask = io.imread(str(mask_path)).astype(np.float32)

        h, w = mask.shape
        if h < pad_h or w < pad_w:
            padded = np.zeros((pad_h, pad_w), dtype=np.float32)
            padded[:h, :w] = mask
            mask = padded

        return torch.from_numpy(mask).unsqueeze(0)

    def _load_and_preprocess(self, path: Path) -> torch.Tensor:
        """
        Load a TIF image, normalize to [0, 1], pad, and return as tensor.

        Returns
        -------
        torch.Tensor
            Shape (1, H, W), dtype float32, values in [0, 1].
        """
        img = io.imread(str(path)).astype(np.float32) / 255.0

        if self.pad_to is not None:
            target_h, target_w = self.pad_to
            h, w = img.shape
            if h < target_h or w < target_w:
                padded = np.zeros((target_h, target_w), dtype=np.float32)
                padded[:h, :w] = img
                img = padded

        return torch.from_numpy(img).unsqueeze(0)
