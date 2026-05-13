from pathlib import Path

import numpy as np
import pytest
import torch
from scripts.cell_tracking.cell_segmentation import CellCropBuilder, CellCropDataset


@pytest.fixture
def builder(tmp_path):
    return CellCropBuilder(
        data_dir=str(tmp_path / 'data'),
        output_dir=str(tmp_path / 'crops'),
        crop_size=224,
    )


def test_compute_centroid_single_cell(builder):
    mask = np.zeros((520, 696), dtype=np.float32)
    mask[100:120, 200:240] = 3.0
    cy, cx = builder._compute_centroid(mask, label=3)
    assert abs(cy - 109.5) < 1.0
    assert abs(cx - 219.5) < 1.0


def test_compute_centroid_missing_label_returns_none(builder):
    mask = np.zeros((520, 696), dtype=np.float32)
    result = builder._compute_centroid(mask, label=5)
    assert result is None


def test_crop_window_center_case(builder):
    left, top, right, bottom = builder._crop_window(
        center_x=348.0, center_y=260.0, img_w=696, img_h=520
    )
    assert right - left == 224
    assert bottom - top == 224
    assert left >= 0 and right <= 696
    assert top >= 0 and bottom <= 520


def test_crop_window_clamps_near_left_edge(builder):
    left, top, right, bottom = builder._crop_window(
        center_x=10.0, center_y=260.0, img_w=696, img_h=520
    )
    assert left == 0
    assert right == 224


def test_crop_window_clamps_near_right_edge(builder):
    left, top, right, bottom = builder._crop_window(
        center_x=690.0, center_y=260.0, img_w=696, img_h=520
    )
    assert right == 696
    assert left == 472


def test_crop_window_clamps_near_bottom_edge(builder):
    left, top, right, bottom = builder._crop_window(
        center_x=348.0, center_y=515.0, img_w=696, img_h=520
    )
    assert bottom == 520
    assert top == 296


# ── CellCropDataset tests ──────────────────────────────────────────────────

def _make_crop(path: Path, seq_prefix: str, label: int) -> None:
    arr = np.random.rand(224, 224).astype(np.float32)
    mask = (arr > 0.5).astype(np.float32)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path / f'{seq_prefix}_f000_f001_c{label}.npz'),
        source=arr, target=arr, source_mask=mask, target_mask=mask,
    )


def test_dataset_split_train(tmp_path):
    _make_crop(tmp_path, '01', 1)
    _make_crop(tmp_path, '01', 2)
    _make_crop(tmp_path, '02', 1)
    ds = CellCropDataset(root_dir=str(tmp_path), split='train', val_sequence='02')
    assert len(ds) == 2


def test_dataset_split_val(tmp_path):
    _make_crop(tmp_path, '01', 1)
    _make_crop(tmp_path, '02', 1)
    ds = CellCropDataset(root_dir=str(tmp_path), split='val', val_sequence='02')
    assert len(ds) == 1


def test_dataset_getitem_shapes(tmp_path):
    _make_crop(tmp_path, '01', 3)
    ds = CellCropDataset(root_dir=str(tmp_path), split='train', val_sequence='02')
    sample = ds[0]
    for key in ('source', 'target', 'source_mask', 'target_mask'):
        assert key in sample
        assert sample[key].shape == (1, 224, 224), f'{key} shape mismatch'
        assert sample[key].dtype == torch.float32


def test_dataset_getitem_values_in_range(tmp_path):
    _make_crop(tmp_path, '01', 1)
    ds = CellCropDataset(root_dir=str(tmp_path), split='train', val_sequence='02')
    sample = ds[0]
    assert sample['source'].min() >= 0.0
    assert sample['source'].max() <= 1.0
