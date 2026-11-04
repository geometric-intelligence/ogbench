"""OmicsDatasetLoader must read split cut points from split_info.json, never guess them."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from ogbench.data.loaders.graph.omics_datasets import OmicsDatasetLoader


class _FakeDataset(SimpleNamespace):
    def __len__(self) -> int:
        return self.n


def _loader(tmp_path, fold: int = 1) -> OmicsDatasetLoader:
    params = OmegaConf.create(
        {
            'data_dir': str(tmp_path),
            'data_name': 'toy',
            'split_type': 'k-fold',
            'k': 5,
            'fold': fold,
            'train_val_test_split': [0.7, 0.15, 0.15],
        }
    )
    return OmicsDatasetLoader(params)


def _write_split_info(raw_dir, **overrides) -> None:
    info = {
        'split_type': 'k-fold',
        'k': 5,
        'fold': 1,
        'train_idx': 60,
        'val_idx': 80,
        'total_samples': 100,
    }
    info.update(overrides)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / 'split_info.json').write_text(json.dumps(info))


def test_reads_contiguous_cut_points(tmp_path):
    raw_dir = tmp_path / 'raw'
    _write_split_info(raw_dir)
    split = _loader(tmp_path)._prepare_split_idx(_FakeDataset(raw_dir=str(raw_dir), n=100))

    np.testing.assert_array_equal(split['train'], np.arange(60))
    np.testing.assert_array_equal(split['valid'], np.arange(60, 80))
    np.testing.assert_array_equal(split['test'], np.arange(80, 100))


def test_missing_split_info_raises(tmp_path):
    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    with pytest.raises(FileNotFoundError, match='split_info.json'):
        _loader(tmp_path)._prepare_split_idx(_FakeDataset(raw_dir=str(raw_dir), n=100))


def test_split_info_from_another_fold_raises(tmp_path):
    raw_dir = tmp_path / 'raw'
    _write_split_info(raw_dir, fold=3)
    with pytest.raises(ValueError, match='fold'):
        _loader(tmp_path, fold=1)._prepare_split_idx(_FakeDataset(raw_dir=str(raw_dir), n=100))


def test_sample_count_mismatch_raises(tmp_path):
    raw_dir = tmp_path / 'raw'
    _write_split_info(raw_dir)
    with pytest.raises(ValueError, match='total_samples'):
        _loader(tmp_path)._prepare_split_idx(_FakeDataset(raw_dir=str(raw_dir), n=90))
