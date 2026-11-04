"""The GNN split loader must keep the precomputed omics fold instead of re-splitting it."""

import fcntl
import threading
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data, InMemoryDataset

from ogbench.data.preprocessor import PreProcessor

N_TRAIN, N_VAL, N_TEST = 30, 10, 10


class _ToyFoldDataset(InMemoryDataset):
    """Samples stored train|val|test like an HFOmics fold cache."""

    def __init__(self, root: str) -> None:
        self._root = root
        super().__init__(root)
        n = N_TRAIN + N_VAL + N_TEST
        graphs = [
            Data(
                x=torch.zeros(3, 1),
                y=torch.tensor([i % 2]),
                sample_id=torch.tensor([i]),
            )
            for i in range(n)
        ]
        self.data, self.slices = self.collate(graphs)

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return []

    def download(self) -> None:
        pass

    def process(self) -> None:
        pass

    def get_data_dir(self) -> str:
        return self._root


def _omics_dataset(root: str) -> _ToyFoldDataset:
    dataset = _ToyFoldDataset(root)
    dataset.uses_precomputed_split = True
    dataset.split_idx = {
        'train': np.arange(N_TRAIN),
        'valid': np.arange(N_TRAIN, N_TRAIN + N_VAL),
        'test': np.arange(N_TRAIN + N_VAL, N_TRAIN + N_VAL + N_TEST),
    }
    return dataset


def _split_params(tmp_path) -> OmegaConf:
    return OmegaConf.create(
        {
            'learning_setting': 'inductive',
            'split_type': 'k-fold',
            'k': 5,
            'data_seed': 0,
            'data_split_dir': str(tmp_path / 'legacy_splits'),
        }
    )


def _ids(split) -> list[int]:
    return [int(graph.sample_id) for graph in split.data_lst]


def test_preprocessor_keeps_precomputed_fold_under_kfold(tmp_path):
    preprocessor = PreProcessor(_omics_dataset(str(tmp_path / 'ds')), str(tmp_path / 'ds'))

    train, val, test = preprocessor.load_dataset_splits(_split_params(tmp_path))

    assert _ids(train) == list(range(N_TRAIN))
    assert _ids(val) == list(range(N_TRAIN, N_TRAIN + N_VAL))
    assert _ids(test) == list(range(N_TRAIN + N_VAL, N_TRAIN + N_VAL + N_TEST))
    assert not (tmp_path / 'legacy_splits').exists()


def test_pre_transformed_cache_is_read_only_after_its_builder_finishes(tmp_path):
    root = str(tmp_path / 'ds')
    transforms = OmegaConf.create({'identity': {'transform_name': 'Identity'}})
    built = PreProcessor(_omics_dataset(root), root, transforms)
    cache = Path(built.processed_data_dir)
    assert (cache / 'data.pt').exists()
    assert not list(cache.glob('*.partial'))

    loaded: list[PreProcessor] = []
    with open(cache / '.lock', 'w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        reader = threading.Thread(
            target=lambda: loaded.append(PreProcessor(_omics_dataset(root), root, transforms))
        )
        reader.start()
        reader.join(timeout=2)
        assert reader.is_alive()
        fcntl.flock(handle, fcntl.LOCK_UN)
    reader.join(timeout=60)

    assert len(loaded[0]) == N_TRAIN + N_VAL + N_TEST
