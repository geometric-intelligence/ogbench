"""End to end: each k-fold run must train, validate and test on exactly its StratifiedKFold samples.

The real cache builder, ``OmicsDatasetLoader``, ``PreProcessor`` and split loader run on a small
synthetic dataset; only the HuggingFace download is replaced. Every feature is a monotone function
of the original sample id, so the per-feature normalisation keeps the id recoverable by ranking.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from ogbench.data.loaders.graph import omics_datasets
from ogbench.data.loaders.graph.omics_datasets import OmicsDatasetLoader
from ogbench.data.preprocessor import PreProcessor

N_SAMPLES, N_FEATURES, K = 60, 12, 5
N_GROUPS = 12


def _write_hub(hub: Path, *, grouped: bool) -> tuple[np.ndarray, np.ndarray | None]:
    rng = np.random.default_rng(0)
    ids = np.arange(N_SAMPLES, dtype=float)
    targets = np.arange(N_SAMPLES) % 3
    features = np.column_stack(
        [(j + 1) * ids + rng.normal(scale=0.01, size=N_SAMPLES) for j in range(N_FEATURES)]
    )
    hub.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(features, columns=[f'g{j}' for j in range(N_FEATURES)]).to_parquet(
        hub / 'toy_data.parquet'
    )
    pd.DataFrame({'target': targets}).to_parquet(hub / 'toy_targets.parquet')
    groups = None
    if grouped:
        groups = np.arange(N_SAMPLES) % N_GROUPS
        pd.DataFrame({'batch': groups}).to_parquet(hub / 'toy_batches.parquet')
    return targets, groups


def _expected_fold_ids(targets: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
    dummy = np.zeros((len(targets), 1))
    if groups is None:
        splits = StratifiedKFold(n_splits=K, shuffle=True, random_state=42).split(dummy, targets)
    else:
        splitter = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=42)
        splits = splitter.split(dummy, targets, groups)
    fold_ids = np.empty(len(targets), dtype=int)
    for fold, (_, test) in enumerate(splits):
        fold_ids[test] = fold
    return fold_ids


def _sample_id(graph, value_to_id: dict[float, int]) -> int:
    return value_to_id[round(float(graph.x[0, 0]), 6)]


def _run_fold(data_dir: Path, fold: int, grouping: str | None):
    loader = OmicsDatasetLoader(
        OmegaConf.create(
            {
                'data_type': 'omics',
                'data_domain': 'graph',
                'data_name': 'toy',
                'data_dir': str(data_dir),
                'adjacency_method': 'wgcna',
                'adjacency_target_connectivity': 0.10,
                'wgcna_binarization': 'target_connectivity',
                'node_sample_ratio': 1.0,
                'method': 'random',
                'imputation_method': 'mean',
                'split_type': 'k-fold',
                'k': K,
                'fold': fold,
                'grouping': grouping,
            }
        )
    )
    dataset, dataset_dir = loader.load()
    preprocessor = PreProcessor(dataset, dataset_dir)
    split_params = OmegaConf.create(
        {
            'learning_setting': 'inductive',
            'split_type': 'k-fold',
            'k': K,
            'data_seed': fold,
            'data_split_dir': str(data_dir / 'legacy_splits'),
        }
    )
    return dataset, preprocessor.load_dataset_splits(split_params)


@pytest.mark.parametrize('grouped', [False, True], ids=['stratified', 'grouped'])
def test_every_fold_sees_exactly_its_kfold_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, grouped: bool
) -> None:
    hub = tmp_path / 'hub'
    targets, groups = _write_hub(hub, grouped=grouped)

    def local_download(self, filename: str) -> str:
        path = hub / filename
        if not path.exists():
            raise FileNotFoundError(filename)
        return str(path)

    # ogbench.data.datasets registers its modules dynamically, so patch the class the loader uses.
    monkeypatch.setattr(omics_datasets.HFOmicsDataset, '_hf_download', local_download)
    fold_ids = _expected_fold_ids(targets, groups)
    tested: list[int] = []

    for fold in range(K):
        dataset, (train, val, test) = _run_fold(
            tmp_path / 'data', fold, 'batch' if grouped else None
        )
        values = np.array([float(dataset[i].x[0, 0]) for i in range(len(dataset))])
        ranks = np.argsort(np.argsort(values))
        value_to_id = {round(v, 6): int(r) for v, r in zip(values, ranks, strict=True)}
        assert len(value_to_id) == N_SAMPLES

        split_ids = {
            name: sorted(_sample_id(graph, value_to_id) for graph in split.data_lst)
            for name, split in (('train', train), ('valid', val), ('test', test))
        }
        val_fold = (fold + 1) % K
        assert split_ids['test'] == np.flatnonzero(fold_ids == fold).tolist()
        assert split_ids['valid'] == np.flatnonzero(fold_ids == val_fold).tolist()
        assert (
            split_ids['train']
            == np.flatnonzero((fold_ids != fold) & (fold_ids != val_fold)).tolist()
        )
        for split in (train, val, test):
            for graph in split.data_lst:
                assert int(graph.y) == targets[_sample_id(graph, value_to_id)]
        if grouped:
            split_groups = {name: set(groups[ids]) for name, ids in split_ids.items()}
            assert not split_groups['train'] & split_groups['valid']
            assert not split_groups['train'] & split_groups['test']
            assert not split_groups['valid'] & split_groups['test']
        tested.extend(split_ids['test'])

    assert sorted(tested) == list(range(N_SAMPLES))
    assert not (tmp_path / 'data' / 'legacy_splits').exists()
