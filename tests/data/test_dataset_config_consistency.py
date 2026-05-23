"""Tests to ensure consistency across dataset YAML configs."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from huggingface_hub import hf_hub_download
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

DATASET_CONFIG_DIR = Path(__file__).resolve().parents[2] / 'configs' / 'dataset'
HF_CONFIG_PATH = Path(__file__).resolve().parents[2] / 'configs' / 'hf' / 'default.yaml'
DATASET_CONFIGS = sorted(DATASET_CONFIG_DIR.glob('*.yaml'))

HF_REPO_ID = 'geometric-intelligence/ogbench'


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope='module')
def all_configs() -> dict[str, dict]:
    return {p.stem: _load_config(p) for p in DATASET_CONFIGS}


class TestDatasetConfigConsistency:
    """Ensure key parameters are consistent across all dataset configs."""

    def test_all_datasets_have_same_train_val_test_split(self, all_configs):
        splits = {
            name: tuple(cfg['loader']['parameters']['train_val_test_split'])
            for name, cfg in all_configs.items()
        }

        unique_splits = set(splits.values())
        assert len(unique_splits) == 1, (
            f'Expected all datasets to share the same train_val_test_split, '
            f'but found {len(unique_splits)} distinct values: '
            + ', '.join(f'{name}={list(s)}' for name, s in splits.items())
        )

    def test_class_weights_consistently_set_or_null(self, all_configs):
        weights = {name: cfg['parameters']['class_weights'] for name, cfg in all_configs.items()}

        has_weights = {name: w is not None for name, w in weights.items()}
        all_null = all(not v for v in has_weights.values())
        all_set = all(v for v in has_weights.values())

        assert all_null or all_set, (
            'class_weights must be consistently null or set across all datasets, '
            'but got a mix: '
            + ', '.join(
                f"{name}={'set' if flag else 'null'}" for name, flag in has_weights.items()
            )
        )

    def test_all_datasets_have_same_monitor_metric(self, all_configs):
        metrics = {name: cfg['parameters']['monitor_metric'] for name, cfg in all_configs.items()}

        unique_metrics = set(metrics.values())
        assert len(unique_metrics) == 1, (
            f'Expected all datasets to share the same monitor_metric, '
            f'but found {len(unique_metrics)} distinct values: '
            + ', '.join(f'{name}={m}' for name, m in metrics.items())
        )

    def test_all_configs_identical_except_dataset_specific_keys(self, all_configs):
        EXCLUDED_PATHS = [
            ('loader', 'parameters', 'data_name'),
            ('loader', 'parameters', 'adjacency_threshold'),
            ('parameters', 'num_classes'),
            ('parameters', 'num_samples'),
            ('parameters', 'full_num_nodes'),
            ('parameters', 'class_weights'),
        ]

        def _remove_paths(cfg: dict) -> dict:
            cfg = copy.deepcopy(cfg)
            for path in EXCLUDED_PATHS:
                d = cfg
                for key in path[:-1]:
                    if isinstance(d, dict) and key in d:
                        d = d[key]
                    else:
                        break
                else:
                    d.pop(path[-1], None)
            return cfg

        stripped = {name: _remove_paths(cfg) for name, cfg in all_configs.items()}
        names = list(stripped)
        reference_name = names[0]
        reference = stripped[reference_name]

        for name in names[1:]:
            assert stripped[name] == reference, (
                f'Config {name} differs from {reference_name} '
                f'(after excluding dataset-specific keys). '
                f'Differences: {_diff(reference, stripped[name])}'
            )


def _load_hf_revision() -> str:
    with open(HF_CONFIG_PATH) as f:
        return yaml.safe_load(f)['revision']


def _download_parquet(data_name: str, suffix: str, revision: str) -> pd.DataFrame:
    path = hf_hub_download(  # nosec B615 - revision is pinned via config
        repo_id=HF_REPO_ID,
        repo_type='dataset',
        revision=revision,
        filename=f'{data_name}_{suffix}.parquet',
    )
    return pd.read_parquet(path)


class TestDatasetConfigMatchesHuggingFace:
    """Verify that config values (num_samples, full_num_nodes, num_classes) match HF data."""

    @pytest.fixture(scope='class')
    def hf_revision(self) -> str:
        return _load_hf_revision()

    @pytest.fixture(scope='class')
    def hf_datasets(self, hf_revision) -> dict[str, dict]:
        """Download data and targets for every dataset once per test class."""
        results = {}
        for config_path in DATASET_CONFIGS:
            cfg = _load_config(config_path)
            data_name = cfg['loader']['parameters']['data_name']
            data_df = _download_parquet(data_name, 'data', hf_revision)
            targets_df = _download_parquet(data_name, 'targets', hf_revision)

            if 'target' in data_df.columns:
                data_df = data_df.drop('target', axis=1)

            results[config_path.stem] = {
                'config': cfg,
                'num_features': data_df.shape[1],
                'num_samples': len(targets_df),
                'num_classes': targets_df['target'].nunique(),
            }
        return results

    @pytest.mark.parametrize('dataset_name', [p.stem for p in DATASET_CONFIGS])
    def test_full_num_nodes_matches_hf(self, hf_datasets, dataset_name):
        ds = hf_datasets[dataset_name]
        expected = ds['num_features']
        actual = ds['config']['parameters']['full_num_nodes']
        assert (
            actual == expected
        ), f'{dataset_name}: config full_num_nodes={actual} but HF data has {expected} features'

    @pytest.mark.parametrize('dataset_name', [p.stem for p in DATASET_CONFIGS])
    def test_num_samples_matches_hf(self, hf_datasets, dataset_name):
        ds = hf_datasets[dataset_name]
        expected = ds['num_samples']
        actual = ds['config']['parameters']['num_samples']
        assert (
            actual == expected
        ), f'{dataset_name}: config num_samples={actual} but HF data has {expected} samples'

    @pytest.mark.parametrize('dataset_name', [p.stem for p in DATASET_CONFIGS])
    def test_num_classes_matches_hf(self, hf_datasets, dataset_name):
        ds = hf_datasets[dataset_name]
        expected = ds['num_classes']
        actual = ds['config']['parameters']['num_classes']
        assert (
            actual == expected
        ), f'{dataset_name}: config num_classes={actual} but HF data has {expected} classes'

    @pytest.mark.parametrize('dataset_name', [p.stem for p in DATASET_CONFIGS])
    def test_class_weights_match_training_data(self, hf_revision, dataset_name):
        cfg = _load_config(DATASET_CONFIG_DIR / f'{dataset_name}.yaml')
        config_weights = cfg['parameters']['class_weights']
        if config_weights is None:
            pytest.skip(f'{dataset_name} has class_weights=null')

        targets_df = _download_parquet(
            cfg['loader']['parameters']['data_name'], 'targets', hf_revision
        )
        targets = targets_df.iloc[:, 0].values
        targets = shuffle(targets, random_state=42)

        split = cfg['loader']['parameters']['train_val_test_split']
        train_idx = int(len(targets) * split[0])
        train_targets = targets[:train_idx]

        classes = np.unique(train_targets)
        expected_weights = compute_class_weight('balanced', classes=classes, y=train_targets)
        expected_weights = np.round(expected_weights, 3).tolist()

        assert config_weights == expected_weights, (
            f'{dataset_name}: config class_weights={config_weights} '
            f'but computed from training data={expected_weights}'
        )


def _diff(a, b, path='') -> list[str]:
    """Return human-readable list of differences between two nested dicts."""
    diffs = []
    all_keys = set(a.keys() if isinstance(a, dict) else []) | set(
        b.keys() if isinstance(b, dict) else []
    )
    for key in sorted(all_keys):
        full = f'{path}.{key}' if path else str(key)
        va, vb = (
            (a.get(key) if isinstance(a, dict) else None),
            (b.get(key) if isinstance(b, dict) else None),
        )
        if va is None and vb is not None:
            diffs.append(f'{full}: missing in first, present in second ({vb!r})')
        elif va is not None and vb is None:
            diffs.append(f'{full}: present in first ({va!r}), missing in second')
        elif isinstance(va, dict) and isinstance(vb, dict):
            diffs.extend(_diff(va, vb, full))
        elif va != vb:
            diffs.append(f'{full}: {va!r} != {vb!r}')
    return diffs
