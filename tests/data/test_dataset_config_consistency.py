"""Tests to ensure consistency across dataset YAML configs."""

from pathlib import Path

import pytest
import yaml

DATASET_CONFIG_DIR = Path(__file__).resolve().parents[2] / 'configs' / 'dataset'
DATASET_CONFIGS = sorted(DATASET_CONFIG_DIR.glob('*.yaml'))


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
