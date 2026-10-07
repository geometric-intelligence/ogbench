"""The single-split grid search must refuse k-fold and split sweeps."""

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        'hyperparam_search', ROOT / 'scripts' / 'hyperparam_search.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def script():
    return _load_script()


def test_missing_split_type_is_rejected(script):
    with pytest.raises(ValueError, match='optuna_search.py'):
        script.require_fixed_split_search({'fixed': {}})


def test_kfold_is_rejected(script):
    with pytest.raises(ValueError, match='optuna_search.py'):
        script.require_fixed_split_search({'fixed': {'dataset.split_params.split_type': 'k-fold'}})


@pytest.mark.parametrize('name', ['smoke_test.yaml', 'multi_dataset_grid_search.yaml'])
def test_committed_grid_configs_pin_fixed_split(script, name):
    config = yaml.safe_load((ROOT / 'configs' / 'hparams_search' / name).read_text())
    script.require_fixed_split_search(config)
