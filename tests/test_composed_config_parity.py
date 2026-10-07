"""Every model must run with the same composed settings on every dataset.

Dataset-specific values (num_samples, num_classes, ...) may only reach the model, trainer,
optimizer, loss and callbacks through ``${dataset.*}`` interpolations, so the unresolved
non-dataset part of the composed config has to be identical across datasets.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ogbench.baseline import _get_hf_omics_raw_dir
from ogbench.utils.config_resolvers import register_all_resolvers

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / 'configs'
DATASETS = sorted(p.stem for p in (CONFIG_DIR / 'dataset').glob('*.yaml'))
MODELS = sorted(p.stem for p in (CONFIG_DIR / 'model').glob('*.yaml'))


def _diff(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b), key=str):
            out += _diff(a.get(key), b.get(key), f'{path}.{key}')
        return out
    return [] if a == b else [f'{path}: {a!r} != {b!r}']


@pytest.fixture(scope='module')
def composed():
    register_all_resolvers()
    configs = {}
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base='1.3'):
        for model in MODELS:
            for dataset in DATASETS:
                configs[model, dataset] = compose(
                    'train.yaml', overrides=[f'model={model}', f'dataset={dataset}']
                )
    return configs


@pytest.mark.parametrize('model', MODELS)
def test_non_dataset_settings_identical_across_datasets(composed, model):
    def strip(cfg):
        container = OmegaConf.to_container(cfg, resolve=False)
        container.pop('dataset')
        return container

    reference = strip(composed[model, DATASETS[0]])
    for dataset in DATASETS[1:]:
        diffs = _diff(reference, strip(composed[model, dataset]))
        assert not diffs, f'{model}: {dataset} differs from {DATASETS[0]}: {diffs[:10]}'


@pytest.mark.parametrize('dataset', DATASETS)
def test_default_protocol_is_kfold_with_train_fold_wgcna(composed, dataset):
    cfg = composed[MODELS[0], dataset]
    loader = cfg.dataset.loader.parameters
    assert cfg.dataset.split_params.split_type == 'k-fold'
    assert cfg.dataset.split_params.k == 5
    assert loader.fold == cfg.dataset.split_params.data_seed
    assert loader.adjacency_method == 'wgcna'
    assert loader.wgcna_binarization == 'target_connectivity'
    assert loader.adjacency_target_connectivity == 0.10
    assert cfg.dataset.parameters.class_weights == 'balanced'


@pytest.mark.parametrize('dataset', DATASETS)
def test_baseline_reads_the_gnn_cache_for_the_same_fold(composed, dataset):
    cfg = composed[MODELS[0], dataset]
    raw_dir = _get_hf_omics_raw_dir(cfg)
    loader = cfg.dataset.loader.parameters
    assert f'hf_{str(loader.revision)[:12]}' in raw_dir
    assert f'impute_{loader.imputation_method}' in raw_dir
    assert f'split_k-fold_k_5_fold_{cfg.dataset.split_params.data_seed}' in raw_dir
    assert raw_dir.endswith('raw')
