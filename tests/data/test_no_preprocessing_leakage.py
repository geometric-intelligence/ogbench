"""End-to-end check that no fitted preprocessing step sees held-out samples.

Each fold is built twice: once on the original matrix and once after replacing the
held-out rows (test, or validation and test) with different values. Everything fitted
on the training fold -- corrections, imputation, gene selection, the WGCNA graph, and
the feature normalizer -- must come out bit-identical, as must the training rows. When
only test rows change, validation rows must be identical too, so transforms are row-wise.
The sklearn baseline loader is checked the same way and must match the GNN cache.
"""

import json
import os.path as osp

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

import ogbench.baseline as baseline
from ogbench.data.datasets.hf_omics import HFOmicsDataset
from ogbench.data.utils.split_utils import compute_omics_split_indices

N_SAMPLES, N_FEATURES, K, FOLD = 120, 260, 5, 0

SCENARIOS = {
    # name: (corrections, selection method, has NaNs)
    'motrpac_like': (['covariate_adjust'], 'variance', True),
    'addneuromed_like': (['combat'], 'correlation', False),
    'smoking_like': (['promoter_min_beta', 'median_center'], 'distance_correlation', True),
}


def _synthetic(scenario: str) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    corrections, _, has_nan = SCENARIOS[scenario]
    labels = np.array([0, 1] * (N_SAMPLES // 2))
    rng.shuffle(labels)
    latent = rng.normal(size=(N_SAMPLES, 6))
    values = latent @ rng.normal(size=(6, N_FEATURES)) + rng.normal(size=(N_SAMPLES, N_FEATURES))
    values[:, :20] += labels[:, None] * 0.8
    batches = np.where(rng.random(N_SAMPLES) < 0.5, 'b0', 'b1')
    values[batches == 'b1'] += 1.5
    if has_nan:
        values[rng.random(values.shape) < 0.02] = np.nan
    columns = [f'f{i}' for i in range(N_FEATURES)]
    files = {
        'data': pd.DataFrame(values, columns=columns),
        'targets': pd.DataFrame({'target': labels}),
        'covariates': pd.DataFrame(
            {
                'age': rng.normal(50, 10, N_SAMPLES),
                'sex': rng.choice(['F', 'M'], N_SAMPLES),
                'bmi': rng.normal(25, 3, N_SAMPLES),
                'race': rng.choice(['A', 'B'], N_SAMPLES),
            }
        ),
        'batches': pd.DataFrame({'batch': batches}),
    }
    if 'promoter_min_beta' in corrections:
        files['probe_map'] = pd.DataFrame(
            {'probe_id': columns, 'gene': [f'g{i // 2}' for i in range(N_FEATURES)]}
        )
    return files


def _perturbed(files: dict[str, pd.DataFrame], rows: np.ndarray) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(1)
    out = {name: frame.copy() for name, frame in files.items()}
    data = out['data'].to_numpy(copy=True)
    data[rows] = rng.normal(5.0, 3.0, size=(len(rows), data.shape[1]))
    if np.isnan(files['data'].to_numpy()).any():
        data[rows[: len(rows) // 2], :50] = np.nan
    out['data'] = pd.DataFrame(data, columns=files['data'].columns)
    out['covariates'].loc[rows, 'age'] = rng.normal(80, 5, len(rows))
    return out


def _held_out(files: dict[str, pd.DataFrame], which: str) -> np.ndarray:
    split = compute_omics_split_indices(
        files['targets']['target'].to_numpy(), split_type='k-fold', k=K, fold=FOLD
    )
    if which == 'test':
        return split['test']
    return np.concatenate([split['valid'], split['test']])


def _write(files: dict[str, pd.DataFrame], directory) -> dict[str, str]:
    directory.mkdir(parents=True)
    paths = {}
    for name, frame in files.items():
        path = directory / f'synthetic_{name}.parquet'
        frame.to_parquet(path)
        paths[f'synthetic_{name}.parquet'] = str(path)
    return paths


def _build_gnn_fold(scenario, files, tmp_path, tag, monkeypatch):
    paths = _write(files, tmp_path / f'hub_{tag}')
    monkeypatch.setattr(HFOmicsDataset, '_hf_download', lambda self, name: paths[name])
    monkeypatch.setattr(
        HFOmicsDataset,
        '_download_optional_parquet',
        lambda self, name: pd.read_parquet(paths[name]) if name in paths else None,
    )
    corrections, method, _ = SCENARIOS[scenario]
    np.random.seed(0)
    dataset = HFOmicsDataset(
        root=str(tmp_path / f'cache_{tag}'),
        data_name='synthetic',
        method=method,
        imputation_method='mean',
        adjacency_method='wgcna',
        adjacency_target_connectivity=0.1,
        node_sample_ratio=0.5,
        revision='test-revision',
        split_type='k-fold',
        k=K,
        fold=FOLD,
        corrections=corrections,
    )
    with open(osp.join(dataset.processed_dir, 'processing_stats.json')) as f:
        normalizer = json.load(f)['feature_normalizer']
    with open(osp.join(dataset.raw_dir, 'split_info.json')) as f:
        split_info = json.load(f)
    return {
        'selected': pd.read_parquet(osp.join(dataset.raw_dir, 'selected_data.parquet')),
        'adjacency': np.load(osp.join(dataset.raw_dir, 'adj_matrix.npy')),
        'normalizer': normalizer,
        'split_info': split_info,
        'x': np.stack([dataset[i].x.numpy().ravel() for i in range(len(dataset))]),
    }


def _build_baseline_fold(scenario, files, tmp_path, tag, monkeypatch):
    paths = _write(files, tmp_path / f'baseline_hub_{tag}')

    def fake_download(*, filename, **_):
        if filename not in paths:
            raise FileNotFoundError(filename)
        return paths[filename]

    monkeypatch.setattr(baseline, 'hf_hub_download', fake_download)
    corrections, method, _ = SCENARIOS[scenario]
    cfg = OmegaConf.create(
        {
            'seed': 0,
            'dataset': {
                'loader': {
                    'parameters': {
                        'data_name': 'synthetic',
                        'revision': 'test-revision',
                        'method': method,
                        'node_sample_ratio': 0.5,
                        'imputation_method': 'mean',
                        'train_val_test_split': [0.7, 0.15, 0.15],
                        'corrections': corrections,
                        'split_type': 'k-fold',
                        'k': K,
                        'fold': FOLD,
                        'grouping': None,
                    }
                },
                'split_params': {'split_type': 'k-fold', 'k': K, 'data_seed': FOLD},
            },
        }
    )
    return baseline.load_and_prepare_data(cfg)


@pytest.mark.parametrize('scenario', list(SCENARIOS))
@pytest.mark.parametrize('held_out', ['test', 'valid_and_test'])
def test_gnn_fold_ignores_held_out_values(scenario, held_out, tmp_path, monkeypatch):
    files = _synthetic(scenario)
    rows = _held_out(files, 'test' if held_out == 'test' else 'both')
    clean = _build_gnn_fold(scenario, files, tmp_path, 'clean', monkeypatch)
    moved = _build_gnn_fold(scenario, _perturbed(files, rows), tmp_path, 'moved', monkeypatch)

    n_train = clean['split_info']['train_idx']
    n_fit = clean['split_info']['val_idx'] if held_out == 'test' else n_train
    for key in ('train_indices', 'valid_indices', 'test_indices'):
        assert clean['split_info'][key] == moved['split_info'][key]
    assert list(clean['selected'].columns) == list(moved['selected'].columns)
    np.testing.assert_array_equal(clean['adjacency'], moved['adjacency'])
    assert clean['normalizer'] == moved['normalizer']
    np.testing.assert_array_equal(
        clean['selected'].to_numpy()[:n_fit], moved['selected'].to_numpy()[:n_fit]
    )
    np.testing.assert_array_equal(clean['x'][:n_fit], moved['x'][:n_fit])
    assert not np.array_equal(clean['x'][n_fit:], moved['x'][n_fit:])


@pytest.mark.parametrize('scenario', list(SCENARIOS))
def test_baseline_fold_ignores_held_out_values(scenario, tmp_path, monkeypatch):
    files = _synthetic(scenario)
    rows = _held_out(files, 'test')
    clean = _build_baseline_fold(scenario, files, tmp_path, 'clean', monkeypatch)
    moved = _build_baseline_fold(scenario, _perturbed(files, rows), tmp_path, 'moved', monkeypatch)

    np.testing.assert_array_equal(clean.X_train_processed, moved.X_train_processed)
    np.testing.assert_array_equal(clean.X_val_processed, moved.X_val_processed)
    np.testing.assert_array_equal(clean.y_test, moved.y_test)
    assert not np.array_equal(clean.X_test_processed, moved.X_test_processed)


@pytest.mark.parametrize('scenario', list(SCENARIOS))
def test_baseline_features_match_gnn_cache(scenario, tmp_path, monkeypatch):
    files = _synthetic(scenario)
    gnn = _build_gnn_fold(scenario, files, tmp_path, 'gnn', monkeypatch)
    table = _build_baseline_fold(scenario, files, tmp_path, 'table', monkeypatch)

    n_train, n_val = gnn['split_info']['train_idx'], gnn['split_info']['val_idx']
    selected = gnn['selected'].to_numpy()
    assert table.X_train_processed.shape[1] == selected.shape[1]
    np.testing.assert_allclose(table.X_train_processed, selected[:n_train], rtol=1e-10)
    np.testing.assert_allclose(table.X_val_processed, selected[n_train:n_val], rtol=1e-10)
    np.testing.assert_allclose(table.X_test_processed, selected[n_val:], rtol=1e-10)
