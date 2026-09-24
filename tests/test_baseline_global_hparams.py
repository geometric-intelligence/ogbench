"""Tests for baseline hyperparameter selection across k-fold splits."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import ogbench.baseline as baseline


def _candidate(params, score):
    return baseline._param_key(params), (params, score)


def test_choose_global_hparams_uses_mean_not_fold_winners():
    weak = {'model__c': 1}
    stable = {'model__c': 2}
    fold_results = {}
    weak_scores = [0.95, 0.20, 0.20, 0.20, 0.20]
    stable_scores = [0.70, 0.70, 0.70, 0.70, 0.70]
    for fold, (weak_score, stable_score) in enumerate(
        zip(weak_scores, stable_scores, strict=True)
    ):
        fold_results[fold] = dict(
            [
                _candidate(weak, weak_score),
                _candidate(stable, stable_score),
            ]
        )

    selected = baseline._choose_global_hparams(fold_results)

    assert selected.best_params == stable
    assert selected.mean_score == pytest.approx(0.70)
    assert selected.fold_scores == dict.fromkeys(range(5), 0.70)


def test_choose_global_hparams_resolves_ties_by_grid_order():
    first = {'model__c': 1}
    second = {'model__c': 2}
    fold_results = {
        fold: dict([_candidate(first, 0.5), _candidate(second, 0.5)]) for fold in range(5)
    }

    selected = baseline._choose_global_hparams(fold_results)

    assert selected.best_params == first


def test_global_selection_scores_all_folds_once_then_reuses_cache(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            'seed': 42,
            'baseline_hparam_cache_dir': str(tmp_path),
            'dataset': {
                'split_params': {
                    'split_type': 'k-fold',
                    'k': 5,
                    'data_seed': 0,
                    'grouping': None,
                },
                'loader': {
                    'parameters': {
                        'data_name': 'toy',
                        'split_type': 'k-fold',
                        'k': 5,
                        'fold': 0,
                        'node_sample_ratio': 0.5,
                        'method': 'variance',
                        'revision': 'test-revision',
                        'corrections': [],
                        'grouping': None,
                    }
                },
            },
        }
    )
    baseline_config = OmegaConf.create({'scoring': 'f1_macro'})
    current_dataset = SimpleNamespace(fold=0, y_train=[0, 1], y_val=[0, 1])
    loaded_folds = []
    scored_folds = []

    def fake_load(fold_cfg):
        fold = int(fold_cfg.dataset.loader.parameters.fold)
        loaded_folds.append(fold)
        return SimpleNamespace(fold=fold, y_train=[0, 1], y_val=[0, 1])

    def fake_score(dataset, *_args):
        scored_folds.append(dataset.fold)
        params_a = {'model__c': 1}
        params_b = {'model__c': 2}
        return dict(
            [
                _candidate(params_a, 0.9 if dataset.fold == 0 else 0.1),
                _candidate(params_b, 0.6),
            ]
        )

    monkeypatch.setattr(baseline, 'load_and_prepare_data', fake_load)
    monkeypatch.setattr(baseline, '_score_param_grid_on_fold', fake_score)
    param_grid = {'model__c': [1, 2]}

    first = baseline._select_global_kfold_hparams(
        cfg,
        'toy_baseline',
        baseline_config,
        'standard',
        param_grid,
        current_dataset,
    )
    second = baseline._select_global_kfold_hparams(
        cfg,
        'toy_baseline',
        baseline_config,
        'standard',
        param_grid,
        current_dataset,
    )

    assert first.best_params == {'model__c': 2}
    assert first.mean_score == pytest.approx(0.6)
    assert second == first
    assert loaded_folds == [1, 2, 3, 4]
    assert scored_folds == [0, 1, 2, 3, 4]


def test_shared_fold_datasets_are_loaded_once_per_job(tmp_path, monkeypatch):
    """Two baselines in one job must not rebuild the same fold's features."""
    cfg = OmegaConf.create(
        {
            'seed': 42,
            'baseline_hparam_cache_dir': str(tmp_path),
            'dataset': {
                'split_params': {
                    'split_type': 'k-fold',
                    'k': 5,
                    'data_seed': 0,
                    'grouping': None,
                },
                'loader': {
                    'parameters': {
                        'data_name': 'toy',
                        'split_type': 'k-fold',
                        'k': 5,
                        'fold': 0,
                        'node_sample_ratio': 0.5,
                        'method': 'distance_correlation',
                        'revision': 'test-revision',
                        'corrections': [],
                        'grouping': None,
                    }
                },
            },
        }
    )
    current_dataset = SimpleNamespace(fold=0, y_train=[0, 1], y_val=[0, 1])
    loaded_folds = []

    def fake_load(fold_cfg):
        fold = int(fold_cfg.dataset.loader.parameters.fold)
        loaded_folds.append(fold)
        return SimpleNamespace(fold=fold, y_train=[0, 1], y_val=[0, 1])

    def fake_score(dataset, *_args):
        return dict([_candidate({'model__c': 1}, 0.5), _candidate({'model__c': 2}, 0.6)])

    monkeypatch.setattr(baseline, 'load_and_prepare_data', fake_load)
    monkeypatch.setattr(baseline, '_score_param_grid_on_fold', fake_score)

    fold_datasets = {}
    for baseline_name in ('svm', 'elastic_net'):
        baseline._select_global_kfold_hparams(
            cfg,
            baseline_name,
            OmegaConf.create({'scoring': 'f1_macro'}),
            'standard',
            {'model__c': [1, 2]},
            current_dataset,
            fold_datasets=fold_datasets,
        )

    # Folds 1-4 are built once in total, not once per baseline.
    assert loaded_folds == [1, 2, 3, 4]
    assert sorted(fold_datasets) == [('standard', fold) for fold in range(5)]
