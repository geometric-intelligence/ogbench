"""Tests for compute_omics_split_indices."""

import numpy as np
import pytest
from sklearn.utils import shuffle as sklearn_shuffle

from ogbench.data.utils.split_utils import (
    compute_omics_split_indices,
    group_kfold_is_feasible,
    omics_cache_split_suffix,
)


def _balanced_labels(n: int = 100, n_classes: int = 2) -> np.ndarray:
    labels = np.arange(n) % n_classes
    return labels.astype(np.int64)


def test_fixed_matches_historical_shuffle_and_cut():
    labels = _balanced_labels(535)
    n = len(labels)
    ratios = [0.7, 0.15, 0.15]

    split = compute_omics_split_indices(
        labels,
        split_type='fixed',
        train_val_test_split=ratios,
        random_state=42,
    )

    order = sklearn_shuffle(np.arange(n), random_state=42)
    train_end = int(n * ratios[0])
    val_end = int(n * (ratios[0] + ratios[1]))
    np.testing.assert_array_equal(split['train'], order[:train_end])
    np.testing.assert_array_equal(split['valid'], order[train_end:val_end])
    np.testing.assert_array_equal(split['test'], order[val_end:])


def test_fixed_is_partition():
    labels = _balanced_labels(200)
    split = compute_omics_split_indices(labels, split_type='fixed')
    concat = np.concatenate([split['train'], split['valid'], split['test']])
    assert len(concat) == len(labels)
    assert len(np.unique(concat)) == len(labels)


def test_kfold_is_three_one_one_rotation():
    labels = _balanced_labels(500, n_classes=2)
    k = 5
    for fold in range(k):
        split = compute_omics_split_indices(
            labels, split_type='k-fold', k=k, fold=fold, random_state=42
        )
        n = len(labels)
        assert len(split['train']) + len(split['valid']) + len(split['test']) == n
        assert abs(len(split['test']) / n - 1 / k) < 0.03
        assert abs(len(split['valid']) / n - 1 / k) < 0.03
        assert abs(len(split['train']) / n - (k - 2) / k) < 0.05


def test_kfold_each_sample_is_test_and_val_once():
    labels = _balanced_labels(100)
    k = 5
    test_sets = []
    val_sets = []
    for fold in range(k):
        split = compute_omics_split_indices(
            labels, split_type='k-fold', k=k, fold=fold, random_state=42
        )
        test_sets.append(set(split['test'].tolist()))
        val_sets.append(set(split['valid'].tolist()))
    assert set.union(*test_sets) == set(range(len(labels)))
    assert set.union(*val_sets) == set(range(len(labels)))
    for i in range(k):
        for j in range(i + 1, k):
            assert test_sets[i].isdisjoint(test_sets[j])
            assert val_sets[i].isdisjoint(val_sets[j])
        # val of fold i is test of fold (i+1) % k
        next_test = test_sets[(i + 1) % k]
        assert val_sets[i] == next_test


def test_kfold_requires_at_least_three_folds():
    labels = _balanced_labels(50)
    with pytest.raises(ValueError, match='k must be >= 3'):
        compute_omics_split_indices(labels, split_type='k-fold', k=2, fold=0)


def test_kfold_invalid_fold_raises():
    labels = _balanced_labels(50)
    with pytest.raises(ValueError, match='fold must satisfy'):
        compute_omics_split_indices(labels, split_type='k-fold', k=5, fold=5)


def test_cache_suffix_fixed_is_none():
    assert omics_cache_split_suffix('fixed') is None
    assert omics_cache_split_suffix('k-fold', k=5, fold=2) == 'split_k-fold_k_5_fold_2'


def test_cache_suffix_separates_grouped_folds():
    plain = omics_cache_split_suffix('k-fold', k=5, fold=2)
    grouped = omics_cache_split_suffix('k-fold', k=5, fold=2, grouping='batch')
    assert grouped == f'{plain}_group_batch'
    # Fixed splits ignore grouping, so they must keep the historical cache path.
    assert omics_cache_split_suffix('fixed', grouping='batch') is None


def test_cache_name_includes_corrections_when_set():
    from ogbench.data.utils.split_utils import build_omics_cache_relative_name

    plain = build_omics_cache_relative_name('addneuromed', 0.1, 'wgcna', 'variance', 0.5, 0.7)
    combat = build_omics_cache_relative_name(
        'addneuromed', 0.1, 'wgcna', 'variance', 0.5, 0.7, corrections=['combat']
    )
    assert 'corr_combat' not in plain
    assert combat.endswith('corr_combat')


def test_wgcna_target_connectivity_replaces_fixed_threshold_in_cache_name():
    from ogbench.data.utils.split_utils import build_omics_cache_relative_name

    name = build_omics_cache_relative_name(
        'motrpac',
        0.0229,
        'wgcna',
        'variance',
        0.5,
        0.7,
        adjacency_target_connectivity=0.1,
    )

    assert 'target_connectivity_0.1' in name
    assert 'adj_thresh_' not in name


def test_group_kfold_keeps_groups_unmixed():
    n_groups = 10
    samples_per_group = 8
    groups = np.repeat(np.arange(n_groups), samples_per_group)
    labels = np.tile([0, 1], n_groups * samples_per_group // 2)
    k = 5
    ok, reason = group_kfold_is_feasible(groups, labels, k)
    assert ok, reason
    for fold in range(k):
        split = compute_omics_split_indices(
            labels, split_type='k-fold', k=k, fold=fold, random_state=42, groups=groups
        )
        train_g = set(groups[split['train']])
        val_g = set(groups[split['valid']])
        test_g = set(groups[split['test']])
        assert train_g.isdisjoint(val_g)
        assert train_g.isdisjoint(test_g)
        assert val_g.isdisjoint(test_g)


def test_group_kfold_infeasible_with_too_few_groups():
    groups = np.array([0, 0, 1, 1, 0, 1])
    labels = np.array([0, 1, 0, 1, 0, 1])
    ok, reason = group_kfold_is_feasible(groups, labels, k=5)
    assert not ok
    assert 'need at least k=5 groups' in reason


def test_fixed_rejects_groups():
    labels = _balanced_labels(20)
    groups = np.arange(20) % 4
    with pytest.raises(ValueError, match='group-aware'):
        compute_omics_split_indices(labels, split_type='fixed', groups=groups)
