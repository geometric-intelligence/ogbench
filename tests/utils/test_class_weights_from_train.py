"""Tests for computing class weights from the training split of each fold."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data

from ogbench.dataloader import DataloadDataset
from ogbench.utils.config_resolvers import resolve_class_weights_from_dataset


def _cfg(class_weights, num_classes=3):
    return OmegaConf.create(
        {
            'dataset': {
                'parameters': {
                    'task': 'classification',
                    'num_classes': num_classes,
                    'class_weights': class_weights,
                }
            }
        }
    )


def _train_split(labels):
    return DataloadDataset(
        [Data(x=torch.zeros(4, 1), y=torch.tensor([label])) for label in labels]
    )


def test_balanced_matches_sklearn_on_training_labels():
    labels = [0, 0, 0, 0, 1, 1, 2]
    cfg = _cfg('balanced')

    weights = resolve_class_weights_from_dataset(cfg, _train_split(labels))

    expected = compute_class_weight('balanced', classes=np.arange(3), y=np.array(labels))
    np.testing.assert_allclose(weights, expected)
    np.testing.assert_allclose(list(cfg.dataset.parameters.class_weights), expected)


def test_each_fold_gets_its_own_weights():
    fold_a = resolve_class_weights_from_dataset(_cfg('balanced'), _train_split([0, 0, 1, 2]))
    fold_b = resolve_class_weights_from_dataset(_cfg('balanced'), _train_split([0, 1, 1, 2]))

    assert fold_a != fold_b


def test_null_leaves_loss_unweighted():
    cfg = _cfg(None)

    assert resolve_class_weights_from_dataset(cfg, _train_split([0, 1, 2])) is None
    assert cfg.dataset.parameters.class_weights is None


def test_hardcoded_weights_are_rejected():
    with pytest.raises(ValueError, match='null or "balanced"'):
        resolve_class_weights_from_dataset(_cfg([1.0, 2.0, 3.0]), _train_split([0, 1, 2]))


def test_class_missing_from_training_split_fails():
    with pytest.raises(ValueError, match=r'no samples for classes \[2\]'):
        resolve_class_weights_from_dataset(_cfg('balanced'), _train_split([0, 0, 1]))
