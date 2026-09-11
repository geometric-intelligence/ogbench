"""Tests for fold-aware num_nodes calculation and model-dim sync."""

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from ogbench.utils.config_resolvers import (
    calculate_num_nodes,
    sync_num_nodes_from_dataset,
)


def test_calculate_num_nodes_fixed_matches_historical():
    assert calculate_num_nodes(711, [0.7, 0.15, 0.15], 0.5, 17198) == 994
    assert calculate_num_nodes(711, [0.7, 0.15, 0.15], 0.5, 17198, split_type='fixed') == 994


def test_calculate_num_nodes_full():
    assert calculate_num_nodes(100, [0.7, 0.15, 0.15], 'full', 5000) == 5000


def test_calculate_num_nodes_kfold_three_one_one():
    fixed = calculate_num_nodes(711, [0.7, 0.15, 0.15], 0.5, 17198, split_type='fixed')
    kfold = calculate_num_nodes(711, [0.7, 0.15, 0.15], 0.5, 17198, split_type='k-fold', k=5)
    assert kfold < fixed
    n_train = int(711 * 3 / 5)
    assert kfold == int(n_train / 0.5)


def test_sync_num_nodes_remaps_omics_readout_hidden_dim():
    old_n, new_n, channels = 994, 852, 64
    cfg = OmegaConf.create(
        {
            'dataset': {'parameters': {'num_nodes': old_n}},
            'model': {
                'backbone': {'num_nodes': old_n},
                'readout': {
                    'num_nodes': old_n,
                    'hidden_dim': old_n * channels,
                    'fc_dim': [16, 8, 4],
                },
            },
        }
    )
    graph = Data(x=torch.randn(new_n, 1), edge_index=torch.zeros(2, 0, dtype=torch.long))

    class _Split:
        data_lst = [graph]

    synced = sync_num_nodes_from_dataset(cfg, _Split())
    assert synced == new_n
    assert cfg.dataset.parameters.num_nodes == new_n
    assert cfg.model.backbone.num_nodes == new_n
    assert cfg.model.readout.num_nodes == new_n
    assert cfg.model.readout.hidden_dim == new_n * channels
    assert list(cfg.model.readout.fc_dim) == [16, 8, 4]


def test_sync_noop_when_matching():
    n = 100
    cfg = OmegaConf.create(
        {
            'dataset': {'parameters': {'num_nodes': n}},
            'model': {'readout': {'hidden_dim': n * 32}},
        }
    )
    graph = Data(x=torch.randn(n, 1))

    class _Split:
        data_lst = [graph]

    assert sync_num_nodes_from_dataset(cfg, _Split()) == n
    assert cfg.model.readout.hidden_dim == n * 32
