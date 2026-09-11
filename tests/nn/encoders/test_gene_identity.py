"""Tests for learnable node-identity injection."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from ogbench.nn.encoders.all_cell_encoder import AllCellFeatureEncoder
from ogbench.nn.encoders.flat_encoder import FlatEncoder
from ogbench.nn.encoders.gene_identity import (
    GeneIdentityFeatureEncoder,
    LearnableGeneIdentityBank,
    apply_gene_identity_to_model,
    is_gene_identity_enabled,
    setup_gene_identity,
)


def test_is_gene_identity_enabled():
    assert not is_gene_identity_enabled(None)
    assert not is_gene_identity_enabled(OmegaConf.create({'mode': 'disabled'}))
    assert is_gene_identity_enabled(OmegaConf.create({'mode': 'learnable'}))


def test_learnable_bank_shape_and_validation():
    bank = LearnableGeneIdentityBank(num_nodes=5, embed_dim=3)
    assert bank().shape == (5, 3)
    assert bank().requires_grad

    with pytest.raises(ValueError, match='num_nodes must be positive'):
        LearnableGeneIdentityBank(num_nodes=0, embed_dim=3)
    with pytest.raises(ValueError, match='embed_dim must be positive'):
        LearnableGeneIdentityBank(num_nodes=5, embed_dim=0)


def test_concat_before_all_cell_encoder():
    num_nodes, feature_dim, embed_dim, hidden_dim = 5, 1, 3, 8
    bank = LearnableGeneIdentityBank(num_nodes, embed_dim)
    base = AllCellFeatureEncoder(
        in_channels=[feature_dim + embed_dim],
        out_channels=hidden_dim,
    )
    encoder = GeneIdentityFeatureEncoder(base, bank, combine='concat')

    data = Data(x=torch.randn(num_nodes, feature_dim))
    data.batch_0 = torch.zeros(num_nodes, dtype=torch.long)
    output = encoder(data)

    assert data.x.shape == (num_nodes, feature_dim + embed_dim)
    assert output.x_0.shape == (num_nodes, hidden_dim)


def test_concat_repeats_identity_for_batched_graphs():
    num_nodes, feature_dim, embed_dim, batch_size = 4, 2, 3, 3
    bank = LearnableGeneIdentityBank(num_nodes, embed_dim)
    base = AllCellFeatureEncoder(
        in_channels=[feature_dim + embed_dim],
        out_channels=6,
    )
    encoder = GeneIdentityFeatureEncoder(base, bank, combine='concat')

    data = Data(x=torch.randn(batch_size * num_nodes, feature_dim))
    data.batch_0 = torch.arange(batch_size).repeat_interleave(num_nodes)
    output = encoder(data)

    expected = bank().detach()
    for graph_index in range(batch_size):
        start = graph_index * num_nodes
        assert torch.allclose(
            data.x[start : start + num_nodes, feature_dim:],
            expected,
        )
    assert output.x_0.shape == (batch_size * num_nodes, 6)


def test_rejects_non_shared_node_count():
    bank = LearnableGeneIdentityBank(num_nodes=4, embed_dim=2)
    base = AllCellFeatureEncoder(in_channels=[3], out_channels=5)
    encoder = GeneIdentityFeatureEncoder(base, bank, combine='concat')
    data = Data(x=torch.randn(5, 1))
    data.batch_0 = torch.zeros(5, dtype=torch.long)

    with pytest.raises(ValueError, match='fixed node order'):
        encoder(data)


def test_flat_encoder_with_identity():
    num_nodes, feature_dim, embed_dim, batch_size = 3, 1, 2, 2
    bank = LearnableGeneIdentityBank(num_nodes, embed_dim)
    base = FlatEncoder(
        in_channels=feature_dim,
        out_channels=num_nodes * (feature_dim + embed_dim),
    )
    encoder = GeneIdentityFeatureEncoder(base, bank, combine='concat')

    data = Data(x=torch.randn(batch_size * num_nodes, feature_dim))
    data.y = torch.zeros(batch_size, dtype=torch.long)
    data.batch_size = batch_size
    output = encoder(data)

    assert output.x_0.shape == (batch_size, num_nodes * (feature_dim + embed_dim))


def test_add_combine_preserves_feature_shape():
    num_nodes, feature_dim, embed_dim = 4, 2, 5
    bank = LearnableGeneIdentityBank(num_nodes, embed_dim)
    base = AllCellFeatureEncoder(in_channels=[feature_dim], out_channels=7)
    encoder = GeneIdentityFeatureEncoder(
        base,
        bank,
        combine='add',
        add_in_channels=feature_dim,
    )
    data = Data(x=torch.randn(num_nodes, feature_dim))
    data.batch_0 = torch.zeros(num_nodes, dtype=torch.long)

    output = encoder(data)

    assert data.x.shape == (num_nodes, feature_dim)
    assert output.x_0.shape == (num_nodes, 7)


def _config(encoder_name: str, in_channels, out_channels: int):
    return OmegaConf.create(
        {
            'gene_identity': {
                'mode': 'learnable',
                'embed_dim': 3,
                'combine': 'concat',
            },
            'dataset': {'parameters': {'num_nodes': 4, 'num_features': 2}},
            'model': {
                'feature_encoder': {
                    'encoder_name': encoder_name,
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                }
            },
        }
    )


def test_setup_bumps_graph_encoder_input_channels():
    cfg = _config('AllCellFeatureEncoder', [2], 8)

    bank = setup_gene_identity(cfg, [Data(x=torch.randn(4, 2))])

    assert bank is not None
    assert cfg.model.feature_encoder.in_channels == [5]
    assert cfg.gene_identity._runtime.num_nodes == 4


def test_setup_recomputes_flat_encoder_output_channels():
    cfg = _config('FlatEncoder', 2, 8)

    setup_gene_identity(cfg, [Data(x=torch.randn(4, 2))])

    assert cfg.model.feature_encoder.out_channels == 20


def test_setup_rejects_configured_node_mismatch():
    cfg = _config('AllCellFeatureEncoder', [2], 8)

    with pytest.raises(ValueError, match='configured=4, actual=5'):
        setup_gene_identity(cfg, [Data(x=torch.randn(5, 2))])


def test_apply_wraps_model_feature_encoder():
    cfg = _config('AllCellFeatureEncoder', [2], 8)
    bank = LearnableGeneIdentityBank(num_nodes=4, embed_dim=3)
    model = SimpleNamespace(feature_encoder=AllCellFeatureEncoder(in_channels=[5], out_channels=8))

    wrapped = apply_gene_identity_to_model(model, bank, cfg)

    assert isinstance(wrapped.feature_encoder, GeneIdentityFeatureEncoder)
    assert wrapped.feature_encoder.bank is bank
