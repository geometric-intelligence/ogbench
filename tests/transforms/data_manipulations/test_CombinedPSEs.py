"""Tests for CombinedPSEs transform with shared_graph caching."""

import torch
from torch_geometric.data import Data

from ogbench.transforms.data_manipulations.combined_positional_and_structural_encodings import (
    CombinedPSEs,
)


def _make_sample(x_values, edge_index, num_nodes):
    """Create a Data sample with the given node features and shared graph structure."""
    return Data(
        x=torch.tensor(x_values, dtype=torch.float32).unsqueeze(1),
        edge_index=edge_index.clone(),
        num_nodes=num_nodes,
    )


class TestCombinedPSEsSharedGraph:
    """Tests for CombinedPSEs with shared_graph caching."""

    def setup_method(self):
        """Set up test fixtures."""
        self.num_nodes = 6
        self.edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]]
        )
        self.encodings = ['LapPE', 'RWSE']
        self.parameters = {
            'LapPE': {'max_pe_dim': 4, 'concat_to_x': True},
            'RWSE': {'max_pe_dim': 4, 'concat_to_x': True},
        }

    def test_shared_graph_produces_same_pses(self):
        """Cached PSEs must match freshly computed PSEs for same graph structure."""
        sample_a = _make_sample([1, 2, 3, 4, 5, 6], self.edge_index, self.num_nodes)
        sample_b = _make_sample([10, 20, 30, 40, 50, 60], self.edge_index, self.num_nodes)

        transform = CombinedPSEs(
            encodings=self.encodings,
            parameters=self.parameters,
            shared_graph=True,
        )

        out_a = transform(sample_a)
        out_b = transform(sample_b)

        # PSE columns (everything after first column) must be identical
        pse_a = out_a.x[:, 1:]
        pse_b = out_b.x[:, 1:]
        assert torch.allclose(pse_a, pse_b, atol=1e-6), 'Cached PSEs differ from computed PSEs'

    def test_shared_graph_preserves_node_features(self):
        """Node features must remain distinct across samples."""
        sample_a = _make_sample([1, 2, 3, 4, 5, 6], self.edge_index, self.num_nodes)
        sample_b = _make_sample([10, 20, 30, 40, 50, 60], self.edge_index, self.num_nodes)

        transform = CombinedPSEs(
            encodings=self.encodings,
            parameters=self.parameters,
            shared_graph=True,
        )

        out_a = transform(sample_a)
        out_b = transform(sample_b)

        # Original feature column must be different
        assert not torch.equal(
            out_a.x[:, 0], out_b.x[:, 0]
        ), 'Node features should differ between samples'

    def test_shared_graph_matches_no_cache(self):
        """Output of shared_graph=True must match shared_graph=False up to sign ambiguity.

        LapPE eigenvectors have inherent sign ambiguity across independent computations, so we
        compare absolute values for the LapPE columns.
        """
        sample_cached = _make_sample([1, 2, 3, 4, 5, 6], self.edge_index, self.num_nodes)
        sample_no_cache = _make_sample([1, 2, 3, 4, 5, 6], self.edge_index, self.num_nodes)

        cached_transform = CombinedPSEs(
            encodings=self.encodings,
            parameters=self.parameters,
            shared_graph=True,
        )
        no_cache_transform = CombinedPSEs(
            encodings=self.encodings,
            parameters=self.parameters,
            shared_graph=False,
        )

        out_cached = cached_transform(sample_cached)
        out_no_cache = no_cache_transform(sample_no_cache)

        # Same output shape
        assert out_cached.x.shape == out_no_cache.x.shape

        # Original features (col 0) must be identical
        assert torch.allclose(out_cached.x[:, 0], out_no_cache.x[:, 0], atol=1e-6)

        # LapPE columns (1..4): compare absolute values due to sign ambiguity
        lap_pe_dim = self.parameters['LapPE']['max_pe_dim']
        assert torch.allclose(
            out_cached.x[:, 1 : 1 + lap_pe_dim].abs(),
            out_no_cache.x[:, 1 : 1 + lap_pe_dim].abs(),
            atol=1e-6,
        )

        # RWSE columns (5..8): must match exactly (no sign ambiguity)
        assert torch.allclose(
            out_cached.x[:, 1 + lap_pe_dim :],
            out_no_cache.x[:, 1 + lap_pe_dim :],
            atol=1e-6,
        )

    def test_no_cache_recomputes_each_time(self):
        """With shared_graph=False, no cache should be created."""
        sample = _make_sample([1, 2, 3, 4, 5, 6], self.edge_index, self.num_nodes)

        transform = CombinedPSEs(
            encodings=self.encodings,
            parameters=self.parameters,
            shared_graph=False,
        )
        transform(sample)

        assert transform._pse_cache is None, 'Cache should not be created with shared_graph=False'

    def test_shared_graph_cache_is_populated(self):
        """With shared_graph=True, cache must be populated after first sample."""
        sample = _make_sample([1, 2, 3, 4, 5, 6], self.edge_index, self.num_nodes)

        transform = CombinedPSEs(
            encodings=self.encodings,
            parameters=self.parameters,
            shared_graph=True,
        )
        transform(sample)

        assert transform._pse_cache is not None, 'Cache should be populated after first sample'
        assert (
            '_concat_pse' in transform._pse_cache
        ), 'Cache should contain concatenated PSE tensor'

    def test_attribute_based_pses(self):
        """Test caching with concat_to_x=False (attribute-based PSEs)."""
        params = {
            'LapPE': {'max_pe_dim': 4, 'concat_to_x': False},
            'RWSE': {'max_pe_dim': 4, 'concat_to_x': False},
        }
        sample_a = _make_sample([1, 2, 3, 4, 5, 6], self.edge_index, self.num_nodes)
        sample_b = _make_sample([10, 20, 30, 40, 50, 60], self.edge_index, self.num_nodes)

        transform = CombinedPSEs(
            encodings=self.encodings,
            parameters=params,
            shared_graph=True,
        )

        out_a = transform(sample_a)
        out_b = transform(sample_b)

        # Attribute-based PSEs must be identical
        assert torch.allclose(out_a.LapPE, out_b.LapPE, atol=1e-6)
        assert torch.allclose(out_a.RWSE, out_b.RWSE, atol=1e-6)
        # Node features must remain unchanged (only original column)
        assert out_a.x.shape[-1] == 1
        assert out_b.x.shape[-1] == 1

    def test_electrostatic_encoding_cached(self):
        """Test caching with ElectrostaticPE included."""
        encodings = ['LapPE', 'RWSE', 'ElectrostaticPE']
        params = {
            'LapPE': {'max_pe_dim': 4, 'concat_to_x': True},
            'RWSE': {'max_pe_dim': 4, 'concat_to_x': True},
            'ElectrostaticPE': {'concat_to_x': True},
        }
        sample_a = _make_sample([1, 2, 3, 4, 5, 6], self.edge_index, self.num_nodes)
        sample_b = _make_sample([10, 20, 30, 40, 50, 60], self.edge_index, self.num_nodes)

        transform = CombinedPSEs(
            encodings=encodings,
            parameters=params,
            shared_graph=True,
        )

        out_a = transform(sample_a)
        out_b = transform(sample_b)

        # PSE columns must be identical (4 LapPE + 4 RWSE + 7 Electrostatic = 15)
        pse_a = out_a.x[:, 1:]
        pse_b = out_b.x[:, 1:]
        assert pse_a.shape[-1] == 15
        assert torch.allclose(pse_a, pse_b, atol=1e-6)
