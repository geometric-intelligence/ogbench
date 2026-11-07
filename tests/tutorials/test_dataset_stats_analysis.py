"""Tests for the canonical fold-0 dataset statistics configuration."""

from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from ogbench.data.datasets import hf_omics
from tutorials import dataset_stats_analysis

HF_CONFIG = Path(__file__).parents[2] / 'configs' / 'hf' / 'default.yaml'


@pytest.mark.parametrize(
    ('dataset_name', 'corrections', 'grouping'),
    [
        ('motrpac', ['covariate_adjust'], None),
        ('addneuromed', ['combat'], None),
        ('smoking', ['promoter_min_beta', 'median_center'], None),
        ('parkinsons', [], 'batch'),
        ('tuberculosis', [], 'batch'),
        ('brca', [], None),
    ],
)
def test_load_dataset_uses_canonical_fold_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset_name: str,
    corrections: list[str],
    grouping: str | None,
) -> None:
    captured: dict = {}

    def fake_dataset(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    # load_dataset imports HFOmicsDataset inside the function body, so patching
    # the module attribute is what intercepts construction.
    monkeypatch.setattr(hf_omics, 'HFOmicsDataset', fake_dataset)

    dataset_stats_analysis.load_dataset(dataset_name, cache_root=str(tmp_path))

    assert captured['split_type'] == 'k-fold'
    assert captured['k'] == 5
    assert captured['fold'] == 0
    assert captured['corrections'] == corrections
    assert captured['grouping'] == grouping
    assert captured['species'] == (83332 if dataset_name == 'tuberculosis' else 9606)
    assert captured['wgcna_binarization'] == 'fixed_threshold'
    assert captured['revision'] == OmegaConf.load(HF_CONFIG).revision


# Rows are nodes, columns are training samples: nodes 0 and 1 are identical, node 2 is their
# negation, and node 3 is orthogonal to all of them.
PROFILES = np.array(
    [
        [1.0, -1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0, -1.0],
    ]
)


def test_feature_homophily_averages_profile_cosine_over_edges() -> None:
    graph = nx.Graph([(0, 1), (1, 2), (2, 3)])

    homophily = dataset_stats_analysis._compute_feature_homophily(graph, PROFILES)

    assert homophily == pytest.approx((1.0 - 1.0 + 0.0) / 3)


def test_feature_homophily_ignores_self_loops() -> None:
    graph = nx.Graph([(0, 2), (0, 0), (1, 1), (2, 2), (3, 3)])

    homophily = dataset_stats_analysis._compute_feature_homophily(graph, PROFILES)

    assert homophily == pytest.approx(-1.0)


def test_feature_homophily_without_edges_is_nan() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(range(4))
    graph.add_edges_from([(n, n) for n in range(4)])

    assert np.isnan(dataset_stats_analysis._compute_feature_homophily(graph, PROFILES))


class _FakeDataset:
    def __init__(self, edge_index: torch.Tensor, num_nodes: int) -> None:
        self.data = SimpleNamespace(edge_index=edge_index, x=torch.zeros(num_nodes, 1))

    def __getitem__(self, idx: int) -> SimpleNamespace:
        return self.data


def test_graph_stats_exclude_pipeline_self_loops() -> None:
    # Path 0-1-2 in both directions plus the self-loop the pipeline adds to every node.
    edges = [(0, 1), (1, 0), (1, 2), (2, 1), (0, 0), (1, 1), (2, 2)]
    dataset = _FakeDataset(torch.tensor(edges).t(), num_nodes=3)

    stats = dataset_stats_analysis.get_graph_stats(dataset, node_profiles=PROFILES[:3])

    assert stats['num_edges'] == 2
    assert stats['avg_degree'] == pytest.approx(4 / 3)
    assert stats['density_pct'] == pytest.approx(100 * 2 / 3)
    assert stats['clustering_coefficient'] == 0.0
    assert stats['diameter'] == 2.0
    assert stats['homophily'] == pytest.approx(0.0)
