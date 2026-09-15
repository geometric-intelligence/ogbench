"""Tests for train-fold adjacency density targeting."""

import numpy as np
import pytest

from ogbench.data.adjacency.density import binarize_to_target_connectivity


def _connectivity(adjacency: np.ndarray) -> float:
    n_nodes = adjacency.shape[0]
    return np.triu(adjacency, k=1).sum() / (n_nodes * (n_nodes - 1) / 2)


def test_target_connectivity_keeps_exact_nearest_edge_count() -> None:
    rng = np.random.default_rng(42)
    continuous = rng.random((11, 11))
    continuous = (continuous + continuous.T) / 2

    binary, _cutoff, achieved = binarize_to_target_connectivity(continuous, 0.10)

    # 11 nodes have 55 possible edges, so 10% rounds to 6/55.
    assert np.triu(binary, k=1).sum() == 6
    assert achieved == pytest.approx(6 / 55)
    assert _connectivity(binary) == pytest.approx(achieved)
    np.testing.assert_array_equal(binary, binary.T)
    np.testing.assert_array_equal(np.diag(binary), np.ones(11))


def test_equal_weights_are_resolved_deterministically() -> None:
    continuous = np.ones((5, 5))

    first, cutoff, achieved = binarize_to_target_connectivity(continuous, 0.30)
    second, _, _ = binarize_to_target_connectivity(continuous, 0.30)

    np.testing.assert_array_equal(first, second)
    assert np.triu(first, k=1).sum() == 3
    assert cutoff == pytest.approx(1.0)
    assert achieved == pytest.approx(0.30)


@pytest.mark.parametrize('target', [-0.01, 1.01])
def test_invalid_target_is_rejected(target: float) -> None:
    with pytest.raises(ValueError, match='between 0 and 1'):
        binarize_to_target_connectivity(np.eye(3), target)
