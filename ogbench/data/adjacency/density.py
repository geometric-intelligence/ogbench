"""Utilities for constructing graphs with a prescribed edge density."""

import numpy as np


def binarize_to_target_connectivity(
    adjacency: np.ndarray, target_connectivity: float
) -> tuple[np.ndarray, float, float]:
    """Keep the strongest undirected edges nearest to a target connectivity.

    The number of retained off-diagonal edges is the nearest integer to
    ``target_connectivity * n * (n - 1) / 2``. Equal-weight edges are resolved
    deterministically in upper-triangle order, so the requested density is
    reached as closely as the finite number of possible edges permits.

    Returns the binary adjacency, the weakest retained edge weight, and the
    achieved connectivity.
    """
    continuous = np.asarray(adjacency, dtype=float)
    if continuous.ndim != 2 or continuous.shape[0] != continuous.shape[1]:
        raise ValueError('adjacency must be a square matrix')
    if not 0.0 <= target_connectivity <= 1.0:
        raise ValueError('target_connectivity must be between 0 and 1')
    if not np.isfinite(continuous).all():
        raise ValueError('adjacency must contain only finite values')

    n_nodes = continuous.shape[0]
    binary = np.eye(n_nodes, dtype=np.int8)
    rows, cols = np.triu_indices(n_nodes, k=1)
    n_possible = len(rows)
    if n_possible == 0:
        return binary, float('inf'), 0.0

    # WGCNA should be symmetric. Averaging also makes the graph definition
    # robust to negligible floating-point asymmetry.
    weights = (continuous[rows, cols] + continuous[cols, rows]) / 2.0
    n_keep = int(np.floor(target_connectivity * n_possible + 0.5))
    order = np.argsort(-weights, kind='stable')
    selected = order[:n_keep]
    binary[rows[selected], cols[selected]] = 1
    binary[cols[selected], rows[selected]] = 1

    cutoff = float(weights[order[n_keep - 1]]) if n_keep else float('inf')
    achieved = n_keep / n_possible
    return binary, cutoff, achieved
