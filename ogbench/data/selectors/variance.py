"""Variance-based node selector."""

import numpy as np

from ogbench.data.selectors.base import AbstractNodeSelector


class VarianceSelector(AbstractNodeSelector):
    """Select nodes based on highest feature variance."""

    def select(
        self, data: np.ndarray, targets: np.ndarray, n_selected: int
    ) -> np.ndarray:
        """Select nodes based on variance."""
        variances = np.std(data, axis=0)
        ranked_nodes = np.argsort(variances)[::-1]
        return ranked_nodes[:n_selected]

