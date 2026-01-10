"""Correlation-based node selector."""

import numpy as np

from ogbench.data.selectors.base import AbstractNodeSelector


class CorrelationSelector(AbstractNodeSelector):
    """Select nodes based on absolute Pearson correlation with targets."""

    def select(
        self, data: np.ndarray, targets: np.ndarray, n_selected: int
    ) -> np.ndarray:
        """Select nodes based on correlation."""
        correlations = np.abs(
            np.array([np.corrcoef(data[:, i], targets)[0, 1] for i in range(data.shape[1])])
        )
        ranked_nodes = np.argsort(correlations)[::-1]
        return ranked_nodes[:n_selected]

