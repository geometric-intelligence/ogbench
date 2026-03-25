"""Distance correlation-based node selector."""

import dcor
import numpy as np

from ogbench.data.selectors.base import AbstractNodeSelector


class DistanceCorrelationSelector(AbstractNodeSelector):
    """Select nodes based on distance correlation with targets."""

    def select(self, data: np.ndarray, targets: np.ndarray, n_selected: int) -> np.ndarray:
        """Select nodes based on distance correlation."""
        dcorrelations = np.array(
            [dcor.distance_correlation(data[:, i], targets) for i in range(data.shape[1])]
        )
        ranked_nodes = np.argsort(dcorrelations)[::-1]
        return ranked_nodes[:n_selected]
