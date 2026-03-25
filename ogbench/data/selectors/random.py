"""Random node selector."""

import numpy as np

from ogbench.data.selectors.base import AbstractNodeSelector


class RandomSelector(AbstractNodeSelector):
    """Select nodes randomly.

    Randomly permutes all features and selects the first n_selected.
    """

    def select(self, data: np.ndarray, targets: np.ndarray, n_selected: int) -> np.ndarray:
        """Select nodes randomly."""
        ranked_nodes = np.random.permutation(data.shape[1])
        return ranked_nodes[:n_selected]
