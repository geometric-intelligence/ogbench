"""Random node selector."""

import numpy as np

from ogbench.data.selectors.base import AbstractNodeSelector


class RandomSelector(AbstractNodeSelector):
    """Select nodes randomly.

    Randomly permutes all features and selects the first n_selected. With a seed the permutation
    comes from its own generator, so it does not depend on the global NumPy RNG state.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def select(self, data: np.ndarray, targets: np.ndarray, n_selected: int) -> np.ndarray:
        """Select nodes randomly."""
        if self.seed is None:
            ranked_nodes = np.random.permutation(data.shape[1])
        else:
            ranked_nodes = np.random.default_rng(self.seed).permutation(data.shape[1])
        return ranked_nodes[:n_selected]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(seed={self.seed})'
