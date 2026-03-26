"""Abstract base class for adjacency matrix builders."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class AbstractAdjacencyBuilder(ABC):
    """Abstract base class for adjacency matrix construction strategies.

    All adjacency builders must implement the `build` method which takes
    node features and returns a continuous adjacency matrix.
    """

    @abstractmethod
    def build(self, node_features: pd.DataFrame, map_df: pd.DataFrame | None = None) -> np.ndarray:
        """Build adjacency matrix from node features."""
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
