"""Abstract base class for node selectors."""

from abc import ABC, abstractmethod

import numpy as np


class AbstractNodeSelector(ABC):
    """Abstract base class for node selection strategies.
    
    All node selectors must implement the `select` method which takes
    training features and labels and returns indices of selected nodes.
    This standardized interface allows different selection strategies
    to be plugged in without modifying graph construction or model training.
    """

    @abstractmethod
    def select(
        self, data: np.ndarray, targets: np.ndarray, n_selected: int
    ) -> np.ndarray:
        """Select nodes based on the implemented strategy."""
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

