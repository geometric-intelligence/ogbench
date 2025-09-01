from typing import Optional

import numpy as np


class MeanStdNormalizer:
    """Normalize features using mean and standard deviation."""

    def __init__(
        self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None
    ) -> None:
        self.mean = mean
        self.std = std

    def fit(self, data: np.ndarray) -> None:
        """Compute mean and standard deviation from data."""
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Normalize input array."""
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer must be fitted before use")
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Convert normalized data back to original scale."""
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer must be fitted before use")
        return x * self.std + self.mean


class MinMaxNormalizer:
    """Normalize target values using min-max scaling."""

    def __init__(self, min_val: Optional[float] = None, max_val: Optional[float] = None) -> None:
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, data: np.ndarray) -> None:
        """Compute min and max values from data."""
        self.min_val = np.min(data)
        self.max_val = np.max(data)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Normalize input array to [0, 1] range."""
        if self.min_val is None or self.max_val is None:
            raise ValueError("Normalizer must be fitted before use")
        return (x - self.min_val) / (self.max_val - self.min_val)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Convert normalized data back to original scale."""
        if self.min_val is None or self.max_val is None:
            raise ValueError("Normalizer must be fitted before use")
        return x * (self.max_val - self.min_val) + self.min_val
