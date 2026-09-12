"""Train-only parametric ComBat (location/scale) correction."""

from __future__ import annotations

import numpy as np
import pandas as pd


class CombatCorrector:
    """Parametric ComBat with estimates frozen on the training split.

    For each gene, training data are standardized with the train grand mean and
    pooled variance. Per-batch location (``gamma``) and scale (``delta``) are
    then estimated on those standardized train values and applied to val/test
    samples whose batch was seen in train.

    Unseen batches raise ``ValueError``. Labels are never used.
    """

    def __init__(self, min_batch_samples: int = 2) -> None:
        self.min_batch_samples = int(min_batch_samples)
        self.batch_levels_: np.ndarray | None = None
        self.grand_mean_: np.ndarray | None = None
        self.var_pooled_: np.ndarray | None = None
        self.gamma_: dict[object, np.ndarray] = {}
        self.delta_: dict[object, np.ndarray] = {}

    def fit(self, data: pd.DataFrame, batches: np.ndarray) -> CombatCorrector:
        """Estimate ComBat parameters on training samples only."""
        x = np.asarray(data, dtype=float)
        batches = np.asarray(batches)
        if x.ndim != 2:
            raise ValueError('data must be 2-D (samples x features)')
        if len(batches) != x.shape[0]:
            raise ValueError('batches length must match number of samples')
        if np.isnan(x).any():
            raise ValueError('ComBat requires finite values; impute after correction or drop NaNs')

        self.batch_levels_ = np.unique(batches)
        n_samples, n_features = x.shape
        self.grand_mean_ = x.mean(axis=0)
        var_pooled = np.zeros(n_features, dtype=float)
        for batch in self.batch_levels_:
            mask = batches == batch
            n_b = int(mask.sum())
            if n_b < self.min_batch_samples:
                raise ValueError(
                    f'batch {batch!r} has {n_b} train samples; need >= {self.min_batch_samples}'
                )
            var_pooled += x[mask].var(axis=0, ddof=1) * (n_b - 1)
        self.var_pooled_ = np.clip(var_pooled / max(n_samples - 1, 1), 1e-8, None)
        scale = np.sqrt(self.var_pooled_)
        standardized = (x - self.grand_mean_) / scale

        self.gamma_.clear()
        self.delta_.clear()
        for batch in self.batch_levels_:
            mask = batches == batch
            batch_z = standardized[mask]
            self.gamma_[batch] = batch_z.mean(axis=0)
            delta = batch_z.var(axis=0, ddof=1)
            self.delta_[batch] = np.clip(delta, 1e-8, None)
        return self

    def transform(self, data: pd.DataFrame, batches: np.ndarray) -> pd.DataFrame:
        """Apply frozen train ComBat parameters."""
        if self.grand_mean_ is None or self.var_pooled_ is None:
            raise RuntimeError('CombatCorrector must be fit before transform')
        x = np.asarray(data, dtype=float)
        batches = np.asarray(batches)
        if len(batches) != x.shape[0]:
            raise ValueError('batches length must match number of samples')
        unseen = sorted(set(np.unique(batches)) - set(self.batch_levels_))
        if unseen:
            raise ValueError(f'Unseen ComBat batches in transform: {unseen}')

        scale = np.sqrt(self.var_pooled_)
        standardized = (x - self.grand_mean_) / scale
        adjusted = np.empty_like(standardized)
        for batch in np.unique(batches):
            mask = batches == batch
            gamma = self.gamma_[batch]
            delta = self.delta_[batch]
            adjusted[mask] = (standardized[mask] - gamma) / np.sqrt(delta)
        restored = adjusted * scale + self.grand_mean_
        return pd.DataFrame(restored, columns=data.columns, index=data.index)
