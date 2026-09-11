"""Train-only per-feature median centering."""

from __future__ import annotations

import pandas as pd


class MedianCenterer:
    """Subtract the training-split column median from every split."""

    def __init__(self) -> None:
        self.median_: pd.Series | None = None

    def fit(self, data: pd.DataFrame) -> MedianCenterer:
        """Store per-column medians from training samples."""
        self.median_ = data.median(axis=0)
        if self.median_.isna().any():
            n_bad = int(self.median_.isna().sum())
            raise ValueError(f'{n_bad} features have undefined training medians')
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply frozen training medians."""
        if self.median_ is None:
            raise RuntimeError('MedianCenterer must be fit before transform')
        missing = [c for c in self.median_.index if c not in data.columns]
        if missing:
            raise ValueError(f'Features missing from data: {missing[:5]}')
        centered = data.loc[:, self.median_.index] - self.median_
        return pd.DataFrame(centered, columns=self.median_.index, index=data.index)
