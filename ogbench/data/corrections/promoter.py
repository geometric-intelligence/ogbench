"""Train-only promoter probe selection (smoking / Illumina 450k)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_min_promoter_per_gene(
    beta: pd.DataFrame, mapping: pd.DataFrame, never_mask: np.ndarray
) -> tuple[pd.DataFrame, pd.Series]:
    """For each gene pick the candidate probe with the minimum mean beta in never-smokers.

    ``never_mask`` must align with ``beta`` rows. Means are computed only on the
    True entries of that mask (typically training never-smokers).

    Returns
    -------
    gene_data:
        DataFrame (samples x genes) of selected probe betas, columns renamed to genes.
    gene_to_probe:
        Series mapping gene -> chosen probe_id.
    """
    if 'probe_id' not in mapping.columns or 'gene' not in mapping.columns:
        raise ValueError('probe mapping must have probe_id and gene columns')
    never_mask = np.asarray(never_mask, dtype=bool)
    if len(never_mask) != len(beta):
        raise ValueError('never_mask length must match number of samples')
    if never_mask.sum() == 0:
        raise ValueError('No never-smoker samples found (label == 0)')

    available = mapping[mapping['probe_id'].isin(beta.columns)].copy()
    if available.empty:
        raise ValueError('No manifest probes overlap with the beta matrix columns')

    candidate_probes = available['probe_id'].unique().tolist()
    never_means = beta.loc[never_mask, candidate_probes].mean(axis=0)
    available['never_mean'] = available['probe_id'].map(never_means)

    available = available.dropna(subset=['never_mean'])
    if available.empty:
        raise ValueError('All candidate probes have NaN mean across never-smoker samples')

    idx_min = available.groupby('gene')['never_mean'].idxmin()
    chosen = available.loc[idx_min, ['gene', 'probe_id']]
    gene_to_probe = pd.Series(
        chosen['probe_id'].values, index=chosen['gene'].values, name='probe_id'
    )

    gene_data = beta.loc[:, gene_to_probe.values].copy()
    gene_data.columns = gene_to_probe.index.astype(str)
    return gene_data, gene_to_probe


class PromoterMinBetaSelector:
    """Collapse promoter probes to genes using train never-smoker means only.

    Never-smokers are samples with label ``0`` (GEO smoking status never).
    The chosen probe per gene is frozen and applied to val/test.
    """

    def __init__(self) -> None:
        self.gene_to_probe_: pd.Series | None = None

    def fit(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        mapping: pd.DataFrame,
    ) -> PromoterMinBetaSelector:
        """Select one promoter probe per gene from training never-smokers."""
        labels = np.asarray(labels).reshape(-1)
        if len(labels) != len(data):
            raise ValueError('labels length must match number of samples')
        never_mask = labels == 0
        gene_data, gene_to_probe = select_min_promoter_per_gene(data, mapping, never_mask)
        all_nan_cols = gene_data.columns[gene_data.isna().all(axis=0)]
        if len(all_nan_cols) > 0:
            gene_to_probe = gene_to_probe.drop(index=all_nan_cols)
        self.gene_to_probe_ = gene_to_probe
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the frozen probe-to-gene map."""
        if self.gene_to_probe_ is None:
            raise RuntimeError('PromoterMinBetaSelector must be fit before transform')
        missing = [p for p in self.gene_to_probe_.values if p not in data.columns]
        if missing:
            raise ValueError(f'Selected probes missing from data: {missing[:5]}')
        gene_data = data.loc[:, self.gene_to_probe_.values].copy()
        gene_data.columns = self.gene_to_probe_.index.astype(str)
        return gene_data
