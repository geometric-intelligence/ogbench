"""Tests for the canonical fold-0 dataset statistics configuration."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from ogbench.data.datasets import hf_omics
from tutorials import dataset_stats_analysis


@pytest.mark.parametrize(
    ('dataset_name', 'corrections', 'grouping'),
    [
        ('motrpac', ['covariate_adjust'], None),
        ('addneuromed', ['combat'], None),
        ('smoking', ['promoter_min_beta', 'median_center'], None),
        ('parkinsons', [], 'batch'),
        ('tuberculosis', [], None),
        ('brca', [], None),
    ],
)
def test_load_dataset_uses_canonical_fold_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset_name: str,
    corrections: list[str],
    grouping: str | None,
) -> None:
    captured: dict = {}

    def fake_dataset(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    # load_dataset imports HFOmicsDataset inside the function body, so patching
    # the module attribute is what intercepts construction.
    monkeypatch.setattr(hf_omics, 'HFOmicsDataset', fake_dataset)

    dataset_stats_analysis.load_dataset(dataset_name, cache_root=str(tmp_path))

    assert captured['split_type'] == 'k-fold'
    assert captured['k'] == 5
    assert captured['fold'] == 0
    assert captured['corrections'] == corrections
    assert captured['grouping'] == grouping
    assert captured['species'] == (83332 if dataset_name == 'tuberculosis' else 9606)
