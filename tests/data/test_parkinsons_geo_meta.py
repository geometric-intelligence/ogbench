"""Tests for Parkinson GEO characteristic parsing and batch inventory."""

from pathlib import Path

import pandas as pd
import yaml

from ogbench.data.utils.split_utils import group_kfold_is_feasible
from scripts.processors.parkinsons import (
    _infer_batch_field,
    _parse_geo_sample_characteristics,
)


def test_parse_geo_sample_characteristics():
    lines = [
        ['moca score: 22', 'moca score: 18'],
        ['batch: hyb1', 'batch: hyb2'],
    ]
    meta = _parse_geo_sample_characteristics(lines)
    assert list(meta.columns) == ['moca score', 'batch']
    assert meta.loc[0, 'moca score'] == '22'
    assert meta.loc[1, 'batch'] == 'hyb2'
    assert _infer_batch_field(meta) == 'batch'


def test_two_batches_cannot_support_five_fold_groups():
    groups = pd.Series(['GPL1'] * 20 + ['GPL2'] * 20).to_numpy()
    labels = [0, 1] * 20
    ok, reason = group_kfold_is_feasible(groups, labels, k=5)
    assert not ok
    assert 'need at least k=5 groups' in reason


def test_parkinsons_config_requests_batch_grouping():
    cfg_path = Path(__file__).resolve().parents[2] / 'configs' / 'dataset' / 'parkinsons.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert cfg['split_params']['grouping'] == 'batch'
