"""Tests for splitting and merging the ratio-0.3 cells run on the A30 servers."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.ratio03_a30_split import ADJACENCY, build_queue, slowdown_factors
from scripts.ratio03_collect import coverage_markdown
from scripts.ratio03_merge import merge_results


def _candidates() -> pd.DataFrame:
    rows = []
    priority = 0
    for model, costs in (('graph_sage', (50, 60, 70, 4000)), ('gatv2', (300, 100, 5000, 200))):
        for index, cost in enumerate(costs):
            priority += 1
            rows.append(
                {
                    'study_name': f'{model}_{index}',
                    'model': model,
                    'dataset': 'brca',
                    ADJACENCY: 'wgcna' if cost > 1000 else 'string',
                    'source_rule': 'same_cell_r0.5',
                    'estimated_fold_seconds': float(cost),
                    'priority': priority,
                }
            )
    return pd.DataFrame(rows)


def _factors() -> pd.Series:
    return pd.Series(
        {('string', 0.5): 1.0, ('wgcna', 0.5): 1.0},
        name='ratio',
    ).rename_axis([ADJACENCY, 'src_ratio'])


def test_queue_splits_gatv2_first_and_graph_sage_in_reverse() -> None:
    queue = build_queue(
        _candidates(),
        started={'graph_sage_0'},
        factors=_factors(),
        servers=['frank', 'hall'],
        first_models=['gatv2'],
        second_models=['graph_sage'],
        a30_factor=1.0,
        timeout=3600,
    )

    first = queue[queue['tier'] == 1]
    second = queue[queue['tier'] == 2]
    assert list(first['study_name']) == ['gatv2_1', 'gatv2_3', 'gatv2_0', 'gatv2_2']
    assert list(first['server']) == ['frank', 'hall', 'frank', 'hall']
    assert list(second['study_name']) == ['graph_sage_2', 'graph_sage_1', 'graph_sage_3']
    assert 'graph_sage_0' not in set(queue['study_name'])
    assert queue['study_name'].is_unique
    assert list(queue.loc[queue['server'] == 'frank', 'server_rank']) == [1, 2, 3, 4]


def test_slowdown_uses_reference_model_folds() -> None:
    candidates = pd.DataFrame(
        {
            'study_name': ['gcn_a', 'mlp_a'],
            'model': ['gcn', 'mlp'],
            ADJACENCY: ['wgcna', 'wgcna'],
            'source_rule': ['same_cell_r0.8', 'same_cell_r0.8'],
            'estimated_fold_seconds': [100.0, 100.0],
        }
    )
    attempts = pd.DataFrame(
        {
            'study_name': ['gcn_a', 'gcn_a', 'mlp_a', 'gcn_a'],
            'status': ['success', 'success', 'success', 'failed'],
            'elapsed_time': [200.0, 400.0, 900.0, 3600.0],
        }
    )

    factors = slowdown_factors(candidates, attempts)

    assert factors[('wgcna', 0.8)] == 3.0


def _results(statuses: dict[str, tuple[str, int, float | None]]) -> pd.DataFrame:
    rows = []
    for priority, name in enumerate(('a', 'b', 'c', 'd'), start=1):
        status, folds, objective = statuses.get(name, ('pending', 0, None))
        rows.append(
            {
                'priority': priority,
                'study_name': name,
                'model': 'gatv2' if name in 'ab' else 'graph_sage',
                'dataset': 'brca',
                'status': status,
                'folds_done': folds,
                'objective': objective,
                'source_rule': 'same_cell_r0.5',
                'sampled_params': f'{{"lr": {priority}}}',
            }
        )
    return pd.DataFrame(rows)


def test_merge_prefers_complete_then_parka() -> None:
    merged = merge_results(
        {
            'parka': _results({'a': ('complete', 5, 0.7), 'c': ('incomplete', 2, None)}),
            'frank': _results({'a': ('complete', 5, 0.6), 'b': ('complete', 5, 0.8)}),
            'hall': _results({'c': ('complete', 5, 0.9), 'd': ('incomplete', 1, None)}),
        }
    ).set_index('study_name')

    assert list(merged.index) == ['a', 'b', 'c', 'd']
    assert merged.loc['a', ['server', 'objective', 'duplicate_servers']].tolist() == [
        'parka',
        0.7,
        'frank',
    ]
    assert merged.loc['b', 'server'] == 'frank'
    assert merged.loc['c', ['server', 'status']].tolist() == ['hall', 'complete']
    assert merged.loc['d', ['server', 'status', 'duplicate_servers']].tolist() == [
        'hall',
        'incomplete',
        '',
    ]


def test_merge_rejects_different_configurations() -> None:
    other = _results({}).assign(sampled_params='{"lr": 0}')

    with pytest.raises(ValueError, match='different configurations'):
        merge_results({'parka': _results({}), 'frank': other})


def test_coverage_notes_follow_the_counts() -> None:
    results = _results({'a': ('complete', 5, 0.7)})
    summary = {
        'time': 'now',
        'folds_succeeded': 5,
        'folds_target': 20,
        'folds_last_hour': 0,
        'failed_attempts': 0,
        'oom_attempts': 0,
        'timeouts': 0,
    }

    text = coverage_markdown(results, summary, final=True, notes=['- Complete by server: x'])

    assert '- Complete cells: 1 / 4' in text
    assert '(out of memory: 0, timeouts: 0)\n- Complete by server: x\n' in text
    assert '| gatv2      | 1/2    |' in text
