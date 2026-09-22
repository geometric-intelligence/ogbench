"""Tests for cross-server Optuna continuation planning."""

from __future__ import annotations

from scripts.optuna_rebalance import RemainingStudy, assign_weighted


def test_weighted_assignment_is_disjoint_exhaustive_and_balanced() -> None:
    studies = [
        RemainingStudy(
            study_name=f'study-{index}',
            model='gcn',
            dataset='motrpac',
            completed_parameter_sets=0,
            remaining_folds=5,
            estimated_seconds=100.0,
        )
        for index in range(40)
    ]

    assignments = assign_weighted(
        studies,
        {'parka': 2.0, 'frank': 1.0, 'hall': 1.0},
    )

    names = [
        study.study_name
        for assigned in assignments.values()
        for study in assigned
    ]
    assert len(names) == len(set(names)) == 40
    assert set(names) == {study.study_name for study in studies}
    assert len(assignments['parka']) == 20
    assert len(assignments['frank']) == 10
    assert len(assignments['hall']) == 10


def test_weighted_assignment_rejects_nonpositive_capacity() -> None:
    try:
        assign_weighted([], {'parka': 0})
    except ValueError as error:
        assert 'positive' in str(error)
    else:
        raise AssertionError('Expected nonpositive capacity to fail')
