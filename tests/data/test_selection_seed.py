"""Random node selection must be reproducible per fold and independent of global RNG state."""

import numpy as np
import pytest

from ogbench.data.selectors import RandomSelector, get_selector, selection_seed


def test_selection_seed_is_stable_and_distinguishes_folds_and_ratios() -> None:
    seed = selection_seed('brca', 5, 2, 0.3)

    assert seed == selection_seed('brca', 5, 2, 0.3)
    assert seed == selection_seed('brca', 5, 2, '0.30')
    # Pinned: changing the derivation silently changes every random-selection cache.
    assert seed == 814453148
    others = {
        selection_seed('brca', 5, 3, 0.3),
        selection_seed('brca', 5, 2, 0.5),
        selection_seed('smoking', 5, 2, 0.3),
        selection_seed('brca', 10, 2, 0.3),
        selection_seed('brca', 5, 2, 'full'),
    }
    assert seed not in others
    assert len(others) == 5


def test_seeded_random_selector_ignores_global_numpy_state() -> None:
    data = np.zeros((4, 100))

    np.random.seed(0)
    first = get_selector('random', seed=7).select(data, np.zeros(4), 10)
    np.random.seed(123)
    np.random.random(50)
    second = get_selector('random', seed=7).select(data, np.zeros(4), 10)

    np.testing.assert_array_equal(first, second)
    assert len(set(first.tolist())) == 10
    assert not np.array_equal(first, RandomSelector(seed=8).select(data, np.zeros(4), 10))


def test_deterministic_selectors_ignore_the_seed() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(20, 30))

    np.testing.assert_array_equal(
        get_selector('variance', seed=1).select(data, np.zeros(20), 5),
        get_selector('variance').select(data, np.zeros(20), 5),
    )


@pytest.mark.parametrize('seed', [None, 3])
def test_random_selector_repr_reports_seed(seed) -> None:
    assert repr(RandomSelector(seed=seed)) == f'RandomSelector(seed={seed})'
