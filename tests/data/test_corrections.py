"""Tests for train-only sample corrections."""

import numpy as np
import pandas as pd
import pytest

from ogbench.data.corrections.center import MedianCenterer
from ogbench.data.corrections.combat import CombatCorrector
from ogbench.data.corrections.covariates import CovariateAdjuster
from ogbench.data.corrections.promoter import PromoterMinBetaSelector, select_min_promoter_per_gene


def test_covariate_adjuster_fit_does_not_use_val_rows():
    rng = np.random.default_rng(0)
    n, p = 80, 5
    age = rng.normal(50, 10, size=n)
    sex = np.where(rng.random(n) > 0.5, 'F', 'M')
    data = pd.DataFrame({f'p{i}': 0.2 * age + rng.normal(0, 0.1, size=n) for i in range(p)})
    cov = pd.DataFrame({'age': age, 'sex': sex, 'bmi': rng.normal(25, 3, n), 'race': 'A'})
    train, val = slice(0, 60), slice(60, 80)
    adj = CovariateAdjuster()
    adj.fit(data.iloc[train], cov.iloc[train])
    coef_before = {k: v.copy() for k, v in adj.coef_.items()}
    adj.transform(data.iloc[val], cov.iloc[val])
    for key, value in coef_before.items():
        np.testing.assert_array_equal(adj.coef_[key], value)


def test_covariate_adjuster_reduces_age_correlation_on_train():
    rng = np.random.default_rng(1)
    n = 100
    age = rng.normal(50, 10, size=n)
    y = 0.5 * age + rng.normal(0, 0.05, size=n)
    data = pd.DataFrame({'prot': y})
    cov = pd.DataFrame({'age': age, 'sex': 'F', 'bmi': 24.0, 'race': 'A'})
    adj = CovariateAdjuster(covariate_names=['age'])
    adj.fit(data, cov)
    out = adj.transform(data, cov)
    corr_before = abs(np.corrcoef(data['prot'], age)[0, 1])
    corr_after = abs(np.corrcoef(out['prot'], age)[0, 1])
    assert corr_after < corr_before * 0.2


def test_combat_train_only_estimates_frozen():
    rng = np.random.default_rng(2)
    n_genes = 12
    b0 = rng.normal(0, 1, size=(40, n_genes)) + 3.0
    b1 = rng.normal(0, 1, size=(40, n_genes)) - 2.0
    x = np.vstack([b0, b1])
    batches = np.array(['a'] * 40 + ['b'] * 40)
    data = pd.DataFrame(x, columns=[f'g{i}' for i in range(n_genes)])
    combat = CombatCorrector()
    combat.fit(data.iloc[:60], batches[:60])
    gamma_before = {k: v.copy() for k, v in combat.gamma_.items()}
    combat.transform(data.iloc[60:], batches[60:])
    for key, value in gamma_before.items():
        np.testing.assert_array_equal(combat.gamma_[key], value)


def test_combat_unseen_batch_raises():
    rng = np.random.default_rng(3)
    data = pd.DataFrame(rng.normal(size=(20, 4)), columns=list('abcd'))
    batches = np.array(['a'] * 10 + ['b'] * 10)
    combat = CombatCorrector()
    combat.fit(data.iloc[:10], batches[:10])
    with pytest.raises(ValueError, match='Unseen ComBat batches'):
        combat.transform(data.iloc[10:], batches[10:])


def test_combat_reduces_batch_mean_gap():
    rng = np.random.default_rng(4)
    n_genes = 8
    a = rng.normal(0, 1, size=(30, n_genes)) + 5.0
    b = rng.normal(0, 1, size=(30, n_genes))
    data = pd.DataFrame(np.vstack([a, b]), columns=[f'g{i}' for i in range(n_genes)])
    batches = np.array(['a'] * 30 + ['b'] * 30)
    gap_before = abs(data.iloc[:30].mean().to_numpy() - data.iloc[30:].mean().to_numpy()).mean()
    combat = CombatCorrector().fit(data, batches)
    out = combat.transform(data, batches)
    gap_after = abs(out.iloc[:30].mean().to_numpy() - out.iloc[30:].mean().to_numpy()).mean()
    assert gap_after < gap_before * 0.25


def test_promoter_selector_ignores_val_never_smokers():
    """Val never-smokers would prefer probe B; train-only fit must keep probe A."""
    probes = ['cgA', 'cgB']
    mapping = pd.DataFrame({'probe_id': probes, 'gene': ['GENE1', 'GENE1']})
    # 4 train never-smokers: A is lower; 4 val never-smokers: B is lower
    train_never = pd.DataFrame({'cgA': [0.1, 0.1, 0.1, 0.1], 'cgB': [0.9, 0.9, 0.9, 0.9]})
    val_never = pd.DataFrame({'cgA': [0.9, 0.9, 0.9, 0.9], 'cgB': [0.1, 0.1, 0.1, 0.1]})
    ever = pd.DataFrame({'cgA': [0.5, 0.5], 'cgB': [0.5, 0.5]})
    train = pd.concat([train_never, ever], ignore_index=True)
    val = pd.concat([val_never, ever], ignore_index=True)
    y_train = np.array([0, 0, 0, 0, 1, 1])
    selector = PromoterMinBetaSelector().fit(train, y_train, mapping)
    assert list(selector.gene_to_probe_.values) == ['cgA']
    out_val = selector.transform(val)
    np.testing.assert_allclose(out_val['GENE1'].to_numpy()[:4], [0.9, 0.9, 0.9, 0.9])


def test_select_min_promoter_matches_full_never_mask():
    mapping = pd.DataFrame({'probe_id': ['p1', 'p2', 'p3'], 'gene': ['G', 'G', 'H']})
    beta = pd.DataFrame(
        {
            'p1': [0.2, 0.3, 0.8],
            'p2': [0.4, 0.5, 0.1],
            'p3': [0.6, 0.7, 0.9],
        }
    )
    never_mask = np.array([True, True, False])
    gene_data, gene_to_probe = select_min_promoter_per_gene(beta, mapping, never_mask)
    assert gene_to_probe['G'] == 'p1'
    assert gene_to_probe['H'] == 'p3'
    np.testing.assert_allclose(gene_data['G'].to_numpy(), beta['p1'].to_numpy())


def test_median_centerer_uses_train_only():
    train = pd.DataFrame({'g': [1.0, 3.0, 5.0]})
    val = pd.DataFrame({'g': [10.0, 20.0]})
    centerer = MedianCenterer().fit(train)
    np.testing.assert_allclose(centerer.median_['g'], 3.0)
    out_val = centerer.transform(val)
    np.testing.assert_allclose(out_val['g'].to_numpy(), [7.0, 17.0])
    out_train = centerer.transform(train)
    np.testing.assert_allclose(out_train['g'].median(), 0.0)
