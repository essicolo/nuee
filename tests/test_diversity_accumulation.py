import numpy as np
import pytest

import nuee
from nuee.diversity.accumulation import _exact_richness, _pool_estimators


def _presence(seed=0, n=20, s=30):
    rng = np.random.default_rng(seed)
    return rng.random((n, s)) < 0.4


# --- specaccum ---------------------------------------------------------------

def test_specaccum_exact_matches_analytic_expectation():
    """Regression: "exact" simply accumulated sites in input order, returning
    integers instead of the expected richness of a random subset."""
    presence = _presence()
    n_sites, n_species = presence.shape
    freqs = presence.sum(axis=0)

    richness = _exact_richness(presence)

    # E(S_1) is the mean per-site richness.
    assert richness[0] == pytest.approx(presence.sum(axis=1).mean())
    # E(S_N) is the total observed richness.
    assert richness[-1] == pytest.approx(float((freqs > 0).sum()))
    # The curve is non-decreasing and generally fractional.
    assert np.all(np.diff(richness) >= -1e-12)
    assert not np.allclose(richness, np.round(richness))


def test_specaccum_exact_is_deterministic():
    species = nuee.datasets.varespec()
    first = nuee.specaccum(species, method="exact")["richness"]
    second = nuee.specaccum(species, method="exact")["richness"]
    assert np.array_equal(first, second)


def test_specaccum_random_is_reproducible_with_a_seed():
    species = nuee.datasets.varespec()
    kwargs = dict(method="random", permutations=20, random_state=7)
    assert np.allclose(nuee.specaccum(species, **kwargs)["richness"],
                       nuee.specaccum(species, **kwargs)["richness"])


def test_specaccum_rejects_unknown_method():
    """"rarefaction" used to fall through to the exact branch silently."""
    with pytest.raises(ValueError, match="method must be one of"):
        nuee.specaccum(nuee.datasets.varespec(), method="rarefaction")


# --- poolaccum ---------------------------------------------------------------

def test_pool_estimators_reduce_to_observed_richness_without_rare_species():
    """With no singletons or doubletons every extrapolation collapses to S."""
    presence = np.ones((6, 5), dtype=bool)
    observed, chao, jack1, jack2, boot = _pool_estimators(presence)

    assert observed == 5
    assert chao == pytest.approx(5.0)
    assert jack1 == pytest.approx(5.0)
    assert jack2 == pytest.approx(5.0)
    assert boot == pytest.approx(5.0)


def test_pool_estimators_extrapolate_above_observed_richness():
    presence = _presence(seed=3, n=8, s=40)
    observed, chao, jack1, jack2, boot = _pool_estimators(presence)

    assert chao >= observed
    assert jack1 >= observed
    assert boot >= observed


def test_poolaccum_returns_estimator_curves():
    """Regression: poolaccum used to return cumulative observed richness in
    input order, with none of vegan's extrapolated estimators."""
    result = nuee.poolaccum(nuee.datasets.varespec(), permutations=25,
                            random_state=1)

    assert set(result).issuperset({"sites", "richness", "chao", "jack1",
                                   "jack2", "boot", "means"})
    assert result["sites"][0] == 3          # vegan's default minsize
    assert result["sites"][-1] == 24
    assert list(result["means"].columns) == ["N", "S", "chao", "jack1",
                                             "jack2", "boot"]
    # Extrapolations sit at or above the observed richness they correct.
    assert np.all(result["chao"] >= result["richness"] - 1e-9)
    assert np.all(result["jack1"] >= result["richness"] - 1e-9)


def test_poolaccum_is_reproducible_with_a_seed():
    species = nuee.datasets.varespec()
    kwargs = dict(permutations=10, random_state=42)
    assert np.allclose(nuee.poolaccum(species, **kwargs)["chao"],
                       nuee.poolaccum(species, **kwargs)["chao"])


def test_poolaccum_rejects_minsize_larger_than_the_data():
    with pytest.raises(ValueError, match="exceeds the number of sites"):
        nuee.poolaccum(nuee.datasets.varespec(), minsize=100)
