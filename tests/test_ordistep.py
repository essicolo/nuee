import numpy as np
import pandas as pd
import pytest

import nuee


@pytest.fixture(scope="module")
def data():
    species = nuee.datasets.varespec()
    env = nuee.datasets.varechem()[["N", "P", "K"]]
    return species, env


def test_ordistep_selects_terms(data):
    """Regression: ordistep was a placeholder returning [] unconditionally,
    which reads as "no predictor matters" rather than as "not implemented"."""
    species, env = data
    result = nuee.ordistep(nuee.rda(species, env), env, direction="both",
                           random_state=123)

    assert result["selected_variables"], "ordistep retained nothing at all"
    assert set(result["selected_variables"]) <= set(env.columns)


def test_ordistep_rejects_a_clearly_irrelevant_term(data):
    """K sits at p ~= 0.42 and must never enter."""
    species, env = data
    result = nuee.ordistep(nuee.rda(species, env), env, direction="both",
                           random_state=123)
    assert "K" not in result["selected_variables"]


def test_ordistep_reports_a_p_value_per_retained_term(data):
    species, env = data
    result = nuee.ordistep(nuee.rda(species, env), env, direction="both",
                           random_state=123)

    assert set(result["p_values"]) == set(result["selected_variables"])
    assert all(0.0 <= p <= 1.0 for p in result["p_values"].values())


def test_ordistep_records_its_steps(data):
    species, env = data
    result = nuee.ordistep(nuee.rda(species, env), env, direction="both",
                           random_state=123)

    assert result["steps"], "no selection step was recorded"
    for step in result["steps"]:
        assert step["action"] in {"add", "drop"}
        assert step["term"] in set(env.columns)


def test_ordistep_is_reproducible_with_a_seed(data):
    species, env = data
    model = nuee.rda(species, env)
    kwargs = dict(direction="both", random_state=7, permutations=99)
    first = nuee.ordistep(model, env, **kwargs)["selected_variables"]
    second = nuee.ordistep(model, env, **kwargs)["selected_variables"]
    assert first == second


@pytest.mark.parametrize("direction", ["forward", "backward", "both"])
def test_ordistep_accepts_every_direction(direction, data):
    species, env = data
    result = nuee.ordistep(nuee.rda(species, env), env, direction=direction,
                           random_state=1, permutations=99)
    assert set(result["selected_variables"]) <= set(env.columns)


def test_ordistep_rejects_an_unknown_direction(data):
    species, env = data
    with pytest.raises(ValueError, match="direction must be"):
        nuee.ordistep(nuee.rda(species, env), env, direction="sideways")


def test_ordistep_accepts_a_plain_array(data):
    species, env = data
    result = nuee.ordistep(nuee.rda(species, env), env.to_numpy(),
                           direction="forward", random_state=1,
                           permutations=99)
    assert set(result["selected_variables"]) <= {"X1", "X2", "X3"}


def test_ordistep_requires_a_response_matrix(data):
    species, env = data
    stripped = nuee.rda(species, env)
    stripped._raw_response = None
    with pytest.raises(ValueError, match="no response matrix"):
        nuee.ordistep(stripped, env)


# --- generic permutation helpers ---------------------------------------------

def test_permtest_reports_its_null_distribution():
    rng = np.random.default_rng(0)
    data = rng.normal(size=50)
    result = nuee.permtest(lambda x: float(np.sum(x)), data, permutations=99,
                           random_state=1)

    assert result["permutations"] == 99
    assert len(result["permuted_stats"]) == 99


def test_permtest_with_zero_permutations_reports_no_p_value():
    data = np.arange(10, dtype=float)
    result = nuee.permtest(lambda x: float(x[0]), data, permutations=0)
    assert result["permutations"] == 0
    assert np.isnan(result["p_value"])


def test_permanova_matches_adonis2(data):
    """permanova is the adonis2 entry point under its other name."""
    species, env = data
    grouping = (env["N"] > env["N"].median()).to_numpy()
    distance = nuee.vegdist(species, method="bray")

    first = nuee.permanova(distance, grouping, permutations=199, random_state=5)
    second = nuee.adonis2(distance, grouping, permutations=199, random_state=5)

    assert isinstance(first, pd.DataFrame)
    pd.testing.assert_frame_equal(first, second)


def test_permtest_counts_ties_as_exceedances():
    """Regression: a bare `>=` read last-bit rounding of an invariant
    statistic as evidence, giving p = 0.87 where it must be 1."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=50)
    result = nuee.permtest(lambda x: float(np.sum(x)), data,
                           permutations=99, random_state=1)
    assert result["p_value"] == pytest.approx(1.0)


def test_permtest_detects_a_real_effect():
    idx = np.arange(50, dtype=float)
    data = np.sort(np.random.default_rng(0).normal(size=50))
    result = nuee.permtest(lambda x: float(np.corrcoef(x, idx)[0, 1]), data,
                           permutations=999, random_state=1)
    assert result["p_value"] < 0.01


def test_permtest_is_reproducible_with_a_seed():
    idx = np.arange(30, dtype=float)
    data = np.random.default_rng(2).normal(size=30)
    stat = lambda x: float(np.corrcoef(x, idx)[0, 1])
    kwargs = dict(permutations=99, random_state=11)
    assert (nuee.permtest(stat, data, **kwargs)["p_value"]
            == nuee.permtest(stat, data, **kwargs)["p_value"])
