import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

import nuee
from nuee.ordination.base import OrdinationResult


def test_metamds_equilateral_has_zero_stress():
    data = np.eye(3, dtype=float)
    with pytest.warns(UserWarning):
        result = nuee.metaMDS(
            data,
            k=2,
            distance="bray",
            trymax=10,
            maxit=200,
            autotransform=False,
            wascores=False,
            random_state=0,
        )

    ordination_points = (
        result.points.values
        if isinstance(result.points, pd.DataFrame)
        else result.points
    )
    assert result.stress < 1e-8
    assert np.allclose(
        pdist(ordination_points), pdist(data, metric="braycurtis"), atol=1e-6
    )


def test_envfit_recovers_axis_aligned_vector():
    scores = np.array(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    ord_result = OrdinationResult(points=scores)

    env = scores[:, 0] * 3.0  # purely aligned with the first axis
    with pytest.warns(UserWarning):
        fit = nuee.envfit(
            ord_result, env.reshape(-1, 1), permutations=0, scaling=None, random_state=0
        )
    fitted_scores = fit["vectors"]["scores"]

    assert fitted_scores.shape == (1, 2)
    assert np.allclose(fitted_scores[0, 0], 1.0, atol=1e-7)
    assert np.allclose(fitted_scores[0, 1], 0.0, atol=1e-7)
    assert np.allclose(fit["vectors"]["r"][0] ** 2, fit["vectors"]["r2"][0], atol=1e-10)


def test_procrustes_identifies_rigid_transform():
    rng = np.random.default_rng(42)
    base = rng.normal(size=(6, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    scale = 2.5
    translation = np.array([0.5, -1.2, 0.3])
    transformed = scale * (base @ q) + translation

    with pytest.warns(UserWarning):
        result = nuee.procrustes(base, transformed, scale=True)
    assert result["ss"] < 1e-10
    assert np.allclose(result["correlation"], 1.0, atol=1e-8)


def test_permutation_tests_are_reproducible_and_bounded():
    rng = np.random.default_rng(123)
    data = rng.normal(size=(6, 4))
    distance = nuee.vegdist(data, method="euclidean")
    grouping = np.array([0, 0, 0, 1, 1, 1])

    anosim_first = nuee.anosim(distance, grouping, permutations=199, random_state=17)
    anosim_second = nuee.anosim(distance, grouping, permutations=199, random_state=17)
    assert np.isclose(anosim_first["p_value"], anosim_second["p_value"])
    assert 0.0 <= anosim_first["p_value"] <= 1.0

    mrpp_first = nuee.mrpp(distance, grouping, permutations=199, random_state=24)
    mrpp_second = nuee.mrpp(distance, grouping, permutations=199, random_state=24)
    assert np.isclose(mrpp_first["p_value"], mrpp_second["p_value"])
    assert 0.0 <= mrpp_first["p_value"] <= 1.0


def test_anova_rda_f_ratio_matches_inertia_partition():
    rng = np.random.default_rng(7)
    species = rng.gamma(shape=2.0, scale=1.0, size=(12, 5))
    env = rng.normal(size=(12, 2))

    rda_result = nuee.rda(species, env, scale=False, center=True)
    anova_result = nuee.anova_cca(rda_result, permutations=0, random_state=0)
    table = anova_result["tab"]

    constrained = np.sum(rda_result.constrained_eig) if rda_result.constrained_eig is not None else 0.0
    residual = np.sum(rda_result.unconstrained_eig) if rda_result.unconstrained_eig is not None else 0.0
    total = constrained + residual
    assert np.isclose(total, rda_result.tot_chi, atol=1e-6)

    df_model = rda_result.rank
    df_resid = rda_result.points.shape[0] - 1 - df_model
    ms_model = constrained / max(df_model, 1)
    ms_resid = residual / max(df_resid, 1)
    expected_f = ms_model / ms_resid if ms_resid > 0 else np.inf

    assert np.isclose(table.loc["Model", "F"], expected_f, rtol=1e-6, atol=1e-9)
    assert np.isnan(table.loc["Residual", "F"])


def test_specaccum_exact_matches_incremental_richness():
    data = np.array(
        [
            [1, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 3, 0],
            [0, 0, 0, 4],
        ],
        dtype=float,
    )
    result = nuee.specaccum(data, method="exact")
    expected = np.array([1, 2, 3, 4], dtype=float)

    assert np.all(result["richness"] == expected)
    assert result["richness"][-1] == np.count_nonzero(data.sum(axis=0) > 0)


def test_specaccum_random_monotonic_and_reaches_total_richness():
    np.random.seed(0)
    data = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=float,
    )
    result = nuee.specaccum(data, method="random", permutations=256)
    richness = result["richness"]

    assert np.all(np.diff(richness) >= -1e-9)
    total_species = np.count_nonzero(data.sum(axis=0) > 0)
    assert np.isclose(richness[-1], total_species, atol=1e-6)
