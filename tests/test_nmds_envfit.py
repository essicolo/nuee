import numpy as np
import pytest

import nuee
from nuee.ordination.envfit import _fit_vector
from nuee.ordination.nmds import _kruskal_stress_1, _principal_axes


# --- NMDS stress and axis convention ----------------------------------------

def test_kruskal_stress_is_scale_free():
    """Regression: the raw SMACOF stress was normalised by a pair count, which
    made the reported value depend on the number of sites."""
    rng = np.random.default_rng(0)
    points = rng.normal(size=(30, 2))
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)

    stress = _kruskal_stress_1(distances, points)
    scaled = _kruskal_stress_1(distances * 10.0, points * 10.0)

    assert stress == pytest.approx(scaled, abs=1e-9)
    assert 0.0 <= stress < 1e-6      # a perfect embedding has ~zero stress


def test_kruskal_stress_is_bounded_and_positive_for_a_poor_fit():
    rng = np.random.default_rng(1)
    distances = rng.random((15, 15))
    distances = (distances + distances.T) / 2
    np.fill_diagonal(distances, 0.0)
    embedding = rng.normal(size=(15, 2))

    stress = _kruskal_stress_1(distances, embedding)
    assert 0.0 < stress < 1.0


def test_principal_axes_preserves_distances_and_orthogonalises():
    rng = np.random.default_rng(2)
    points = rng.normal(size=(25, 2)) @ np.array([[2.0, 1.0], [0.5, 1.5]])

    rotated = _principal_axes(points)

    before = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    after = np.linalg.norm(rotated[:, None, :] - rotated[None, :, :], axis=-1)
    assert np.allclose(before, after)

    centred = rotated - rotated.mean(axis=0)
    assert abs(np.corrcoef(centred[:, 0], centred[:, 1])[0, 1]) < 1e-10
    # Axes ordered by decreasing spread.
    assert centred[:, 0].var() >= centred[:, 1].var()


def test_metamds_axes_are_orthogonal():
    result = nuee.metaMDS(nuee.datasets.varespec(), k=2, distance="bray",
                          random_state=123)
    points = np.asarray(result.points)
    centred = points - points.mean(axis=0)
    assert abs(np.corrcoef(centred[:, 0], centred[:, 1])[0, 1]) < 1e-10


def test_metamds_reports_kruskal_stress_in_a_plausible_range():
    result = nuee.metaMDS(nuee.datasets.varespec(), k=2, distance="bray",
                          random_state=123)
    assert 0.0 < result.stress < 1.0
    # vegan reports 0.18432 for this dataset with the same autotransformation.
    assert result.stress == pytest.approx(0.18432, abs=0.005)


# --- envfit ------------------------------------------------------------------

def test_fit_vector_matches_regression_r_squared_on_oblique_axes():
    """Regression: r2 was the sum of squared per-axis correlations, which only
    equals the regression R-squared when the axes are orthogonal."""
    rng = np.random.default_rng(3)
    axes = rng.normal(size=(40, 2))
    axes[:, 1] += 0.6 * axes[:, 0]              # deliberately oblique
    axes -= axes.mean(axis=0)
    env = axes @ np.array([1.5, -0.7]) + rng.normal(size=40) * 0.5
    env -= env.mean()

    _, r_squared = _fit_vector(env, axes)

    beta, *_ = np.linalg.lstsq(axes, env, rcond=None)
    expected = 1 - np.sum((env - axes @ beta) ** 2) / np.sum(env ** 2)
    assert r_squared == pytest.approx(expected, rel=1e-10)

    naive = sum(np.corrcoef(env, axes[:, a])[0, 1] ** 2 for a in range(2))
    assert not np.isclose(naive, expected, rtol=1e-3)


def test_fit_vector_arrow_length_encodes_the_correlation():
    rng = np.random.default_rng(4)
    axes = rng.normal(size=(30, 2))
    axes -= axes.mean(axis=0)
    env = axes @ np.array([1.0, 0.0])

    arrow, r_squared = _fit_vector(env, axes)
    assert np.linalg.norm(arrow) == pytest.approx(np.sqrt(r_squared))


def test_envfit_r2_matches_direct_regression():
    species = nuee.datasets.varespec()
    env = nuee.datasets.varechem()[["N", "P", "K", "Ca", "Mg"]]
    result = nuee.metaMDS(species, k=2, distance="bray", random_state=123)

    vectors = nuee.envfit(result, env, permutations=99, random_state=123)["vectors"]

    points = np.asarray(result.points)
    points = points - points.mean(axis=0)
    for idx, column in enumerate(env.columns):
        y = env[column].to_numpy() - env[column].to_numpy().mean()
        beta, *_ = np.linalg.lstsq(points, y, rcond=None)
        expected = 1 - np.sum((y - points @ beta) ** 2) / np.sum(y ** 2)
        assert vectors["r2"][idx] == pytest.approx(expected, rel=1e-9)


def test_envfit_r_is_the_root_of_r2():
    species = nuee.datasets.varespec()
    env = nuee.datasets.varechem()[["N", "P"]]
    result = nuee.metaMDS(species, k=2, distance="bray", random_state=123)
    vectors = nuee.envfit(result, env, permutations=49, random_state=1)["vectors"]

    assert np.allclose(vectors["r"], np.sqrt(vectors["r2"]))
