import numpy as np
import pandas as pd
import pytest
import nuee


def _make_data():
    """Create a simple 3-class dataset for LDA tests."""
    rng = np.random.default_rng(42)
    n = 20
    X = np.vstack([
        rng.normal(loc=[0, 0, 0, 0], scale=0.5, size=(n, 4)),
        rng.normal(loc=[3, 3, 0, 0], scale=0.5, size=(n, 4)),
        rng.normal(loc=[0, 0, 3, 3], scale=0.5, size=(n, 4)),
    ])
    y = np.array(["A"] * n + ["B"] * n + ["C"] * n)
    return X, y


def test_lda_returns_ordination_result():
    X, y = _make_data()
    result = nuee.lda(X, y)
    assert hasattr(result, "points")
    assert hasattr(result, "species")
    assert hasattr(result, "eigenvalues")
    assert result.points.shape[0] == X.shape[0]
    assert result.call["method"] == "LDA"


def test_lda_scores_shape():
    X, y = _make_data()
    result = nuee.lda(X, y)
    # 3 classes → max 2 discriminant axes
    assert result.points.shape == (60, 2)
    assert result.species.shape == (4, 2)


def test_lda_groups_attached():
    X, y = _make_data()
    result = nuee.lda(X, y)
    assert hasattr(result, "groups")
    np.testing.assert_array_equal(result.groups, y)


def test_lda_n_components():
    X, y = _make_data()
    result = nuee.lda(X, y, n_components=1)
    assert result.points.shape[1] == 1


def test_lda_biplot_runs():
    X, y = _make_data()
    result = nuee.lda(X, y)
    fig = result.biplot()
    assert fig is not None


def test_lda_with_dataframe():
    X, y = _make_data()
    df = pd.DataFrame(X, columns=["sp1", "sp2", "sp3", "sp4"])
    result = nuee.lda(df, y)
    assert result._species_names == ["sp1", "sp2", "sp3", "sp4"]


def test_biplot_groups_parameter():
    """biplot() accepts groups for any ordination type."""
    X, y = _make_data()
    result = nuee.pca(pd.DataFrame(X))
    fig = nuee.biplot(result, groups=y)
    assert fig is not None


def test_biplot_color_by_parameter():
    """biplot() supports continuous coloring."""
    X, _ = _make_data()
    result = nuee.pca(pd.DataFrame(X))
    values = np.random.default_rng(0).random(X.shape[0])
    fig = nuee.biplot(result, color_by=values)
    assert fig is not None


def test_biplot_groups_and_color_by_exclusive():
    X, y = _make_data()
    result = nuee.pca(pd.DataFrame(X))
    with pytest.raises(ValueError, match="mutually exclusive"):
        nuee.biplot(result, groups=y, color_by=np.ones(len(y)))


# --- regression tests -------------------------------------------------------

def test_lsqr_solver_is_rejected_with_a_clear_message():
    """sklearn's lsqr has no transform and no scalings_, so it cannot back
    an ordination; it used to crash deep inside fit_transform."""
    X, y = _make_data()
    with pytest.raises(ValueError, match="solver must be one of"):
        nuee.lda(X, y, solver="lsqr")


def test_eigen_solver_is_accepted():
    X, y = _make_data()
    result = nuee.lda(X, y, solver="eigen")
    assert result.points.shape[0] == X.shape[0]


def test_proportion_explained_is_not_renormalised_when_truncated():
    """Regression: a single retained axis reported 100% of the between-group
    variance because proportion_explained re-divided by the retained sum."""
    X, y = _make_data()
    full = nuee.lda(X, y)
    truncated = nuee.lda(X, y, n_components=1)

    assert full.proportion_explained.sum() == pytest.approx(1.0)
    assert truncated.proportion_explained[0] == pytest.approx(
        full.proportion_explained[0])
    assert truncated.proportion_explained[0] < 1.0


def test_proportion_explained_agrees_with_biplot_axis_labels():
    X, y = _make_data()
    result = nuee.lda(X, y, n_components=1)
    assert np.allclose(result.proportion_explained,
                       result.explained_variance_ratio_)


def test_loadings_are_invariant_to_feature_rescaling():
    """Regression: raw scalings_ shrank with the units of X while the scores
    stayed put, collapsing every biplot arrow onto the origin."""
    X, y = _make_data()
    base = nuee.lda(X, y)
    rescaled = nuee.lda(X * 1000.0, y)

    assert np.max(np.abs(rescaled.species)) == pytest.approx(
        np.max(np.abs(base.species)), rel=1e-8)
    ratio = np.max(np.abs(base.species)) / np.max(np.abs(base.points))
    ratio_rescaled = np.max(np.abs(rescaled.species)) / np.max(np.abs(rescaled.points))
    assert ratio == pytest.approx(ratio_rescaled, rel=1e-8)


def test_raw_loadings_are_preserved():
    X, y = _make_data()
    result = nuee.lda(X, y)
    assert result.raw_loadings.shape == result.species.shape
    assert np.allclose(result.raw_loadings, result.model.scalings_[:, :result.species.shape[1]])
