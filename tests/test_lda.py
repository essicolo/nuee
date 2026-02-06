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
