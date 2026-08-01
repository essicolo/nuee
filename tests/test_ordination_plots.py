import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

import nuee
from nuee.plotting.ordination_plots import _to_1d, _unique_groups, _group_mask


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _lda_data(seed=42):
    rng = np.random.default_rng(seed)
    n = 20
    X = np.vstack([
        rng.normal(loc=[0, 0, 0, 0], scale=0.5, size=(n, 4)),
        rng.normal(loc=[3, 3, 0, 0], scale=0.5, size=(n, 4)),
        rng.normal(loc=[0, 0, 3, 3], scale=0.5, size=(n, 4)),
    ])
    y = np.array(["A"] * n + ["B"] * n + ["C"] * n)
    return X, y


def _sites(fig):
    """Total number of scattered points and the legend labels of a biplot."""
    ax = fig.axes[0]
    plotted = sum(coll.get_offsets().shape[0] for coll in ax.collections)
    return plotted, ax.get_legend_handles_labels()[1]


# --- groups / color_by resolution -------------------------------------------

def test_color_by_works_on_lda_result():
    """Regression: LDA results always carry .groups, which used to be picked
    up before the exclusivity check and made color_by look like a conflict."""
    X, y = _lda_data()
    rng = np.random.default_rng(0)
    fig = nuee.biplot(nuee.lda(X, y), color_by=rng.normal(size=X.shape[0]))

    plotted, _ = _sites(fig)
    assert plotted == X.shape[0]


def test_explicit_groups_still_work_on_lda_result():
    X, y = _lda_data()
    plotted, labels = _sites(nuee.biplot(nuee.lda(X, y), groups=y))

    assert plotted == X.shape[0]
    assert sorted(labels) == ["A", "B", "C"]


def test_groups_are_auto_detected_from_lda_result():
    X, y = _lda_data()
    plotted, labels = _sites(nuee.biplot(nuee.lda(X, y)))

    assert plotted == X.shape[0]
    assert sorted(labels) == ["A", "B", "C"]


def test_groups_and_color_by_together_still_rejected():
    X, y = _lda_data()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        nuee.biplot(nuee.lda(X, y), groups=y, color_by=rng.normal(size=X.shape[0]))


# --- missing group labels ----------------------------------------------------

def test_nan_groups_keep_every_site_and_collapse_to_one_label():
    """Regression: each float NaN was its own dict key, so NaN-labelled sites
    matched nothing and were dropped while spawning phantom legend entries."""
    rng = np.random.default_rng(3)
    result = nuee.pca(rng.normal(size=(12, 4)))
    groups = pd.Series([1.0, 2.0, np.nan] * 4)

    plotted, labels = _sites(nuee.biplot(result, groups=groups))

    assert plotted == 12
    assert labels == ["1.0", "2.0", "NA"]


def test_unique_groups_collapses_missing_values():
    assert _unique_groups(np.array([1.0, 2.0, np.nan, np.nan])).count(2.0) == 1
    assert sum(pd.isna(g) for g in _unique_groups(np.array([1.0, np.nan, np.nan]))) == 1


def test_group_mask_selects_missing_values():
    groups = np.array([1.0, np.nan, 2.0, np.nan])
    assert _group_mask(groups, np.nan).tolist() == [False, True, False, True]
    assert _group_mask(groups, 1.0).tolist() == [True, False, False, False]


def test_object_dtype_none_groups_are_kept():
    rng = np.random.default_rng(4)
    result = nuee.pca(rng.normal(size=(6, 4)))
    groups = pd.Series(["a", "b", None, "a", "b", None], dtype=object)

    plotted, labels = _sites(nuee.biplot(result, groups=groups))
    assert plotted == 6
    assert "NA" in labels


# --- input validation --------------------------------------------------------

@pytest.mark.parametrize("kwarg", ["groups", "color_by"])
def test_length_mismatch_names_the_problem(kwarg):
    rng = np.random.default_rng(5)
    result = nuee.pca(rng.normal(size=(10, 4)))
    values = np.arange(5) if kwarg == "color_by" else np.array(["a"] * 5)

    with pytest.raises(ValueError, match="5 entries but the ordination has 10 sites"):
        nuee.biplot(result, **{kwarg: values})


def test_to_1d_rejects_multi_column_frames():
    frame = pd.DataFrame({"a": range(10), "b": range(10)})
    with pytest.raises(ValueError, match="received 2 columns"):
        _to_1d(frame)


def test_to_1d_accepts_single_column_shapes():
    assert _to_1d(pd.DataFrame({"a": range(4)})).shape == (4,)
    assert _to_1d(pd.Series(range(4))).shape == (4,)
    assert _to_1d(np.arange(4).reshape(4, 1)).shape == (4,)
    assert _to_1d([1, 2, 3]).shape == (3,)


# --- keyword handling --------------------------------------------------------

def test_c_kwarg_does_not_collide_with_group_colors():
    """Regression: 'c' survived into the group branch next to 'color'."""
    rng = np.random.default_rng(6)
    result = nuee.pca(rng.normal(size=(12, 4)))
    groups = pd.Series(["a", "b"] * 6)

    plotted, _ = _sites(nuee.biplot(result, groups=groups, site_kw={"c": "red"}))
    assert plotted == 12


def test_c_kwarg_does_not_collide_with_color_by():
    rng = np.random.default_rng(7)
    result = nuee.pca(rng.normal(size=(12, 4)))

    plotted, _ = _sites(nuee.biplot(result, color_by=np.arange(12),
                                    site_kw={"c": "red"}))
    assert plotted == 12


@pytest.mark.parametrize("n_groups", [12, 20])
def test_qualitative_colormap_gives_distinct_colors(n_groups):
    """tab20 holds 20 entries; sampling it on a 0-1 ramp collapsed neighbours."""
    rng = np.random.default_rng(8)
    result = nuee.pca(rng.normal(size=(n_groups, 4)))
    groups = np.array([f"g{i}" for i in range(n_groups)])

    fig = nuee.biplot(result, groups=groups)
    colors = {
        tuple(np.round(coll.get_facecolor()[0], 5))
        for coll in fig.axes[0].collections
        if coll.get_offsets().shape[0]
    }
    assert len(colors) == n_groups
