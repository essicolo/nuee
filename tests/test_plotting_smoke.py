"""Smoke tests for the plotting surface.

These do not check appearance; they check that every public plotting entry
point runs on real data, returns a figure, and actually draws something.
Several of these functions had no test at all.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import nuee


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def species():
    return nuee.datasets.varespec()


@pytest.fixture(scope="module")
def env():
    return nuee.datasets.varechem()


@pytest.fixture(scope="module")
def groups(env):
    return pd.Series(np.where(env["pH"] > env["pH"].median(), "high", "low"),
                     index=env.index)


def _drawn(figure):
    """A figure counts as drawn when some axes holds marks."""
    return any(axes.collections or axes.lines or axes.patches or axes.images
               for axes in figure.axes)


def test_ordiplot_runs(species, env):
    figure = nuee.ordiplot(nuee.rda(species, env[["N", "P", "K"]]))
    assert isinstance(figure, plt.Figure)
    assert _drawn(figure)


def test_plot_ordination_runs(species):
    figure = nuee.plot_ordination(nuee.pca(species))
    assert isinstance(figure, plt.Figure)
    assert _drawn(figure)


def test_plot_ordination_with_groups(species, groups):
    figure = nuee.plot_ordination(nuee.pca(species), groups=groups)
    assert _drawn(figure)


def test_ordiellipse_runs(species, groups):
    figure = nuee.ordiellipse(nuee.pca(species), groups=groups)
    assert isinstance(figure, plt.Figure)
    # An ellipse per group is a patch.
    assert any(axes.patches for axes in figure.axes)


def test_ordispider_runs(species, groups):
    figure = nuee.ordispider(nuee.pca(species), groups=groups)
    assert isinstance(figure, plt.Figure)
    assert any(axes.lines or axes.collections for axes in figure.axes)


def test_biplot_runs_on_a_constrained_ordination(species, env):
    figure = nuee.biplot(nuee.rda(species, env[["N", "P", "K"]]))
    assert _drawn(figure)


def test_plot_diversity_runs(species):
    figure = nuee.plot_diversity(species)
    assert isinstance(figure, plt.Figure)
    assert _drawn(figure)


def test_plot_dissimilarity_runs(species):
    figure = nuee.plot_dissimilarity(nuee.vegdist(species, method="bray"))
    assert isinstance(figure, plt.Figure)
    assert _drawn(figure)


def test_plot_specaccum_runs(species):
    figure = nuee.plot_specaccum(nuee.specaccum(species, method="exact"))
    assert isinstance(figure, plt.Figure)
    assert _drawn(figure)


def test_plot_rarecurve_runs(species):
    counts = np.round(species.values).astype(int)
    figure = nuee.plot_rarecurve(nuee.rarecurve(counts))
    assert isinstance(figure, plt.Figure)
    assert _drawn(figure)


def test_plot_betadisper_runs(species, groups):
    result = nuee.betadisper(nuee.vegdist(species, method="bray"), groups)
    figure = nuee.plot_betadisper(result)
    assert isinstance(figure, plt.Figure)
    assert _drawn(figure)


@pytest.mark.parametrize("scaling", [1, 2, 3])
def test_ordiplot_accepts_every_scaling(species, env, scaling):
    figure = nuee.ordiplot(nuee.rda(species, env[["N", "P", "K"]]),
                           scaling=scaling)
    assert _drawn(figure)


def test_plot_betadisper_draws_one_box_per_group(species, groups):
    """Regression: this rendered the literal text "Implementation pending"."""
    result = nuee.betadisper(nuee.vegdist(species, method="bray"), groups)
    figure = nuee.plot_betadisper(result)
    axes = figure.axes[0]

    n_groups = len(set(groups))
    # One box (a Line2D/PathPatch per box) plus the mean markers.
    assert axes.collections, "group means were not marked"
    assert axes.collections[0].get_offsets().shape[0] == n_groups
    assert axes.get_ylabel() == "Distance to centroid"
    assert not any("pending" in text.get_text().lower() for text in axes.texts)


def test_betadisper_result_carries_the_grouping(species, groups):
    result = nuee.betadisper(nuee.vegdist(species, method="bray"), groups)
    assert "groups" in result
    assert len(result["groups"]) == len(result["distances"])
    assert set(map(str, result["groups"])) == set(map(str, groups))


def test_plot_betadisper_requires_the_grouping(species, groups):
    result = nuee.betadisper(nuee.vegdist(species, method="bray"), groups)
    del result["groups"]
    with pytest.raises(ValueError, match="no 'groups'"):
        nuee.plot_betadisper(result)
