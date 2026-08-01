import copy

import numpy as np
import pandas as pd
import pytest
import nuee
from nuee.permutation.permtest import _compute_rda_f, _conditioning_rank


def _manual_partial_rda(X, Y, Z):
    """Textbook partial RDA inertia decomposition on the vegan variance scale.

    Returns (total, conditioned, constrained, unconstrained). The response is
    centred and divided by sqrt(n - 1) so that the total sum of squares equals
    the total variance, which is the inertia convention nuee/vegan report.
    """
    n = X.shape[0]
    Xc = (X - X.mean(axis=0)) / np.sqrt(n - 1)
    Yc = Y - Y.mean(axis=0)
    Zc = Z - Z.mean(axis=0)

    total = float(np.sum(Xc ** 2))
    X_res = Xc - Zc @ np.linalg.lstsq(Zc, Xc, rcond=None)[0]
    Y_res = Yc - Zc @ np.linalg.lstsq(Zc, Yc, rcond=None)[0]
    fitted = Y_res @ np.linalg.lstsq(Y_res, X_res, rcond=None)[0]

    return (total,
            total - float(np.sum(X_res ** 2)),
            float(np.sum(fitted ** 2)),
            float(np.sum((X_res - fitted) ** 2)))


def _make_data(n=60, seed=11):
    """Response driven by both the constraining and the conditioning matrix."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2))
    Y = rng.normal(size=(n, 3))
    X = Y @ (rng.normal(size=(3, 8)) * 3.0) + Z @ (rng.normal(size=(2, 8)) * 3.0)
    return X + rng.normal(size=(n, 8)) * 0.5, Y, Z


@pytest.mark.parametrize("n", [30, 100, 500])
def test_partial_rda_inertia_matches_manual_decomposition(n):
    """Inertias must be on the same scale as tot_chi, independent of n.

    Regression test: the conditioned residual was normalised by sqrt(n - 1)
    twice, which deflated every eigenvalue by a factor of (n - 1) while
    tot_chi stayed on the correct scale.
    """
    X, Y, Z = _make_data(n=n)
    total, conditioned, constrained, unconstrained = _manual_partial_rda(X, Y, Z)
    result = nuee.rda(X, Y, Z=Z)

    assert result.tot_chi == pytest.approx(total, rel=1e-10)
    assert result.partial_chi == pytest.approx(conditioned, rel=1e-10)
    assert float(np.sum(result.constrained_eig)) == pytest.approx(constrained, rel=1e-10)
    assert float(np.sum(result.unconstrained_eig)) == pytest.approx(unconstrained, rel=1e-10)


@pytest.mark.parametrize("n", [30, 100, 500])
def test_partial_rda_inertia_is_additive(n):
    """conditioned + constrained + unconstrained must equal the total."""
    X, Y, Z = _make_data(n=n)
    result = nuee.rda(X, Y, Z=Z)
    parts = (result.partial_chi
             + float(np.sum(result.constrained_eig))
             + float(np.sum(result.unconstrained_eig)))
    assert parts == pytest.approx(result.tot_chi, rel=1e-10)


def test_partial_rda_constrained_proportion_is_meaningful():
    """A strong constraining signal must not report a near-zero proportion."""
    X, Y, Z = _make_data()
    total, _, constrained, _ = _manual_partial_rda(X, Y, Z)
    result = nuee.rda(X, Y, Z=Z)

    proportion = float(np.sum(result.constrained_proportion))
    assert proportion == pytest.approx(constrained / total, rel=1e-10)
    assert proportion > 0.1


def test_partial_rda_with_orthogonal_conditioner_matches_simple_rda():
    """Conditioning on a variable orthogonal to X and Y must change nothing."""
    rng = np.random.default_rng(7)
    n = 60
    Y = rng.normal(size=(n, 3))
    X = Y @ rng.normal(size=(3, 6)) + rng.normal(size=(n, 6)) * 0.4

    basis, _ = np.linalg.qr(np.hstack([np.ones((n, 1)), Y, X]))
    Z = rng.normal(size=(n, 1))
    Z = Z - basis @ (basis.T @ Z)

    partial = nuee.rda(X, Y, Z=Z)
    simple = nuee.rda(X, Y)

    assert partial.partial_chi == pytest.approx(0.0, abs=1e-8)
    assert partial.tot_chi == pytest.approx(simple.tot_chi, rel=1e-10)
    assert float(np.sum(partial.constrained_eig)) == pytest.approx(
        float(np.sum(simple.constrained_eig)), rel=1e-8)


def test_partial_rda_on_varespec_matches_manual_decomposition():
    varespec = nuee.datasets.varespec()
    varechem = nuee.datasets.varechem()
    X = varespec.values
    Y = varechem[["N", "P", "K"]].values
    Z = varechem[["Al", "pH"]].values

    total, conditioned, constrained, unconstrained = _manual_partial_rda(X, Y, Z)
    result = nuee.rda(X, Y, Z=Z)

    assert result.tot_chi == pytest.approx(total, rel=1e-10)
    assert result.partial_chi == pytest.approx(conditioned, rel=1e-10)
    assert float(np.sum(result.constrained_eig)) == pytest.approx(constrained, rel=1e-10)
    assert float(np.sum(result.unconstrained_eig)) == pytest.approx(unconstrained, rel=1e-10)


def test_inertia_partition_matches_manual_decomposition():
    X, Y, Z = _make_data()
    total, conditioned, constrained, unconstrained = _manual_partial_rda(X, Y, Z)
    table = nuee.rda(X, Y, Z=Z).inertia_partition()

    assert list(table.index) == ["Total", "Conditioned", "Constrained", "Unconstrained"]
    assert list(table.columns) == ["Inertia", "Proportion"]
    assert table.loc["Total", "Inertia"] == pytest.approx(total, rel=1e-10)
    assert table.loc["Conditioned", "Inertia"] == pytest.approx(conditioned, rel=1e-10)
    assert table.loc["Constrained", "Inertia"] == pytest.approx(constrained, rel=1e-10)
    assert table.loc["Unconstrained", "Inertia"] == pytest.approx(unconstrained, rel=1e-10)


def test_inertia_partition_components_sum_to_total():
    X, Y, Z = _make_data()
    table = nuee.rda(X, Y, Z=Z).inertia_partition()
    components = ["Conditioned", "Constrained", "Unconstrained"]

    assert table.loc[components, "Inertia"].sum() == pytest.approx(
        table.loc["Total", "Inertia"], rel=1e-10)
    assert table.loc[components, "Proportion"].sum() == pytest.approx(1.0, rel=1e-10)
    assert table.loc["Total", "Proportion"] == pytest.approx(1.0, rel=1e-12)


def test_inertia_partition_omits_conditioned_row_when_not_partial():
    X, Y, _ = _make_data()
    table = nuee.rda(X, Y).inertia_partition()

    assert list(table.index) == ["Total", "Constrained", "Unconstrained"]
    assert table.loc[["Constrained", "Unconstrained"], "Inertia"].sum() == pytest.approx(
        table.loc["Total", "Inertia"], rel=1e-10)


def test_inertia_partition_works_for_cca():
    varespec = nuee.datasets.varespec()
    varechem = nuee.datasets.varechem()
    table = nuee.cca(varespec, varechem[["N", "P", "K"]]).inertia_partition()

    assert list(table.index) == ["Total", "Constrained", "Unconstrained"]
    assert table.loc[["Constrained", "Unconstrained"], "Inertia"].sum() == pytest.approx(
        table.loc["Total", "Inertia"], rel=1e-10)


def test_permutest_residual_df_excludes_conditioning_rank():
    """Conditioning variables consume residual degrees of freedom.

    Regression test: df_resid was n - 1 - df_model, ignoring the rank of Z,
    which inflated F for every partial RDA.
    """
    varespec = nuee.datasets.varespec()
    varechem = nuee.datasets.varechem()
    result = nuee.rda(varespec, varechem[["N", "P", "K"]], Z=varechem[["Al", "pH"]])
    stats = _compute_rda_f(result)

    n_samples = result.call["n_samples"]
    assert _conditioning_rank(result) == 2
    assert stats["df_resid"] == n_samples - 1 - stats["df_model"] - 2

    expected_f = ((stats["constrained"] / stats["df_model"])
                  / (stats["residual"] / stats["df_resid"]))
    assert stats["f_value"] == pytest.approx(expected_f, rel=1e-12)


def test_permutest_df_unchanged_without_conditioning():
    varespec = nuee.datasets.varespec()
    varechem = nuee.datasets.varechem()
    result = nuee.rda(varespec, varechem[["N", "P", "K"]])
    stats = _compute_rda_f(result)

    assert _conditioning_rank(result) == 0
    assert stats["df_resid"] == result.call["n_samples"] - 1 - stats["df_model"]


def test_conditioning_rank_ignores_collinear_columns():
    varespec = nuee.datasets.varespec()
    varechem = nuee.datasets.varechem()
    duplicated = pd.DataFrame({"Al": varechem["Al"], "Al_copy": varechem["Al"]})
    result = nuee.rda(varespec, varechem[["N", "P", "K"]], Z=duplicated)

    assert _conditioning_rank(result) == 1


def test_residual_fallback_excludes_conditioned_inertia():
    """When unconstrained eigenvalues are unavailable, the residual is
    recovered from tot_chi; the conditioned share must not leak into it."""
    X, Y, Z = _make_data()
    result = nuee.rda(X, Y, Z=Z)
    expected_residual = result.unconstrained_inertia

    stripped = copy.deepcopy(result)
    stripped.unconstrained_eig = np.array([])

    assert _compute_rda_f(stripped)["residual"] == pytest.approx(
        expected_residual, rel=1e-10)


@pytest.mark.parametrize("scaling", [1, 2, 3])
def test_partial_rda_scores_are_finite(scaling):
    X, Y, Z = _make_data()
    result = nuee.rda(X, Y, Z=Z)
    sites = np.asarray(result.get_scores(display="sites", scaling=scaling))
    species = np.asarray(result.get_scores(display="species", scaling=scaling))

    assert np.all(np.isfinite(sites))
    assert np.all(np.isfinite(species))
    assert sites.shape[0] == X.shape[0]
