"""
Comprehensive diagnostics that compare nuee's implementations against
reference outputs generated with vegan (see tests/reference/export_vegan.R).

Run this script manually with:

    PYTHONPATH=src python tests/diagnostics/compare_vegan_reference.py

The script logs per-function comparisons and aggregates any failures so the
differences are easy to triage while debugging nuee.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

import nuee
from nuee.permutation.permtest import _compute_rda_f


LOGGER = logging.getLogger("nuee.diagnostics.vegan")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

REFERENCE_PATH = Path(__file__).resolve().parents[1] / "reference" / "vegan_reference.json"

# nuee and vegan draw independent permutation streams, so their p-values agree
# only up to Monte Carlo error. With 199 permutations a p-value moves in steps
# of 1/200 = 0.005, so anything tighter than a couple of steps is testing the
# RNG rather than the statistic.
PERMUTATION_P_TOL = 0.011


# ---------------------------------------------------------------------------
# Helpers to decode the JSON payload
# ---------------------------------------------------------------------------

def matrix_from_payload(payload: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    if payload is None:
        return None
    frame = pd.DataFrame(payload["data"])
    frame = frame.apply(pd.to_numeric, errors="coerce")
    colnames = payload.get("colnames")
    rownames = payload.get("rownames")
    # An unnamed R matrix serialises its dimnames as [], which must be left
    # alone rather than assigned as an empty index.
    if colnames:
        frame.columns = list(colnames)[: frame.shape[1]]
    if rownames:
        frame.index = list(rownames)[: frame.shape[0]]
    return frame


def vector_from_payload(payload: Optional[Dict[str, Any]]) -> Optional[pd.Series]:
    if payload is None:
        return None
    series = pd.Series(payload["data"])
    series = pd.to_numeric(series, errors="coerce")
    names = payload.get("names")
    if names is not None and len(names) == series.shape[0]:
        series.index = list(names)
    return series


def dist_from_payload(payload: Dict[str, Any]) -> pd.DataFrame:
    condensed = np.asarray(payload["data"], dtype=float)
    matrix = squareform(condensed)
    labels = payload.get("labels")
    df = pd.DataFrame(matrix)
    if labels is not None and len(labels) == df.shape[0]:
        df.index = list(labels)
        df.columns = list(labels)
    return df


def ensure_frame(data: Any,
                 rows: Optional[Iterable[str]] = None,
                 cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if data is None:
        raise AssertionError("Expected data for DataFrame comparison, received None.")
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, dict):
        frame = pd.DataFrame.from_dict(data, orient="index")
    else:
        arr = np.asarray(data)
        if arr.ndim == 1:
            arr = arr[:, None]
        frame = pd.DataFrame(arr)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if rows is not None and len(list(rows)) == frame.shape[0]:
        frame.index = list(rows)
    if cols is not None and len(list(cols)) == frame.shape[1]:
        frame.columns = list(cols)
    return frame


def ensure_series(data: Any, index: Optional[Iterable[str]] = None) -> pd.Series:
    if data is None:
        raise AssertionError("Expected data for Series comparison, received None.")
    if isinstance(data, pd.Series):
        series = data.copy()
    elif isinstance(data, dict):
        series = pd.Series(data, dtype=float)
    else:
        series = pd.Series(np.asarray(data))
    series = pd.to_numeric(series, errors="coerce")
    if index is not None and len(list(index)) == series.shape[0]:
        series.index = list(index)
    return series


def scalar_from_payload(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("value", "data"):
            if key in value:
                payload = value[key]
                if isinstance(payload, list):
                    return float(payload[0]) if payload else None
                if payload is not None:
                    return float(payload)
        return None
    if isinstance(value, (list, tuple)):
        return float(value[0]) if value else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def _tolerance(expected: np.ndarray, abs_tol: float, rel_tol: float) -> float:
    scale = np.nanmax(np.abs(expected))
    if math.isnan(scale) or scale == 0.0:
        return abs_tol
    return max(abs_tol, rel_tol * scale)


def align_signs(actual: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """Flip each column of *actual* to the sign convention of *expected*.

    The sign of a singular vector is arbitrary -- it depends on the LAPACK
    build, so the same code gives opposite axes on different machines. Any
    comparison of eigenvectors, ordination axes or the rotation between two
    configurations has to align first or it is testing the linear algebra
    backend rather than the statistic.
    """
    # Reindex first: the caller may hold every axis while the reference keeps
    # only the first few, and the flip has to be decided per aligned column.
    aligned = actual.reindex(index=expected.index, columns=expected.columns)
    actual_values = aligned.to_numpy()
    expected_values = expected.to_numpy()
    agreement = np.nansum(actual_values * expected_values, axis=0)
    flips = np.where(agreement < 0, -1.0, 1.0)
    return pd.DataFrame(actual_values * flips,
                        index=aligned.index, columns=aligned.columns)


def align_signs_both_ways(actual: pd.DataFrame,
                          expected: pd.DataFrame) -> pd.DataFrame:
    """Align a matrix that is defined up to row *and* column sign flips.

    A Procrustes rotation maps one configuration onto another, so it inherits
    the arbitrary axis signs of both: flipping an axis of the source turns R
    into ``D @ R`` and flipping one of the target into ``R @ D'``. Flipping
    columns alone cannot undo the first, and a single flipped axis turns the
    rotation into a reflection -- which is what made this comparison pass on
    one machine and fail on another.

    The search is exhaustive, which is exact and cheap for the small matrices
    involved here.
    """
    aligned = actual.reindex(index=expected.index, columns=expected.columns)
    actual_values = aligned.to_numpy()
    expected_values = expected.to_numpy()
    n_rows, n_cols = actual_values.shape

    if n_rows + n_cols > 12:
        raise AssertionError(
            f"align_signs_both_ways: {n_rows}x{n_cols} is too large for an "
            "exhaustive sign search."
        )

    best, best_error = actual_values, np.inf
    for row_bits in range(2 ** n_rows):
        row_signs = np.array([1.0 if row_bits >> i & 1 else -1.0
                              for i in range(n_rows)])
        for col_bits in range(2 ** n_cols):
            col_signs = np.array([1.0 if col_bits >> j & 1 else -1.0
                                  for j in range(n_cols)])
            candidate = actual_values * row_signs[:, None] * col_signs[None, :]
            error = np.nanmax(np.abs(candidate - expected_values))
            if error < best_error:
                best, best_error = candidate, error

    return pd.DataFrame(best, index=aligned.index, columns=aligned.columns)


def _reject_empty_comparison(name: str, difference: np.ndarray) -> None:
    """Refuse a comparison in which nothing lines up.

    Reindexing onto labels that do not match yields an all-NaN difference;
    np.nanmax then returns NaN and `NaN > tol` is False, so the assertion
    passes while testing nothing. Two comparisons sat in that state.
    """
    if difference.size == 0 or np.all(np.isnan(difference)):
        raise AssertionError(
            f"{name}: nothing to compare -- actual and expected share no "
            "aligned values. Check the labels and shapes on both sides."
        )


def assert_close_scalar(name: str,
                        actual: float,
                        expected: float,
                        abs_tol: float = 1e-8,
                        rel_tol: float = 1e-6) -> None:
    if math.isnan(expected):
        if not math.isnan(actual):
            raise AssertionError(f"{name}: expected NaN, received {actual}")
        return
    diff = abs(actual - expected)
    tol = max(abs_tol, rel_tol * max(abs(actual), abs(expected)))
    LOGGER.debug("%s difference %.6g (tolerance %.6g)", name, diff, tol)
    if diff > tol:
        raise AssertionError(f"{name}: difference {diff:.6g} exceeds tolerance {tol:.6g}")


def assert_close_series(name: str,
                        actual: pd.Series,
                        expected: pd.Series,
                        abs_tol: float = 1e-8,
                        rel_tol: float = 1e-6) -> None:
    actual_aligned = actual.reindex(expected.index)
    residual = np.abs(actual_aligned.to_numpy() - expected.to_numpy())
    _reject_empty_comparison(name, residual)
    diff = np.nanmax(residual)
    tol = _tolerance(expected.to_numpy(), abs_tol, rel_tol)
    LOGGER.debug("%s max abs diff %.6g (tolerance %.6g)", name, diff, tol)
    if diff > tol:
        raise AssertionError(f"{name}: max abs diff {diff:.6g} exceeds tolerance {tol:.6g}")


def assert_close_frame(name: str,
                       actual: pd.DataFrame,
                       expected: pd.DataFrame,
                       abs_tol: float = 1e-8,
                       rel_tol: float = 1e-6) -> None:
    actual_aligned = actual.reindex(index=expected.index, columns=expected.columns)
    residual = np.abs(actual_aligned.to_numpy() - expected.to_numpy())
    _reject_empty_comparison(name, residual)
    diff = np.nanmax(residual)
    tol = _tolerance(expected.to_numpy(), abs_tol, rel_tol)
    LOGGER.debug("%s max abs diff %.6g (tolerance %.6g)", name, diff, tol)
    if diff > tol:
        raise AssertionError(f"{name}: max abs diff {diff:.6g} exceeds tolerance {tol:.6g}")


# ---------------------------------------------------------------------------
# Comparison routines
# ---------------------------------------------------------------------------

def compare_meta_mds(reference: Dict[str, Any],
                     species: pd.DataFrame) -> None:
    ref = reference["ordination"]["metaMDS"]
    result = nuee.metaMDS(species, k=2, distance="bray", random_state=123)

    assert_close_scalar("metaMDS.stress", float(result.stress), float(ref["stress"]))
    # No assertion on the number of tries: it echoes an input parameter rather
    # than a fitted quantity, and vegan exposes it as $tries, not $trymax.

    # An NMDS configuration is only defined up to translation, rotation,
    # reflection and (because vegan rescales to half-change units) a global
    # scale factor. Comparing coordinates elementwise would test those
    # conventions rather than the fit, so align the two configurations first
    # and require the residual to vanish.
    ref_points = matrix_from_payload(ref["points"])
    actual_points = ensure_frame(result.points, rows=ref_points.index, cols=ref_points.columns)
    fit = nuee.procrustes(ref_points.to_numpy(), actual_points.to_numpy())
    residual = float(np.asarray(fit["ss"]).ravel()[0])
    assert_close_scalar("metaMDS.points.procrustes_residual", residual, 0.0,
                        abs_tol=1e-6, rel_tol=0.0)


def compare_rda(reference: Dict[str, Any],
                species: pd.DataFrame,
                env: pd.DataFrame) -> None:
    ref = reference["ordination"]["rda"]
    result = nuee.rda(species, formula="~ N + P + K + Ca + Mg", data=env)

    expected_constrained = pd.Series(ref["eigenvalues"]["constrained"])
    expected_unconstrained = pd.Series(ref["eigenvalues"]["unconstrained"])
    actual_constrained = pd.Series(result.constrained_eig)
    actual_unconstrained = pd.Series(result.unconstrained_eig)

    assert_close_series("rda.constrained_eigenvalues", actual_constrained, expected_constrained)
    assert_close_series("rda.unconstrained_eigenvalues", actual_unconstrained, expected_unconstrained)

    bases = ref["bases"]
    for key in ("cca_wa", "ca_u", "cca_v", "ca_v"):
        expected = matrix_from_payload(bases[key])
        actual = ensure_frame(getattr(result, key), rows=expected.index, cols=expected.columns)
        assert_close_frame(f"rda.{key}", align_signs(actual, expected), expected,
                           abs_tol=1e-6, rel_tol=1e-6)

    scores_ref = ref["scores"]
    for scaling_key in ("scaling1", "scaling2", "scaling3"):
        scaling_id = int(scaling_key[-1])
        ref_sites = matrix_from_payload(scores_ref[scaling_key]["sites"])
        ref_species = matrix_from_payload(scores_ref[scaling_key]["species"])

        sites = result.get_scores(display="sites", scaling=scaling_id)
        species_scores = result.get_scores(display="species", scaling=scaling_id)

        sites_df = ensure_frame(sites, rows=ref_sites.index, cols=ref_sites.columns)
        species_df = ensure_frame(species_scores, rows=ref_species.index, cols=ref_species.columns)

        assert_close_frame(f"rda.{scaling_key}.sites", align_signs(sites_df, ref_sites), ref_sites, abs_tol=1e-6, rel_tol=1e-6)
        assert_close_frame(f"rda.{scaling_key}.species", align_signs(species_df, ref_species), ref_species, abs_tol=1e-6, rel_tol=1e-6)


def compare_rda_partial(reference: Dict[str, Any],
                        species: pd.DataFrame,
                        env: pd.DataFrame) -> None:
    """Validate the partial RDA path against vegan's Condition() model.

    Covers what the non-partial comparison cannot: the inertia partition
    (total / conditioned / constrained / unconstrained) and the residual
    degrees of freedom, which must exclude the rank of the conditioning
    matrix.
    """
    ref = reference["ordination"]["rda_partial"]
    constraints = list(ref["constraints"])
    conditions = list(ref["conditions"])
    result = nuee.rda(species, env[constraints], Z=env[conditions])

    assert_close_series("rda_partial.constrained_eigenvalues",
                        pd.Series(result.constrained_eig),
                        pd.Series(ref["eigenvalues"]["constrained"]),
                        abs_tol=1e-6, rel_tol=1e-6)
    assert_close_series("rda_partial.unconstrained_eigenvalues",
                        pd.Series(result.unconstrained_eig),
                        pd.Series(ref["eigenvalues"]["unconstrained"]),
                        abs_tol=1e-6, rel_tol=1e-6)

    inertia = ref["inertia"]
    assert_close_scalar("rda_partial.total_inertia",
                        float(result.tot_chi), float(inertia["total"]),
                        abs_tol=1e-6, rel_tol=1e-6)
    assert_close_scalar("rda_partial.conditioned_inertia",
                        float(result.conditioned_inertia), float(inertia["conditioned"]),
                        abs_tol=1e-6, rel_tol=1e-6)
    assert_close_scalar("rda_partial.constrained_inertia",
                        float(result.constrained_inertia), float(inertia["constrained"]),
                        abs_tol=1e-6, rel_tol=1e-6)
    assert_close_scalar("rda_partial.unconstrained_inertia",
                        float(result.unconstrained_inertia), float(inertia["unconstrained"]),
                        abs_tol=1e-6, rel_tol=1e-6)

    table = result.inertia_partition()
    assert_close_scalar("rda_partial.inertia_partition.conditioned",
                        float(table.loc["Conditioned", "Inertia"]),
                        float(inertia["conditioned"]), abs_tol=1e-6, rel_tol=1e-6)

    anova = matrix_from_payload(ref["anova"])
    stats = _compute_rda_f(result)
    assert_close_scalar("rda_partial.df_model",
                        float(stats["df_model"]), float(anova.loc["Model", "Df"]),
                        abs_tol=0.0, rel_tol=0.0)
    assert_close_scalar("rda_partial.df_residual",
                        float(stats["df_resid"]), float(anova.loc["Residual", "Df"]),
                        abs_tol=0.0, rel_tol=0.0)
    assert_close_scalar("rda_partial.f_statistic",
                        float(stats["f_value"]), float(anova.loc["Model", "F"]),
                        abs_tol=1e-6, rel_tol=1e-6)

    scores_ref = ref["scores"]
    for scaling_key in ("scaling1", "scaling2"):
        scaling_id = int(scaling_key[-1])
        ref_sites = matrix_from_payload(scores_ref[scaling_key]["sites"])
        sites = result.get_scores(display="sites", scaling=scaling_id)
        sites_df = ensure_frame(sites, rows=ref_sites.index, cols=ref_sites.columns)
        assert_close_frame(f"rda_partial.{scaling_key}.sites", align_signs(sites_df, ref_sites), ref_sites,
                           abs_tol=1e-6, rel_tol=1e-6)


def compare_ca(reference: Dict[str, Any],
               species: pd.DataFrame) -> None:
    """Unconstrained correspondence analysis, i.e. vegan's cca() with no
    constraints."""
    ref = reference["ordination"]["ca"]
    result = nuee.ca(species)

    assert_close_series("ca.eigenvalues",
                        pd.Series(np.asarray(result.eigenvalues)),
                        pd.Series(ref["eigenvalues"]),
                        abs_tol=1e-8, rel_tol=1e-8)
    assert_close_scalar("ca.tot_chi", float(result.tot_chi), float(ref["tot_chi"]),
                        abs_tol=1e-8, rel_tol=1e-8)

    # Eigenvector signs are arbitrary, so align each axis before comparing.
    for key, attribute in (("u", "site_u"), ("v", "species_v")):
        expected = matrix_from_payload(ref["bases"][key])
        actual = getattr(result, attribute, None)
        if actual is None or expected is None:
            continue
        actual = ensure_frame(actual, rows=expected.index,
                              cols=expected.columns).to_numpy()
        signs = np.where((actual * expected.to_numpy()).sum(axis=0) < 0, -1.0, 1.0)
        assert_close_frame(f"ca.{key}",
                           pd.DataFrame(actual * signs, index=expected.index,
                                        columns=expected.columns),
                           expected, abs_tol=1e-6, rel_tol=1e-6)


def compare_ordistep(reference: Dict[str, Any],
                     species: pd.DataFrame,
                     env_full: pd.DataFrame) -> None:
    """Stepwise selection must agree with vegan on which terms are eligible.

    Exact set equality is not testable on this dataset: N sits at p ~= 0.05
    once P is in the model, so the outcome turns on the permutation draw.
    Re-running vegan across six seeds yields N, P, P, nothing, nothing and
    N,P in turn. What is stable, and what is asserted here, is that K
    (p ~= 0.42) never enters and that nothing outside vegan's own candidate
    set does either.
    """
    ref = reference["ordination"]["ordistep"]
    candidates = env_full[["N", "P", "K"]]
    full_model = nuee.rda(species, candidates)
    result = nuee.ordistep(full_model, candidates, direction="both",
                           random_state=123)

    selected = result.get("selected_variables")
    if selected is None:
        raise AssertionError(
            f"ordistep returned no 'selected_variables'; keys: {sorted(result)}")

    selected = set(map(str, selected))
    if "K" in selected:
        raise AssertionError(
            "ordistep.selected: K entered the model despite p ~= 0.42")
    if not selected <= {"N", "P"}:
        raise AssertionError(
            f"ordistep.selected: {sorted(selected)} contains terms outside "
            f"vegan's eligible set {sorted(map(str, ref['selected']))}")
    # A pure placeholder returning [] would pass the checks above, so require
    # that selection actually happened.
    if not selected:
        raise AssertionError(
            "ordistep.selected: nothing was retained, yet P is significant "
            "at p ~= 0.05 on its own")


def compare_cca(reference: Dict[str, Any],
                species: pd.DataFrame,
                env: pd.DataFrame) -> None:
    ref = reference["ordination"]["cca"]
    result = nuee.cca(species, X=env, formula="~ N + P + K + Ca + Mg")

    expected_constrained = pd.Series(ref["eigenvalues"]["constrained"])
    expected_unconstrained = pd.Series(ref["eigenvalues"]["unconstrained"])
    actual_constrained = pd.Series(result.constrained_eig)
    actual_unconstrained = pd.Series(result.unconstrained_eig)

    assert_close_series("cca.constrained_eigenvalues", actual_constrained, expected_constrained, abs_tol=1e-6, rel_tol=1e-6)
    assert_close_series("cca.unconstrained_eigenvalues", actual_unconstrained, expected_unconstrained, abs_tol=1e-6, rel_tol=1e-6)

    bases = ref["bases"]
    for key in ("cca_u", "cca_v", "cca_wa", "ca_u", "ca_v"):
        expected = matrix_from_payload(bases[key])
        actual = ensure_frame(getattr(result, key), rows=expected.index,
                              cols=expected.columns)
        assert_close_frame(f"cca.{key}", align_signs(actual, expected), expected,
                           abs_tol=1e-6, rel_tol=1e-6)

    scores_ref = ref["scores"]
    for scaling_key in ("scaling1", "scaling2", "scaling3"):
        scaling_id = int(scaling_key[-1])
        ref_sites = matrix_from_payload(scores_ref[scaling_key]["sites"])
        ref_species = matrix_from_payload(scores_ref[scaling_key]["species"])

        sites = result.get_scores(display="sites", scaling=scaling_id)
        species_scores = result.get_scores(display="species", scaling=scaling_id)

        sites_df = ensure_frame(sites, rows=ref_sites.index, cols=ref_sites.columns)
        species_df = ensure_frame(species_scores, rows=ref_species.index, cols=ref_species.columns)

        assert_close_frame(f"cca.{scaling_key}.sites",
                           align_signs(sites_df, ref_sites), ref_sites,
                           abs_tol=1e-6, rel_tol=1e-6)
        assert_close_frame(f"cca.{scaling_key}.species",
                           align_signs(species_df, ref_species), ref_species,
                           abs_tol=1e-6, rel_tol=1e-6)


def compare_pca(reference: Dict[str, Any],
                species: pd.DataFrame) -> None:
    ref = reference["ordination"]["pca"]
    result = nuee.pca(species, scale=True, center=True)

    expected_sdev = pd.Series(ref["sdev"], index=ref["rotation"]["colnames"])
    actual_sdev = pd.Series(result.singular_values,
                            index=ref["rotation"]["colnames"][:len(result.singular_values)])
    assert_close_series("pca.singular_values", actual_sdev, expected_sdev.iloc[:len(actual_sdev)], abs_tol=1e-6, rel_tol=1e-6)

    rotation_ref = matrix_from_payload(ref["rotation"])
    rotation_actual = ensure_frame(result.species, rows=rotation_ref.index, cols=rotation_ref.columns)

    nonzero_mask = expected_sdev.values > 1e-8
    rotation_ref_nz = rotation_ref.loc[:, nonzero_mask]
    rotation_actual_nz = rotation_actual.loc[:, nonzero_mask]

    assert_close_frame("pca.rotation", rotation_actual_nz, rotation_ref_nz, abs_tol=1e-6, rel_tol=1e-6)

    scores_ref = matrix_from_payload(ref["x"])
    scores_actual = ensure_frame(result.points, rows=scores_ref.index, cols=scores_ref.columns)
    assert_close_frame("pca.scores", scores_actual, scores_ref, abs_tol=1e-6, rel_tol=1e-6)

    assert_close_series("pca.center", pd.Series(result.column_means), pd.Series(ref["center"]))
    assert_close_series("pca.scale", pd.Series(result.column_scale), pd.Series(ref["scale"]))


def compare_envfit(reference: Dict[str, Any],
                   meta_result: Any,
                   env: pd.DataFrame) -> None:
    ref = reference["ordination"]["envfit"]
    result = nuee.envfit(meta_result, env, permutations=199, random_state=123)
    vectors = result["vectors"]

    # r2 and the p-values are invariant to how the ordination axes are
    # oriented; the arrow coordinates are not, so they are compared per axis
    # up to a sign flip.
    ref_r2 = vector_from_payload(ref["r2"])
    ref_p = vector_from_payload(ref["pvals"])

    assert_close_series("envfit.r2", ensure_series(vectors["r2"], ref_r2.index),
                        ref_r2, abs_tol=1e-4, rel_tol=1e-4)
    # These p-values sit in the 0.02-0.17 range, where 199 permutations give a
    # Monte Carlo standard deviation near 0.025. Allow three of those; anything
    # tighter fails on the permutation draw rather than on the statistic.
    assert_close_series("envfit.pvalues", ensure_series(vectors["pvals"], ref_p.index),
                        ref_p, abs_tol=0.08, rel_tol=0.0)

    ref_scores = matrix_from_payload(ref["scores"])
    actual_scores = ensure_frame(vectors["scores"], rows=ref_scores.index,
                                 cols=ref_scores.columns).to_numpy()
    expected_scores = ref_scores.to_numpy()
    aligned = actual_scores * np.where(
        (actual_scores * expected_scores).sum(axis=0) < 0, -1.0, 1.0)
    assert_close_frame("envfit.scores",
                       pd.DataFrame(aligned, index=ref_scores.index,
                                    columns=ref_scores.columns),
                       ref_scores, abs_tol=1e-4, rel_tol=1e-4)


def compare_procrustes(reference: Dict[str, Any],
                       rda_result: Any,
                       meta_result: Any) -> None:
    ref = reference["ordination"]["procrustes"]
    rda_scores = rda_result.get_scores(display="sites", scaling=1)
    sites_df = ensure_frame(rda_scores)
    n_axes = meta_result.points.shape[1]
    sites_df = sites_df.iloc[:, :n_axes]
    sites_df.index = meta_result.points.index
    sites_df.columns = meta_result.points.columns
    np.random.seed(123)
    result = nuee.procrustes(sites_df.values, meta_result.points.values)

    ref_corr = scalar_from_payload(ref.get("correlation"))
    if ref_corr is not None:
        assert_close_scalar("procrustes.correlation", float(result["correlation"]), ref_corr, abs_tol=1e-6, rel_tol=1e-6)
    ref_ss = scalar_from_payload(ref.get("ss"))
    if ref_ss is not None and "ss" in result:
        # nuee and vegan land on the same NMDS configuration only to within the
        # optimiser's own convergence, which leaves a residual near 2e-6 here.
        assert_close_scalar("procrustes.ss", float(result["ss"]), ref_ss, abs_tol=1e-5, rel_tol=1e-5)
    # The rotation maps the RDA configuration onto the NMDS one, so it
    # inherits the arbitrary axis signs of both -- rows from the source,
    # columns from the target.
    ref_rotation = matrix_from_payload(ref["rotation"])
    actual_rotation = ensure_frame(result["rotation"], rows=ref_rotation.index, cols=ref_rotation.columns)
    assert_close_frame("procrustes.rotation",
                       align_signs_both_ways(actual_rotation, ref_rotation),
                       ref_rotation, abs_tol=1e-4, rel_tol=1e-4)


def compare_vegdist(reference: Dict[str, Any],
                    species: pd.DataFrame) -> None:
    ref_matrix = dist_from_payload(reference["dissimilarity"]["vegdist_bray"])
    actual_matrix = nuee.vegdist(species, method="bray")
    actual_df = ensure_frame(actual_matrix, rows=ref_matrix.index, cols=ref_matrix.columns)
    assert_close_frame("vegdist.bray", actual_df, ref_matrix, abs_tol=1e-6, rel_tol=1e-6)


def compare_diversity_metrics(reference: Dict[str, Any],
                              species: pd.DataFrame) -> None:
    ref = reference["diversity"]
    species_counts = species.round().astype(int)

    shannon_ref = vector_from_payload(ref["shannon"])
    shannon_actual = ensure_series(nuee.shannon(species), shannon_ref.index)
    assert_close_series("diversity.shannon", shannon_actual, shannon_ref, abs_tol=1e-6, rel_tol=1e-6)

    simpson_ref = vector_from_payload(ref["simpson"])
    simpson_actual = ensure_series(nuee.simpson(species), simpson_ref.index)
    assert_close_series("diversity.simpson", simpson_actual, simpson_ref, abs_tol=1e-6, rel_tol=1e-6)

    evenness_ref = vector_from_payload(ref["evenness_shannon"])
    evenness_actual = ensure_series(nuee.evenness(species), evenness_ref.index)
    assert_close_series("diversity.evenness", evenness_actual, evenness_ref, abs_tol=1e-6, rel_tol=1e-6)

    fisher_ref = vector_from_payload(ref["fisher_alpha"])
    fisher_actual = ensure_series(nuee.fisher_alpha(species_counts), fisher_ref.index)
    assert_close_series("diversity.fisher_alpha", fisher_actual, fisher_ref, abs_tol=1e-6, rel_tol=1e-6)

    # The reference was generated for scales c(0, 0.5, 1, 2, 3, 4); calling
    # nuee.renyi() on its defaults compared a different set of scales
    # entirely, which is why nothing lined up.
    renyi_ref = matrix_from_payload(ref["renyi"])
    renyi_scales = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
    renyi_actual = ensure_frame(nuee.renyi(species, scales=renyi_scales),
                                rows=renyi_ref.index, cols=renyi_ref.columns)
    assert_close_frame("diversity.renyi", renyi_actual, renyi_ref, abs_tol=1e-6, rel_tol=1e-6)

    rarefy_ref = vector_from_payload(ref["rarefy"])
    rarefy_actual = ensure_series(nuee.rarefy(species_counts, sample=20), rarefy_ref.index)
    assert_close_series("diversity.rarefy", rarefy_actual, rarefy_ref, abs_tol=1e-6, rel_tol=1e-6)

    specnumber_ref = vector_from_payload(ref["specnumber"])
    specnumber_actual = ensure_series(nuee.specnumber(species), specnumber_ref.index)
    assert_close_series("diversity.specnumber", specnumber_actual, specnumber_ref, abs_tol=1e-6, rel_tol=1e-6)

    # vegan lays estimateR out as statistics x sites; nuee as sites x
    # statistics, and the two report overlapping but not identical sets (nuee
    # adds f1/f2, vegan adds the standard errors). Transpose, then compare the
    # statistics both provide.
    estimate_ref = matrix_from_payload(ref["estimateR"]).T
    estimate_actual = nuee.estimateR(species_counts)
    shared = [name for name in estimate_ref.columns if name in estimate_actual.columns]
    if not shared:
        raise AssertionError(
            f"estimateR shares no statistic with vegan: nuee has "
            f"{list(estimate_actual.columns)}, vegan has {list(estimate_ref.columns)}")
    estimate_actual = ensure_frame(estimate_actual[shared].to_numpy(),
                                   rows=estimate_ref.index, cols=shared)
    assert_close_frame("diversity.estimateR", estimate_actual,
                       estimate_ref[shared], abs_tol=1e-6, rel_tol=1e-6)

    # The analytic curve is deterministic, so it can be pinned down exactly.
    specaccum_exact_ref = ref["specaccum_exact"]
    specaccum_exact = nuee.specaccum(species, method="exact")
    assert_close_series("diversity.specaccum_exact.sites",
                        ensure_series(specaccum_exact["sites"]),
                        pd.Series(specaccum_exact_ref["sites"]))
    assert_close_series("diversity.specaccum_exact.richness",
                        ensure_series(specaccum_exact["richness"]),
                        pd.Series(specaccum_exact_ref["richness"]),
                        abs_tol=1e-8, rel_tol=1e-8)

    # method="random" averages 50 random orderings, and nuee and vegan draw
    # different ones, so only the shape of the curve can be compared. A whole
    # species either side of vegan's mean is well inside that sampling noise.
    specaccum_ref = ref["specaccum"]
    specaccum_actual = nuee.specaccum(species, method="random",
                                      permutations=50, random_state=123)
    assert_close_series("diversity.specaccum.sites",
                        ensure_series(specaccum_actual["sites"]),
                        pd.Series(specaccum_ref["sites"]))
    assert_close_series("diversity.specaccum.richness",
                        ensure_series(specaccum_actual["richness"]),
                        pd.Series(specaccum_ref["richness"]),
                        abs_tol=1.5, rel_tol=0.0)

    # Rarefaction curves: one analytic curve per site, compared over the
    # sample sizes vegan was asked for.
    rarecurve_ref = ref["rarecurve"]
    ref_sizes = np.asarray(rarecurve_ref["sample_sizes"], dtype=int)
    ref_expected = matrix_from_payload(rarecurve_ref["expected"]).to_numpy()
    rarecurve_actual = nuee.rarecurve(species_counts)
    site_keys = list(rarecurve_actual)
    for row, key in enumerate(site_keys):
        curve = rarecurve_actual[key]
        sizes = np.asarray(curve["sample_sizes"], dtype=int)
        richness = np.asarray(curve["richness"], dtype=float)
        lookup = dict(zip(sizes.tolist(), richness.tolist()))
        actual = np.array([lookup.get(int(size), np.nan) for size in ref_sizes])
        assert_close_series(f"diversity.rarecurve.{key}",
                            pd.Series(actual), pd.Series(ref_expected[row]),
                            abs_tol=1e-6, rel_tol=1e-6)

    # vegan's reference curve averages only 50 random orderings. Re-running it
    # across six seeds moves the estimators by up to 0.72 (S), 2.52 (Chao),
    # 1.20 (jack1), 1.99 (jack2) and 0.96 (bootstrap), so those spreads are the
    # floor for any comparison here. nuee averages more orderings to sit near
    # the middle of that band.
    poolaccum_ref = matrix_from_payload(ref["poolaccum"]["means"])
    poolaccum_actual = nuee.poolaccum(species, permutations=2000, random_state=123)
    column_tolerances = {
        "N": 0.0,
        "S": 1.0,
        "Chao": 3.0,
        "Jackknife 1": 1.5,
        "Jackknife 2": 2.5,
        "Bootstrap": 1.25,
    }
    actual_columns = {
        "N": poolaccum_actual["sites"],
        "S": poolaccum_actual["richness"],
        "Chao": poolaccum_actual["chao"],
        "Jackknife 1": poolaccum_actual["jack1"],
        "Jackknife 2": poolaccum_actual["jack2"],
        "Bootstrap": poolaccum_actual["boot"],
    }
    for column, tolerance in column_tolerances.items():
        assert_close_series(f"diversity.poolaccum.{column}",
                            ensure_series(actual_columns[column]),
                            pd.Series(poolaccum_ref[column].to_numpy()),
                            abs_tol=max(tolerance, 1e-9), rel_tol=0.0)


def compare_adonis2(reference: Dict[str, Any],
                    species: pd.DataFrame,
                    env: pd.DataFrame) -> None:
    ref_table = matrix_from_payload(reference["dissimilarity"]["adonis2"])
    result = nuee.adonis2(species, env, permutations=199)
    if isinstance(result, dict):
        raise AssertionError(
            f"Expected PERMANOVA table with columns {list(ref_table.columns)}, "
            f"received dict with keys {sorted(result.keys())}."
        )
    actual_table = ensure_frame(result, rows=ref_table.index, cols=ref_table.columns)
    assert_close_frame("adonis2.table", actual_table, ref_table, abs_tol=1e-6, rel_tol=1e-6)


def compare_anosim(reference: Dict[str, Any],
                   distance: np.ndarray,
                   grouping: pd.Series) -> None:
    ref = reference["dissimilarity"]["anosim"]
    np.random.seed(123)
    result = nuee.anosim(distance, grouping, permutations=199, random_state=123)
    if not isinstance(result, dict):
        raise AssertionError(f"Expected dict from nuee.anosim, got {type(result).__name__}")
    if "r_statistic" not in result or "p_value" not in result:
        raise AssertionError(f"ANOSIM result missing keys: {sorted(result.keys())}")
    assert_close_scalar("anosim.statistic", float(result["r_statistic"]), float(ref["statistic"]), abs_tol=1e-6, rel_tol=1e-6)
    assert_close_scalar("anosim.p_value", float(result["p_value"]), float(ref["signif"]), abs_tol=PERMUTATION_P_TOL, rel_tol=0.0)
    if ref.get("permutations") is not None and "permutations" in result:
        assert_close_scalar("anosim.permutations", float(result["permutations"]), float(ref["permutations"]), abs_tol=0.0, rel_tol=0.0)


def compare_mrpp(reference: Dict[str, Any],
                 species: pd.DataFrame,
                 grouping: pd.Series) -> None:
    ref = reference["dissimilarity"]["mrpp"]
    np.random.seed(123)
    result = nuee.mrpp(species, grouping, permutations=199, random_state=123)
    if not isinstance(result, dict):
        raise AssertionError(f"Expected dict from nuee.mrpp, got {type(result).__name__}")
    assert_close_scalar("mrpp.delta", float(result.get("delta")), float(ref["delta"]), abs_tol=1e-6, rel_tol=1e-6)
    expected_delta = result.get("expected_delta")
    if expected_delta is None:
        raise AssertionError("MRPP result missing expected_delta field.")
    assert_close_scalar("mrpp.expected_delta", float(expected_delta), float(ref["e_delta"]), abs_tol=1e-6, rel_tol=1e-6)
    if "a_statistic" in result:
        ref_a = 1.0 - float(ref["delta"]) / float(ref["e_delta"]) if ref["e_delta"] else 0.0
        assert_close_scalar("mrpp.a_statistic", float(result["a_statistic"]), ref_a, abs_tol=1e-6, rel_tol=1e-6)
    assert_close_scalar("mrpp.p_value", float(result["p_value"]), float(ref["p_value"]), abs_tol=PERMUTATION_P_TOL, rel_tol=0.0)


def compare_betadisper(reference: Dict[str, Any],
                       distance: np.ndarray,
                       grouping: pd.Series) -> None:
    ref = reference["dissimilarity"]["betadisper"]
    result = nuee.betadisper(distance, grouping)
    if not isinstance(result, dict):
        raise AssertionError(f"Expected dict from nuee.betadisper, got {type(result).__name__}")

    ref_distances = vector_from_payload(ref["distances"])
    actual_distances = ensure_series(result.get("distances"), ref_distances.index)
    assert_close_series("betadisper.distances", actual_distances, ref_distances, abs_tol=1e-6, rel_tol=1e-6)

    ref_centroids = matrix_from_payload(ref["centroids"])
    if "centroids" in result:
        centroids_df = ensure_frame(result["centroids"], rows=ref_centroids.index, cols=ref_centroids.columns)
        for col in ref_centroids.columns:
            diff = (centroids_df[col] - ref_centroids[col]).abs().max()
            diff_flip = (-centroids_df[col] - ref_centroids[col]).abs().max()
            if diff_flip < diff:
                centroids_df[col] *= -1
        assert_close_frame("betadisper.centroids", centroids_df, ref_centroids, abs_tol=1e-6, rel_tol=1e-6)
    else:
        raise AssertionError("betadisper result missing centroids.")


def compare_permutest_betadisper(reference: Dict[str, Any]) -> None:
    ref = reference["dissimilarity"]["permutest_betadisper"]
    permutations = ref["permutations"]
    if permutations:
        raise AssertionError("permutest(betadisper) not yet implemented in nuee, so permutation table is unavailable.")


def compare_mantel(reference: Dict[str, Any],
                   distance_species: np.ndarray,
                   distance_env: np.ndarray) -> None:
    ref = reference["dissimilarity"]["mantel"]
    np.random.seed(123)
    result = nuee.mantel(distance_species, distance_env, permutations=199, random_state=123)
    if not isinstance(result, dict):
        raise AssertionError(f"Expected dict from nuee.mantel, got {type(result).__name__}")
    assert_close_scalar("mantel.statistic", float(result["r_statistic"]), float(ref["statistic"]), abs_tol=1e-6, rel_tol=1e-6)
    assert_close_scalar("mantel.p_value", float(result["p_value"]), float(ref["signif"]), abs_tol=PERMUTATION_P_TOL, rel_tol=0.0)


def compare_mantel_partial(reference: Dict[str, Any],
                           distance_species: np.ndarray,
                           distance_env: np.ndarray,
                           distance_covariate: np.ndarray) -> None:
    ref = reference["dissimilarity"]["mantel_partial"]
    np.random.seed(123)
    result = nuee.mantel_partial(distance_species, distance_env, distance_covariate, permutations=199, random_state=123)
    if not isinstance(result, dict):
        raise AssertionError(f"Expected dict from nuee.mantel_partial, got {type(result).__name__}")
    assert_close_scalar("mantel_partial.statistic",
                        float(result["r_statistic"]),
                        float(ref["statistic"]),
                        abs_tol=1e-6,
                        rel_tol=1e-6)
    assert_close_scalar("mantel_partial.p_value",
                        float(result["p_value"]),
                        float(ref["signif"]),
                        abs_tol=1e-3,
                        rel_tol=1e-3)


def compare_protest(reference: Dict[str, Any],
                    x: np.ndarray,
                    y: np.ndarray) -> None:
    ref = reference["dissimilarity"]["protest"]
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    result = nuee.protest(x_arr, y_arr, permutations=199, random_state=123)
    if not isinstance(result, dict):
        raise AssertionError(f"Expected dict from nuee.protest, got {type(result).__name__}")
    ref_corr_payload = ref.get("correlation", ref.get("statistic"))
    ref_corr = scalar_from_payload(ref_corr_payload)
    if ref_corr is None:
        raise AssertionError("Protest reference statistic unavailable.")
    ref_p_payload = ref.get("signif")
    ref_p = scalar_from_payload(ref_p_payload)
    if ref_p is None:
        raise AssertionError("Protest reference p-value unavailable.")
    assert_close_scalar("protest.correlation", float(result["correlation"]), float(ref_corr), abs_tol=1e-6, rel_tol=1e-6)
    assert_close_scalar("protest.p_value", float(result["p_value"]), float(ref_p), abs_tol=PERMUTATION_P_TOL, rel_tol=0.0)


def compare_permutest_rda(reference: Dict[str, Any],
                          rda_result: Any) -> None:
    ref = reference["dissimilarity"]["permutest_rda"]
    ref_table = matrix_from_payload(ref.get("tab"))
    if ref_table is None:
        raise AssertionError("Vegan reference missing permutest(rda) table; regenerate export_vegan.R.")
    ref_permutations = ref.get("permutations", 0) or 0
    np.random.seed(123)
    result = nuee.permutest(rda_result, permutations=199, random_state=123)
    if not isinstance(result, dict):
        raise AssertionError(f"Expected dict from nuee.permutest, got {type(result).__name__}")
    result_table = ensure_frame(result.get("tab"), rows=ref_table.index if ref_table is not None else None,
                                cols=ref_table.columns if ref_table is not None else None)
    if ref_table is not None:
        assert_close_frame("permutest(rda).table", result_table, ref_table, abs_tol=1e-6, rel_tol=1e-6)
    if ref_permutations:
        if int(result.get("permutations", 0)) != int(ref_permutations):
            raise AssertionError(f"permutest(rda).permutations mismatch: {result.get('permutations')} vs {ref_permutations}")


def compare_anova_rda(reference: Dict[str, Any],
                      rda_result: Any) -> None:
    ref = matrix_from_payload(reference["dissimilarity"]["anova_rda"])
    result = nuee.anova_cca(rda_result, permutations=199)
    if not isinstance(result, dict):
        raise AssertionError(f"Expected dict from nuee.anova_cca, got {type(result).__name__}")
    result_table = result.get("tab")
    if result_table is None:
        raise AssertionError("anova_cca did not return an ANOVA table.")
    result_frame = ensure_frame(result_table, rows=ref.index, cols=ref.columns)
    assert_close_frame("anova_cca.table", result_frame, ref, abs_tol=1e-6, rel_tol=1e-6)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_reference() -> Dict[str, Any]:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"Reference file not found: {REFERENCE_PATH}")
    with REFERENCE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    reference = load_reference()
    species = nuee.datasets.varespec()
    env_full = nuee.datasets.varechem()

    env_columns = reference["datasets"]["env_vars"]
    env = env_full[env_columns]

    np.random.seed(123)
    meta = nuee.metaMDS(species, k=2, distance="bray", random_state=123)
    rda_result = nuee.rda(species, formula="~ N + P + K + Ca + Mg", data=env)
    distance_species = nuee.vegdist(species, method="bray")
    distance_env = nuee.vegdist(env, method="bray")
    distance_covariate = nuee.vegdist(env.iloc[:, 1:], method="bray")
    grouping = pd.Series(
        np.where(env_full["Humdepth"] > env_full["Humdepth"].median(), "High", "Low"),
        index=env_full.index,
        name="grouping"
    )
    rda_sites_full = ensure_frame(rda_result.get_scores(display="sites", scaling=1))
    rda_sites_for_procrustes = rda_sites_full.iloc[:, :meta.points.shape[1]]
    rda_sites_for_procrustes.index = meta.points.index
    rda_sites_for_procrustes.columns = meta.points.columns

    protest_meta_points = matrix_from_payload(reference["ordination"]["metaMDS"]["points"])
    protest_rda_sites = matrix_from_payload(reference["ordination"]["rda"]["scores"]["scaling1"]["sites"])
    if protest_rda_sites is not None and protest_meta_points is not None:
        protest_rda_sites = protest_rda_sites.iloc[:, :protest_meta_points.shape[1]]

    failures: List[str] = []

    def run(label: str, func: Callable[[], None]) -> None:
        try:
            func()
            LOGGER.info("%s: OK", label)
        except AssertionError as exc:
            LOGGER.error("%s: %s", label, exc)
            failures.append(f"{label}: {exc}")
        except Exception as exc:  # pragma: no cover - unexpected failures
            LOGGER.exception("%s: unexpected error", label)
            failures.append(f"{label}: unexpected error {exc}")

    run("metaMDS", lambda: compare_meta_mds(reference, species))
    run("rda", lambda: compare_rda(reference, species, env))
    # env is subset to env_vars; the partial model conditions on Al + pH,
    # which live outside that subset.
    run("rda_partial", lambda: compare_rda_partial(reference, species, env_full))
    run("ca", lambda: compare_ca(reference, species))
    run("ordistep", lambda: compare_ordistep(reference, species, env_full))
    run("cca", lambda: compare_cca(reference, species, env))
    run("pca", lambda: compare_pca(reference, species))
    run("envfit", lambda: compare_envfit(reference, meta, env))
    run("procrustes", lambda: compare_procrustes(reference, rda_result, meta))
    run("vegdist", lambda: compare_vegdist(reference, species))
    run("adonis2", lambda: compare_adonis2(reference, species, env))
    run("anosim", lambda: compare_anosim(reference, distance_species, grouping))
    run("mrpp", lambda: compare_mrpp(reference, species, grouping))
    run("betadisper", lambda: compare_betadisper(reference, distance_species, grouping))
    run("permutest_betadisper", lambda: compare_permutest_betadisper(reference))
    run("mantel", lambda: compare_mantel(reference, distance_species, distance_env))
    run("mantel_partial", lambda: compare_mantel_partial(reference, distance_species, distance_env, distance_covariate))
    run("protest", lambda: compare_protest(reference,
                                           protest_meta_points.to_numpy() if protest_meta_points is not None else meta.points.values,
                                           protest_rda_sites.to_numpy() if protest_rda_sites is not None else rda_sites_for_procrustes.values))
    run("permutest_rda", lambda: compare_permutest_rda(reference, rda_result))
    run("anova_rda", lambda: compare_anova_rda(reference, rda_result))
    run("diversity", lambda: compare_diversity_metrics(reference, species))

    if failures:
        summary = "\n".join(failures)
        raise AssertionError(f"Vegan comparison failures detected:\n{summary}")

    LOGGER.info("All vegan reference comparisons passed.")


if __name__ == "__main__":
    main()
