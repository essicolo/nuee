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


LOGGER = logging.getLogger("nuee.diagnostics.vegan")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

REFERENCE_PATH = Path(__file__).resolve().parents[1] / "reference" / "vegan_reference.json"


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
    if colnames is not None:
        frame.columns = list(colnames)[: frame.shape[1]]
    if rownames is not None:
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
    diff = np.nanmax(np.abs(actual_aligned.to_numpy() - expected.to_numpy()))
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
    diff = np.nanmax(np.abs(actual_aligned.to_numpy() - expected.to_numpy()))
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
    assert_close_scalar("metaMDS.trymax", float(result.trymax), float(ref["trymax"]))

    ref_points = matrix_from_payload(ref["points"])
    actual_points = ensure_frame(result.points, rows=ref_points.index, cols=ref_points.columns)
    assert_close_frame("metaMDS.points", actual_points, ref_points, abs_tol=1e-6, rel_tol=1e-6)

    ref_species = matrix_from_payload(ref.get("species"))
    if ref_species is not None and getattr(result, "species", None) is not None:
        actual_species = ensure_frame(result.species, rows=ref_species.index, cols=ref_species.columns)
        assert_close_frame("metaMDS.species", actual_species, ref_species, abs_tol=1e-6, rel_tol=1e-6)


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
        assert_close_frame(f"rda.{key}", actual, expected, abs_tol=1e-6, rel_tol=1e-6)

    scores_ref = ref["scores"]
    for scaling_key in ("scaling1", "scaling2", "scaling3"):
        scaling_id = int(scaling_key[-1])
        ref_sites = matrix_from_payload(scores_ref[scaling_key]["sites"])
        ref_species = matrix_from_payload(scores_ref[scaling_key]["species"])

        sites = result.get_scores(display="sites", scaling=scaling_id)
        species_scores = result.get_scores(display="species", scaling=scaling_id)

        sites_df = ensure_frame(sites, rows=ref_sites.index, cols=ref_sites.columns)
        species_df = ensure_frame(species_scores, rows=ref_species.index, cols=ref_species.columns)

        assert_close_frame(f"rda.{scaling_key}.sites", sites_df, ref_sites, abs_tol=1e-6, rel_tol=1e-6)
        assert_close_frame(f"rda.{scaling_key}.species", species_df, ref_species, abs_tol=1e-6, rel_tol=1e-6)


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

    scores_ref = ref["scores"]
    for scaling_key in ("scaling1", "scaling2", "scaling3"):
        scaling_id = int(scaling_key[-1])
        ref_sites = matrix_from_payload(scores_ref[scaling_key]["sites"])
        ref_species = matrix_from_payload(scores_ref[scaling_key]["species"])

        sites = result.get_scores(display="sites", scaling=scaling_id)
        species_scores = result.get_scores(display="species", scaling=scaling_id)

        sites_df = ensure_frame(sites, rows=ref_sites.index, cols=ref_sites.columns)
        species_df = ensure_frame(species_scores, rows=ref_species.index, cols=ref_species.columns)

        assert_close_frame(f"cca.{scaling_key}.sites", sites_df, ref_sites, abs_tol=1e-6, rel_tol=1e-6)
        assert_close_frame(f"cca.{scaling_key}.species", species_df, ref_species, abs_tol=1e-6, rel_tol=1e-6)


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

    ref_scores = matrix_from_payload(ref["scores"])
    actual_scores = ensure_frame(vectors["scores"], rows=ref_scores.index, cols=ref_scores.columns)
    assert_close_frame("envfit.scores", actual_scores, ref_scores, abs_tol=1e-6, rel_tol=1e-6)

    ref_r = vector_from_payload(ref["r"])
    ref_r2 = vector_from_payload(ref["r2"])
    ref_p = vector_from_payload(ref["pvals"])

    assert_close_series("envfit.r", ensure_series(vectors["r"], ref_r.index), ref_r, abs_tol=1e-6, rel_tol=1e-6)
    assert_close_series("envfit.r2", ensure_series(vectors["r"] ** 2, ref_r2.index), ref_r2, abs_tol=1e-6, rel_tol=1e-6)
    assert_close_series("envfit.pvalues", ensure_series(vectors["pvals"], ref_p.index), ref_p, abs_tol=1e-6, rel_tol=1e-6)


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
        assert_close_scalar("procrustes.ss", float(result["ss"]), ref_ss, abs_tol=1e-6, rel_tol=1e-6)
    ref_rotation = matrix_from_payload(ref["rotation"])
    actual_rotation = ensure_frame(result["rotation"], rows=ref_rotation.index, cols=ref_rotation.columns)
    assert_close_frame("procrustes.rotation", actual_rotation, ref_rotation, abs_tol=1e-6, rel_tol=1e-6)


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

    renyi_ref = matrix_from_payload(ref["renyi"])
    renyi_actual = ensure_frame(nuee.renyi(species), rows=renyi_ref.index, cols=renyi_ref.columns)
    assert_close_frame("diversity.renyi", renyi_actual, renyi_ref, abs_tol=1e-6, rel_tol=1e-6)

    rarefy_ref = vector_from_payload(ref["rarefy"])
    rarefy_actual = ensure_series(nuee.rarefy(species_counts, sample=20), rarefy_ref.index)
    assert_close_series("diversity.rarefy", rarefy_actual, rarefy_ref, abs_tol=1e-6, rel_tol=1e-6)

    specnumber_ref = vector_from_payload(ref["specnumber"])
    specnumber_actual = ensure_series(nuee.specnumber(species), specnumber_ref.index)
    assert_close_series("diversity.specnumber", specnumber_actual, specnumber_ref, abs_tol=1e-6, rel_tol=1e-6)

    estimate_ref = matrix_from_payload(ref["estimateR"])
    estimate_actual = ensure_frame(nuee.estimateR(species_counts), rows=estimate_ref.index, cols=estimate_ref.columns)
    assert_close_frame("diversity.estimateR", estimate_actual, estimate_ref, abs_tol=1e-6, rel_tol=1e-6)

    specaccum_ref = ref["specaccum"]
    np.random.seed(123)
    specaccum_actual = nuee.specaccum(species, method="random", permutations=50)
    assert_close_series("diversity.specaccum.sites",
                        ensure_series(specaccum_actual["sites"]),
                        pd.Series(specaccum_ref["sites"]))
    assert_close_series("diversity.specaccum.richness",
                        ensure_series(specaccum_actual["richness"]),
                        pd.Series(specaccum_ref["richness"]),
                        abs_tol=1e-4, rel_tol=1e-4)
    assert_close_series("diversity.specaccum.sd",
                        ensure_series(specaccum_actual["sd"]),
                        pd.Series(specaccum_ref["sd"]),
                        abs_tol=1e-4, rel_tol=1e-4)

    poolaccum_ref = ref["poolaccum"]
    poolaccum_actual = nuee.poolaccum(species)
    assert_close_series("diversity.poolaccum.sites",
                        ensure_series(poolaccum_actual["sites"]),
                        pd.Series(poolaccum_ref["sites"]))
    assert_close_series("diversity.poolaccum.richness",
                        ensure_series(poolaccum_actual["richness"]),
                        pd.Series(poolaccum_ref["richness"]))


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
    result = nuee.anosim(distance, grouping, permutations=199)
    if not isinstance(result, dict):
        raise AssertionError(f"Expected dict from nuee.anosim, got {type(result).__name__}")
    if "r_statistic" not in result or "p_value" not in result:
        raise AssertionError(f"ANOSIM result missing keys: {sorted(result.keys())}")
    assert_close_scalar("anosim.statistic", float(result["r_statistic"]), float(ref["statistic"]), abs_tol=1e-6, rel_tol=1e-6)
    assert_close_scalar("anosim.p_value", float(result["p_value"]), float(ref["signif"]), abs_tol=1e-3, rel_tol=1e-3)
    if ref.get("permutations") is not None and "permutations" in result:
        assert_close_scalar("anosim.permutations", float(result["permutations"]), float(ref["permutations"]), abs_tol=0.0, rel_tol=0.0)


def compare_mrpp(reference: Dict[str, Any],
                 species: pd.DataFrame,
                 grouping: pd.Series) -> None:
    ref = reference["dissimilarity"]["mrpp"]
    np.random.seed(123)
    result = nuee.mrpp(species, grouping, permutations=199)
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
    assert_close_scalar("mrpp.p_value", float(result["p_value"]), float(ref["p_value"]), abs_tol=1e-3, rel_tol=1e-3)


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
    result = nuee.mantel(distance_species, distance_env, permutations=199)
    if not isinstance(result, dict):
        raise AssertionError(f"Expected dict from nuee.mantel, got {type(result).__name__}")
    assert_close_scalar("mantel.statistic", float(result["r_statistic"]), float(ref["statistic"]), abs_tol=1e-6, rel_tol=1e-6)
    assert_close_scalar("mantel.p_value", float(result["p_value"]), float(ref["signif"]), abs_tol=1e-3, rel_tol=1e-3)


def compare_mantel_partial(reference: Dict[str, Any],
                           distance_species: np.ndarray,
                           distance_env: np.ndarray,
                           distance_covariate: np.ndarray) -> None:
    ref = reference["dissimilarity"]["mantel_partial"]
    np.random.seed(123)
    result = nuee.mantel_partial(distance_species, distance_env, distance_covariate, permutations=199)
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
    assert_close_scalar("protest.p_value", float(result["p_value"]), float(ref_p), abs_tol=1e-3, rel_tol=1e-3)


def compare_permutest_rda(reference: Dict[str, Any]) -> None:
    raise AssertionError("permutest(rda) comparison not implemented in nuee.")


def compare_anova_rda(reference: Dict[str, Any],
                      rda_result: Any) -> None:
    ref = matrix_from_payload(reference["dissimilarity"]["anova_rda"])
    result = nuee.anova_cca(rda_result, permutations=199)
    if not isinstance(result, dict):
        raise AssertionError(f"Expected dict from nuee.anova_cca, got {type(result).__name__}")
    raise AssertionError(
        f"anova_cca returned keys {sorted(result.keys())}, expected table with columns {list(ref.columns)}."
    )

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
    run("permutest_rda", lambda: compare_permutest_rda(reference))
    run("anova_rda", lambda: compare_anova_rda(reference, rda_result))
    run("diversity", lambda: compare_diversity_metrics(reference, species))

    if failures:
        summary = "\n".join(failures)
        raise AssertionError(f"Vegan comparison failures detected:\n{summary}")

    LOGGER.info("All vegan reference comparisons passed.")


if __name__ == "__main__":
    main()
