"""
Environmental variable fitting to ordination.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Union, Optional, Sequence, Dict, Any, Tuple

from .base import OrdinationResult


def _fit_vector(env_centered: np.ndarray,
                scores_centered: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit one environmental vector onto the ordination, as vegan does.

    The environmental variable is regressed on the ordination axes jointly.
    The arrow is the unit-length vector of regression coefficients, and the
    goodness of fit is the regression R-squared.

    Summing the squared correlations with each axis instead is only equal to
    the R-squared when the axes are orthogonal; ordination axes often are not,
    and the shortcut then understates the fit.
    """
    coefficients, *_ = np.linalg.lstsq(scores_centered, env_centered, rcond=None)
    fitted = scores_centered @ coefficients

    total_ss = float(np.sum(env_centered ** 2))
    r_squared = (0.0 if total_ss <= 0
                 else float(1.0 - np.sum((env_centered - fitted) ** 2) / total_ss))
    r_squared = float(np.clip(r_squared, 0.0, 1.0))

    norm = float(np.linalg.norm(coefficients))
    arrow = coefficients / norm if norm > 0 else np.zeros_like(coefficients)
    # Scale the unit arrow by the correlation, so its length shows the
    # strength of the fit. This is what vegan's scores(fit, "vectors") returns.
    return arrow * np.sqrt(r_squared), r_squared


def envfit(ordination: OrdinationResult,
           env: Union[np.ndarray, pd.DataFrame],
           permutations: int = 999,
           scaling: Optional[Union[int, str]] = 2,
           choices: Optional[Sequence[int]] = None,
           random_state: Optional[int] = None,
           **kwargs) -> Dict[str, Any]:
    """
    Fit environmental vectors to an ordination configuration.

    Parameters
    ----------
    ordination:
        OrdinationResult providing site scores.
    env:
        Environmental data matrix (matching site order).
    permutations:
        Number of permutations used for significance testing.
    scaling:
        Scaling option passed to `ordination.get_scores`.
    choices:
        Optional 1-based axis indices to include; defaults to all axes.
    random_state:
        Seed for the permutation generator.

    Returns
    -------
    dict
        Dictionary mirroring vegan's `envfit` output layout with a
        ``vectors`` entry containing scores, r, r², and p-values.
    """
    env_array, var_names = _prepare_environment_matrix(env)
    site_scores = _fetch_site_scores(ordination, scaling)
    if site_scores is None:
        raise ValueError("Ordination result does not provide site scores for envfit.")

    if choices is not None:
        axis_idx = np.asarray(list(choices), dtype=int) - 1
        site_scores = site_scores[:, axis_idx]
    site_scores = np.asarray(site_scores, dtype=float)

    if site_scores.ndim != 2:
        raise ValueError("Site scores must be a 2D array.")

    scores_centered = site_scores - np.mean(site_scores, axis=0, keepdims=True)
    axis_norms = np.linalg.norm(scores_centered, axis=0)
    axis_norms[axis_norms == 0] = 1.0

    rng = np.random.default_rng(random_state)

    arrow_scores = []
    vector_r = []
    vector_r2 = []
    p_values = []

    for col in range(env_array.shape[1]):
        env_var = env_array[:, col]
        env_centered = env_var - np.mean(env_var)
        env_norm = np.linalg.norm(env_centered)
        if env_norm == 0:
            correlations = np.zeros(site_scores.shape[1], dtype=float)
            r_squared = 0.0
            p_val = 1.0
        else:
            correlations, r_squared = _fit_vector(env_centered, scores_centered)
            p_val = _permutation_test(env_centered, scores_centered, axis_norms, env_norm,
                                      r_squared, permutations, rng)
        arrow_scores.append(correlations)
        vector_r.append(np.sqrt(r_squared))
        vector_r2.append(r_squared)
        p_values.append(p_val)

    vectors = {
        "scores": np.vstack(arrow_scores),
        "r": np.asarray(vector_r, dtype=float),
        "r2": np.asarray(vector_r2, dtype=float),
        "pvals": np.asarray(p_values, dtype=float),
        "nperm": int(permutations),
        "variables": var_names
    }


    return {"vectors": vectors, "call": {"scaling": scaling, "choices": choices}}


def _prepare_environment_matrix(env: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, Sequence[str]]:
    if isinstance(env, pd.DataFrame):
        return env.values.astype(float), list(env.columns)
    env_array = np.asarray(env, dtype=float)
    if env_array.ndim != 2:
        raise ValueError("Environmental matrix must be 2-dimensional.")
    var_names = [f"Var{i+1}" for i in range(env_array.shape[1])]
    return env_array, var_names


def _fetch_site_scores(ordination: OrdinationResult,
                       scaling: Optional[Union[int, str]]) -> Optional[np.ndarray]:
    try:
        scores = ordination.get_scores(display="sites", scaling=scaling)
        if scores is not None:
            return np.asarray(scores, dtype=float)
    except (AttributeError, ValueError):
        pass

    points = getattr(ordination, "points", None)
    if points is None:
        return None
    if isinstance(points, pd.DataFrame):
        return points.values.astype(float)
    return np.asarray(points, dtype=float)


def _permutation_test(env_centered: np.ndarray,
                      scores_centered: np.ndarray,
                      axis_norms: np.ndarray,
                      env_norm: float,
                      observed_r2: float,
                      permutations: int,
                      rng: np.random.Generator) -> float:
    if permutations <= 0 or env_norm == 0.0:
        return 1.0
    exceed = 0
    for _ in range(permutations):
        permuted = rng.permutation(env_centered)
        # Same statistic as the observed fit, otherwise the null distribution
        # is not comparable to the value being tested.
        _, perm_r2 = _fit_vector(permuted, scores_centered)
        if perm_r2 >= observed_r2 - 1e-12:
            exceed += 1
    return (exceed + 1) / (permutations + 1)
