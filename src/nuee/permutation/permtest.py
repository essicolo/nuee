"""
General permutation tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Dict, Optional, Union

from ..ordination.base import ConstrainedOrdinationResult
from ..ordination.rda import RDA
from ..dissimilarity.mantel import _normalize_rng, _draw_permutation


def permtest(statistic_func: Callable,
             data: np.ndarray,
             permutations: int = 999,
             **kwargs) -> dict:
    """
    General permutation test.
    
    Parameters:
        statistic_func: Function to calculate test statistic
        data: Data array
        permutations: Number of permutations
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with test results
    """
    observed = statistic_func(data, **kwargs)
    if permutations <= 0:
        return {
            "statistic": observed,
            "p_value": np.nan,
            "permutations": 0,
            "permuted_stats": np.array([], dtype=float)
        }

    permuted_stats = np.empty(permutations, dtype=float)
    for idx in range(permutations):
        permuted_data = np.random.permutation(data)
        permuted_stats[idx] = statistic_func(permuted_data, **kwargs)

    p_value = (np.sum(permuted_stats >= observed) + 1) / (permutations + 1)

    return {
        "statistic": observed,
        "p_value": p_value,
        "permutations": permutations,
        "permuted_stats": permuted_stats
    }


def _compute_rda_f(ordination_result: ConstrainedOrdinationResult) -> Dict[str, float]:
    constrained = float(np.sum(ordination_result.constrained_eig)) if ordination_result.constrained_eig is not None else 0.0
    residual = float(np.sum(ordination_result.unconstrained_eig)) if ordination_result.unconstrained_eig is not None else 0.0
    if residual <= 0.0 and ordination_result.tot_chi is not None:
        residual = max(float(ordination_result.tot_chi) - constrained, 0.0)

    df_model = int(ordination_result.rank)
    n_samples = int(ordination_result.call.get("n_samples", ordination_result.points.shape[0] if ordination_result.points is not None else 0))
    df_resid = max(n_samples - 1 - df_model, 1)

    ms_model = constrained / max(df_model, 1)
    ms_resid = residual / df_resid if df_resid > 0 else np.nan
    f_value = ms_model / ms_resid if ms_resid and ms_resid > 0 else np.inf

    return {
        "constrained": constrained,
        "residual": residual,
        "df_model": df_model,
        "df_resid": df_resid,
        "f_value": f_value
    }


def permutest(ordination_result: ConstrainedOrdinationResult,
              permutations: int = 999,
              *,
              random_state: Optional[Union[int, np.random.Generator]] = None) -> Dict[str, Union[pd.DataFrame, int, float, np.ndarray]]:
    """
    Permutation test for constrained ordination (RDA/CCA).

    Parameters
    ----------
    ordination_result:
        Constrained ordination result (e.g., from :func:`nuee.rda`).
    permutations:
        Number of permutations used to build the null distribution.
    random_state:
        Optional random seed or Generator for reproducible results.

    Returns
    -------
    dict
        Dictionary containing the ANOVA-style table and permutation details.
    """
    if not isinstance(ordination_result, ConstrainedOrdinationResult):
        raise TypeError("permutest currently supports constrained ordination results (e.g., RDA/CCA).")

    if ordination_result.rank == 0:
        raise ValueError("Permutation test requires at least one constrained axis.")

    raw_response = getattr(ordination_result, "_raw_response", None)
    raw_env = getattr(ordination_result, "_raw_constraints", None)
    if raw_response is None or raw_env is None:
        raise ValueError("Raw response and constraint data are unavailable; re-fit the model with nuee>=0.1.2.")

    raw_conditioning = getattr(ordination_result, "_raw_conditioning", None)
    response_is_df = getattr(ordination_result, "_response_is_dataframe", False)
    response_columns = getattr(ordination_result, "_response_columns", None)
    response_index = getattr(ordination_result, "_response_index", None)
    env_is_df = getattr(ordination_result, "_constraints_is_dataframe", False)
    conditioning_is_df = getattr(ordination_result, "_conditioning_is_dataframe", False)
    perm_spec = getattr(ordination_result, "_permutation_spec", {})

    formula = perm_spec.get("formula")
    scale = perm_spec.get("scale", False)
    center = perm_spec.get("center", True)

    n_samples = raw_response.shape[0]
    stats_obs = _compute_rda_f(ordination_result)

    rng = _normalize_rng(random_state)
    permuted_f = np.empty(permutations, dtype=float) if permutations and permutations > 0 else np.array([], dtype=float)
    exceed = 0

    if permutations and permutations > 0:
        estimator = RDA(formula=formula, scale=scale, center=center)
        for idx in range(permutations):
            perm = _draw_permutation(n_samples, rng)
            X_perm = raw_response[perm, :]
            if response_is_df and response_columns is not None:
                X_input = pd.DataFrame(X_perm, columns=response_columns)
                if response_index is not None:
                    X_input.index = [response_index[i] for i in perm]
            else:
                X_input = X_perm

            if env_is_df:
                env_input = raw_env.copy(deep=True)
            else:
                env_input = np.array(raw_env, copy=True)

            if raw_conditioning is not None:
                if conditioning_is_df:
                    conditioning_input = raw_conditioning.copy(deep=True)
                else:
                    conditioning_input = np.array(raw_conditioning, copy=True)
                perm_result = estimator.fit(X_input, env_input, conditioning_input)
            else:
                perm_result = estimator.fit(X_input, env_input)

            stats_perm = _compute_rda_f(perm_result)
            permuted_f[idx] = stats_perm["f_value"]
            if stats_perm["f_value"] >= stats_obs["f_value"] - 1e-12:
                exceed += 1

        p_value = (exceed + 1) / (permutations + 1)
    else:
        p_value = np.nan

    table = pd.DataFrame(
        {
            "Df": [stats_obs["df_model"], stats_obs["df_resid"]],
            "Variance": [stats_obs["constrained"], stats_obs["residual"]],
            "F": [stats_obs["f_value"], np.nan],
            "Pr(>F)": [p_value, np.nan],
        },
        index=["Model", "Residual"]
    )

    return {
        "tab": table,
        "permutations": permutations,
        "f_observed": stats_obs["f_value"],
        "f_permutations": permuted_f
    }
