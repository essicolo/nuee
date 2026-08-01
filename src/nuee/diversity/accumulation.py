"""
Species accumulation curves.
"""

import numpy as np
import pandas as pd
from scipy.special import gammaln
from typing import Union, Optional


def _exact_richness(presence: np.ndarray) -> np.ndarray:
    """Analytic expected richness for every subset size (vegan's "exact").

    For a subset of ``n`` of the ``N`` sites, a species occurring in ``f`` of
    them is absent with probability ``C(N - f, n) / C(N, n)``, so the expected
    richness is ``sum_j (1 - C(N - f_j, n) / C(N, n))``. The binomial ratio is
    evaluated through log-gamma to stay stable for large site counts.
    """
    n_sites = presence.shape[0]
    freqs = presence.sum(axis=0).astype(float)
    richness = np.empty(n_sites, dtype=float)

    log_denom_const = gammaln(n_sites + 1)
    for idx in range(n_sites):
        n = idx + 1
        remaining = n_sites - freqs
        # A species present in more than N - n sites cannot be missed.
        feasible = remaining >= n
        log_p = (gammaln(remaining + 1)
                 - gammaln(np.where(feasible, remaining - n, 0) + 1)
                 - log_denom_const
                 + gammaln(n_sites - n + 1))
        p_absent = np.where(feasible, np.exp(log_p), 0.0)
        richness[idx] = float(np.sum(1.0 - p_absent))

    return richness


def specaccum(x: Union[np.ndarray, pd.DataFrame],
              method: str = "random",
              permutations: int = 100,
              random_state: Optional[int] = None) -> dict:
    """
    Calculate species accumulation curves.

    Parameters:
        x: Community data matrix
        method: Accumulation method. ``"random"`` averages the accumulation
            over random site orderings; ``"exact"`` returns the analytic
            expected richness for each subset size, which is deterministic.
        permutations: Number of permutations for the random method
        random_state: Seed for the random method, for reproducible curves

    Returns:
        Dictionary with accumulation results

    Raises:
        ValueError: if *method* is not one of the supported accumulators.
    """
    supported = ("random", "exact")
    if method not in supported:
        raise ValueError(
            f"method must be one of {supported}, got {method!r}."
        )

    if isinstance(x, pd.DataFrame):
        x_array = x.values
    else:
        x_array = np.asarray(x)

    n_sites = x_array.shape[0]
    sites = np.arange(1, n_sites + 1)
    presence = x_array > 0

    if method == "random":
        rng = np.random.default_rng(random_state)
        richness_curves = []

        for _ in range(permutations):
            # Random permutation of sites
            site_order = rng.permutation(n_sites)

            # Calculate cumulative richness
            cumulative_richness = []
            seen_species = np.zeros(x_array.shape[1], dtype=bool)
            
            for i in range(n_sites):
                site_idx = site_order[i]
                seen_species |= (x_array[site_idx] > 0)
                cumulative_richness.append(np.sum(seen_species))
            
            richness_curves.append(cumulative_richness)
        
        # Calculate mean and standard deviation
        richness_curves = np.array(richness_curves)
        mean_richness = np.mean(richness_curves, axis=0)
        sd_richness = np.std(richness_curves, axis=0)
        
        return {
            'sites': sites,
            'richness': mean_richness,
            'sd': sd_richness,
            'method': method,
            'permutations': permutations
        }
    
    return {
        'sites': sites,
        'richness': _exact_richness(presence),
        'method': method
    }


def _pool_estimators(presence: np.ndarray) -> tuple:
    """Incidence-based richness estimators for one pooled set of sites.

    Mirrors vegan's ``specpool``: observed richness plus the bias-corrected
    Chao, first- and second-order jackknife, and bootstrap extrapolations,
    all driven by the number of species found in exactly one (``a1``) or
    exactly two (``a2``) of the pooled sites.
    """
    n = presence.shape[0]
    freq = presence.sum(axis=0)
    observed = freq > 0
    n_obs = float(observed.sum())
    a1 = float((freq == 1).sum())
    a2 = float((freq == 2).sum())

    if n < 2:
        return n_obs, n_obs, n_obs, n_obs, n_obs

    if a2 > 0:
        chao = n_obs + a1 * a1 / (2.0 * a2) * (n - 1) / n
    else:
        chao = n_obs + a1 * (a1 - 1) / 2.0 * (n - 1) / n

    jack1 = n_obs + a1 * (n - 1) / n
    jack2 = n_obs + a1 * (2 * n - 3) / n - a2 * (n - 2) ** 2 / (n * (n - 1))

    proportions = freq[observed] / n
    boot = n_obs + float(np.sum((1.0 - proportions) ** n))

    return n_obs, chao, jack1, jack2, boot


def poolaccum(x: Union[np.ndarray, pd.DataFrame],
              permutations: int = 100,
              minsize: int = 3,
              random_state: Optional[int] = None) -> dict:
    """
    Accumulate extrapolated richness estimators over pooled sites.

    For every subset size from *minsize* to the number of sites, sites are
    pooled in a random order and the ``specpool`` estimators are computed;
    the curves are then averaged over *permutations* orderings. This mirrors
    vegan's ``poolaccum``.

    Parameters:
        x: Community data matrix
        permutations: Number of random site orderings to average over
        minsize: Smallest number of pooled sites to report (vegan uses 3)
        random_state: Seed, for reproducible curves

    Returns:
        Dictionary with the number of pooled sites (``sites``), the mean
        observed richness (``richness``), and the mean ``chao``, ``jack1``,
        ``jack2`` and ``boot`` estimates, plus a ``means`` DataFrame holding
        them together.

    Raises:
        ValueError: if *minsize* exceeds the number of sites.
    """
    if isinstance(x, pd.DataFrame):
        x_array = x.values
    else:
        x_array = np.asarray(x)

    presence = x_array > 0
    n_sites = presence.shape[0]
    if minsize > n_sites:
        raise ValueError(
            f"minsize ({minsize}) exceeds the number of sites ({n_sites})."
        )

    sizes = np.arange(minsize, n_sites + 1)
    rng = np.random.default_rng(random_state)
    accumulated = np.zeros((len(sizes), 5), dtype=float)

    for _ in range(permutations):
        order = rng.permutation(n_sites)
        for row, size in enumerate(sizes):
            accumulated[row] += _pool_estimators(presence[order[:size]])

    means = accumulated / permutations
    frame = pd.DataFrame(
        np.column_stack([sizes, means]),
        columns=["N", "S", "chao", "jack1", "jack2", "boot"],
    )

    return {
        'sites': sizes,
        'richness': means[:, 0],
        'chao': means[:, 1],
        'jack1': means[:, 2],
        'jack2': means[:, 3],
        'boot': means[:, 4],
        'means': frame,
        'permutations': permutations,
        'method': 'pooled'
    }