"""
Non-metric Multidimensional Scaling (NMDS) implementation.

This module provides the metaMDS function, which is a wrapper around
scikit-learn's MDS with additional functionality for community ecology.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Callable, Dict, Any, List, Tuple
from sklearn.manifold import smacof
from scipy.spatial.distance import squareform
import warnings


def _cmdscale_init(dist_matrix: np.ndarray, n_components: int) -> Optional[np.ndarray]:
    n_samples = dist_matrix.shape[0]
    if n_samples == 0:
        return None
    J = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
    B = -0.5 * J @ (dist_matrix ** 2) @ J
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    positive = evals > 0
    if not np.any(positive):
        return None
    evals = np.sqrt(evals[positive])
    evecs = evecs[:, positive]
    k = min(n_components, evecs.shape[1])
    return evecs[:, :k] * evals[:k]

from .base import OrdinationResult, OrdinationMethod
from ..dissimilarity import vegdist


class NMDS(OrdinationMethod):
    """
    Non-metric Multidimensional Scaling for community ecology.

    This implementation follows the approach used in vegan's metaMDS function,
    including multiple random starts and stress evaluation. NMDS is a rank-based
    ordination method that attempts to represent ecological distances in a
    reduced dimensional space.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions for the embedding (ordination axes).
    max_iter : int, default=200
        Maximum number of iterations for the optimization algorithm.
    n_init : int, default=20
        Number of random initializations. The solution with minimum stress
        is returned.
    eps : float, default=1e-3
        Convergence tolerance for the stress value.
    random_state : int, optional
        Random seed for reproducibility. If None, the random state is not fixed.
    dissimilarity : str, default="bray"
        Distance metric to use. See :func:`nuee.vegdist` for available options.
    n_jobs : int, optional
        Number of parallel jobs for computation. If None, uses a single core.

    Attributes
    ----------
    n_components : int
        Number of dimensions.
    max_iter : int
        Maximum iterations.
    n_init : int
        Number of random starts.
    eps : float
        Convergence tolerance.

    See Also
    --------
    metaMDS : High-level interface with automatic transformations
    MDS : Scikit-learn's MDS implementation

    Notes
    -----
    NMDS is particularly useful when:
    - Relationships between samples are non-linear
    - You want to use a specific distance metric
    - You have presence/absence or abundance data

    The stress value indicates goodness-of-fit:
    - Values < 0.05 indicate excellent fit
    - Values 0.05-0.10 indicate good fit
    - Values > 0.20 indicate poor fit

    References
    ----------
    .. [1] Kruskal, J.B. (1964). Nonmetric multidimensional scaling.
           Psychometrika 29, 115-129.

    Examples
    --------
    >>> from nuee.ordination import NMDS
    >>> import nuee
    >>> species = nuee.datasets.varespec()
    >>> nmds = NMDS(n_components=2, n_init=20)
    >>> result = nmds.fit(species)
    >>> print(f"Stress: {result.stress:.3f}")
    """

    def __init__(self, n_components: int = 2, max_iter: int = 200,
                 n_init: int = 20, eps: float = 1e-3, random_state: Optional[int] = None,
                 dissimilarity: str = "bray", n_jobs: Optional[int] = None):
        """Initialize NMDS with specified parameters."""
        self.n_components = n_components
        self.max_iter = max_iter
        self.n_init = n_init
        self.eps = eps
        self.random_state = random_state
        self.dissimilarity = dissimilarity
        self.n_jobs = n_jobs
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            distance: Optional[np.ndarray] = None,
            **kwargs) -> OrdinationResult:
        """
        Fit NMDS to the data.
        
        Parameters:
            X: Community data matrix (samples x species) or distance matrix
            distance: Precomputed distance matrix (optional)
            **kwargs: Additional parameters
            
        Returns:
            OrdinationResult with NMDS coordinates and stress
        """
        X = self._validate_data(X)
        
        # Calculate or use provided distance matrix
        if distance is not None:
            if isinstance(distance, pd.DataFrame):
                distance = distance.values
            dist_matrix = np.asarray(distance)
        else:
            # Calculate distance matrix using vegdist
            dist_matrix = vegdist(X, method=self.dissimilarity)
            
        # Ensure distance matrix is symmetric
        if dist_matrix.shape[0] != dist_matrix.shape[1]:
            # If it's a condensed distance matrix, convert to square
            if dist_matrix.shape[0] == X.shape[0] * (X.shape[0] - 1) // 2:
                dist_matrix = squareform(dist_matrix)
            else:
                raise ValueError("Distance matrix dimensions are incompatible")
                
        # Perform NMDS using SMACOF, keeping track of each random start
        best_points = None
        best_stress = np.inf
        best_n_iter = None
        best_run = None
        best_seed = None
        stress_history = []
        tol = 1e-9

        rng = np.random.default_rng(self.random_state)

        stress_scale = dist_matrix.shape[0] * max(dist_matrix.shape[0] - 1, 1)
        best_raw_stress = None

        cmdscale_init = _cmdscale_init(dist_matrix, self.n_components)

        for run_idx in range(self.n_init):
            if self.random_state is None:
                seed = None
            else:
                seed = int(rng.integers(0, 2**32 - 1))

            init = None
            if run_idx == 0 and cmdscale_init is not None:
                init = cmdscale_init

            embedding, raw_stress, n_iter = smacof(
                dist_matrix,
                metric=False,
                n_components=self.n_components,
                init=init,
                n_init=1,
                max_iter=self.max_iter,
                verbose=0,
                eps=self.eps,
                random_state=seed,
                return_n_iter=True,
                n_jobs=self.n_jobs,
                normalized_stress=False,
            )

            raw_stress = float(raw_stress)
            stress = float(np.sqrt(raw_stress / stress_scale)) if stress_scale > 0 else 0.0
            stress_history.append({
                'run': run_idx + 1,
                'stress': stress,
                'raw_stress': raw_stress,
                'n_iter': n_iter,
                'random_state': seed,
            })

            if stress + tol < best_stress:
                best_points = embedding
                best_stress = stress
                best_n_iter = n_iter
                best_run = run_idx + 1
                best_seed = seed
                best_raw_stress = raw_stress

        best_repeats = sum(
            1 for entry in stress_history
            if abs(entry['stress'] - best_stress) <= tol
        )
        
        call_info = {
            'method': 'NMDS',
            'dissimilarity': self.dissimilarity,
            'n_components': self.n_components,
            'n_init': self.n_init,
            'max_iter': self.max_iter,
            'best_run': best_run,
            'best_seed': best_seed,
        }
        
        result = OrdinationResult(
            points=best_points,
            stress=best_stress,
            converged=best_n_iter is not None and best_n_iter < self.max_iter,
            call=call_info
        )

        result.stress_history = stress_history
        result.best_run = best_run
        result.best_n_iter = best_n_iter
        result.best_repeats = best_repeats
        result.n_init = self.n_init
        result.distance_method = self.dissimilarity
        result.raw_stress = best_raw_stress if best_stress is not None else None

        return result


def metaMDS(X: Union[np.ndarray, pd.DataFrame],
            k: int = 2,
            distance: str = "bray",
            trymax: int = 20,
            maxit: int = 200,
            trace: bool = False,
            autotransform: bool = True,
            wascores: bool = True,
            expand: bool = True,
            random_state: Optional[int] = None,
            **kwargs) -> OrdinationResult:
    """
    Non-metric Multidimensional Scaling with automatic transformation.

    This function provides a high-level interface for NMDS ordination, following
    the conventions of the R vegan package's metaMDS function. It automatically
    handles data transformation and uses multiple random starts to find the best
    ordination solution.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Community data matrix with samples in rows and species in columns.
        Values should be non-negative abundances or counts.
    k : int, default=2
        Number of dimensions for the ordination. Common choices are 2 or 3.
    distance : str, default="bray"
        Distance metric to use for calculating dissimilarities.
        See :func:`nuee.vegdist` for available options.
    trymax : int, default=20
        Maximum number of random starts to find the best solution.
        Higher values increase computation time but may find better solutions.
    maxit : int, default=200
        Maximum number of iterations for each random start.
    trace : bool, default=False
        If True, print progress information including stress values.
    autotransform : bool, default=True
        If True, automatically apply square root transformation to abundance data
        (values > 1) to reduce the influence of dominant species.
    wascores : bool, default=True
        If True, calculate weighted average species scores based on site scores
        and species abundances.
    expand : bool, default=True
        If True, expand the result to include additional information.
    random_state : int, optional
        Seed used for reproducible random starts. If ``None``, each run is
        initialised independently.
    **kwargs : dict
        Additional parameters passed to the NMDS class.

    Returns
    -------
    OrdinationResult
        Result object containing:

        - points : pd.DataFrame
            Site (sample) scores in the ordination space
        - species : pd.DataFrame, optional
            Species scores (if wascores=True)
        - stress : float
            Final stress value (lower is better)
        - converged : bool
            Whether the solution converged

    Notes
    -----
    NMDS is a rank-based ordination method that attempts to preserve the rank
    order of dissimilarities between samples. Unlike metric methods like PCA,
    NMDS makes no assumptions about the linearity of relationships.

    Stress values provide a measure of fit:
    - < 0.05: excellent
    - 0.05 - 0.10: good
    - 0.10 - 0.20: acceptable
    - > 0.20: poor (consider increasing k or using a different method)

    The function uses multiple random starts (trymax) because NMDS can get stuck
    in local optima. The solution with the lowest stress is returned.

    Examples
    --------
    Basic NMDS ordination:

    >>> import nuee
    >>> species = nuee.datasets.varespec()
    >>> nmds = nuee.metaMDS(species, k=2, distance="bray")
    >>> print(f"Stress: {nmds.stress:.3f}")

    With custom parameters:

    >>> nmds = nuee.metaMDS(species, k=3, distance="bray", trymax=50, trace=True)

    Visualize the ordination:

    >>> import matplotlib.pyplot as plt
    >>> fig = nuee.plot_ordination(nmds, display="sites")
    >>> plt.show()

    See Also
    --------
    NMDS : Lower-level NMDS class
    rda : Redundancy Analysis (constrained ordination)
    cca : Canonical Correspondence Analysis
    pca : Principal Component Analysis

    References
    ----------
    .. [1] Kruskal, J.B. (1964). Nonmetric multidimensional scaling: a numerical
           method. Psychometrika 29, 115-129.
    .. [2] Minchin, P.R. (1987). An evaluation of relative robustness of techniques
           for ecological ordinations. Vegetatio 69, 89-107.
    .. [3] Oksanen, J., et al. (2020). vegan: Community Ecology Package.
           https://CRAN.R-project.org/package=vegan
    """
    warnings.warn(
        "nuee.metaMDS currently relies on a SMACOF implementation that may "
        "produce slightly different stress values than vegan::metaMDS. The "
        "ordination is still valid, but expect mild numerical differences.",
        UserWarning,
        stacklevel=2,
    )
    # Validate input
    if isinstance(X, pd.DataFrame):
        row_names = X.index.tolist()
        col_names = X.columns.tolist()
        X_array = X.values
    else:
        X_array = np.asarray(X)
        row_names = [f"Site{i+1}" for i in range(X_array.shape[0])]
        col_names = [f"Species{i+1}" for i in range(X_array.shape[1])]
    
    # Data transformation
    transform = kwargs.pop('transform', None)
    transformations_applied: List[str] = []

    if autotransform:
        if np.max(X_array) > 1:
            if trace:
                print("Applying square root transformation")
            X_array = np.sqrt(X_array)
            transformations_applied.append('sqrt')
        if trace:
            print("Applying Wisconsin double standardization")
        X_array = _wisconsin_transform(X_array)
        transformations_applied.append('wisconsin')
    elif transform:
        if isinstance(transform, (list, tuple, set)):
            transforms = transform
        else:
            transforms = [transform]
        for name in transforms:
            if name == 'sqrt':
                if trace:
                    print("Applying square root transformation")
                X_array = np.sqrt(X_array)
                transformations_applied.append('sqrt')
            elif name == 'wisconsin':
                if trace:
                    print("Applying Wisconsin double standardization")
                X_array = _wisconsin_transform(X_array)
                transformations_applied.append('wisconsin')
            else:
                raise ValueError(f"Unknown transform '{name}'")
    
    # Perform NMDS
    nmds = NMDS(n_components=k,
                max_iter=maxit,
                n_init=trymax,
                dissimilarity=distance,
                random_state=random_state,
                **kwargs)
    
    result = nmds.fit(X_array)
    
    # Calculate species scores if requested
    if wascores and X_array.shape[1] > 0:
        scores, shrinkage, centre = _calculate_species_scores(
            X_array, result.points, expand=expand
        )
        result.species = scores
        if shrinkage is not None:
            result.species_shrinkage = shrinkage
        if centre is not None:
            result.species_centre = centre
    
    # Add row and column names
    if hasattr(result, 'points'):
        result.points = pd.DataFrame(result.points, 
                                   index=row_names,
                                   columns=[f"NMDS{i+1}" for i in range(k)])
    
    if hasattr(result, 'species') and result.species is not None:
        result.species = pd.DataFrame(result.species,
                                    index=col_names,
                                    columns=[f"NMDS{i+1}" for i in range(k)])
        if hasattr(result, "species_shrinkage"):
            result.species.attrs["shrinkage"] = result.species_shrinkage
            delattr(result, "species_shrinkage")
        if hasattr(result, "species_centre"):
            result.species.attrs["centre"] = result.species_centre
            delattr(result, "species_centre")
    
    if trace:
        print(f"NMDS stress: {result.stress:.4f}")
        if result.converged:
            print("NMDS converged")
        else:
            print("NMDS did not converge")

    # Attach metadata similar to vegan's output
    result.transformations = transformations_applied
    result.distance_method = distance
    result.trymax = trymax
    result.maxit = maxit
    result.random_state = random_state
    result.stress_per_run = [entry['stress'] for entry in getattr(result, 'stress_history', [])]
    result.raw_stress_per_run = [entry.get('raw_stress') for entry in getattr(result, 'stress_history', [])]
    result.best_stress = result.stress
    result.best_run_repeats = getattr(result, 'best_repeats', 1)

    if result.call is not None:
        result.call.update({
            'transformations': transformations_applied,
            'trymax': trymax,
            'maxit': maxit,
            'distance': distance,
            'random_state': random_state,
        })

    return result


def _wisconsin_transform(X: np.ndarray) -> np.ndarray:
    """
    Apply Wisconsin double standardization.
    
    First standardizes by species maxima, then by site totals.
    """
    # Standardize by species maxima
    species_max = np.max(X, axis=0)
    species_max[species_max == 0] = 1  # Avoid division by zero
    X_std = X / species_max
    
    # Standardize by site totals
    site_totals = np.sum(X_std, axis=1)
    site_totals[site_totals == 0] = 1  # Avoid division by zero
    X_std = X_std / site_totals[:, np.newaxis]
    
    return X_std


def _calculate_species_scores(X: np.ndarray,
                              points: np.ndarray,
                              expand: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Calculate weighted average species scores using vegan's wascores logic.

    Parameters
    ----------
    X : ndarray
        Community matrix (sites x species) in the same order as `points`.
    points : ndarray
        Site scores from the NMDS ordination.
    expand : bool, optional
        Apply variance inflation correction to match vegan::wascores(expand=TRUE).

    Returns
    -------
    scores : ndarray
        Weighted average species scores.
    shrinkage : ndarray or None
        Shrinkage factors applied during expansion (1 / multiplier^2).
    centre : ndarray or None
        Weighted centroid used for expansion.
    """
    X = np.asarray(X, dtype=float)
    points = np.asarray(points, dtype=float)
    species_totals = np.sum(X, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        weights = np.divide(X, species_totals[np.newaxis, :],
                            out=np.zeros_like(X, dtype=float),
                            where=species_totals[np.newaxis, :] > 0)
    species_scores = weights.T @ points

    shrinkage = None
    centre = None
    if expand and species_scores.size:
        valid = species_totals > 0
        x_weights = np.sum(X, axis=1)
        x_cov, x_center = _weighted_covariance(points, x_weights)

        if np.any(valid):
            ewa = species_scores[valid, :]
            ewa_weights = species_totals[valid]
            wa_cov, wa_center = _weighted_covariance(ewa, ewa_weights)

            if wa_cov is not None:
                diag_wa = np.diag(wa_cov)
                diag_x = np.diag(x_cov) if x_cov is not None else np.zeros_like(diag_wa)
                with np.errstate(divide="ignore", invalid="ignore"):
                    mul = np.sqrt(np.divide(diag_x, diag_wa,
                                            out=np.ones_like(diag_wa),
                                            where=diag_wa > 0))
                ewa_centered = ewa - wa_center
                ewa = ewa_centered * mul + wa_center
                species_scores[valid, :] = ewa
                with np.errstate(divide="ignore", invalid="ignore"):
                    shrinkage = np.divide(1.0, mul**2,
                                          out=np.zeros_like(mul),
                                          where=mul != 0)
                centre = wa_center

    return species_scores, shrinkage, centre


def _weighted_covariance(matrix: np.ndarray,
                         weights: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Replicate R's cov.wt(..., method='ML') for non-negative weights."""
    matrix = np.asarray(matrix, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    mask = weights > 0
    matrix = matrix[mask]
    weights = weights[mask]
    if matrix.size == 0 or np.sum(weights) == 0:
        return None, None

    w_sum = np.sum(weights)
    norm_weights = weights / w_sum
    centre = (matrix * norm_weights[:, None]).sum(axis=0)
    centred = matrix - centre
    cov = (centring := centred * norm_weights[:, None]).T @ centred
    return cov, centre
