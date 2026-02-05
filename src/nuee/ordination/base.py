"""
Base classes and utilities for ordination methods.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any, Tuple, Callable, List


class OrdinationResult:
    """
    Base class for ordination results.
    
    Attributes:
        points: Sample coordinates in ordination space
        species: Species coordinates (if applicable)
        eigenvalues: Eigenvalues for ordination axes
        stress: Stress value (for NMDS)
        converged: Whether the ordination converged
        nobj: Number of objects (samples)
        ndim: Number of dimensions
        call: Dictionary containing method call parameters
    """
    
    def __init__(self,
                 points: Optional[np.ndarray],
                 species: Optional[np.ndarray] = None,
                 eigenvalues: Optional[np.ndarray] = None,
                 stress: Optional[float] = None,
                 converged: bool = True,
                 call: Optional[Dict[str, Any]] = None,
                 site_u: Optional[np.ndarray] = None,
                 species_v: Optional[np.ndarray] = None,
                 singular_values: Optional[np.ndarray] = None,
                 site_u_unconstrained: Optional[np.ndarray] = None,
                 species_v_unconstrained: Optional[np.ndarray] = None,
                 singular_values_unconstrained: Optional[np.ndarray] = None,
                 row_weights: Optional[np.ndarray] = None,
                 column_weights: Optional[np.ndarray] = None,
                 scaling_backend: Optional[
                     Callable[[int], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]] = None):
        self._raw_points = None if points is None else np.array(points, dtype=float, copy=True)
        self._raw_species = None if species is None else np.array(species, dtype=float, copy=True)

        self.points = None if self._raw_points is None else np.array(self._raw_points, copy=True)
        self.species = None if self._raw_species is None else np.array(self._raw_species, copy=True)
        self.eigenvalues = None if eigenvalues is None else np.array(eigenvalues, dtype=float, copy=True)
        self.stress = stress
        self.converged = converged
        self.call = call or {}

        self._site_u_constrained = self._to_2d(site_u)
        self._species_v_constrained = self._to_2d(species_v)
        self._singular_constrained = self._to_1d(singular_values)

        self._site_u_unconstrained = self._to_2d(site_u_unconstrained)
        self._species_v_unconstrained = self._to_2d(species_v_unconstrained)
        self._singular_unconstrained = self._to_1d(singular_values_unconstrained)

        self._row_weights = self._to_1d(row_weights)
        self._column_weights = self._to_1d(column_weights)

        self._site_axis_labels: Optional[List[str]] = None
        self._species_axis_labels: Optional[List[str]] = None
        self._site_names: Optional[List[str]] = None
        self._species_names: Optional[List[str]] = None

        self._site_u_all: Optional[np.ndarray] = None
        self._species_v_all: Optional[np.ndarray] = None
        self._singular_all: Optional[np.ndarray] = None

        self._scaling_backend = scaling_backend

        self.singular_values = (None if self._singular_constrained is None
                                else np.array(self._singular_constrained, copy=True))
        self.singular_values_unconstrained = (None if self._singular_unconstrained is None
                                              else np.array(self._singular_unconstrained, copy=True))

        self._assemble_components()
        self._ensure_defaults()

        self.nobj = self.points.shape[0] if self.points is not None else (
            self._site_u_all.shape[0] if self._site_u_all is not None else 0
        )
        self.ndim = self.points.shape[1] if self.points is not None else (
            self._site_u_all.shape[1] if self._site_u_all is not None else 0
        )
        self.n_samples = self.nobj
        
    def __repr__(self):
        return f"OrdinationResult(nobj={self.nobj}, ndim={self.ndim})"
    
    def plot(self, axes=(0, 1), display="sites", type="points", scaling=None, **kwargs):
        """
        Plot the ordination result.
        
        Parameters:
            axes: Which axes to plot (tuple of axis indices)
            display: What to display ("sites", "species", "both")
            type: Plot type ("points", "text", "none")
            **kwargs: Additional plotting arguments
            
        Returns:
            matplotlib Figure object
        """
        from ..plotting.ordination_plots import plot_ordination
        return plot_ordination(self, axes=axes, display=display, type=type,
                               scaling=scaling, **kwargs)
    
    def biplot(self, scaling: Union[int, str, None] = None, **kwargs):
        """
        Create a biplot.

        For PCA the species loadings are shown as arrows from the origin.
        For constrained ordination (RDA / CCA), environmental variables
        are shown as arrows while species are shown as points.

        Parameters:
            scaling: Scaling mode for site/species scores
            **kwargs: Additional plotting arguments

        Returns:
            matplotlib Figure object
        """
        from ..plotting.ordination_plots import biplot
        return biplot(self, scaling=scaling or 2, **kwargs)

    def get_scores(self, display: str = "sites",
                   scaling: Optional[Union[int, str]] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Retrieve (optionally scaled) ordination scores.

        Parameters:
            display: "sites", "species", or "both"
            scaling: None, 1/2/3 or "sites"/"species"/"symmetric"

        Returns:
            Requested scores as numpy arrays.
        """
        display = display.lower()
        scaling_id = self._normalize_scaling(scaling)

        custom_backend = self._scaling_backend if scaling_id is not None else None

        if scaling_id is None:
            sites = None if self._raw_points is None else np.array(self._raw_points, copy=True)
            species = None if self._raw_species is None else np.array(self._raw_species, copy=True)
        elif custom_backend is not None:
            sites, species = custom_backend(scaling_id)
            if sites is not None:
                sites = np.array(sites, copy=True)
                sites = self._wrap_scores_dataframe(
                    sites, self._site_axis_labels, self._site_names
                )
            if species is not None:
                species = np.array(species, copy=True)
                species = self._wrap_scores_dataframe(
                    species, self._species_axis_labels, self._species_names
                )
        else:
            sites, species = self._compute_scaled_scores(scaling_id)

        if display == "sites":
            return sites
        if display == "species":
            return species
        if display == "both":
            return sites, species
        raise ValueError("display must be one of 'sites', 'species', or 'both'")

    scores = get_scores

    def _wrap_scores_dataframe(self,
                               array: np.ndarray,
                               axis_labels: Optional[List[str]],
                               index_labels: Optional[List[str]]):
        if axis_labels is None and index_labels is None:
            return array
        df = pd.DataFrame(array)
        if axis_labels is not None:
            labels = [str(lbl) for lbl in axis_labels[:df.shape[1]]]
            df.columns = pd.Index(labels, dtype=object)
        if index_labels is not None and len(index_labels) == df.shape[0]:
            df.index = pd.Index([str(lbl) for lbl in index_labels], dtype=object)
        return df

    def _normalize_scaling(self, scaling: Optional[Union[int, str]]) -> Optional[int]:
        if scaling is None:
            return None
        if isinstance(scaling, str):
            mapping = {
                "sites": 1,
                "species": 2,
                "symmetric": 3,
                "sym": 3
            }
            scaling = mapping.get(scaling.lower())
        if scaling in (1, 2, 3):
            return scaling
        if scaling in (None, 0):
            return None
        raise ValueError("scaling must be None, 1, 2, 3 or one of "
                         "'sites', 'species', 'symmetric'")

    def _compute_scaled_scores(self, scaling: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._singular_all is None:
            raise ValueError("Scaled scores requested but no decomposition is stored.")

        singular = self._singular_all
        site_mult, species_mult = self._scaling_multipliers(scaling, singular)

        sites = self._unweighted_left_vectors()
        if sites is not None:
            sites = np.array(sites, copy=True)
            cols = min(sites.shape[1], len(site_mult))
            sites = sites[:, :cols]
            site_mult = site_mult[:cols]
            sites *= site_mult

        species = None
        if self._species_v_all is not None:
            species = np.array(self._species_v_all, copy=True)
            cols = min(species.shape[1], len(species_mult))
            species = species[:, :cols]
            species_mult = species_mult[:cols]
            species *= species_mult

        return sites, species

    def _scaling_multipliers(self, scaling: int,
                             singular: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if singular is None:
            if self._singular_all is None:
                raise ValueError("No singular values stored for scaling.")
            singular = self._singular_all

        singular = np.asarray(singular, dtype=float)
        singular = np.clip(singular, a_min=0.0, a_max=None)

        if scaling == 1:
            site_mult = singular
            species_mult = np.ones_like(singular)
        elif scaling == 2:
            site_mult = np.ones_like(singular)
            species_mult = singular
        elif scaling == 3:
            fourth = np.sqrt(singular)
            site_mult = fourth
            species_mult = fourth
        else:
            raise ValueError("Unsupported scaling value.")

        return site_mult, species_mult

    def _assemble_components(self) -> None:
        site_blocks = []
        singular_blocks = []

        if self._site_u_constrained is not None and self._singular_constrained is not None:
            site_blocks.append(self._site_u_constrained)
            singular_blocks.append(self._singular_constrained)
        if self._site_u_unconstrained is not None and self._singular_unconstrained is not None:
            site_blocks.append(self._site_u_unconstrained)
            singular_blocks.append(self._singular_unconstrained)

        if site_blocks:
            self._site_u_all = np.concatenate(site_blocks, axis=1)
        if singular_blocks:
            self._singular_all = np.concatenate(singular_blocks)

        species_blocks = []
        if self._species_v_constrained is not None and self._singular_constrained is not None:
            species_blocks.append(self._species_v_constrained)
        if self._species_v_unconstrained is not None and self._singular_unconstrained is not None:
            species_blocks.append(self._species_v_unconstrained)

        if species_blocks:
            self._species_v_all = np.concatenate(species_blocks, axis=1)

    def _ensure_defaults(self) -> None:
        if self.points is None and self._singular_all is not None:
            sites, _ = self._scores_from_components(default_scaling=1)
            if sites is not None:
                self._raw_points = sites
                self.points = np.array(sites, copy=True)

        if self.species is None and self._singular_all is not None:
            _, species = self._scores_from_components(default_scaling=2)
            if species is not None:
                self._raw_species = species
                self.species = np.array(species, copy=True)

    def _scores_from_components(self, default_scaling: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._singular_all is None:
            return None, None

        site_mult, species_mult = self._scaling_multipliers(default_scaling, self._singular_all)

        sites = self._unweighted_left_vectors()
        if sites is not None:
            sites = np.array(sites, copy=True)
            cols = min(sites.shape[1], len(site_mult))
            sites = sites[:, :cols]
            site_mult = site_mult[:cols]
            sites *= site_mult

        species = None
        if self._species_v_all is not None:
            species = np.array(self._species_v_all, copy=True)
            cols = min(species.shape[1], len(species_mult))
            species = species[:, :cols]
            species_mult = species_mult[:cols]
            species *= species_mult

        return sites, species

    def _unweighted_left_vectors(self) -> Optional[np.ndarray]:
        if self._site_u_all is None:
            return None

        site = np.array(self._site_u_all, copy=True)
        if self._row_weights is None:
            return site

        weights = np.asarray(self._row_weights, dtype=float).reshape(-1)
        if weights.shape[0] != site.shape[0]:
            raise ValueError("Row weights must match the number of sites.")

        sqrt_weights = np.sqrt(weights)
        inv_sqrt = np.divide(
            1.0,
            sqrt_weights,
            out=np.zeros_like(weights, dtype=float),
            where=sqrt_weights > 0
        )
        site *= inv_sqrt[:, None]
        return site

    @staticmethod
    def _to_2d(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        elif arr.ndim != 2:
            raise ValueError("Expected a 2D array.")
        return np.array(arr, copy=True)

    @staticmethod
    def _to_1d(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if value is None:
            return None
        arr = np.asarray(value, dtype=float).reshape(-1)
        return np.array(arr, copy=True)


class ConstrainedOrdinationResult(OrdinationResult):
    """
    Result class for constrained ordination methods (RDA, CCA).
    
    Additional attributes:
        constrained_eig: Eigenvalues for constrained axes
        unconstrained_eig: Eigenvalues for unconstrained axes  
        biplot: Biplot scores for environmental variables
        centroids: Centroids for factor variables
        envfit_result: Environmental fitting results
        tot_chi: Total inertia
        partial_chi: Partial inertia (if partial ordination)
    """
    
    def __init__(self, points: Optional[np.ndarray] = None, species: Optional[np.ndarray] = None,
                 constrained_eig: Optional[np.ndarray] = None,
                 unconstrained_eig: Optional[np.ndarray] = None,
                 biplot: Optional[np.ndarray] = None,
                 centroids: Optional[np.ndarray] = None,
                 tot_chi: Optional[float] = None,
                 partial_chi: Optional[float] = None,
                 call: Optional[Dict[str, Any]] = None,
                 site_u: Optional[np.ndarray] = None,
                 species_v: Optional[np.ndarray] = None,
                 singular_values: Optional[np.ndarray] = None,
                 site_u_unconstrained: Optional[np.ndarray] = None,
                 species_v_unconstrained: Optional[np.ndarray] = None,
                 singular_values_unconstrained: Optional[np.ndarray] = None,
                 row_weights: Optional[np.ndarray] = None,
                 column_weights: Optional[np.ndarray] = None,
                 scaling_backend: Optional[
                     Callable[[int], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]] = None):

        # Handle eigenvalues concatenation safely
        eig_parts = []
        if constrained_eig is not None and len(constrained_eig) > 0:
            eig_parts.append(constrained_eig)
        if unconstrained_eig is not None and len(unconstrained_eig) > 0:
            eig_parts.append(unconstrained_eig)

        all_eig = np.concatenate(eig_parts) if eig_parts else np.array([])
        super().__init__(
            points=points,
            species=species,
            eigenvalues=all_eig,
            call=call,
            site_u=site_u,
            species_v=species_v,
            singular_values=singular_values,
            site_u_unconstrained=site_u_unconstrained,
            species_v_unconstrained=species_v_unconstrained,
            singular_values_unconstrained=singular_values_unconstrained,
            row_weights=row_weights,
            column_weights=column_weights,
            scaling_backend=scaling_backend
        )

        self.constrained_eig = constrained_eig
        self.unconstrained_eig = unconstrained_eig
        self.biplot_scores = biplot
        self.centroids = centroids
        self.tot_chi = tot_chi
        self.partial_chi = partial_chi
        self._raw_response = None
        self._response_is_dataframe = False
        self._response_columns = None
        self._response_index = None
        self._raw_constraints = None
        self._constraints_is_dataframe = False
        self._raw_conditioning = None
        self._conditioning_is_dataframe = False
        self._permutation_spec = {}
        
    @property
    def rank(self) -> int:
        """Number of constrained axes."""
        return len(self.constrained_eig) if self.constrained_eig is not None else 0
    
    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Proportion of variance explained by each axis."""
        if self.eigenvalues is not None and self.tot_chi is not None:
            return self.eigenvalues / self.tot_chi
        return np.array([])


class OrdinationMethod(ABC):
    """
    Abstract base class for ordination methods.
    """
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> OrdinationResult:
        """
        Fit the ordination method to the data.
        
        Parameters:
            X: Community data matrix (samples x species)
            **kwargs: Method-specific parameters
            
        Returns:
            OrdinationResult object
        """
        pass
    
    def _validate_data(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Validate and convert input data to numpy array.
        
        Parameters:
            X: Input data matrix
            
        Returns:
            Validated numpy array
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array")
            
        if np.any(np.isnan(X)):
            raise ValueError("Input data contains NaN values")
            
        if np.any(np.isinf(X)):
            raise ValueError("Input data contains infinite values")
            
        return X
    
    def _center_data(self, X: np.ndarray) -> np.ndarray:
        """Center the data by subtracting column means."""
        return X - np.mean(X, axis=0)
    
    def _standardize_data(self, X: np.ndarray) -> np.ndarray:
        """Standardize the data to unit variance."""
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)





