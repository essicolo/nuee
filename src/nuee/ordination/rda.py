"""
Redundancy Analysis (RDA) implementation.

RDA is a constrained ordination method that combines regression and PCA.
It finds linear combinations of explanatory variables that best explain
the variation in the response matrix.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, Tuple, Callable
import patsy

from .base import ConstrainedOrdinationResult, OrdinationMethod


class RDA(OrdinationMethod):
    """
    Redundancy Analysis for constrained ordination.

    This implementation follows the classical algorithm:
      1. Centre/scale the response matrix as requested.
      2. Centre the constraining variables.
      3. Solve the weighted least-squares regression analytically.
      4. Perform SVD on fitted (constrained) and residual (unconstrained) matrices.
      5. Store the full decomposition so downstream tools can reconstruct
         scaled scores mirroring vegan's behaviour.
    """

    def __init__(self, formula: Optional[str] = None,
                 scale: bool = False,
                 center: bool = True):
        """
        Parameters
        ----------
        formula:
            R-style model formula describing the constraints (optional).
        scale:
            Whether to standardise species to unit variance before analysis.
        center:
            Whether to centre species (response) data column-wise.
        """
        self.formula = formula
        self.scale = scale
        self.center = center

    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            Y: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            Z: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            **kwargs) -> ConstrainedOrdinationResult:
        """
        Fit RDA to the data.

        Parameters
        ----------
        X:
            Response matrix (samples x species).
        Y:
            Constraining variables (samples x predictors) or DataFrame used with a formula.
        Z:
            Conditioning matrix for partial RDA.

        Returns
        -------
        ConstrainedOrdinationResult
        """
        X_is_df = isinstance(X, pd.DataFrame)
        sample_names = list(X.index) if X_is_df else None
        species_names = list(X.columns) if X_is_df else None
        raw_response_matrix = np.asarray(X.values if X_is_df else X, dtype=float).copy()
        X_matrix = self._validate_data(X)

        constraint_names = None
        raw_constraints: Union[pd.DataFrame, np.ndarray]
        raw_conditioning_matrix: Optional[Union[pd.DataFrame, np.ndarray]] = None
        conditioning_is_df = isinstance(Z, pd.DataFrame)
        if self.formula is not None:
            if not isinstance(Y, pd.DataFrame):
                raise ValueError("DataFrame required for formula interface")
            Y_matrix, constraint_names = self._parse_formula(self.formula, Y)
            raw_constraints = Y.copy(deep=True)
        else:
            if Y is None:
                raise ValueError("Either formula or Y matrix must be provided")
            if isinstance(Y, pd.DataFrame):
                constraint_names = list(Y.columns)
                raw_constraints = Y.copy(deep=True)
            Y_matrix = self._validate_data(Y)
            if not isinstance(Y, pd.DataFrame):
                raw_constraints = np.asarray(Y, dtype=float).copy()

        if Y_matrix.shape[1] == 0:
            raise ValueError("No constraining variables supplied for RDA.")

        call_info: Dict[str, Any] = {
            "method": "RDA",
            "formula": self.formula,
            "scale": self.scale,
            "center": self.center,
            "n_samples": int(X_matrix.shape[0]),
            "n_species": int(X_matrix.shape[1]),
            "n_constraints": int(Y_matrix.shape[1]),
        }
        if constraint_names is not None:
            call_info["constraints"] = constraint_names

        if Z is not None:
            Z_matrix = self._validate_data(Z)
            call_info["mode"] = "partial"
            raw_conditioning_matrix = Z.copy(deep=True) if conditioning_is_df else np.asarray(Z, dtype=float).copy()
            result = self._partial_rda(
                X_matrix,
                Y_matrix,
                Z_matrix,
                sample_names=sample_names,
                species_names=species_names,
                call_info=call_info
            )
            result._raw_response = raw_response_matrix
            result._response_is_dataframe = X_is_df
            if X_is_df:
                result._response_columns = list(species_names) if species_names is not None else None
                result._response_index = list(sample_names) if sample_names is not None else None
            else:
                result._response_columns = None
                result._response_index = None
            result._raw_constraints = raw_constraints
            result._constraints_is_dataframe = isinstance(raw_constraints, pd.DataFrame)
            result._raw_conditioning = raw_conditioning_matrix
            result._conditioning_is_dataframe = conditioning_is_df
            result._permutation_spec = {
                "formula": self.formula,
                "scale": self.scale,
                "center": self.center
            }
            return result

        call_info["mode"] = "simple"
        result = self._simple_rda(
            X_matrix,
            Y_matrix,
            sample_names=sample_names,
            species_names=species_names,
            call_info=call_info
        )
        result._raw_response = raw_response_matrix
        result._response_is_dataframe = X_is_df
        if X_is_df:
            result._response_columns = list(species_names) if species_names is not None else None
            result._response_index = list(sample_names) if sample_names is not None else None
        else:
            result._response_columns = None
            result._response_index = None
        result._raw_constraints = raw_constraints
        result._constraints_is_dataframe = isinstance(raw_constraints, pd.DataFrame)
        result._raw_conditioning = None
        result._conditioning_is_dataframe = False
        result._permutation_spec = {
            "formula": self.formula,
            "scale": self.scale,
            "center": self.center
        }
        return result

    def _parse_formula(self, formula: str, data: pd.DataFrame) -> Tuple[np.ndarray, list[str]]:
        """Parse formula string to create design matrix and retain column names."""
        design = patsy.dmatrix(formula, data, return_type="dataframe")
        if "Intercept" in design.columns:
            design = design.drop("Intercept", axis=1)
        return design.values.astype(float), design.columns.tolist()

    def _simple_rda(self, X: np.ndarray, Y: np.ndarray,
                    sample_names: Optional[list[str]] = None,
                    species_names: Optional[list[str]] = None,
                    call_info: Optional[Dict[str, Any]] = None) -> ConstrainedOrdinationResult:
        """Perform simple (non-partial) RDA."""
        response_matrix, response_center, response_scale = self._prepare_response_matrix(X)
        env_matrix, env_center = self._prepare_explanatory_matrix(Y)

        coefficients = self._solve_least_squares(env_matrix, response_matrix)
        fitted = env_matrix @ coefficients if env_matrix.size else np.zeros_like(response_matrix)
        residual = response_matrix - fitted

        n_samples, n_species = response_matrix.shape
        row_weights = np.ones(n_samples, dtype=float)
        column_weights = np.ones(n_species, dtype=float)
        sqrt_weights = np.sqrt(row_weights)[:, None]

        weighted_fitted = fitted * sqrt_weights
        weighted_residual = residual * sqrt_weights

        constrained_u, constrained_s, constrained_vt = self._svd_with_rank(weighted_fitted)
        unconstrained_u, unconstrained_s, unconstrained_vt = self._svd_with_rank(weighted_residual)

        if constrained_u.size:
            constrained_u = constrained_u / sqrt_weights
        if unconstrained_u.size:
            unconstrained_u = unconstrained_u / sqrt_weights

        constrained_eig = constrained_s ** 2
        unconstrained_eig = unconstrained_s ** 2

        cca_v = constrained_vt.T if constrained_s.size else None
        ca_v = unconstrained_vt.T if unconstrained_s.size else None

        cca_wa = None
        if cca_v is not None and constrained_s.size:
            inv_s = np.divide(1.0, constrained_s,
                              out=np.zeros_like(constrained_s),
                              where=constrained_s > 0)
            cca_wa = response_matrix @ (cca_v * inv_s)
            cca_wa = cca_wa / sqrt_weights

        ca_u = unconstrained_u if unconstrained_s.size else None

        site_blocks = []
        if cca_wa is not None:
            site_blocks.append(cca_wa)
        if ca_u is not None:
            site_blocks.append(ca_u)
        sites_default = np.hstack(site_blocks) if site_blocks else None

        species_blocks = []
        if cca_v is not None:
            species_blocks.append(cca_v)
        if ca_v is not None:
            species_blocks.append(ca_v)
        species_default = np.hstack(species_blocks) if species_blocks else None

        constrained_labels = ([f"RDA{i + 1}" for i in range(cca_wa.shape[1])]
                               if cca_wa is not None and cca_wa.size else [])
        unconstrained_labels = ([f"PC{i + 1}" for i in range(ca_u.shape[1])]
                                 if ca_u is not None and ca_u.size else [])
        site_labels = constrained_labels + unconstrained_labels
        species_labels = constrained_labels + unconstrained_labels

        biplot_scores = self._compute_biplot_scores(
            env_matrix, constrained_u, constrained_s, row_weights
        )

        tot_chi = float(np.sum(response_matrix ** 2))
        eig_full = np.concatenate([
            constrained_eig,
            unconstrained_eig
        ]) if constrained_eig.size or unconstrained_eig.size else np.array([], dtype=float)

        scaling_backend = _build_rda_scaling_backend(
            cca_wa=cca_wa if (cca_wa is not None and cca_wa.size) else None,
            ca_u=ca_u if (ca_u is not None and ca_u.size) else None,
            cca_v=cca_v if (cca_v is not None and cca_v.size) else None,
            ca_v=ca_v if (ca_v is not None and ca_v.size) else None,
            eigenvalues=eig_full,
            tot_chi=tot_chi,
            n_samples=n_samples
        )

        result = ConstrainedOrdinationResult(
            points=sites_default,
            species=species_default,
            constrained_eig=constrained_eig,
            unconstrained_eig=unconstrained_eig,
            biplot=biplot_scores,
            tot_chi=tot_chi,
            call=dict(call_info or {}),
            site_u=constrained_u if constrained_s.size else None,
            species_v=constrained_vt.T if constrained_s.size else None,
            singular_values=constrained_s if constrained_s.size else None,
            site_u_unconstrained=unconstrained_u if unconstrained_s.size else None,
            species_v_unconstrained=unconstrained_vt.T if unconstrained_s.size else None,
            singular_values_unconstrained=unconstrained_s if unconstrained_s.size else None,
            row_weights=row_weights,
            column_weights=column_weights,
            scaling_backend=scaling_backend
        )

        result._site_axis_labels = site_labels if site_labels else None
        result._species_axis_labels = species_labels if species_labels else None
        if sample_names is not None:
            result._site_names = list(sample_names)
        if species_names is not None:
            result._species_names = list(species_names)

        result.cca_wa = None if cca_wa is None else np.array(cca_wa, copy=True)
        result.ca_u = None if ca_u is None else np.array(ca_u, copy=True)
        result.cca_v = None if cca_v is None else np.array(cca_v, copy=True)
        result.ca_v = None if ca_v is None else np.array(ca_v, copy=True)
        result.response_matrix = response_matrix
        result.response_center = response_center
        result.response_scale = response_scale
        result.explanatory_center = env_center
        result.coefficients = coefficients
        result.fitted = fitted
        result.residual = residual
        result.row_weights = row_weights
        result.column_weights = column_weights
        return result

    def _partial_rda(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                     sample_names: Optional[list[str]] = None,
                     species_names: Optional[list[str]] = None,
                     call_info: Optional[Dict[str, Any]] = None) -> ConstrainedOrdinationResult:
        """Perform partial RDA with conditioning variables."""
        response_matrix, response_center, response_scale = self._prepare_response_matrix(X)
        env_matrix, env_center = self._prepare_explanatory_matrix(Y)
        conditioning_matrix, conditioning_center = self._prepare_explanatory_matrix(Z)

        coef_response = self._solve_least_squares(conditioning_matrix, response_matrix)
        X_residual = response_matrix - conditioning_matrix @ coef_response

        coef_env = self._solve_least_squares(conditioning_matrix, env_matrix)
        Y_residual = env_matrix - conditioning_matrix @ coef_env

        temp = RDA(scale=False, center=False)
        result = temp._simple_rda(
            X_residual,
            Y_residual,
            sample_names=sample_names,
            species_names=species_names,
            call_info=call_info
        )

        total_inertia = float(np.sum(response_matrix ** 2))
        partial_chi = float(np.sum((response_matrix - X_residual) ** 2))

        result.tot_chi = total_inertia
        result.partial_chi = partial_chi
        result.response_center = response_center
        result.response_scale = response_scale
        result.explanatory_center = env_center
        result.conditioning_center = conditioning_center
        return result

    @staticmethod
    def _solve_least_squares(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Solve A * coef = B in the least squares sense, column by column."""
        if A.size == 0:
            return np.zeros((A.shape[1], B.shape[1]), dtype=float)
        coef, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        return coef

    def _prepare_response_matrix(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Centre and/or scale the response matrix as configured."""
        X = np.asarray(X, dtype=float)
        column_means = np.mean(X, axis=0)
        if self.center:
            centred = X - column_means
        else:
            centred = X.copy()
        column_scale = np.ones_like(column_means)
        if self.scale:
            column_scale = np.std(centred, axis=0, ddof=1)
            zero_mask = column_scale == 0
            column_scale[zero_mask] = 1.0
        transformed = centred / column_scale
        n_samples = X.shape[0]
        if n_samples > 1:
            transformed = transformed / np.sqrt(n_samples - 1)
        return transformed, column_means, column_scale

    @staticmethod
    def _prepare_explanatory_matrix(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Centre the environmental matrix and record column means."""
        Y = np.asarray(Y, dtype=float)
        column_means = np.mean(Y, axis=0)
        return Y - column_means, column_means

    @staticmethod
    def _svd_with_rank(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute a trimmed SVD, removing numerically zero singular values."""
        if matrix.size == 0:
            return (
                np.zeros((matrix.shape[0], 0), dtype=float),
                np.array([], dtype=float),
                np.zeros((0, matrix.shape[1]), dtype=float)
            )
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
        if s.size == 0:
            return U[:, :0], s, Vt[:0, :]
        tol = np.finfo(float).eps * max(matrix.shape) * s[0]
        mask = s > tol
        if not np.any(mask):
            return U[:, :0], np.array([], dtype=float), Vt[:0, :]
        return U[:, mask], s[mask], Vt[mask, :]

    def _compute_biplot_scores(self, env_matrix: np.ndarray,
                                site_u: np.ndarray,
                                singular: np.ndarray,
                                row_weights: np.ndarray) -> Optional[np.ndarray]:
        if site_u is None or singular is None or site_u.size == 0 or singular.size == 0:
            return None

        inv_sqrt_weights = np.divide(
            1.0,
            np.sqrt(row_weights),
            out=np.zeros_like(row_weights, dtype=float),
            where=row_weights > 0
        )
        canonical_sites = (site_u * inv_sqrt_weights[:, None]) * singular
        if canonical_sites.size == 0 or env_matrix.size == 0:
            return None

        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(env_matrix.T, canonical_sites.T)
        # Extract the block of correlations between env vars and canonical axes
        env_dim = env_matrix.shape[1]
        correlations = corr[:env_dim, env_dim:]
        correlations = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)

        eig = singular ** 2
        cols = min(correlations.shape[1], eig.shape[0])
        if cols == 0:
            return None
        scaled = correlations[:, :cols] * np.sqrt(eig[:cols])
        return scaled


def _build_rda_scaling_backend(cca_wa: Optional[np.ndarray],
                               ca_u: Optional[np.ndarray],
                               cca_v: Optional[np.ndarray],
                               ca_v: Optional[np.ndarray],
                               eigenvalues: np.ndarray,
                               tot_chi: float,
                               n_samples: int
                               ) -> Optional[Callable[[int], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]]:
    if ((cca_wa is None or cca_wa.size == 0) and
            (ca_u is None or ca_u.size == 0) and
            (cca_v is None or cca_v.size == 0) and
            (ca_v is None or ca_v.size == 0)):
        return None

    slam = np.sqrt(np.clip(eigenvalues, a_min=0.0, a_max=None))

    if tot_chi > 0 and n_samples > 1:
        site_const = ((n_samples - 1) / tot_chi) ** 0.25
        species_const = np.sqrt(n_samples - 1) / site_const
        sym_const = np.power(n_samples - 1, 0.25)
    else:
        site_const = 1.0
        species_const = 1.0
        sym_const = 1.0

    def scaler(scaling: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if scaling not in (1, 2, 3):
            raise ValueError("scaling must be 1, 2, or 3 for RDA results.")

        site_scalers = {
            1: slam,
            2: np.ones_like(slam),
            3: np.sqrt(slam)
        }
        species_scalers = {
            1: np.ones_like(slam),
            2: slam,
            3: np.sqrt(slam)
        }

        axis_offset = 0
        site_blocks = []
        if cca_wa is not None and cca_wa.size:
            mult = site_scalers[scaling][axis_offset:axis_offset + cca_wa.shape[1]]
            site_blocks.append(cca_wa * mult)
            axis_offset += cca_wa.shape[1]
        if ca_u is not None and ca_u.size:
            mult = site_scalers[scaling][axis_offset:axis_offset + ca_u.shape[1]]
            site_blocks.append(ca_u * mult)
            axis_offset += ca_u.shape[1]

        axis_offset = 0
        species_blocks = []
        if cca_v is not None and cca_v.size:
            mult = species_scalers[scaling][axis_offset:axis_offset + cca_v.shape[1]]
            species_blocks.append(cca_v * mult)
            axis_offset += cca_v.shape[1]
        if ca_v is not None and ca_v.size:
            mult = species_scalers[scaling][axis_offset:axis_offset + ca_v.shape[1]]
            species_blocks.append(ca_v * mult)
            axis_offset += ca_v.shape[1]

        site_scaled = np.hstack(site_blocks) if site_blocks else None
        if site_scaled is not None:
            if scaling == 1:
                site_scaled = site_scaled * site_const
            elif scaling == 2:
                site_scaled = site_scaled * species_const
            else:
                site_scaled = site_scaled * sym_const
        species_scaled = np.hstack(species_blocks) if species_blocks else None
        if species_scaled is not None:
            if scaling == 1:
                species_scaled = species_scaled * species_const
            elif scaling == 2:
                species_scaled = species_scaled * site_const
            else:
                species_scaled = species_scaled * sym_const

        return site_scaled, species_scaled

    return scaler


def rda(X: Union[np.ndarray, pd.DataFrame], 
        Y: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        Z: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        scale: bool = False,
        center: bool = True,
        **kwargs) -> ConstrainedOrdinationResult:
    """
    Redundancy Analysis (RDA).
    
    RDA is a constrained ordination method that finds linear combinations
    of explanatory variables that best explain the variation in the response matrix.
    
    Parameters:
        X: Response matrix (samples x species)
        Y: Explanatory matrix (samples x variables)
        Z: Conditioning matrix for partial RDA (optional)
        formula: Formula string (e.g., "~ var1 + var2")
        data: DataFrame containing variables for formula
        scale: Whether to scale species to unit variance
        **kwargs: Additional parameters
        
    Returns:
        ConstrainedOrdinationResult with RDA results
        
    Examples:
        # Simple RDA
        result = rda(species_data, environmental_data)
        
        # RDA with formula
        result = rda(species_data, formula="~ pH + temperature", data=env_data)
        
        # Partial RDA
        result = rda(species_data, environmental_data, conditioning_data)
    """
    rda_obj = RDA(formula=formula, scale=scale, center=center)
    
    # Use data parameter if provided with formula
    if formula is not None and data is not None:
        Y = data
    
    return rda_obj.fit(X, Y, Z, **kwargs)
