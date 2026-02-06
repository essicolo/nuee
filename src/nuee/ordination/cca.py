"""
Canonical Correspondence Analysis (CCA) implementation.

The algorithm mirrors the approach used in vegan:
  1. Translate species data to relative frequencies and apply
     chi-square (CA) standardisation.
  2. Perform weighted least-squares regression in the space defined by
     row masses to obtain fitted (constrained) and residual matrices.
  3. Conduct SVD of both parts to expose canonical axes.
  4. Store all decomposition pieces (U/V/singular values, weights) so
     downstream scoring can faithfully reproduce vegan-style scalings.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, Tuple
import patsy

from .base import OrdinationResult, ConstrainedOrdinationResult, OrdinationMethod


class CCA(OrdinationMethod):
    """Canonical Correspondence Analysis."""

    def __init__(self, formula: Optional[str] = None):
        self.formula = formula

    def fit(self, Y: Union[np.ndarray, pd.DataFrame],
            X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            **kwargs) -> Union[ConstrainedOrdinationResult, OrdinationResult]:
        """Fit CCA (or CA when X is None) to the data."""
        species = self._validate_data(Y)

        species_names = list(Y.columns) if isinstance(Y, pd.DataFrame) else None
        site_names = list(Y.index) if isinstance(Y, pd.DataFrame) else None

        if X is None and self.formula is None:
            return self._fit_ca(species, site_names=site_names,
                                species_names=species_names)

        env_names = None
        if self.formula is not None:
            if not isinstance(X, pd.DataFrame):
                raise ValueError("DataFrame required for formula interface in CCA.")
            X_matrix, env_names = self._parse_formula(self.formula, X)
        else:
            if X is None:
                raise ValueError("Environmental matrix Y is required for CCA")
            if isinstance(X, pd.DataFrame):
                env_names = list(X.columns)
            X_matrix = self._validate_data(X)

        totals = species.sum()
        if totals == 0:
            raise ValueError("Species matrix must contain positive totals for CCA.")

        row_masses, col_masses, centered, Z_w, row_sqrt, col_sqrt = self._chi_square_components(species)

        env_centered = self._weighted_center(X_matrix, row_masses)
        sqrt_row = row_sqrt[:, None]
        X_weighted = sqrt_row * env_centered

        constrained = self._fit_weighted_regression(X_weighted, Z_w)
        residual = Z_w - constrained

        constr_u, constr_s, constr_vt = self._svd_with_rank(constrained)
        resid_u, resid_s, resid_vt = self._svd_with_rank(residual)

        if constr_u.size:
            constr_u = np.divide(constr_u,
                                 row_sqrt[:, None],
                                 out=np.zeros_like(constr_u),
                                 where=row_sqrt[:, None] > 0)
        if resid_u.size:
            resid_u = np.divide(resid_u,
                                row_sqrt[:, None],
                                out=np.zeros_like(resid_u),
                                where=row_sqrt[:, None] > 0)

        constrained_eig = constr_s ** 2
        unconstrained_eig = resid_s ** 2

        biplot_scores = self._compute_biplot_scores(
            env_centered, constr_u, constr_s, row_masses
        )

        tot_chi = float(np.sum((centered ** 2) / (row_masses[:, None] * col_masses[None, :])))

        call_info: Dict[str, Any] = {
            "method": "CCA",
            "formula": self.formula,
            "n_samples": int(species.shape[0]),
            "n_species": int(species.shape[1]),
            "n_constraints": int(X_matrix.shape[1]),
        }
        if env_names is not None:
            call_info["constraints"] = env_names

        result = ConstrainedOrdinationResult(
            points=None,
            species=None,
            constrained_eig=constrained_eig,
            unconstrained_eig=unconstrained_eig,
            biplot=biplot_scores,
            tot_chi=tot_chi,
            call=call_info,
            site_u=constr_u if constr_s.size else None,
            species_v=self._scale_species_vectors(constr_vt.T, col_sqrt) if constr_s.size else None,
            singular_values=constr_s if constr_s.size else None,
            site_u_unconstrained=resid_u if resid_s.size else None,
            species_v_unconstrained=self._scale_species_vectors(resid_vt.T, col_sqrt) if resid_s.size else None,
            singular_values_unconstrained=resid_s if resid_s.size else None,
            row_weights=row_masses,
            column_weights=col_masses
        )

        result.row_masses = row_masses
        result.column_masses = col_masses
        result.column_totals = species.sum(axis=0)
        result.row_totals = species.sum(axis=1)
        result.species_centered = centered
        result.environment_centered = env_centered
        result.barycentric_species = self._species_barycentric_scores(centered, col_masses, result)
        return result

    def _fit_ca(self, species: np.ndarray,
                site_names=None, species_names=None) -> OrdinationResult:
        """Unconstrained Correspondence Analysis."""
        totals = species.sum()
        if totals == 0:
            raise ValueError("Species matrix must contain positive totals for CA.")

        row_masses, col_masses, centered, Z_w, row_sqrt, col_sqrt = \
            self._chi_square_components(species)

        u, s, vt = self._svd_with_rank(Z_w)

        if u.size:
            u = np.divide(u, row_sqrt[:, None],
                          out=np.zeros_like(u),
                          where=row_sqrt[:, None] > 0)

        eigenvalues = s ** 2
        species_v = self._scale_species_vectors(vt.T, col_sqrt) if s.size else None

        tot_chi = float(np.sum(
            (centered ** 2) / (row_masses[:, None] * col_masses[None, :])
        ))

        call_info: Dict[str, Any] = {
            "method": "CA",
            "n_samples": int(species.shape[0]),
            "n_species": int(species.shape[1]),
        }

        scaling_backend = _build_ca_scaling_backend(
            site_u=u if s.size else None,
            species_v=species_v,
            eigenvalues=eigenvalues,
        )

        result = OrdinationResult(
            points=None,
            species=None,
            eigenvalues=eigenvalues,
            call=call_info,
            site_u=u if s.size else None,
            species_v=species_v,
            singular_values=s if s.size else None,
            row_weights=row_masses,
            column_weights=col_masses,
            scaling_backend=scaling_backend,
        )

        result.row_masses = row_masses
        result.column_masses = col_masses
        result.tot_chi = tot_chi
        result.species_centered = centered

        n_axes = u.shape[1] if u.size else 0
        sp_axes = species_v.shape[1] if species_v is not None else 0

        if site_names:
            result._site_names = list(site_names)
        result._site_axis_labels = [f"CA{i+1}" for i in range(n_axes)]

        if species_names:
            result._species_names = list(species_names)
        result._species_axis_labels = [f"CA{i+1}" for i in range(sp_axes)]

        return result

    def _parse_formula(self, formula: str, data: pd.DataFrame) -> Tuple[np.ndarray, list[str]]:
        design = patsy.dmatrix(formula, data, return_type="dataframe")
        if "Intercept" in design.columns:
            design = design.drop("Intercept", axis=1)
        return design.values.astype(float), design.columns.tolist()

    def _chi_square_components(self, species: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        totals = np.sum(species)
        row_totals = species.sum(axis=1)
        col_totals = species.sum(axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            row_masses = np.divide(row_totals, totals, out=np.zeros_like(row_totals, dtype=float), where=totals > 0)
            col_masses = np.divide(col_totals, totals, out=np.zeros_like(col_totals, dtype=float), where=totals > 0)

        row_masses[row_masses == 0] = 1e-12
        col_masses[col_masses == 0] = 1e-12

        F = species / totals
        expected = np.outer(row_masses, col_masses)
        centered = F - expected

        row_sqrt = np.sqrt(row_masses)
        col_sqrt = np.sqrt(col_masses)

        denom = row_sqrt[:, None] * col_sqrt[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            Z_w = np.divide(centered,
                            denom,
                            out=np.zeros_like(centered),
                            where=denom > 0)
        return row_masses, col_masses, centered, Z_w, row_sqrt, col_sqrt

    def _weighted_center(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        weights = weights / weights.sum()
        weighted_mean = np.average(X, axis=0, weights=weights)
        return X - weighted_mean

    def _fit_weighted_regression(self, Xw: np.ndarray, Zw: np.ndarray) -> np.ndarray:
        if Xw.size == 0:
            return np.zeros_like(Zw)
        coef, _, _, _ = np.linalg.lstsq(Xw, Zw, rcond=None)
        return Xw @ coef

    @staticmethod
    def _svd_with_rank(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def _compute_biplot_scores(self, X_centered: np.ndarray,
                                site_u: np.ndarray,
                                singular: np.ndarray,
                                row_masses: np.ndarray) -> Optional[np.ndarray]:
        if site_u is None or singular is None or site_u.size == 0 or singular.size == 0:
            return None
        inv_sqrt = np.divide(
            1.0,
            np.sqrt(row_masses),
            out=np.zeros_like(row_masses, dtype=float),
            where=row_masses > 0
        )
        canonical_sites = (site_u * inv_sqrt[:, None]) * singular
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(X_centered.T, canonical_sites.T)
        env_dim = X_centered.shape[1]
        correlations = corr[:env_dim, env_dim:]
        correlations = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)
        eig = singular ** 2
        cols = min(correlations.shape[1], eig.shape[0])
        if cols == 0:
            return None
        return correlations[:, :cols] * np.sqrt(eig[:cols])

    def _scale_species_vectors(self, V: np.ndarray, col_sqrt: np.ndarray) -> np.ndarray:
        return V / col_sqrt[:, None]

    def _species_barycentric_scores(self, centered: np.ndarray,
                                    col_masses: np.ndarray,
                                    result: ConstrainedOrdinationResult) -> Optional[np.ndarray]:
        try:
            site_scores = result.get_scores(display="sites", scaling=1)
        except ValueError:
            return None
        if site_scores is None or site_scores.size == 0:
            return None
        row_masses = getattr(result, "row_masses", None)
        if row_masses is None:
            return None
        F = centered + np.outer(row_masses, col_masses)
        weighted_profiles = np.divide(
            F,
            col_masses[None, :],
            out=np.zeros_like(F),
            where=col_masses[None, :] > 0
        )
        species_scores = weighted_profiles.T @ site_scores
        weights = weighted_profiles.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            species_scores = np.divide(
                species_scores,
                weights[:, None],
                out=np.zeros_like(species_scores),
                where=weights[:, None] > 0
            )
        return species_scores


def _build_ca_scaling_backend(site_u, species_v, eigenvalues):
    """Build a scaling backend for CA results (chi-square metric scaling)."""
    if site_u is None and species_v is None:
        return None

    slam = np.sqrt(np.clip(eigenvalues, a_min=0.0, a_max=None))

    def scaler(scaling: int):
        if scaling not in (1, 2, 3):
            raise ValueError("scaling must be 1, 2, or 3")

        if scaling == 1:
            site_mult = slam
            sp_mult = np.ones_like(slam)
        elif scaling == 2:
            site_mult = np.ones_like(slam)
            sp_mult = slam
        else:
            site_mult = np.sqrt(slam)
            sp_mult = np.sqrt(slam)

        sites = None
        if site_u is not None:
            sites = np.array(site_u, copy=True)
            cols = min(sites.shape[1], len(site_mult))
            sites = sites[:, :cols] * site_mult[:cols]

        species = None
        if species_v is not None:
            species = np.array(species_v, copy=True)
            cols = min(species.shape[1], len(sp_mult))
            species = species[:, :cols] * sp_mult[:cols]

        return sites, species

    return scaler


def cca(Y: Union[np.ndarray, pd.DataFrame],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        formula: Optional[str] = None,
        **kwargs) -> Union[ConstrainedOrdinationResult, OrdinationResult]:
    """
    Canonical Correspondence Analysis (or CA when X is None).

    Parameters
    ----------
    Y:
        Species data matrix (sites x species).
    X:
        Environmental data matrix (sites x variables) or DataFrame for
        formula evaluation.  When *None* and no formula is given, an
        unconstrained Correspondence Analysis (CA) is performed.
    formula:
        R-style formula string referencing columns in `X`.

    Returns
    -------
    ConstrainedOrdinationResult (CCA) or OrdinationResult (CA).
    """
    cca_obj = CCA(formula=formula)
    return cca_obj.fit(Y, X, **kwargs)


def ca(Y: Union[np.ndarray, pd.DataFrame], **kwargs) -> OrdinationResult:
    """
    Correspondence Analysis (unconstrained).

    Parameters
    ----------
    Y:
        Species data matrix (sites x species).

    Returns
    -------
    OrdinationResult with CA results.
    """
    cca_obj = CCA()
    return cca_obj.fit(Y, X=None, **kwargs)
