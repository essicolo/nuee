"""
Redundancy Analysis (RDA) implementation.

RDA is a constrained ordination method that combines regression and PCA.
It finds linear combinations of explanatory variables that best explain
the variation in the response matrix.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict, Any
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import patsy

from .base import ConstrainedOrdinationResult, OrdinationMethod


class RDA(OrdinationMethod):
    """
    Redundancy Analysis for constrained ordination.
    
    RDA combines multiple regression and PCA to find axes that are
    linear combinations of explanatory variables and explain maximum
    variance in the response data.
    """
    
    def __init__(self, formula: Optional[str] = None, 
                 scale: bool = False, 
                 center: bool = True):
        """
        Initialize RDA.
        
        Parameters:
            formula: Formula string for specifying the model (e.g., "~ var1 + var2")
            scale: Whether to scale species to unit variance
            center: Whether to center species data
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
        
        Parameters:
            X: Response matrix (samples x species)
            Y: Explanatory matrix (samples x variables) or DataFrame for formula
            Z: Conditioning matrix for partial RDA (optional)
            **kwargs: Additional parameters
            
        Returns:
            ConstrainedOrdinationResult with RDA results
        """
        X = self._validate_data(X)
        
        # Handle formula interface
        if self.formula is not None:
            if not isinstance(Y, pd.DataFrame):
                raise ValueError("DataFrame required for formula interface")
            X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
            Y_matrix = self._parse_formula(self.formula, Y)
        else:
            if Y is None:
                raise ValueError("Either formula or Y matrix must be provided")
            Y_matrix = self._validate_data(Y)
            
        # Handle conditioning matrix for partial RDA
        if Z is not None:
            Z_matrix = self._validate_data(Z)
            return self._partial_rda(X, Y_matrix, Z_matrix)
        else:
            return self._simple_rda(X, Y_matrix)
    
    def _parse_formula(self, formula: str, data: pd.DataFrame) -> np.ndarray:
        """Parse formula string to create design matrix."""
        # Use patsy to parse the formula
        design_matrix = patsy.dmatrix(formula, data, return_type='dataframe')
        
        # Remove intercept if present (RDA handles centering explicitly)
        if 'Intercept' in design_matrix.columns:
            design_matrix = design_matrix.drop('Intercept', axis=1)
            
        return design_matrix.values
    
    def _simple_rda(self, X: np.ndarray, Y: np.ndarray) -> ConstrainedOrdinationResult:
        """Perform simple (non-partial) RDA."""
        n_samples, n_species = X.shape
        n_vars = Y.shape[1]
        
        # Center and optionally scale response matrix
        if self.center:
            X_centered = self._center_data(X)
        else:
            X_centered = X.copy()
            
        if self.scale:
            X_centered = self._standardize_data(X_centered)
            
        # Center explanatory variables
        Y_centered = self._center_data(Y)
        
        # Perform multiple regression of each species on explanatory variables
        fitted_values = np.zeros_like(X_centered)
        coefficients = np.zeros((n_vars, n_species))
        
        for i in range(n_species):
            reg = LinearRegression(fit_intercept=False)
            reg.fit(Y_centered, X_centered[:, i])
            fitted_values[:, i] = reg.predict(Y_centered)
            coefficients[:, i] = reg.coef_
        
        # PCA on fitted values (constrained ordination)
        pca_constrained = PCA()
        constrained_scores = pca_constrained.fit_transform(fitted_values)
        constrained_eig = pca_constrained.explained_variance_
        
        # Calculate residuals and perform PCA on residuals (unconstrained)
        residuals = X_centered - fitted_values
        pca_unconstrained = PCA()
        unconstrained_scores = pca_unconstrained.fit_transform(residuals)
        unconstrained_eig = pca_unconstrained.explained_variance_
        
        # Combine constrained and unconstrained scores
        n_constrained = min(n_vars, n_species)
        n_unconstrained = min(n_species - n_constrained, unconstrained_scores.shape[1])
        
        if n_unconstrained > 0:
            all_scores = np.hstack([
                constrained_scores[:, :n_constrained],
                unconstrained_scores[:, :n_unconstrained]
            ])
        else:
            all_scores = constrained_scores[:, :n_constrained]
        
        # Calculate species scores  
        constrained_components = pca_constrained.components_[:n_constrained, :].T
        
        if n_unconstrained > 0:
            unconstrained_components = pca_unconstrained.components_[:n_unconstrained, :].T
            species_scores = np.hstack([constrained_components, unconstrained_components])
        else:
            species_scores = constrained_components
        
        # Calculate biplot scores for explanatory variables
        biplot_scores = self._calculate_biplot_scores(
            Y_centered, constrained_scores[:, :n_constrained], constrained_eig[:n_constrained]
        )
        
        # Calculate total inertia
        tot_chi = np.sum(np.var(X_centered, axis=0, ddof=1))
        
        call_info = {
            'method': 'RDA',
            'formula': self.formula,
            'scale': self.scale,
            'center': self.center
        }
        
        return ConstrainedOrdinationResult(
            points=all_scores,
            species=species_scores,
            constrained_eig=constrained_eig[:n_constrained],
            unconstrained_eig=unconstrained_eig[:n_unconstrained] if n_unconstrained > 0 else np.array([]),
            biplot=biplot_scores,
            tot_chi=tot_chi,
            call=call_info
        )
    
    def _partial_rda(self, X: np.ndarray, Y: np.ndarray, 
                    Z: np.ndarray) -> ConstrainedOrdinationResult:
        """Perform partial RDA with conditioning variables."""
        # Center all matrices
        X_centered = self._center_data(X)
        Y_centered = self._center_data(Y)
        Z_centered = self._center_data(Z)
        
        # Remove effect of conditioning variables from both X and Y
        # Regress X on Z
        X_residual = np.zeros_like(X_centered)
        for i in range(X_centered.shape[1]):
            reg = LinearRegression(fit_intercept=False)
            reg.fit(Z_centered, X_centered[:, i])
            X_residual[:, i] = X_centered[:, i] - reg.predict(Z_centered)
        
        # Regress Y on Z
        Y_residual = np.zeros_like(Y_centered)
        for i in range(Y_centered.shape[1]):
            reg = LinearRegression(fit_intercept=False)
            reg.fit(Z_centered, Y_centered[:, i])
            Y_residual[:, i] = Y_centered[:, i] - reg.predict(Z_centered)
        
        # Perform RDA on residuals
        temp_rda = RDA(scale=self.scale, center=False)  # Already centered
        result = temp_rda._simple_rda(X_residual, Y_residual)
        
        # Calculate partial inertia
        partial_chi = np.sum(np.var(X_centered - X_residual, axis=0, ddof=1))
        result.partial_chi = partial_chi
        
        return result
    
    def _calculate_biplot_scores(self, Y: np.ndarray, scores: np.ndarray, 
                               eigenvalues: np.ndarray) -> np.ndarray:
        """Calculate biplot scores for explanatory variables."""
        # Correlation between explanatory variables and site scores
        correlations = np.corrcoef(Y.T, scores.T)[:Y.shape[1], Y.shape[1]:]
        
        # Scale by square root of eigenvalues
        biplot_scores = correlations * np.sqrt(eigenvalues[:correlations.shape[1]])
        
        return biplot_scores


def rda(X: Union[np.ndarray, pd.DataFrame], 
        Y: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        Z: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        scale: bool = False,
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
    rda_obj = RDA(formula=formula, scale=scale)
    
    # Use data parameter if provided with formula
    if formula is not None and data is not None:
        Y = data
        
    return rda_obj.fit(X, Y, Z, **kwargs)