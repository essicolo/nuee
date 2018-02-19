import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as sklearn_PCA
from .ordi_plot import ordiplot, screeplot


class PrincipalComponentsAnalysis():
    r"""Just a wrapper around the sklearn.decomposition.pca class from
    scikit-sklearn to make it coherent with nuee.

    scaling: either 1 (distance biplot) or 2 (correlation biplot). In scaling 2,
    both sample_scores (obervations) and biplot_scores (eigenvectors) are multiplied
    by the standard deviation matrix (square roots of eigenvalues).

    p 444 L&L

    """

    def __init__(self, n_components=None, copy=True, whiten=False,
                 svd_solver='auto', tol=0.0, iterated_power='auto',
                 random_state=None,
                 scaling=1):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.scaling = scaling

    def fit(self, X):
        self.X = X # store X in the output
        sample_ids = X.index
        feature_ids = X.columns
        X = X.as_matrix()
        nuee_PCA = sklearn_PCA(n_components=self.n_components,
                               copy=self.copy,
                               whiten=self.whiten,
                               svd_solver=self.svd_solver,
                               tol=self.tol,
                               iterated_power=self.iterated_power,
                               random_state=self.random_state)
        nuee_PCA.fit(X)

        ordi_column_names = ['PCA%d' % (i+1) for i in range(nuee_PCA.explained_variance_.shape[0])]

        # prepare output
        eigenvalues = nuee_PCA.explained_variance_
        p_explained = pd.Series(eigenvalues / eigenvalues.sum(), index=ordi_column_names)

        if self.scaling == 1:
            sample_scores = nuee_PCA.transform(X)
            biplot_scores = nuee_PCA.components_.T
        elif self.scaling == 2 or scaling == 'correlation':
            sample_scores = nuee_PCA.transform(X).dot(np.diag(eigenvalues**(-0.5)))
            biplot_scores = nuee_PCA.components_.dot(np.diag(eigenvalues**0.5)).T

        # Add PCA ordination object names to self
        self.ordiobject_type = 'PCA'
        self.method_name = 'Principal Components Analysis'
        self.ordi_fitted = nuee_PCA
        self.eigenvalues = eigenvalues
        self.proportion_explained = p_explained
        self.sample_scores = pd.DataFrame(sample_scores,
                                          index=sample_ids,
                                          columns = ordi_column_names)
        self.sample_scores.index.name = 'ID'
        self.biplot_scores = pd.DataFrame(biplot_scores,
                                          index=feature_ids,
                                          columns=ordi_column_names)
        self.biplot_scores.index.name = 'ID'
        return self

    def ordiplot(self, axes=[0, 1],
             arrow_scale=1,
             sample_scatter='labels', group=None,
             level=0.95,
             deviation_ellipses = True,
             error_ellipses = True):

        # Check
        if not hasattr(self, 'ordiobject_type'):
            raise ValueError("Not an ordination object. Have you fitted (.fit) beforehand?")

        return ordiplot(self, axes=axes,
                 arrow_scale=arrow_scale,
                 sample_scatter=sample_scatter,
                 group=group,
                 level=level,
                 deviation_ellipses = deviation_ellipses,
                 error_ellipses = error_ellipses)

    def screeplot(self):
        # Check
        if not hasattr(self, 'ordiobject_type'):
            raise ValueError("Not an ordination object. Have you fitted (.fit) beforehand?")
        return screeplot(self)
