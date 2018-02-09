import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as sklearn_LDA
from sklearn.discriminant_analysis import _cov
from .ordi_plot import ordiplot, screeplot
import warnings

class LinearDiscriminantAnalysis():
    r"""Just a wrapper around the sklearn.discriminant_analysis class from
    scikit-sklearn to make it coherent with nuee.

    EXPLAIN WHEN TO USE DA
    """

    def __init__(self,
                 solver='svd',
                 shrinkage=None,
                 priors=None,
                 n_components=None,
                 store_covariance=True,
                 tol=1e-4,
                 scaling=1):

        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.store_covariance = store_covariance
        self.tol = tol
        self.scaling = scaling

    def fit(self, X, y):
        if not np.all(X.index == y.index):
            warnings.warn("Warning: Indexes in X and y are different. Are you sure they are correctly alligned?")

        self.X = X
        self.y = y

        sample_ids = X.index
        feature_ids = X.columns
        X = X.as_matrix()
        y = y.as_matrix()

        nuee_LDA = sklearn_LDA(solver=self.solver,
                              shrinkage=self.shrinkage,
                              priors=self.priors,
                              n_components=self.n_components,
                              store_covariance=self.store_covariance,
                              tol=self.tol)

        nuee_LDA.fit(X, y)
        ordi_column_names = ['LDA%d' % (i+1) for i in range(nuee_LDA.coef_.shape[1])]

        # prepare output
        ## Compute eigenvalues. sklearn doesn't export them,
        ## so they have to be generated
        Sw = nuee_LDA.covariance_
        St = _cov(X, nuee_LDA.shrinkage)
        Sb = St - Sw  # between scatter
        eigenvalues, _ = linalg.eigh(Sb, Sw)
        eigenvalues = eigenvalues[::-1]
        p_explained = pd.Series(nuee_LDA.explained_variance_ratio_, index=ordi_column_names[:len(nuee_LDA.explained_variance_ratio_)])

        sample_scores = nuee_LDA.transform(X)
        biplot_scores = nuee_LDA.scalings_
        if self.scaling == 2:
            sample_scores = sample_scores.dot(np.diag(eigenvalues[:sample_scores.shape[1]]**(-0.5)))
            biplot_scores = biplot_scores.dot(np.diag(eigenvalues[:biplot_scores.shape[1]]**0.5))

        # Add LCA ordination object names to self
        self.ordiobject_type = 'LDA'
        self.method_name = 'Linear Discriminant Analysis'
        self.ordi_fitted = nuee_LDA
        self.eigenvalues = eigenvalues
        self.proportion_explained = p_explained
        self.sample_scores = pd.DataFrame(sample_scores,
                                          index=sample_ids,
                                          columns = ordi_column_names[:sample_scores.shape[1]])
        self.sample_scores.index.name = 'ID'
        self.biplot_scores = pd.DataFrame(biplot_scores,
                                          index=feature_ids,
                                          columns=ordi_column_names[:biplot_scores.shape[1]])
        self.biplot_scores.index.name = 'ID'

        return self

    def ordiplot(self, axes=[0, 1],
             arrow_scale=1,
             sample_scatter='labels', group='from_self',
             level=0.95,
             deviation_ellipses = True,
             error_ellipses = True):

        # Check
        if not hasattr(self, 'ordiobject_type'):
            raise ValueError("Not an ordination object. Have you fitted (.fit) beforehand?")
        if group=='from_self':
            group=self.y

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
