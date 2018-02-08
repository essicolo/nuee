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
        self.short_method_name = 'LDA'
        self.long_method_name = 'Linear Discriminant Analysis'
        self.ordi_fitted = nuee_LDA
        self.eigenvalues = eigenvalues
        self.proportion_explained = p_explained
        self.sample_scores = pd.DataFrame(sample_scores,
                                          index=sample_ids,
                                          columns = ordi_column_names[:sample_scores.shape[1]])
        self.biplot_scores = pd.DataFrame(biplot_scores,
                                          index=feature_ids,
                                          columns=ordi_column_names[:biplot_scores.shape[1]])

        return self

    def ordiplot(self, axes=[0, 1], title='', cmap=None, arrow_scale=1):
        # Check if the object is plottable
        ordi_objects = ['short_method_name', 'long_method_name', 'eigenvalues',
                        'proportion_explained', 'sample_scores','biplot_scores']
        is_ordi_object = []
        for i in ordi_objects: is_ordi_object.append(i in dir(self))
        if not np.all(is_ordi_object):
            raise ValueError("Not an ordination object. Have you fitted (.fit) beforehand?")

        return ordiplot(self, axes=axes, title=title, cmap=cmap, arrow_scale=arrow_scale)

    def screeplot(self):
        # Check if the object is plottable
        ordi_objects = ['short_method_name', 'long_method_name', 'eigenvalues',
                        'proportion_explained', 'sample_scores',
                        'biplot_scores']
        is_ordi_object = []
        for i in ordi_objects: is_ordi_object.append(i in dir(self))
        if not np.all(is_ordi_object):
            raise ValueError("Not an ordination object. Have you fitted (.fit) beforehand?")

        return screeplot(self)
