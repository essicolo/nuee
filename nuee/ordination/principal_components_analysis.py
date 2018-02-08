import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as sklearn_PCA
from .ordi_plot import ordiplot, screeplot


class PrincipalComponentsAnalysis():
    r"""Just a wrapper around the sklearn.decomposition.pca class from
    scikit-sklearn to make it coherent with nuee.

    EXPLAIN WHEN TO USE PCA
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
        self.short_method_name = 'PCA'
        self.long_method_name = 'Principal Components Analysis'
        self.ordi_fitted = nuee_PCA
        self.eigenvalues = eigenvalues
        self.proportion_explained = p_explained
        self.sample_scores = pd.DataFrame(sample_scores,
                                          index=sample_ids,
                                          columns = ordi_column_names)
        self.biplot_scores = pd.DataFrame(biplot_scores,
                                          index=feature_ids,
                                          columns=ordi_column_names)

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
