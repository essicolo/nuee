import numpy as np
import pandas as pd
from scipy.linalg import svd, lstsq
from ..stats._utils import corr, scale
from .ordi_plot import ordiplot, screeplot

class RedundancyAnalysis():
    r"""Compute redundancy analysis, a type of canonical analysis.

    Redundancy analysis (RDA) is a principal component analysis on predicted
    values :math:`\hat{Y}` obtained by fitting response variables :math:`Y` with
    explanatory variables :math:`X` using a multiple regression.

    EXPLAIN WHEN TO USE RDA

    Parameters
    ----------
    y : pd.DataFrame
        :math:`n \times p` response matrix, where :math:`n` is the number
        of samples and :math:`p` is the number of features. Its columns
        need be dimensionally homogeneous (or you can set `scale_Y=True`).
        This matrix is also referred to as the community matrix that
        commonly stores information about species abundances
    x : pd.DataFrame
        :math:`n \times m, n \geq m` matrix of explanatory
        variables, where :math:`n` is the number of samples and
        :math:`m` is the number of metadata variables. Its columns
        need not be standardized, but doing so turns regression
        coefficients into standard regression coefficients.
    scale_Y : bool, optional
        Controls whether the response matrix columns are scaled to
        have unit standard deviation. Defaults to `False`.
    scaling : int
        Scaling type 1 (scaling=1) produces a distance biplot. It focuses on
        the ordination of rows (samples) because their transformed
        distances approximate their original euclidean
        distances. Especially interesting when most explanatory
        variables are binary.
        Scaling type 2 produces a correlation biplot. It focuses
        on the relationships among explained variables (`y`). It
        is interpreted like scaling type 1, but taking into
        account that distances between objects don't approximate
        their euclidean distances.
        See more details about distance and correlation biplots in
        [1]_, \S 9.1.4.
    sample_scores_type : str
        Type of sample score to output, either 'lc' and 'wa'.

    Returns
    -------
    Ordination object, Ordonation plot, Screeplot

    See Also
    --------
    ca
    cca

    Notes
    -----
    The algorithm is based on [1]_, \S 11.1.

    References
    ----------
    .. [1] Legendre P. and Legendre L. 1998. Numerical
       Ecology. Elsevier, Amsterdam.
    """

    def __init__(self, scale_Y=True, scaling=1, sample_scores_type='wa',
                 n_permutations = None, seed=None, permutation_method = 'reduced'):
        # initialize the self object

        if not isinstance(scale_Y, bool):
            raise ValueError("scale_Y must be either True or False.")

        if not (scaling == 1 or scaling == 2):
            raise ValueError("scaling must be either 1 (distance analysis) or 2 (correlation analysis).")

        if not (sample_scores_type == 'wa' or sample_scores_type == 'lc'):
            raise ValueError("sample_scores_type must be either 'wa' or 'lc'.")

        self.scale_Y = scale_Y
        self.scaling = scaling
        self.sample_scores_type = sample_scores_type
        self.n_permutations = n_permutations
        self.seed = seed
        self.permutation_method = permutation_method

    def fit(self, X, Y, W=None):

        # I use Y as the community matrix and X as the constraining as_matrix.
        # vegan uses the inverse, which is confusing since the response set
        # is usually Y and the explaination set is usually X.


        # These steps are numbered as in Legendre and Legendre, Numerical Ecology,
        # 3rd edition, section 11.1.3

        # 0) Preparation of data
        feature_ids = X.columns
        sample_ids = X.index # x index and y index should be the same
        response_ids = Y.columns
        X = X.as_matrix() # Constraining matrix, typically of environmental variables
        Y = Y.as_matrix() # Community data matrix
        if W is not None:
            condition_ids = W.columns
            W = W.as_matrix()

        # dimensions
        n_x, m = X.shape
        n_y, p = Y.shape
        if n_x == n_y:
            n = n_x
        else:
            raise ValueError("Tables x and y must contain same number of rows.")

        # scale
        if self.scale_Y:
            Y = (Y - Y.mean(axis=0)) / Y.std(axis=0, ddof=1)
        X = X - X.mean(axis=0)

        # If there is a covariable matrix W, the explanatory matrix X becomes the
        # residuals of a regression between X as response and W as explanatory.
        if W is not None:
            W = W - W.mean(axis=0)
            B_XW = lstsq(W, X)[0]
            X_hat = W.dot(B_XW)
            X_ = X - X_hat # X is now the residual
        else:
            X_ = X

        B = lstsq(X_, Y)[0]
        Y_hat = X_.dot(B)
        Y_res = Y - Y_hat # residuals

        # 3) Perform a PCA on Y_hat
        ## perform singular value decomposition.
        ## eigenvalues can be extracted from u
        ## eigenvectors can be extracted from vt
        u, s, vt = svd(Y_hat, full_matrices=False)
        u_res, s_res, vt_res = svd(Y_res, full_matrices=False)

        # compute eigenvalues from singular values
        eigenvalues = s**2/(n-1)
        eigenvalues_res = s_res**2/(n-1)

        ## determine rank kc
        kc = np.linalg.matrix_rank(Y_hat)
        kc_res = np.linalg.matrix_rank(Y_res)

        ## retain only eigenvs superior to tolerance
        eigenvalues = eigenvalues[:kc]
        eigenvalues_res = eigenvalues_res[:kc_res]
        eigenvalues_values_all = np.r_[eigenvalues, eigenvalues_res]

        trace = np.sum(np.diag(np.cov(Y.T)))
        trace_res = np.sum(np.diag(np.cov(Y_res.T)))
        eigenvectors = vt.T[:,:kc]
        eigenvectors_res = vt_res.T[:,:kc_res]

        ## axes names
        ordi_column_names = ['RDA%d' % (i+1) for i in range(kc)]
        ordi_column_names_res = ['RDA_res%d' % (i+1) for i in range(kc_res)]

        # 4) Ordination of objects (site scores, or vegan's wa scores)
        F = Y.dot(eigenvectors) # columns of F are the ordination vectors
        F_res = Y_res.dot(eigenvectors_res) # columns of F are the ordination vectors

        # 5) F in space X (site constraints, or vegan's lc scores)
        Z = Y_hat.dot(eigenvectors)
        Z_res = Y_res.dot(eigenvectors_res)

        # 6) Correlation between the ordination vectors in spaces Y and X
        rk = np.corrcoef(F, Z) # not used yet
        rk_res = np.corrcoef(F_res, Z_res) # not used yet

        # 7) Contribution of the explanatory variables X to the canonical ordination
        # axes
        # 7.1) C (canonical coefficient): the weights of the explanatory variables X in
        # the formation of the matrix of fitted site scores
        C = B.dot(eigenvectors)  # not used yet
        C_res = B.dot(eigenvectors_res)  # not used yet

        # 7.2) The correlations between X and the ordination vectors in space X are
        # used to represent the explanatory variables in biplots.
        corXZ = corr(X_, Z)
        corXZ_res = corr(X_, Z_res)


        # 8) Compute triplot objects
        # I combine fitted and residuals scores into the DataFrames
        singular_values_all = np.r_[s[:kc], s_res[:kc_res]]
        ordi_column_names_all = ordi_column_names + ordi_column_names_res
        const = np.sum(singular_values_all**2)**0.25

        if self.scaling == 1:
            scaling_factor = const
            D = np.diag(np.sqrt(eigenvalues/trace)) # Diagonal matrix of weights (Numerical Ecology with R, p. 196)
            D_res = np.diag(np.sqrt(eigenvalues_res/trace_res))
        elif self.scaling == 2:
            scaling_factor = singular_values_all / const
            D = np.diag(np.ones(kc)) # Diagonal matrix of weights
            D_res = np.diag(np.ones(kc_res))

        response_scores = pd.DataFrame(np.hstack((eigenvectors, eigenvectors_res)) * scaling_factor,
                                      index=response_ids,
                                      columns=ordi_column_names_all)
        response_scores.index.name = 'ID'


        if self.sample_scores_type == 'wa':
                sample_scores = pd.DataFrame(np.hstack((F, F_res)) / scaling_factor,
                                     index=sample_ids,
                                     columns=ordi_column_names_all)
        elif self.sample_scores_type == 'lc':
                sample_scores = pd.DataFrame(np.hstack((Z, Z_res)) / scaling_factor,
                                     index=sample_ids,
                                     columns=ordi_column_names_all)
        sample_scores.index.name = 'ID'

        biplot_scores = pd.DataFrame(np.hstack((corXZ.dot(D), corXZ_res.dot(D_res))) * scaling_factor,
                                     index=feature_ids,
                                     columns=ordi_column_names_all)

        biplot_scores.index.name = 'ID'

        sample_constraints = pd.DataFrame(np.hstack((Z, F_res)) / scaling_factor,
                                          index=sample_ids,
                                          columns=ordi_column_names_all)
        sample_constraints.index.name = 'ID'

        p_explained = pd.Series(singular_values_all / singular_values_all.sum(), index=ordi_column_names_all)

        # Statistics
        ## Response statistics
        ### Unadjusted R2
        SSY_i = np.sum(Y**2, axis=0)#np.array([np.sum((Y[:, i] - Y[:, i].mean())**2) for i in range(p)])
        SSYhat_i = np.sum(Y_hat**2, axis=0)#np.array([np.sum((Y_hat[:, i] - Y_hat[:, i].mean())**2) for i in range(p)])
        SSYres_i = np.sum(Y_res**2, axis=0)#np.array([np.sum((Y_res[:, i] - Y_res[:, i].mean())**2) for i in range(p)])
        R2_i = SSYhat_i/SSY_i
        R2 = np.mean(R2_i)

        ### Adjusted R2
        R2a_i = 1-((n-1)/(n-m-1))*(1-R2_i)
        R2a = np.mean(R2a_i)

        ### F-statistic
        F_stat_i = (R2_i/m) / ((1-R2_i) / (n-m-1))
        F_stat = (R2/m) / ((1-R2) / (n-m-1))

        response_stats_each = pd.DataFrame({'R2': R2_i, 'Adjusted R2': R2a_i, 'F': F_stat_i},
                                      index = response_ids)
        response_stats_summary = pd.DataFrame({'R2': R2, 'Adjusted R2': R2a, 'F':F_stat},
                                              index = ['Summary'])
        response_stats = pd.DataFrame(pd.concat([response_stats_each, response_stats_summary], axis=0),
                                      columns = ['F', 'R2', 'Adjusted R2'])

        ## Canonical axes statistics
        ### see Legendre et al., 2011, doi 10.1111/j.2041-210X.2010.00078.x
        if self.n_permutations is not None:
            # compute F marginal statistic of the original set
            if W is not None:
                XW = np.c_[X, W]
                B_orig = lstsq(XW, Y)[0]
                Y_hat_orig = XW.dot(B_orig)
                Y_res_orig = Y - Y_hat_orig
                kc_XW = np.linalg.matrix_rank(XW)
                u_orig, s_orig, vt_orig = svd(Y_hat_orig, full_matrices=False)
                eigenvalues_orig = s_orig**2 / (n-1)
                F_marginal_orig = eigenvalues_orig[:kc_XW] * (n-1-m) / np.sum(Y_res_orig**2)
            else:
                kc_XW = kc
                XW = X_
                F_marginal_orig = eigenvalues[:kc_XW] * (n-1-m) / np.sum(Y_res**2)

            # Select the permutaion method
            # Legendre et al., 2011, doi 10.1111/j.2041-210X.2010.00078.x
            # * direct method just permutes Y
            # * reduced method permutes the residuals of Y fitted on W then add the permuted
            #    residuals to the non-permuted Y fitted on W
            # * full method permutes the residuals of Y fitted on the concatenation of X and W
            #    then add the permuted residuals to the non-permuted Y fitted on X and W
            # these method share a permuted part, Y_res_perm and non permuted part
            #    Y_hat_method. For the direct method, the non permuted part is zero and
            #    Y_res_method is just Y
            if self.permutation_method == 'direct':
                Y_res_method = Y
                Y_hat_method = 0
            elif self.permutation_method == 'reduced':
                if W is None:
                    raise ValueError("""
                    Always use permutation_method = 'direct' if W = None (no covariables)
                    """)
                B_method = lstsq(W, Y)[0]
                Y_hat_method = W.dot(B_method)
                Y_res_method = Y - Y_hat_method
            elif self.permutation_method == 'full':
                if W is None:
                    raise ValueError("""
                    Always use permutation_method = 'direct' if W = None (no covariables)
                    """)
                B_method = lstsq(np.c_[X, W], Y)[0]
                Y_hat_method = np.c_[X, W].dot(B_method)
                Y_res_method = Y - Y_hat_method

            # Permutation loop
            np.random.seed(seed=self.seed)
            F_marginal_perm = np.zeros([self.n_permutations, kc_XW])
            axes_ids = ['RDA%d' % (i+1) for i in range(kc_XW)]
            for i in range(self.n_permutations):
                Y_perm = Y_hat_method + Y_res_method[np.random.permutation(n), :]
                B_perm = lstsq(XW, Y_perm)[0]
                Y_hat_perm = XW.dot(B_perm)
                Y_res_perm = Y_perm - Y_hat_perm
                u_perm, s_perm, vt_perm = svd(Y_hat_perm, full_matrices=False)
                eigenvalues_perm = s_perm**2 / (n-1)
                F_marginal_perm[i, :] = eigenvalues_perm[:kc_XW] * (n-1-m) / np.sum(Y_res_perm**2)

            F_marginal_test_elements = np.apply_along_axis(lambda x: x > F_marginal_orig, axis=1, arr=F_marginal_perm)
            pvalues_marginal = F_marginal_test_elements.sum(axis=0) / self.n_permutations
            axes_stats = pd.DataFrame({'F marginal': F_marginal_orig, 'P value (>F)': pvalues_marginal},
                     index=axes_ids)
        else:
            feature_stats = None


        # trace: total variance
        # R2, R2a: R-squared, Adjusted R-Squared
        # eigenvalues
        # C: canonical coefficients
        # biplot_scores:
        # response_scores (eigenvalues): Species scores
        # sample_wa_scores (F): sites scores of "wa" type
        # sample_lc_scores (Z): sites scores of "lc" type, site constraints

        # Add RDA ordination object names to self
        self.ordiobject_type = 'RDA'
        self.method_name = 'Redundancy Analysis'
        self.ordi_fitted = None # To do, see the PCA counterpart
        self.eigenvalues = eigenvalues
        self.proportion_explained = p_explained
        self.response_scores = response_scores
        self.sample_scores = sample_scores
        self.biplot_scores = biplot_scores
        self.sample_constraints = sample_constraints
        self.statistics = {'canonical_coeficient': C, # add feature_stats and response_stats
                                              'r_squared': R2,
                                              'adjusted_r_squared': R2a,
                                              'total_variance': trace}
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
