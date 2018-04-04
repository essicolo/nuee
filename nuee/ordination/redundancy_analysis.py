import numpy as np
import pandas as pd
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
                 n_permutations = 199, permute_by=[], seed=None):
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
        self.permute_by = permute_by
        self.seed = seed

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
            q = W.shape[1] # number of covariables (used in permutations)
        else:
            q=0

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
        X = X - X.mean(axis=0)# / X.std(axis=0, ddof=1)
        # Note: Legendre 2011 does not scale X.

        # If there is a covariable matrix W, the explanatory matrix X becomes the
        # residuals of a regression between X as response and W as explanatory.
        if W is not None:
            W = (W - W.mean(axis=0))# / W.std(axis=0, ddof=1)
            # Note: Legendre 2011 does not scale W.
            B_XW = np.linalg.lstsq(W, X)[0]
            X_hat = W.dot(B_XW)
            X_ = X - X_hat # X is now the residual
        else:
            X_ = X

        B = np.linalg.lstsq(X_, Y)[0]
        Y_hat = X_.dot(B)
        Y_res = Y - Y_hat # residuals

        # 3) Perform a PCA on Y_hat
        ## perform singular value decomposition.
        ## eigenvalues can be extracted from u
        ## eigenvectors can be extracted from vt
        u, s, vt = np.linalg.svd(Y_hat, full_matrices=False)
        u_res, s_res, vt_res = np.linalg.svd(Y_res, full_matrices=False)

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

        ## cannonical axes used to compute F_marginal
        canonical_axes = u[:, :kc]

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

        ## Canonical axis statistics
        """
        the permutation algorithm is inspired by the supplementary material
        published i Legendre et al., 2011, doi 10.1111/j.2041-210X.2010.00078.x
        """
        if 'axes' in self.permute_by:

            if W is None:
                F_m = s[0]**2 / (np.sum(Y**2) - np.sum(Y_hat**2))
                F_m_perm = np.array([])
                for j in range(self.n_permutations):
                    Y_perm = Y[np.random.permutation(n), :] # full permutation model
                    B_perm = np.linalg.lstsq(X_, Y_perm)[0]
                    Y_hat_perm = X_.dot(B_perm)
                    s_perm = np.linalg.svd(Y_hat_perm, full_matrices=False)[1]
                    F_m_perm = np.r_[F_m_perm, s_perm[0]**2 / (np.sum(Y_perm**2) - np.sum(Y_hat_perm**2))]

                F_marginal = F_m
                p_values = (1 + np.sum(F_m_perm >= F_m)) / (1 + self.n_permutations)
                begin = 1
            else:
                F_marginal = np.array([])
                p_values = np.array([])
                begin = 0

            if (W is not None) or (W is None and kc > 1):
                if W is None:
                    XW = X_
                else:
                    XW = np.c_[X_, W]

                # Compute F_marginal
                B_XW = np.linalg.lstsq(XW, Y)[0]
                Y_hat_XW = XW.dot(B_XW)
                kc_XW = np.linalg.matrix_rank(XW)
                F_marginal_XW = s[begin:kc]**2 / (np.sum(Y**2) - np.sum(Y_hat_XW**2))
                F_marginal = np.r_[F_marginal, F_marginal_XW]

                for i in range(begin, kc):
                    # set features to compute the object to fit with Y_perm
                    # and to compute Y_perm from Y_res_i
                    if W is None:
                        features_i = np.c_[np.repeat(1, n), canonical_axes[:, :i]]
                    else:
                        features_i = np.c_[W, canonical_axes[:, :i]] # if i==0, then np.c_[W, X[:, :i]] == W

                    B_fX_ = np.linalg.lstsq(features_i, X_)[0]
                    X_hat_i = features_i.dot(B_fX_)
                    X_res_i = X_ - X_hat_i

                    # to avoid collinearity
                    X_res_i = X_res_i[:, :np.linalg.matrix_rank(X_res_i)]

                    # find Y residuals for permutations with residuals model (only model available)
                    B_i = np.linalg.lstsq(features_i, Y)[0] # coefficients for axis i
                    Y_hat_i = features_i.dot(B_i) # Y estimation for axis i
                    Y_res_i = Y - Y_hat_i # Y residuals for axis i

                    F_m_perm = np.array([])
                    for j in range(self.n_permutations):
                        Y_perm = Y_hat_i + Y_res_i[np.random.permutation(n), :] # reduced permutation model
                        B_perm = np.linalg.lstsq(X_res_i, Y_perm)[0]
                        Y_hat_perm = X_res_i.dot(B_perm)
                        u_perm, s_perm, vt_perm = np.linalg.svd(Y_hat_perm, full_matrices=False)
                        B_tot_perm = np.linalg.lstsq(XW, Y_perm)[0]
                        Y_hat_tot_perm = XW.dot(B_tot_perm)
                        F_m_perm = np.r_[F_m_perm, s_perm[0]**2 / (np.sum(Y_perm**2) - np.sum(Y_hat_tot_perm**2))]

                    p_values = np.r_[p_values, (1 + np.sum(F_m_perm >= F_marginal[i])) / (1 + self.n_permutations)]

            axes_stats = pd.DataFrame({'F marginal': F_marginal  * (n-1-kc_XW), 'P value (>F)': p_values},
                                      index=['RDA%d' % (i+1) for i in range(kc)])
        else:
            axes_stats = None

        if 'features' in self.permute_by:
            p_values_coef = np.array([]) # initiate empty vector for p-values
            F_coef = np.array([]) # initiate empty vector for F-scores
            for i in range(X_.shape[1]):
                feature_i = np.c_[X_[:, i]] # isolate the explanatory variable to test
                B_i = np.linalg.lstsq(feature_i, Y)[0] # coefficients for variable i
                Y_hat_i = feature_i.dot(B_i) # Y estimation for explanatory variable i
                Y_res_i = Y - Y_hat_i # Y residuals for variable i
                if W is None:
                    rsq_i = np.sum(Y_hat_i**2) / np.sum(Y**2) # r-square for variable i
                    F_coef = np.r_[F_coef, (rsq_i/m) / ((1-rsq_i) / (n-m-1))] # F-score for variable i, from eq. 7 in LOtB, 2011
                else:
                    F_coef = np.r_[F_coef, (np.sum(Y_hat_i**2) / m) / (np.sum(Y_res_i**2) / (n-m-q-1))] # F-score for variable i, from eq 8 in LOtB, 2011

                F_coef_perm = np.array([]) # initiate permutation vector of F-scores
                for j in range(self.n_permutations):
                    Y_perm = Y_hat_i + Y_res_i[np.random.permutation(n), :] # reduced permutation model
                    B_perm = np.linalg.lstsq(feature_i, Y_perm)[0] # fit the permuted model to obtain coefficients
                    Y_hat_perm = feature_i.dot(B_perm) # obtain estimated Y matrix
                    Y_res_perm = Y_perm - Y_hat_perm
                    if W is None:
                        rsq_perm = np.sum(Y_hat_perm**2) / np.sum(Y_perm**2) # r-suare of regression
                        F_coef_perm = np.r_[F_coef_perm, (rsq_perm/m) / ((1-rsq_perm) / (n-m-1))] # F-score of permuted regression
                    else:
                        F_coef_perm = np.r_[F_coef_perm, (np.sum(Y_hat_perm**2) / m) / (np.sum(Y_res_perm**2) / (n-m-q-1))] # F-score for variable i, from eq 8 in LOtB, 2011

                p_values_coef = np.r_[p_values_coef, (1 + np.sum(F_coef_perm >= F_coef[i])) / (1 + self.n_permutations)]

            coef_stats = pd.DataFrame({'F': F_coef  * (n-m-q-1), 'P value (>F)': p_values_coef}, index=feature_ids)
        else:
            coef_stats = None

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
        self.statistics = {'canonical_coeficient': C,
                           'r_squared': R2,
                           'adjusted_r_squared': R2a,
                           'total_variance': trace,
                           'axes': axes_stats,
                           'features': coef_stats}
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
