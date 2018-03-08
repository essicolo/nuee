%load_ext autoreload
%autoreload 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
iris = sns.load_dataset("iris")


# PCA
## The warning comes from statsmodel loaded by plotnine. It can be safely
## ignored
from nuee.ordination import PrincipalComponentsAnalysis as pca
pca_results = pca(scaling=1)
pca_results.fit(iris.iloc[:, :4]);
pca_results.ordiplot(axes=[0,1], group=iris.species, sample_scatter='points', level=0.99)

pca_results.screeplot()


# LDA
from plotnine import *
from nuee.ordination import LinearDiscriminantAnalysis as lda
lda_results = lda(solver='svd')
lda_results.fit(iris.iloc[:, :4], iris.species);
lda_results.ordiplot() + theme_bw()
lda_results.screeplot()


# RDA
varechem = pd.read_csv('data/varechem.csv', index_col=0, delimiter=';')
varespec = pd.read_csv('data/varespec.csv', index_col=0, delimiter=';')
from nuee.ordination import RedundancyAnalysis as rda
rda_results = rda(scale_Y=True, scaling=1)
rda_results.fit(X=varechem, Y=varespec);
rda_results.ordiplot(axes=[0,1], sample_scatter='labels')
rda_results.screeplot()


rda_results = rda(scale_Y=True, scaling=1, n_permutations = 99,
                  permutation_method = 'reduced')
rda_results.fit(X=doubs_env[['pH', 'dur', 'pho', 'nit', 'amm', 'oxy', 'dbo']],
                Y=doubs_species,
                W=doubs_env[['das', 'alt', 'pen', 'deb']]);
rda_results.ordiplot(axes=[0,1], sample_scatter='labels')




# CoDa
varechem = pd.read_csv('data/varechem.csv',
                       index_col=0, delimiter=';')

from nuee.stats import coda

parts = varechem.loc[:, ['N', 'P', 'K']]
comp = coda.closure(parts)
tern = coda.PlotTriangle(labels = parts.columns)
plt.figure(figsize=(8, 8));
tern.plot_triangle()
coda.plot_comp(comp)

# Testing RDA

from nuee.stats._utils import *
import pandas as pd
import numpy as np
from scipy.linalg import svd, lstsq
from nuee.ordination import RedundancyAnalysis as rda
%matplotlib inline

doubs_species = pd.read_csv('data/DoubsSpe.csv', index_col=0)
doubs_env = pd.read_csv('data/DoubsEnv.csv', index_col=0)

Y = doubs_species
X = doubs_env[['pH', 'dur', 'pho', 'nit', 'amm', 'oxy', 'dbo']]
W = doubs_env[['das', 'alt', 'pen', 'deb']]
scaling=1
scale_Y = True
n_permutations = 999
seed = None
permutation_method = 'direct'
sample_scores_type = 'wa'

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
if scale_Y:
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0, ddof=1)
X = (X - X.mean(axis=0))

# If there is a covariable matrix W, the explanatory matrix X becomes the
# residuals of a regression between X as response and W as explanatory.
if W is not None:
    W = (W - W.mean(axis=0)) / W.std(axis=0, ddof=1)
    B_XW = lstsq(W, X)[0]
    X_hat = W.dot(B_XW)
    X_ = X - X_hat # X is now the residual
else:
    X_ = X

B = lstsq(X_, Y)[0]
Y_hat = X.dot(B)
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

if scaling == 1:
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


sample_scores = pd.DataFrame(np.hstack((F, F_res)) / scaling_factor,
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

## Feature statistics
### see Legendre et al., 2011, doi 10.1111/j.2041-210X.2010.00078.x
if n_permutations is not None:
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
        stats_ids = ['RDA%d' % (i+1) for i in range(kc_XW)]#np.r_[feature_ids, condition_ids]
    else:
        kc_XW = kc
        XW = X_
        F_marginal_orig = eigenvalues[:kc_XW] * (n-1-m) / np.sum(Y_res**2)
        stats_ids = feature_ids

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
    if permutation_method == 'direct':
        Y_res_method = Y
        Y_hat_method = 0
    elif permutation_method == 'reduced':
        if W is None:
            raise ValueError("""
            Always use permutation_method = 'direct' if W = None (no covariables)
            """)
        B_method = lstsq(W, Y)[0]
        Y_hat_method = W.dot(B_method)
        Y_res_method = Y - Y_hat_method
    elif permutation_method == 'full':
        if W is None:
            raise ValueError("""
            Always use permutation_method = 'direct' if W = None (no covariables)
            """)
        B_method = lstsq(np.c_[X, W], Y)[0]
        Y_hat_method = np.c_[X, W].dot(B_method)
        Y_res_method = Y - Y_hat_method

    # Permutation loop
    np.random.seed(seed=seed)
    F_marginal_perm = np.zeros([n_permutations, kc_XW])
    for i in range(n_permutations):
        Y_perm = Y_hat_method + Y_res_method[np.random.permutation(n), :]
        B_perm = lstsq(XW, Y_perm)[0]
        Y_hat_perm = XW.dot(B_perm)
        Y_res_perm = Y_perm - Y_hat_perm
        u_perm, s_perm, vt_perm = svd(Y_hat_perm, full_matrices=False)
        eigenvalues_perm = s_perm**2 / (n-1)
        F_marginal_perm[i, :] = eigenvalues_perm[:kc_XW] * (n-1-m) / np.sum(Y_res_perm**2)

    F_marginal_test_elements = np.apply_along_axis(lambda x: x > F_marginal_orig, axis=1, arr=F_marginal_perm)
    pvalues_marginal = F_marginal_test_elements.sum(axis=0) / n_permutations
    axes_stats = pd.DataFrame({'F marginal': F_marginal_orig, 'P value (>F)': pvalues_marginal},
             index=stats_ids)
else:
    axes_stats = None
