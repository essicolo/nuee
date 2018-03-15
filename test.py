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

# RDA with another dataset

doubs_species = pd.read_csv('data/DoubsSpe.csv', index_col=0)
doubs_env = pd.read_csv('data/DoubsEnv.csv', index_col=0)

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



Y = doubs_species
X = doubs_env[['pH', 'dur', 'pho', 'nit', 'amm', 'oxy', 'dbo']]
W = doubs_env[['das', 'alt', 'pen', 'deb']]
scaling=1
scale_Y = True
n_permutations = 999
seed = None
permutation_method = 'reduced'
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
X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1) # !!!!!!!!!!


# If there is a covariable matrix W, the explanatory matrix X becomes the
# residuals of a regression between X as response and W as explanatory.
if W is not None:
    W = (W - W.mean(axis=0))# / W.std(axis=0, ddof=1) !!!!!!!!!
    B_XW = lstsq(W, X)[0]
    X_hat = W.dot(B_XW)
    X_ = X - X_hat # X_ is now the residual
else:
    X_ = X

# at this point, X_ is the residual of X if W exists. Else, it's X.
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

## trace is the total variance (of Y and Y_res)
trace = np.sum(np.diag(np.cov(Y.T)))
trace_res = np.sum(np.diag(np.cov(Y_res.T)))

## eigenvectors
eigenvectors = vt.T[:,:kc]
eigenvectors_res = vt_res.T[:,:kc_res]

## cannonical axes used to compute F_marginal
axes = u[:, :kc]

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
SSY_i = np.sum(Y**2, axis=0)
SSYhat_i = np.sum(Y_hat**2, axis=0)
SSYres_i = np.sum(Y_res**2, axis=0)
R2_i = SSYhat_i/SSY_i
R2 = np.mean(R2_i)

### Adjusted R2 (not valid if W is not None, Legendre et al. 2011)
if W is None:
    R2a_i = np.repeat(np.nan, R2_i.shape[0])
    R2a = np.nan
else:
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
n_permutations = 999
if n_permutations is not None:
    # compute F marginal statistic of the original set
    if W is not None:
        # Compute F_marginal
        XW = np.c_[X_, W]
        B_original = lstsq(XW, Y)[0]
        Y_hat_original = XW.dot(B_original)
        kc_XW = np.linalg.matrix_rank(XW)
        F_marginal_original = s[:kc]**2 / (np.sum(Y**2) - np.sum(Y_hat_original**2)) # recall that s is the singular value of the original RDA
        stats_ids = ['RDA%d' % (i+1) for i in range(kc)]
    else:
        kc_XW = kc
        XW = X_
        F_marginal_original = s[:kc_XW]**2 / (np.sum(Y**2) - np.sum(Y_hat**2))
        stats_ids = feature_ids

    # Canonical axes are tested in sequence
    canonical_axis_pvalue = np.array([])
    for i in range(kc):
        features_i = np.c_[W, axes[:, :i]] # if i==0, then np.c_[W, X[:, :i]] == W
        B_i = lstsq(features_i, Y)[0]
        Y_hat_i = features_i.dot(B_i)
        Y_res_i = Y - Y_hat_i

        B_fX_ = lstsq(features_i, X_)[0]
        X_hat = features_i.dot(B_fX_)
        X_res = X_ - X_hat

        F_j = np.array([])
        for j in range(n_permutations):
            #Y_j = Y_hat_i + Y_res_i[np.random.permutation(n), :] # reduced permutation model
            Y_j = Y_hat_i + Y_res_i[pd.read_csv('sample_n.csv').iloc[:,0].as_matrix()-1, :] # reduced permutation model
            B_Xres_j = lstsq(X_res, Y_j)[0] # not working with i=1, huge coefficients (~1E12), maybe X_res is too collinear, but how to solve this?????
            Y_hat_j = X_res.dot(B_Xres_j)
            u_j, s_j, vt_j = svd(Y_hat_j, full_matrices=False)
            B_tot_j = lstsq(XW, Y_j)[0]
            Y_hat_tot_j = XW.dot(B_tot_j)
            F_j = np.r_[F_j, s_j[0]**2 / (np.sum(Y_j**2) - np.sum(Y_hat_tot_j**2))]

        canonical_axis_pvalue = np.r_[canonical_axis_pvalue, (1 + np.sum(F_j >= F_marginal_original[i])) / (1 + n_permutations)]

    axes_stats = pd.DataFrame({'F marginal': F_marginal_original[:kc]  * (n-1-kc_XW), 'P value (>F)': canonical_axis_pvalue},
             index=stats_ids)
else:
    axes_stats = None

axes_stats
