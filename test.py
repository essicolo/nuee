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
from nuee.ordination import RedundancyAnalysis as rda

## Case 1: verechem and varespec datasets
### data
import pandas as pd
varechem = pd.read_csv('data/varechem.csv', index_col=0, delimiter=';')
varespec = pd.read_csv('data/varespec.csv', index_col=0, delimiter=';')

### Compute RDA
rda_results = rda(scale_Y=True, scaling=1, n_permutations=999)
rda_results.fit(X=varechem, Y=varespec);
rda_results.ordiplot(axes=[0,1], sample_scatter='labels')
rda_results.screeplot()

## Case 2: Doubs data set
### data
doubs_species = pd.read_csv('data/DoubsSpe.csv', index_col=0)
doubs_env = pd.read_csv('data/DoubsEnv.csv', index_col=0)

###  Compute RDA with condition W
rda_results = rda(scale_Y=True, scaling=1, permute_by=['axes', 'features'], n_permutations = 99)
rda_results.fit(X=doubs_env[['pH', 'dur', 'pho', 'nit', 'amm', 'oxy', 'dbo']],
                Y=doubs_species,
                W=doubs_env[['das', 'alt', 'pen', 'deb']])
rda_results.statistics['axes']
rda_results.statistics['features']
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
