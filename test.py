%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import nuee
import nuee

# PCA
import seaborn as sns
from nuee.ordination import PrincipalComponentsAnalysis as pca
iris = sns.load_dataset("iris")
pca_results = pca(scaling=1)
pca_results.fit(iris.iloc[:, :4]);
pca_results.ordiplot(axes=[0,1])

pca_results.screeplot()


# LDA
from nuee.ordination import LinearDiscriminantAnalysis as lda
lda_results = lda(solver='svd')
lda_results.fit(iris.iloc[:, :4], iris.species);
lda_results.ordiplot(axes=[0,1])

lda_results.screeplot()


# RDA
varechem = pd.read_csv('https://gist.githubusercontent.com/essicolo/087e49c8feb436f45df5d9e8fa9597f8/raw/d708dea9a107d453dec3073bc7daf70e85870e2c/varechem.csv',
                       index_col=0)
varespec = pd.read_csv('https://gist.githubusercontent.com/essicolo/cd5c8b77c91e14b9fe648d63f9afaed9/raw/4c44a3d6e39f3d9cfb21aa0ac2d99c13c2997fcb/varespec.csv',
                       index_col=0)

from nuee.ordination import RedundancyAnalysis as rda
rda_results = rda(scale_Y=True, scaling=1)
rda_results.fit(X=varechem, Y=varespec);
rda_results.ordiplot(axes=[0,1])

rda_results.screeplot()


# multinormal (mardia) - OK
# pairsplot - plus tard
# PCA - OK
# pcoa
# lda
# cca
# ca
# rda - OK
# impute
# envfit
# diversity --> scikit bio
# rarefaction
# ordiplot - OK (manque lda avec ellipses)
## option: show_sample_scores_as = 'index' or 'points'
# screeplot - OK
# ellipse - OK
# coda
