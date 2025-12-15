User Guide
==========

This guide provides detailed information on using nuee for community ecology analysis.

.. contents::
   :local:
   :depth: 2

Overview
--------

nuee is designed to provide a Pythonic interface to community ecology analyses,
following the conventions of the R vegan package while leveraging the power
of the scientific Python ecosystem.

Data Format
-----------

Community Data
~~~~~~~~~~~~~~

Community data should be provided as a matrix where:

* Rows represent samples (sites, plots, etc.)
* Columns represent species (taxa, OTUs, etc.)
* Values represent abundances (counts, biomass, etc.)

nuee accepts data in several formats:

.. code-block:: python

   import nuee
   import numpy as np
   import pandas as pd

   # NumPy array
   data_array = np.random.rand(10, 20)  # 10 samples, 20 species

   # Pandas DataFrame (recommended)
   data_df = pd.DataFrame(
       data_array,
       index=[f"Site{i}" for i in range(10)],
       columns=[f"Species{i}" for i in range(20)]
   )

Environmental Data
~~~~~~~~~~~~~~~~~~

Environmental data should have the same number of rows as the community data:

.. code-block:: python

   env_data = pd.DataFrame({
       "Temperature": np.random.rand(10),
       "pH": np.random.rand(10),
       "Moisture": np.random.rand(10)
   }, index=data_df.index)

Distance Matrices
~~~~~~~~~~~~~~~~~

Distance matrices should be square, symmetric matrices:

.. code-block:: python

   # Calculate distances
   distances = nuee.vegdist(data_df, method="bray")

Workflow Examples
-----------------

Basic Ordination Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Load and prepare data
2. Choose an ordination method
3. Fit the model
4. Visualize results
5. Interpret

.. code-block:: python

   import nuee
   import matplotlib.pyplot as plt

   # 1. Load data
   species = nuee.datasets.varespec()
   env = nuee.datasets.varechem()

   # 2. Choose method (NMDS)
   # 3. Fit the model
   nmds_result = nuee.metaMDS(species, k=2)

   # 4. Visualize
   fig = nuee.plot_ordination(nmds_result)
   plt.show()

   # 5. Interpret stress value
   print(f"Stress: {nmds_result.stress:.3f}")
   # Stress < 0.05: excellent
   # Stress < 0.10: good
   # Stress < 0.20: acceptable
   # Stress > 0.20: poor

.. note::

   ``nuee.metaMDS`` follows vegan's data transformations and SMACOF optimisation,
   but the underlying implementation is still evolving. Recent regression tests
   show small differences in the reported stress compared to ``vegan::metaMDS``.
   This does not invalidate the ordination, but if you require vegan-identical
   results you should re-run the analysis in R for the time being.

Diversity Analysis Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import nuee
   import pandas as pd

   # Load data
   species = nuee.datasets.BCI()

   # Calculate multiple diversity indices
   diversity_df = pd.DataFrame({
       "Shannon": nuee.shannon(species),
       "Gini-Simpson": nuee.simpson(species),
       "Richness": nuee.specnumber(species),
       "Fisher": nuee.fisher_alpha(species)
   })

   # Summary statistics
   print(diversity_df.describe())

   # Compare groups
   # If you have grouping information
   # diversity_by_group = diversity_df.groupby(groups).mean()

Hypothesis Testing Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doctest::

   >>> import nuee

   >>> # Load data
   >>> species = nuee.datasets.dune()
   >>> env = nuee.datasets.dune_env()

   >>> # Calculate distances
   >>> dist = nuee.vegdist(species, method="bray")

   >>> # Test for group differences (PERMANOVA)
   >>> perm_result = nuee.adonis2(dist, env['Management'])
   >>> print(f"R^2: {perm_result.R2[0]:.3f}")
   R^2: 0.342
   >>> print(f"p-value: {perm_result["Pr(>F)"][0]:.3f}") # doctest: +ELLIPSIS
   p-value: ...

   >>> # Test for homogeneity of dispersions
   >>> betadisp = nuee.betadisper(dist, env['Management'])
   >>> print(betadisp) # doctest: +SKIP

Tips and Best Practices
-----------------------

Choosing an Ordination Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **NMDS**: Robust, works with any distance metric, no linearity assumptions
* **RDA**: Linear relationships, environmental variables available
* **CCA**: Unimodal relationships, long environmental gradients
* **PCA**: Quick exploration, linear relationships

Choosing a Distance Metric
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Bray-Curtis**: General purpose, abundance data
* **Jaccard**: Presence/absence data
* **Euclidean**: Environmental data, PCA
* **Hellinger**: Before RDA, avoids double-zero problem

Data Transformation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   # Hellinger transformation (for RDA)
   def hellinger(x):
       row_sums = x.sum(axis=1, keepdims=True)
       return np.sqrt(x / row_sums)

   # Wisconsin double standardization
   def wisconsin(x):
       # By species maxima
       x_std = x / x.max(axis=0)
       # By site totals
       x_std = x_std / x_std.sum(axis=1, keepdims=True)
       return x_std

Compositional Data Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``nuee.composition`` brings compositional data analysis tools into the package
without requiring SciPy.  These utilities are NumPy-only ports of scikit-bio's
composition module.

.. code-block:: python

   from nuee import composition
   import numpy as np

   # Raw counts with zeros
   counts = np.array([[0, 5, 10], [3, 0, 9]])

   # Replace zeros and apply closure
   replaced = composition.multiplicative_replacement(counts)
   closed = composition.closure(replaced)

   # Transform to log-ratio space
   clr_coords = composition.clr(closed)
   ilr_coords = composition.ilr(closed)

   # Invert transforms if required
   recovered = composition.ilr_inv(ilr_coords)

Mathematical Definitions
------------------------

The following formulas summarise the core quantities computed by nuee.

Shannon Diversity
~~~~~~~~~~~~~~~~~

.. math::

   H' = -\sum_{i=1}^{S} p_i \ln p_i

where :math:`p_i = \frac{x_i}{\sum_{j=1}^{S} x_j}` is the relative abundance of
species :math:`i` in a community of size :math:`S`.

Gini-Simpson Diversity
~~~~~~~~~~~~~~~~~~~~~~

.. math::

   D = 1 - \sum_{i=1}^{S} p_i^2

which measures the probability that two individuals drawn at random belong to
different species.

Bray-Curtis Dissimilarity
~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   BC_{ij} = 1 - \frac{2 \sum_{k=1}^{S} \min(x_{ik}, x_{jk})}{\sum_{k=1}^{S} x_{ik} + \sum_{k=1}^{S} x_{jk}}

where :math:`x_{ik}` and :math:`x_{jk}` denote the abundances of species
:math:`k` in sites :math:`i` and :math:`j`.

Hellinger Transformation
~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   h_{ik} = \sqrt{\frac{x_{ik}}{\sum_{j=1}^{S} x_{ij}}}

which stabilises variances prior to linear ordination methods such as RDA.

PERMANOVA F-statistic
~~~~~~~~~~~~~~~~~~~~~

.. math::

   F = \frac{SS_{\text{between}} / (g - 1)}{SS_{\text{within}} / (N - g)}

where :math:`g` is the number of groups and :math:`N` is the number of
observations. Permutation p-values are obtained by recalculating :math:`F`
across random group assignments.

Interpretation Guidelines
-------------------------

NMDS Stress Values
~~~~~~~~~~~~~~~~~~

* < 0.05: Excellent representation
* 0.05 - 0.10: Good representation
* 0.10 - 0.20: Acceptable
* > 0.20: Poor (try different k or method)

RDA/CCA Interpretation
~~~~~~~~~~~~~~~~~~~~~~

* Eigenvalues: Variance explained by each axis
* Species scores: Optimal position for each species
* Site scores: Position of each site
* Environmental vectors: Direction and strength of correlation

PERMANOVA Results
~~~~~~~~~~~~~~~~~

* R^2: Proportion of variance explained
* F-statistic: Ratio of between-group to within-group variance
* p-value: Significance (typically alpha = 0.05)
