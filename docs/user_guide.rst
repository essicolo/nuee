User Guide
==========

This guide provides detailed information on using nuee for community ecology analysis.

.. toctree::
   :maxdepth: 2

   user_guide/ordination
   user_guide/diversity
   user_guide/dissimilarity
   user_guide/permutation_tests
   user_guide/plotting

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
       'Temperature': np.random.rand(10),
       'pH': np.random.rand(10),
       'Moisture': np.random.rand(10)
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
~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import nuee
   import pandas as pd

   # Load data
   species = nuee.datasets.BCI()

   # Calculate multiple diversity indices
   diversity_df = pd.DataFrame({
       'Shannon': nuee.shannon(species),
       'Gini-Simpson': nuee.simpson(species),
       'Richness': nuee.specnumber(species),
       'Fisher': nuee.fisher_alpha(species)
   })

   # Summary statistics
   print(diversity_df.describe())

   # Compare groups
   # If you have grouping information
   # diversity_by_group = diversity_df.groupby(groups).mean()

Hypothesis Testing Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import nuee

   # Load data
   species = nuee.datasets.dune()
   env = nuee.datasets.dune_env()

   # Calculate distances
   dist = nuee.vegdist(species, method="bray")

   # Test for group differences (PERMANOVA)
   perm_result = nuee.adonis2(dist, env['Management'])
   print(f"R²: {perm_result.r_squared:.3f}")
   print(f"p-value: {perm_result.p_value:.3f}")

   # Test for homogeneity of dispersions
   betadisp = nuee.betadisper(dist, env['Management'])
   print(betadisp)

Tips and Best Practices
------------------------

Choosing an Ordination Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **NMDS**: Robust, works with any distance metric, no linearity assumptions
* **RDA**: Linear relationships, environmental variables available
* **CCA**: Unimodal relationships, long environmental gradients
* **PCA**: Quick exploration, linear relationships

Choosing a Distance Metric
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Interpretation Guidelines
--------------------------

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

* R²: Proportion of variance explained
* F-statistic: Ratio of between-group to within-group variance
* p-value: Significance (typically α = 0.05)
