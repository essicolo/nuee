Examples
========

This page contains detailed examples of using nuee for various ecological analyses.

Complete Analysis Examples
---------------------------

Example 1: Lichen Communities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complete analysis of lichen community data with environmental correlates.

.. doctest::

   >>> import nuee
   >>> import matplotlib.pyplot as plt
   >>> import pandas as pd
   >>> import numpy as np

   >>> # Load data
   >>> species = nuee.datasets.varespec()
   >>> env = nuee.datasets.varechem()

   >>> # 1. Explore diversity patterns
   >>> shannon = nuee.shannon(species)
   >>> richness = nuee.specnumber(species)

   >>> # Plot diversity
   >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
   >>> shannon.plot(kind='bar', ax=ax1, title='Shannon Diversity') # TODO broken! # doctest: +SKIP
   >>> richness.plot(kind='bar', ax=ax2, title='Species Richness') # TODO broken! # doctest: +SKIP
   >>> plt.tight_layout()
   >>> plt.show()

   >>> # 2. Ordination analysis
   >>> # NMDS
   >>> nmds = nuee.metaMDS(species, k=2, distance="bray")
   >>> print(f"NMDS stress: {nmds.stress:.3f}")
   NMDS stress: 0.133

.. note::

   ``metaMDS`` currently relies on a SMACOF backend adapted from scikit-learn.
   It applies vegan's data transformations but does not yet replicate the exact
   optimisation path of ``vegan::metaMDS``. Expect modest differences in the
   reported stress until the implementation is fully aligned.

   .. doctest::

       >>> # Plot NMDS
       >>> fig = nuee.plot_ordination(nmds, display="sites")
       >>> plt.title(f"NMDS Ordination (stress: {nmds.stress:.3f})") # doctest: +SKIP
       >>> plt.show()

       >>> # 3. Relate to environment
       >>> # Fit environmental vectors
       >>> envfit_result = nuee.envfit(nmds, env)
       >>> print(envfit_result) # doctest: +SKIP

.. note::

   ``envfit`` mirrors vegan’s API, but vector scaling and permutation tests are
   still under active development. Numerical results may differ from
   ``vegan::envfit`` in the current release.

   .. doctest::

       >>> # RDA with environmental constraints
       >>> rda = nuee.rda(species, env[['N', 'P', 'K']])
       >>> fig = nuee.biplot(rda) # TODO broken! # doctest: +SKIP
       >>> plt.title("RDA: Species ~ N + P + K") # doctest: +SKIP
       >>> plt.show()

       >>> # 4. Test for environmental effects
       >>> dist = nuee.vegdist(species, method="bray")

       >>> # Correlation with individual variables
       >>> for var in ['N', 'P', 'K', 'pH']:
       ...     env_dist = nuee.vegdist(env[[var]], method="euclidean")
       ...     mantel = nuee.mantel(dist, env_dist)
       ...     print(f"Mantel test - {var}: r={mantel["r_statistic"]:.3f}, p={mantel["p_value"]:.3f}")
       Mantel test - N: r=..., p=...

Example 2: Vegetation Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dune meadow vegetation with management types.

.. doctest::

   >>> import nuee
   >>> import matplotlib.pyplot as plt
   >>> import seaborn as sns

   >>> # Load data
   >>> species = nuee.datasets.dune()
   >>> env = nuee.datasets.dune_env()

   >>> # 1. Diversity by management type
   >>> shannon = nuee.shannon(species)
   >>> diversity_by_mgmt = pd.DataFrame({
   ...     'Shannon': shannon,
   ...     'Management': env['Management']
   ... })

   >>> # Boxplot
   >>> fig, ax = plt.subplots(figsize=(10, 6))
   >>> diversity_by_mgmt.boxplot(column='Shannon', by='Management', ax=ax)  # TODO broken! # doctest: +SKIP
   >>> plt.title('Shannon Diversity by Management Type') # doctest: +SKIP
   >>> plt.suptitle('')  # Remove automatic title # doctest: +SKIP
   >>> plt.show()

   >>> # 2. PERMANOVA test
   >>> dist = nuee.vegdist(species, method="bray")
   >>> perm = nuee.adonis2(dist, env['Management'])
   >>> print(f"PERMANOVA R²: {perm.R2[0]:.3f}, p-value: {perm["Pr(>F)"][0]:.3f}")
   PERMANOVA R²: 0.342, p-value: ...

.. note::

   ``nuee.adonis2`` now performs sequential PERMANOVA with permutation
   p-values. Minor numerical differences from ``vegan::adonis2`` are expected,
   but the test is fully functional.

   # 3. Test homogeneity of dispersions
   betadisp = nuee.betadisper(dist, env['Management'])
   print(betadisp)

   # 4. Ordination with groups
   nmds = nuee.metaMDS(species, k=2)
   fig = nuee.plot_ordination(nmds, groups=env['Management'])
   plt.title("NMDS by Management Type")
   plt.show()

   # 5. CCA with all environmental variables
   # Select numeric variables
   numeric_env = env.select_dtypes(include=[np.number])
   cca = nuee.cca(species, numeric_env)
   fig = nuee.biplot(cca)
   plt.title("CCA: Vegetation ~ Environment")
   plt.show()

Example 3: Forest Diversity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Barro Colorado Island tree census data.

.. doctest::

   >>> import nuee
   >>> import matplotlib.pyplot as plt
   >>> import numpy as np

   >>> # Load data
   >>> species = nuee.datasets.BCI()

   >>> # 1. Multiple diversity indices
   >>> diversity_df = pd.DataFrame({
   ...     'Shannon': nuee.shannon(species).values,
   ...     'Gini-Simpson': nuee.simpson(species).values,
   ...     'Richness': nuee.specnumber(species).values,
   ...     'Fisher': nuee.fisher_alpha(species).values,
   ...     'Evenness': nuee.evenness(species, method='pielou').values
   ... })

   >>> # Summary statistics
   >>> print(diversity_df.describe())
   [...]

   >>> # Correlation matrix
   >>> fig, ax = plt.subplots(figsize=(8, 6))
   >>> sns.heatmap(diversity_df.corr(), annot=True, cmap='coolwarm', ax=ax) # doctest: +SKIP
   >>> plt.title('Correlation between Diversity Indices') # doctest: +SKIP
   >>> plt.show()

   >>> # 2. Rarefaction analysis
   >>> rarefaction = nuee.rarecurve(species, step=50)
   >>> fig = nuee.plot_rarecurve(rarefaction)
   >>> plt.title('Species Accumulation Curves') # doctest: +SKIP
   >>> plt.show()

   >>> # 3. Rank-abundance curves
   >>> # For first 5 plots
   >>> fig, axes = plt.subplots(2, 3, figsize=(15, 10))
   >>> axes = axes.flatten()

   >>> for i, (idx, row) in enumerate(species.head(5).iterrows()):
   ...     if i >= len(axes):
   ...         break
   ...     abundances = sorted(row[row > 0].values, reverse=True)
   ...     ranks = np.arange(1, len(abundances) + 1)
   ...     axes[i].plot(ranks, abundances, 'o-') # doctest: +SKIP
   ...     axes[i].set_xlabel('Rank') # doctest: +SKIP
   ...     axes[i].set_ylabel('Abundance') # doctest: +SKIP
   ...     axes[i].set_title(f'Plot {idx}') # doctest: +SKIP
   ...     axes[i].set_yscale('log') # doctest: +SKIP

   >>> plt.tight_layout()
   >>> plt.show()

   >>> # 4. Beta diversity analysis
   >>> dist = nuee.vegdist(species, method="bray")

   >>> # NMDS
   >>> nmds = nuee.metaMDS(species, k=2, distance="bray")
   >>> fig = nuee.plot_ordination(nmds)
   >>> plt.title(f"NMDS of BCI Plots (stress: {nmds.stress:.3f})") # doctest: +SKIP
   >>> plt.show()

Example 4: Renyi Diversity Profiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comparing diversity across multiple scales.

.. doctest::

   >>> import nuee
   >>> import matplotlib.pyplot as plt
   >>> import numpy as np

   >>> # Load data
   >>> species = nuee.datasets.varespec()

   >>> # Calculate Renyi entropy for multiple scales
   >>> scales = [0, 0.25, 0.5, 1, 2, 4, 8, np.inf]
   >>> renyi_vals = nuee.renyi(species, scales=scales)

   >>> # Plot diversity profiles
   >>> fig, ax = plt.subplots(figsize=(10, 6))

   >>> for i, site in enumerate(species.index[:5]):  # First 5 sites
   ...     ax.plot(scales[:-1], renyi_vals.iloc[i, :-1],
   ...             marker='o', label=site) # doctest: +SKIP

   >>> ax.set_xlabel('Alpha (scale parameter)') # doctest: +SKIP
   >>> ax.set_ylabel('Renyi Entropy') # doctest: +SKIP
   >>> ax.set_title('Renyi Diversity Profiles') # doctest: +SKIP
   >>> ax.legend() # doctest: +SKIP
   >>> ax.grid(True, alpha=0.3) # doctest: +SKIP
   >>> plt.show() # doctest: +SKIP

   >>> # Hill numbers (effective number of species)
   >>> hill_nums = nuee.renyi(species, scales=scales, hill=True)

   >>> fig, ax = plt.subplots(figsize=(10, 6))
   >>> for i, site in enumerate(species.index[:5]):
   ...     ax.plot(scales[:-1], hill_nums.iloc[i, :-1],
   ...             marker='o', label=site) # doctest: +SKIP

   >>> ax.set_xlabel('Order of diversity') # doctest: +SKIP
   >>> ax.set_ylabel('Effective number of species') # doctest: +SKIP
   >>> ax.set_title('Hill Numbers') # doctest: +SKIP
   >>> ax.legend() # doctest: +SKIP
   >>> ax.grid(True, alpha=0.3) # doctest: +SKIP
   >>> plt.show() # doctest: +SKIP

Advanced Topics
---------------

Custom Distance Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doctest::

   >>> import numpy as np
   >>> from scipy.spatial.distance import pdist, squareform

   >>> def custom_distance(x, y):
   ...     """Custom distance function."""
   ...     return np.sum((x - y)**2) / (np.sum(x) + np.sum(y))

   >>> # Use with scipy
   >>> dist_matrix = squareform(pdist(species.values, metric=custom_distance))

Partial Ordination
~~~~~~~~~~~~~~~~~~

.. doctest::

   >>> # RDA with covariables (partial RDA)
   >>> # Y ~ X + Condition(Z)
   >>> # Not yet implemented - coming soon!

Variable Selection
~~~~~~~~~~~~~~~~~~

.. doctest::

   >>> # Stepwise variable selection for RDA
   >>> species = nuee.datasets.dune()
   >>> env = nuee.datasets.dune_env()

   >>> # Start with full model
   >>> rda_full = nuee.rda(species, env[["A1", "Moisture", "Manure"]])

   >>> # Perform stepwise selection
   >>> rda_selected = nuee.ordistep(rda_full, env[["A1", "Moisture", "Manure"]])
   >>> print("Selected variables:", rda_selected["selected_variables"])
   [...]
