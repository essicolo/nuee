Quick Start
===========

This guide will get you started with nuee quickly.

Basic NMDS Analysis
-------------------

.. doctest::

   >>> import nuee
   >>> import matplotlib.pyplot as plt

   >>> # Load sample data
   >>> species_data = nuee.datasets.varespec()
   >>> env_data = nuee.datasets.varechem()

   >>> # Perform NMDS ordination
   >>> nmds_result = nuee.metaMDS(species_data, k=2, distance="bray")
   >>> print(f"NMDS Stress: {nmds_result.stress:.3f}")
   NMDS Stress: 0.133

   >>> # Plot the ordination
   >>> fig = nuee.plot_ordination(nmds_result, display="sites")
   >>> plt.title("NMDS Ordination of Lichen Communities") # doctest: +SKIP
   >>> plt.show()

Diversity Analysis
------------------

.. doctest::

   >>> import nuee

   >>> # Load data
   >>> species = nuee.datasets.BCI()

   >>> # Calculate Shannon diversity
   >>> shannon_div = nuee.shannon(species)
   >>> print(f"Mean Shannon diversity: {shannon_div.mean():.3f}")
   Mean Shannon diversity: 3.821

   >>> # Calculate Gini-Simpson diversity (1 - sum(p^2))
   >>> simpson_div = nuee.simpson(species)
   >>> print(f"Mean Gini-Simpson diversity: {simpson_div.mean():.3f}")
   Mean Gini-Simpson diversity: 0.959

   >>> # Calculate species richness
   >>> richness = nuee.specnumber(species)
   >>> print(f"Mean species richness: {richness.mean():.1f}")
   Mean species richness: 90.8

   >>> # Plot diversity
   >>> fig = nuee.plot_diversity(shannon_div)
   >>> plt.show()

Constrained Ordination (RDA)
-----------------------------

.. code-block:: python

   import nuee
   import matplotlib.pyplot as plt

   # Load data
   species = nuee.datasets.dune()
   env = nuee.datasets.dune_env()

   # Perform RDA
   rda_result = nuee.rda(species, env)

   # Create biplot
   fig = nuee.biplot(rda_result)
   plt.title("RDA Biplot of Dune Meadow Vegetation")
   plt.show()

   # Fit environmental vectors
   envfit_result = nuee.envfit(rda_result, env)
   print(envfit_result)

.. note::

   ``envfit`` replicates vegan’s API but the permutation p-values and vector
   scaling are still being tuned. Results may differ slightly from
   ``vegan::envfit``.

PERMANOVA Test
--------------

.. code-block:: python

   import nuee

   # Load data
   species = nuee.datasets.mite()
   env = nuee.datasets.mite_env()

   # Calculate distance matrix
   distances = nuee.vegdist(species, method="bray")

   # Run PERMANOVA
   permanova_result = nuee.adonis2(distances, env[['SubsDens', 'WatrCont']])
   print(permanova_result)

Rarefaction Analysis
--------------------

.. code-block:: python

   import nuee
   import matplotlib.pyplot as plt

   # Load data
   species = nuee.datasets.BCI()

   # Calculate rarefaction curves
   rarefaction = nuee.rarecurve(species, step=10)

   # Plot rarefaction curves
   fig = nuee.plot_rarecurve(rarefaction)
   plt.title("Species Accumulation Curves")
   plt.show()

Next Steps
----------

* Check out the :doc:`user_guide` for more detailed information
* Browse the :doc:`api_reference` for complete function documentation
* See :doc:`examples` for more advanced use cases
