Quick Start
===========

This guide will get you started with nuee quickly.

Basic NMDS Analysis
-------------------

.. code-block:: python

   import nuee
   import matplotlib.pyplot as plt

   # Load sample data
   species_data = nuee.datasets.varespec()
   env_data = nuee.datasets.varechem()

   # Perform NMDS ordination
   nmds_result = nuee.metaMDS(species_data, k=2, distance="bray")
   print(f"NMDS Stress: {nmds_result.stress:.3f}")

   # Plot the ordination
   fig = nuee.plot_ordination(nmds_result, display="sites")
   plt.title("NMDS Ordination of Lichen Communities")
   plt.show()

Diversity Analysis
------------------

.. code-block:: python

   import nuee

   # Load data
   species = nuee.datasets.BCI()

   # Calculate Shannon diversity
   shannon_div = nuee.shannon(species)
   print(f"Mean Shannon diversity: {shannon_div.mean():.3f}")

   # Calculate Gini-Simpson diversity (1 - sum(p^2))
   simpson_div = nuee.simpson(species)
   print(f"Mean Gini-Simpson diversity: {simpson_div.mean():.3f}")

   # Calculate species richness
   richness = nuee.specnumber(species)
   print(f"Mean species richness: {richness.mean():.1f}")

   # Plot diversity
   fig = nuee.plot_diversity(shannon_div)
   plt.show()

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

   # Create ordiplot (automatically balances scores and vectors)
   fig = nuee.ordiplot(rda_result, loading_factor=0, predictor_factor=0)
   plt.title("RDA Ordiplot of Dune Meadow Vegetation")
   plt.show()

   # Any numeric factor can be supplied to loading_factor / predictor_factor
   # to shrink or stretch species loadings and environmental arrows manually.

.. tip::

   ``loading_factor`` and ``predictor_factor`` accept any numeric value. Use
   ``0`` to rescale vectors so they match the longest sample score, or provide
   a manual multiplier when you need tighter or looser arrows.

   To inspect residual (unconstrained) axes, supply ``axes_source="unconstrained"``:

   .. code-block:: python

      nuee.ordiplot(rda_result, axes_source="unconstrained", axes=(0, 1))

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
