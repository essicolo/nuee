nuee: Community Ecology Analysis in Python
==========================================

**``nuee`` has just been released, and is likely to contain errors and bugs. The code has not been thoroughly reviewed. Do NOT trust its results blindly.**

``nuee`` is a comprehensive Python implementation of the popular R package ``vegan`` for community ecology analysis. It provides tools for ordination, diversity analysis, dissimilarity measures, and statistical testing commonly used in ecological research.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide
   api_reference
   examples

Installation
============

Install nuee using pip:

.. code-block:: bash

   pip install nuee

Quick Start
===========

.. code-block:: python

   import nuee
   import matplotlib.pyplot as plt

   # Load sample data
   species_data = nuee.datasets.varespec()
   env_data = nuee.datasets.varechem()

   # NMDS Ordination
   nmds_result = nuee.metaMDS(species_data, k=2, distance="bray")
   print(f"NMDS Stress: {nmds_result.stress:.3f}")

   # Plot ordination
   fig = nuee.plot_ordination(nmds_result, display="sites")
   plt.show()

Features
========

Ordination Methods
------------------

* **NMDS**. Non-metric Multidimensional Scaling with ``nuee.metaMDS()`` (stress
  values may differ slightly from ``vegan::metaMDS`` pending full parity)
* **RDA**. Redundancy Analysis with ``nuee.rda()``
* **CCA**. Canonical Correspondence Analysis with ``nuee.cca()``
* **PCA**. Principal Component Analysis with ``nuee.pca()``
* **Environmental fitting** with ``nuee.envfit()`` (permutation statistics are
  evolving and can diverge from vegan’s output)
* **Procrustes analysis** with ``nuee.procrustes()`` (currently reports slightly
  different residual sums-of-squares than ``vegan::procrustes`` because of
  upstream NMDS differences)

Diversity Analysis
------------------

* **Shannon diversity** with ``nuee.shannon()``
* **Gini-Simpson diversity** with ``nuee.simpson()``
* **Fisher's alpha** with ``nuee.fisher_alpha()``
* **Renyi entropy** with ``nuee.renyi()``
* **Species richness** with ``nuee.specnumber()``
* **Evenness measures** with ``nuee.evenness()``
* **Rarefaction** with ``nuee.rarefy()`` and ``nuee.rarecurve()``

Dissimilarity Measures
----------------------

* **Bray-Curtis**, **Jaccard**, **Euclidean**, and 15+ other distances with ``nuee.vegdist()``
* **PERMANOVA** with ``nuee.adonis2()`` (sequential sums-of-squares with
  permutation p-values; minor numeric differences from vegan can occur)
* **ANOSIM** with ``nuee.anosim()`` (rank-based test with permutation p-values)
* **MRPP** with ``nuee.mrpp()`` (observed/expected within-group distances with permutation p-values; defaults to Euclidean distances matching vegan)
* **Mantel test** with ``nuee.mantel()`` (Pearson/Spearman correlation with permutation p-values)
* **Beta dispersion** with ``nuee.betadisper()``

Visualization
-------------

* **Ordination plots** with ``nuee.plot_ordination()``
* **Biplots** with ``nuee.biplot()``
* **Diversity plots** with ``nuee.plot_diversity()``
* **Rarefaction curves** with ``nuee.plot_rarecurve()``
* **Confidence ellipses** with ``nuee.ordiellipse()``

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
