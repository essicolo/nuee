nuee: Community Ecology Analysis in Python
==========================================

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

* **NMDS** (Non-metric Multidimensional Scaling) with ``metaMDS()``
* **RDA** (Redundancy Analysis) with ``rda()``
* **CCA** (Canonical Correspondence Analysis) with ``cca()``
* **PCA** (Principal Component Analysis) with ``pca()``
* **Environmental fitting** with ``envfit()``
* **Procrustes analysis** with ``procrustes()``

Diversity Analysis
------------------

* **Shannon diversity** with ``shannon()``
* **Simpson diversity** with ``simpson()``
* **Fisher's alpha** with ``fisher_alpha()``
* **Renyi entropy** with ``renyi()``
* **Species richness** with ``specnumber()``
* **Evenness measures** with ``evenness()``
* **Rarefaction** with ``rarefy()`` and ``rarecurve()``

Dissimilarity Measures
----------------------

* **Bray-Curtis**, **Jaccard**, **Euclidean**, and 15+ other distances with ``vegdist()``
* **PERMANOVA** with ``adonis2()``
* **ANOSIM** with ``anosim()``
* **MRPP** with ``mrpp()``
* **Mantel test** with ``mantel()``
* **Beta dispersion** with ``betadisper()``

Visualization
-------------

* **Ordination plots** with ``plot_ordination()``
* **Biplots** with ``biplot()``
* **Diversity plots** with ``plot_diversity()``
* **Rarefaction curves** with ``plot_rarecurve()``
* **Confidence ellipses** with ``ordiellipse()``

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
