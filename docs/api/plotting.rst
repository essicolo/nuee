Plotting
========

The plotting module provides visualization functions for ecological data.

.. currentmodule:: nuee.plotting

Ordination Plots
----------------

.. autofunction:: plot_ordination

.. autofunction:: ordiplot

.. autofunction:: biplot

.. deprecated:: 0.1.2
   :func:`biplot` is kept for compatibility only. New code should call
   :func:`ordiplot`, which also exposes the ``loading_factor`` and
   ``predictor_factor`` scaling parameters.

Diversity Plots
---------------

.. autofunction:: plot_diversity

.. autofunction:: plot_rarecurve

Dissimilarity Plots
-------------------

.. autofunction:: plot_dissimilarity
