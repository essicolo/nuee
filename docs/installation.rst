Installation
============

Requirements
------------

nuee requires Python 3.8 or later and the following packages:

* numpy >= 1.20.0
* scipy >= 1.7.0
* pandas >= 1.3.0
* matplotlib >= 3.4.0
* seaborn >= 0.11.0
* scikit-learn >= 1.0.0
* patsy >= 0.5.0
* statsmodels >= 0.12.0

Installing from PyPI
--------------------

The easiest way to install nuee is using pip:

.. code-block:: bash

   pip install nuee

Installing from Source
----------------------

To install from source:

.. code-block:: bash

   git clone https://github.com/essicolo/nuee.git
   cd nuee
   pip install -e .

Development Installation
------------------------

For development, install with the optional development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

This will install additional packages for testing and documentation:

* pytest
* pytest-cov
* black
* flake8
* mypy
* sphinx
* sphinx-rtd-theme

Verifying Installation
----------------------

You can verify that nuee is installed correctly by running:

.. code-block:: python

   import nuee
   print(nuee.__version__)

   # Load a sample dataset
   varespec = nuee.datasets.varespec()
   print(varespec.head())
