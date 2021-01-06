.. crampon documentation master file, created by
   sphinx-quickstart on Wed Jul 25 14:49:30 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CRAMPON - Cryospheric Monitoring and Prediction Online!
==================================================================

CRAMPON is a modular system that aims to combine glaciological modeling with data assimilation from different sources.
To view our latest updates on the state of glaciers in Switzerland , go to our `our website <https://crampon.glamos.ch/>`_.

The scope of CRAMPON is Swiss glacier mass balance for now.

CRAMPON implements a lot of features itself, but especially the GIS preprocessing relies heavily on OGGM - the Open Global Glacier Model.
Go to `oggm.org <http://oggm.org>`_ for latest updates on OGGM or visit `OGGM on readthedocs <https://docs.oggm.org/en/latest/>`_ to learn more about how OGGM works in detail.


Glacier Modeling
^^^^^^^^^^^^^^^^


* :doc:`introduction`
* :doc:`massbalance-ensemble`
* :doc:`potential-irradiation`
* :doc:`input-uncertainty`
* :doc:`calibration`



.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Glacier Modeling

    introduction.rst
    massbalance-ensemble.rst
    potential-irradiation.rst
    input-uncertainty.rst
    calibration.rst


Observations
^^^^^^^^^^^^

Direct and indirect observations of glacier mass balance and its related parameters are key to drive a combined modeling and data assimilation system.
In Switzerland, there are some well-observed glaciers which profit from the long-term measurement program `GLAMOS <https://www.glamos.ch/>`_.
However, most of the glaciers in Switzerland are not observed at all.
To still get an optimal analysis and prediction of mass balance for these glaciers, we try to gather as much indirect information on them as we can - some by close-range remote sensing, but most of them coming from space observations.

* :doc:`camera-massbalances`
* :doc:`satellite-albedo`
* :doc:`satellite-snowlines`


.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Observations

    camera-massbalances.rst
    satellite-albedo.rst
    satellite-snowlines.rst


Data Assimilation
^^^^^^^^^^^^^^^^^^

In order to combine uncertain modeling results with uncertain observations, CRAMPON comes with an implementation of data assimilation techniques.
These techniques are able to minimize uncertainties and reduce the amount of model runs needed to get an optimal analysis and prediction.

* :doc:`particle-filter`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Data Assimilation

    particle-filter.rst


Use CRAMPON
^^^^^^^^^^^

CRAMPON is intended to be an automated, operational data assimilation framework.
However, the variety of implemented methods makes it also possible to use CRAMPON for experiments.
When planning an experiment with CRAMPON it is important to know that our input is free for scientific use, but might not always be easy to access.
Find an overview here on what is possible with CRAMPON and how data can be accessed.

* :doc:`api`
* :doc:`data-sources`
* :doc:`pitfalls`
* :doc:`whats-new`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Use CRAMPON

    api.rst
    data-sources.rst
    pitfalls.rst
    whats-new.rst

About
-----
    Most of CRAMPON's work has been implemented by `Johannes Landmann <https://vaw.ethz.ch/personen/person-detail.html?persid=234293>`_

    For OGGM authors, see the `OGGM version history`_ for a list of all OGGM contributors.

    .. _OGGM version history: http://docs.oggm.org/en/latest/whats-new.html
