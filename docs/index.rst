======================
 Lagrangian Filtering
======================

**lagrangian-filtering** is a library for performing temporal
filtering of data in a Lagrangian frame of reference. The library is
written in Python, leveraging the flexible OceanParcels_ library for
efficient Lagrangian transformation offloaded to a just-in-time
compiled C kernel. The source code of this library can be found on
GitHub_.

.. _OceanParcels: http://oceanparcels.org
.. _GitHub: https://github.com/angus-g/lagrangian-filtering

.. toctree::
   :caption: Documentation
   :maxdepth: 1

   installation
   algorithm
   data
   filter
   exploring
   examples
   issues
   contributing
   api

Citing
------

The description and analysis of the Lagrangian filtering method has
been published in JAMES, and can be cited as:

    Shakespeare, C. J., Gibson, A. H., Hogg, A. M., Bachman, S. D.,
    Keating, S. R., & Velzeboer, N. (2021). A new open source
    implementation of Lagrangian filtering: A method to identify
    internal waves in high-resolution simulations. *Journal of
    Advances in Modeling Earth Systems*, 13,
    e2021MS002616. https://doi.org/10.1029/2021MS002616


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
