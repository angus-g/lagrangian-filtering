=======================
 Specifying Input Data
=======================

When we construct a :py:class:`~filtering.filtering.LagrangeFilter`
object, the ``filenames``, ``variables`` and ``dimensions``
arguments are passed straight into OceanParcels. There are some
examples of how these arguments should be constructed in the
`OceanParcels tutorial`_, but we will summarise some of the important takeaways here.

.. _OceanParcels tutorial: https://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/parcels_tutorial.ipynb


Filenames dictionary
====================

The ``filenames`` argument is more properly called the
``filenames_or_dataset`` argument in the
:py:class:`~filtering.filtering.LagrangeFilter` initialiser. We'll
start by describing the more common usecase, providing filenames,
rather than a dataset. In all cases where you provide filenames, the
files should be in the NetCDF format. In the most simple case, all
your data is in a single file::

  filenames = "data_file.nc"

The filenames can contain wildcard characters, for example::

  filenames = "data_directory/output*/diags.nc"

If your variables are in separate files, you can pass a dictionary::

  filenames = {
    "U": "u_velocity.nc",
    "V": "v_velocity.nc",
    "rho": "diags.nc",
  }

Finally, you can pass a dictionary of dictionaries, separating the
files containing latitude, longitude, depth and variable data. This is
particularly useful when your data is on a B- or C-grid, as
:ref:`detailed below <bcgrid-data>`. The format of the dictionaries
follows, noting that the ``depth`` entry is not required if you're
only using two-dimensional data::

  filenames = {
    "U": {"lat": "mask.nc", "lon": "mask.nc", "depth": "depth.nc", "data": "u_velocity.nc"},
    "V": {"lat": "mask.nc", "lon": "mask.nc", "depth": "depth.nc", "data": "v_velocity.nc"},
  }

As an alternative to passing filenames, an ``xarray`` dataset can be
given to the ``filenames_or_dataset`` argument. This is probably more
useful when using synthetic data, without requiring that it first be
written to a file.


Variables dictionary
====================

OceanParcels uses particular names for the velocity components and
dimensions of data. These names may differ from those actualy used
within your files. The first bridge between these two conventions is
the ``variables`` dictionary. This is a map between a variable name
used within OceanParcels, and the name within the data files
themselves. Note that if you have extra data beyond just the velocity
components, it still requires an entry in ``variables``. ::

  variables = {"U": "UVEL", "V": "VVEL", "P": "PHIHYD", "RHO": "RHOAnoma"}

This mapping defines the usual ``U`` and ``V`` velocity components,
and the additional ``P`` and ``RHO`` variables, named ``PHIHYD`` and
``RHOAnoma`` in the source data files, respectively.


Dimensions dictionary
=====================

The other bridge between conventions relates to the dimensions of the
data. There are two considerations here: first is to simply inform
OceanParcels of the latitude, longitude, depth and time dimensions
within the data. However, the second consideration is to redefine the
data locality of the variables, which is required when using :ref:`B-
or C-grid interpolation <bcgrid-data>`.

If all data is on the same grid, i.e. Arawaka A-grid, ``dimensions``
can be a single dictionary mapping the OceanParcels dimension names
``lat``, ``lon``, ``time`` and ``depth`` to those found within the
data files. As before, ``depth`` isn't required for two-dimensional
data. However, if your data is three-dimensional and you're choosing a
single depth-level with the index mechanism below, ``depth`` must
still be present in the ``dimensions`` dictionary. ::

  dimensions = {"lon": "X", "lat": "Y", "time": "T", "depth": "Zmd000200"}

It is also possible to separately specify the dimensions for each of
the variables defined in the ``variables`` dictionary. This is often
used when variables have different spatial staggering. ::

  dimensions = {
    "U":   {"lon": "xu_ocean", "lat": "yu_ocean", "time": "time"},
    "V":   {"lon": "xu_ocean", "lat": "yu_ocean", "time": "time"},
    "RHO": {"lon": "xt_ocean", "lat": "yt_ocean", "time": "time"},
  }


Index dictionary
================

In some cases, we might want to restrict the extent of the data that
OceanParcels sees. This is different from using
:py:func:`~filtering.filtering.LagrangeFilter.seed_subdomain` to use
the full domain for advection, but restrict the domain size used for
filtering. This functionality is most useful considering that we
perform filtering in two-dimensional slices: if we provide a full
three-dimensional data file, we may run into some problems. Instead of
requiring a pre-processing step to split out separate vertical levels,
we can tell OceanParcels to consider only a particular level by its
index through the ``indices`` dictionary. This is an optional argument
to the :py:class:`~filtering.filtering.LagrangeFilter`
initialiser. For example, to use only the surface data (for a file
where the indices increase downwards)::

  indices = {"depth": [0]}


.. _bcgrid-data:

B- and C-grid data
==================

Compared to the Arakawa A-grid, where all variables are collocated
within a grid cell, the different variables are staggered differently
in the B- and C-grid conventions. In particular, on a B-grid, velocity
is defined on cell edges, and tracers are taken as a cell mean. This
means that velocity is interpolated bilinearly, as you may expect. The
behaviour with three-dimensional data is more complicated, but we will
not discuss this because the filtering library is aimed at
two-dimensional slices.

OceanParcels assumes that C-grid velocity data is constant along
faces. The U component is defined on the eastern face of a cell, and
the V component on the northern face. To interpolate in this manner,
OceanParcels needs the grid information for velocities to refer to the
*corner* of a cell. Perhaps confusingly, this means that although U
and V are staggered relative to each other, they need to have the same
grid information in ``dimensions``. OceanParcels assumes the NEMO grid
convention, where ``U[i, j]`` is on the cell edge between corners
``[i, j-1]`` and ``[i, j]``. Similarly, ``V[i, j]`` is on the edge
between corners ``[i-1, j]`` and ``[i, j]``. If your data doesn't
follow this convention, new coordinate data will need to be generated
in order to work correctly. More detail is available in the `indexing
documentation`_.

.. _indexing documentation: https://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/documentation_indexing.ipynb
