==========
 Examples
==========

Here are some example excerpts for some things you may want to do with
the filtering library.

Load ROMS data
==============

Suppose we have some ROMS data, with a somewhat complex rotated grid,
velocity staggered according to the Arakawa C-grid. Moreover, the
velocity and grid data are stored in separate files, and the velocity
data is spread over multiple files (they may even have only a single
timestep per file)! Due to the C-grid staggering, we want to specify
the cell corners (*psi* points) as our lat/lon dimensions. This is
consistent with the C-grid interpolation method that OceanParcels will
use in this case. Further, because lat/lon are in degrees, we specify
this as a *spherical* mesh.

.. code-block:: python

   filenames = {
       "U": {"lon": "grid.nc", "lat": "grid.nc", "data": "ocean_*.nc"},
       "V": {"lon": "grid.nc", "lat": "grid.nc", "data": "ocean_*.nc"},
   }
   variables = {"U": "Usur", "V": "Vsur"}
   dimensions = {"lon": "lon_psi", "lat": "lat_psi", "time": "ocean_time"}

   f = LagrangeFilter(
      "roms_experiment", filenames, variables, dimensions, sample_variables,
      mesh="spherical", c_grid=True,
   )

Now the filtering library will automatically seed particles at cell
corners when performing advection/filtering. Underneath, OceanParcels
will interpret the grid correctly, and interpolate the velocity
components onto cell corners as well.


.. _xarray example:

Load data through xarray
========================

This example is a bit more complicated, but demonstrates that we can
coalesce all of the input and grid files into a single xarray
dataset. We use a similar dataset to the previous example.

.. code-block:: python

   variables = {"U": "Usur", "V": "Vsur", "zeta": "zeta"}
   dimensions = {"lon": "lon_psi", "lat": "lat_psi", "time": "ocean_time"}

   # load the data as a dataset:
   # only load the variables we want (preprocess)
   # and make sure to chunk the data
   data_in = xr.open_mfdataset(
       "ocean_*.nc",
        combine="by_coords",
        preprocess=lambda d: d[list(variables.values())],
        chunks={
            "eta_rho": 1025, "eta_u": 1025, "eta_v": 1025,
            "xi_rho": 2161, "xi_u": 2161, "xi_v": 2161,
        },
        decode_times=False,
   )
   gridfile = "grid.nc"

   # merge in the grid files
   ds = xr.merge((data_in, xr.open_dataset(gridfile)[["lon_psi", "lat_psi"]]))

   # because all the variables need to be on the same grid for C-grid interpolation
   # we will be using swap_dims, but first we need to make the variables the right size
   # (the non-corner dimensions are one point larger)
   ds = ds.isel(
       eta_rho=slice(None, -1), eta_u=slice(None, -1),
       xi_rho=slice(None, -1), xi_v=slice(None, -1),
   )
   ds = ds.swap_dims(
       {
           "eta_rho": "eta_psi", "eta_u": "eta_psi", "eta_v": "eta_psi",
           "xi_rho": "xi_psi", "xi_u": "xi_psi", "xi_v": "xi_psi",
       }
   )

   # finally, we need to set the lat/lon variables as "coordinates"
   ds = ds.set_coords(("lon_psi", "lat_psi"))

   f = LagrangeFilter(
      "xarray_experiment", ds, variables, dimensions, sample_variables,
      mesh="spherical", c_grid=True,
   )


.. _masking example:

Dynamic land masking
====================

Using linear interpolation for tracer data or pointwise velocity data
can have unintended behaviour around land: by default, OceanParcels'
interpolation doesn't know about land. If land is masked off by NaN
values, these values will be interpolated into surrounding points, so
ocean cells near land will erroneously have no valid data. If the land
is externally masked, or has junk values, this will be interpolated
into valid cells, giving them incorrect values.

The solution comes in the form of OceanParcels'
``linear_invdist_land_tracer`` interpolation method. This is identical
to regular linear interpolation away from land. Near land, determined
by points with value of exactly zero, a weighted interpolation scheme
is used. The weight is the inverse square of the distance from the
interpolation point to the data location, using only the valid data
points, i.e. ignoring land.

It may not be common to mask land values with zeroes. Instead of
re-masking your entire dataset, you can leverage xarray's delayed
calculations to mask land to zero on the fly. It's important here to
explicitly set the land-aware interpolation method on any tracer
fields. When accessing the field, use the variable names in
OceanParcels' namespace, i.e. the keys of the ``variables``
dictionary.

.. code-block:: python

   # suppose ds is an already-loaded xarray dataset

   # if land is NaN-masked:
   ds = ds.fillna(0)

   # if there is a separate mask array, where True
   # values should be masked out
   ds = ds.where(~mask, 0)

   f = LagrangeFilter(
      "land_masking", ds, variables, dimensions, sample_variables,
   )
   # set tracer interp method
   f.fieldset.RHO.interp_method = "linear_invdist_land_tracer"
