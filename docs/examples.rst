==========
 Examples
==========

Here are some example excerpts for some things you may want to do with
the filtering library.

Load ROMS Data
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
