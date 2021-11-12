"""The main lagrangian-filtering module.

This module contains the crucial datastructure for
lagrangian-filtering, `LagrangeFilter`. See project documentation
for examples on how to construct a filtering workflow using this
library.

"""

from datetime import timedelta
from glob import iglob
import logging
import os.path

import dask.array as da
import netCDF4
import numpy as np
import parcels
from scipy import signal
import xarray as xr

from filtering.file import LagrangeParticleFile
from filtering.filter import Filter


class LagrangeFilter(object):
    """The main class for a Lagrangian filtering workflow.

    The workflow is set up using the input files and the filtering
    parameters. Filtering can be performed all at once, or on
    individual time levels.

    Data must contain horizontal velocity components `U` and `V` to
    perform the Lagrangian frame transformation. Any variables that should
    be filtered must be specified in the `sample_variables` list (this
    includes velocity).

    Note:
        We use the OceanParcels convention for variable names. This means that
        ``U``, ``V``, ``lon``, ``lat``, ``time`` and ``depth`` are canonical
        names for properties required for particle advection. The mapping from
        the actual variable name in your data files to these canonical names
        is contained in the `variables` and `dimensions` dictionaries. When
        specifying `filenames` or `sample_variables`, the canonical names
        must be used, however any other variables may use whatever name you
        would like.

    Once the `LagrangeFilter` has been constructed, you may call it as
    a function to perform the filtering workflow. See :func:`~filter`
    for documentation.

    Example:
        A straightforward filtering workflow::

            f = LagrangeFilter(
                name, filenames, variables, dimensions, sample_variables,
            )
            f()

        Would result in a new file with the given `name` and an appropriate
        extension containing the filtered data for each of the `sample_variables`.

    Args:
        name (str): The name of the workflow
        filenames_or_dataset (Union[Dict[str, str], xarray.Dataset]):
            Either a mapping from data variable names to the files
            containing the data, or an xarray Dataset containing the
            input data.

            Filenames can contain globs if the data is spread across
            multiple files.
        variables (Dict[str, str]): A mapping from canonical
            variable names to the variable names in your data files.
        dimensions (Dict[str, str]): A mapping from canonical dimension
            names to the dimension names in your data files.
        sample_variables (List[str]): A list of variable names that should be sampled
            into the Lagrangian frame of reference and filtered.
        init_only_variables (Optional[List[str]]): An optional list of variable
            names that should be sampled where particles are spawned, but not
            participate in advection. These variables can be passed through to
            filters.
        mesh (Optional[str]): The OceanParcels mesh type, either "flat"
            or "spherical". "flat" meshes are expected to have dimensions
            in metres, and "spherical" meshes in degrees.
        c_grid (Optional[bool]): Whether to interpolate velocity
            components on an Arakawa C grid (defaults to no).
        indices (Optional[Dict[str, List[int]]]): An optional dictionary
            specifying the indices to which a certain dimension should
            be restricted.
        uneven_window (Optional[bool]): Whether to allow different
            lengths for the forward and backward advection phases.
        window_size (Optional[float]): The nominal length of the both
            the forward and backward advection windows, in seconds. A
            longer window may better capture the low-frequency signal to be
            removed.
        minimum_window (Optional[float]): If provided, particles will be
            filtered if they successfully advected for at least this long
            in total. This can increase the yield of filtered data by
            salvaging particles that would otherwise be considered dead.
        highpass_frequency (Optional[float]): The 3dB cutoff frequency
            for filtering, below which spectral components will be
            attenuated. This should be an angular frequency, in [rad/s].
        advection_dt (Optional[datetime.timedelta]): The timestep
            to use for advection. May need to be adjusted depending on the
            resolution/frequency of your data.
        **kwargs (Optional): Additional arguments are passed to the Parcels
            FieldSet constructor.

    """

    def __init__(
        self,
        name,
        filenames_or_dataset,
        variables,
        dimensions,
        sample_variables,
        init_only_variables=[],
        c_grid=False,
        uneven_window=False,
        window_size=None,
        minimum_window=None,
        highpass_frequency=5e-5,
        advection_dt=timedelta(minutes=5),
        **kwargs,
    ):
        # The name of this filter
        self.name = name
        # Width of window over which our filter computes a meaningful result
        # in seconds. Default to 3.5 days on either side
        if window_size is None:
            self.window_size = timedelta(days=3.5).total_seconds()
        else:
            self.window_size = window_size
        # Whether we're permitted to use uneven windows on either side
        self.uneven_window = uneven_window

        # copy input file dictionaries so we can construct the output file
        # filenames dictionary is modified to expand globs when
        # the fieldset is constructed
        self._filenames = filenames_or_dataset
        self._variables = variables
        self._dimensions = dimensions
        # sample variables without the "var_" prefix
        self._sample_variables = sample_variables

        # choose the fieldset constructor depending on the format
        # of the input data
        if isinstance(filenames_or_dataset, xr.Dataset):
            fieldset_constructor = parcels.FieldSet.from_xarray_dataset
        else:
            fieldset_constructor = parcels.FieldSet.from_netcdf

        # for C-grid data, we have to change the interpolation method
        fieldset_kwargs = kwargs
        fieldset_kwargs.setdefault("mesh", "flat")
        if c_grid:
            interp_method = kwargs.get("interp_method", {})
            for v in variables:
                if v in interp_method:
                    continue

                if v in ["U", "V", "W"]:
                    interp_method[v] = "cgrid_velocity"
                else:
                    interp_method[v] = "cgrid_tracer"

            fieldset_kwargs["interp_method"] = interp_method

        # construct the OceanParcels FieldSet to use for particle advection
        self.fieldset = fieldset_constructor(
            filenames_or_dataset, variables, dimensions, **fieldset_kwargs
        )

        self._output_field = self.fieldset.get_fields()[0].name
        logging.warning(
            "Seeding particles and output times on the same grid as '%s'. "
            "You can change this with .set_particle_grid()",
            self._output_field,
        )

        # save the lon/lat on which to seed particles
        # this is saved here because if the grid is later made periodic, the
        # underlying grids will be modified, and we'll seed particles in the halos
        self._set_grid(self.fieldset.gridset.grids[0])

        # starts off non-periodic
        self._is_zonally_periodic = False
        self._is_meridionally_periodic = False

        # guess the output timestep from the seed grid
        self.output_dt = self._output_grid.time[1] - self._output_grid.time[0]
        # default filter frequency
        self.filter_frequency = highpass_frequency / (2 * np.pi)
        self.inertial_filter = None

        # timestep for advection
        self.advection_dt = advection_dt

        # default class for caching advection data before filtering
        self._advection_cache_class = LagrangeParticleFile
        self._advection_cache_kwargs = {}

        # the sample variable attribute has 'var_' prepended to map to
        # variables on particles
        self.sample_variables = ["var_" + v for v in sample_variables]
        self.init_only_variables = ["init_" + v for v in init_only_variables]

        # create the particle class and kernel for sampling
        # map sampled variables to fields
        self.particleclass = ParticleFactory(
            self.sample_variables + self.init_only_variables
        )
        # if we're using cgrid, we need to set the lon/lat dtypes to 64-bit,
        # otherwise things get reordered when we create the particleclass
        if c_grid:
            self.particleclass.set_lonlatdepth_dtype(np.float64)

        self.sample_kernel = self._create_sample_kernel(sample_variables)
        self.init_kernel = self.sample_kernel + self._create_sample_kernel(
            init_only_variables, "init_kernel", "init"
        )
        self.kernel = parcels.AdvectionRK4 + self.sample_kernel

        # compile kernels
        self._compile(self.sample_kernel)
        self._compile(self.init_kernel)
        self._compile(self.kernel)

        # options (compression, etc.) for creating output variables
        self._output_variable_kwargs = {}

        # width of the minimum valid seeding window
        self._min_window = None
        if minimum_window is not None:
            self._min_window = minimum_window / self.output_dt

    @property
    def seed_lat(self):
        """The 2D grid of seed particle latitudes.

        Note:
            This is determined by :func:`~set_particle_grid` and
            :func:`~seed_subdomain`.

        Returns:
            numpy.ndarray: The seed particle latitudes.

        """

        return self._grid_lat

    @property
    def seed_lon(self):
        """The 2D grid of seed particle longitudes.

        Note:
            This is determined by :func:`~set_particle_grid` and
            :func:`~seed_subdomain`.

        Returns:
            numpy.ndarray: The seed particle longitudes.

        """

        return self._grid_lon

    def _set_grid(self, grid):
        """Set the seeding grid from a parcels grid"""

        if grid.gtype in [
            parcels.GridCode.CurvilinearZGrid,
            parcels.GridCode.CurvilinearSGrid,
        ]:
            self._curvilinear = True
            self._grid_lon = grid.lon
            self._grid_lat = grid.lat
        else:
            self._curvilinear = False
            self._grid_lon, self._grid_lat = np.meshgrid(grid.lon, grid.lat)

        # save the original grid to allow subdomain seeding
        self._orig_grid = self._grid_lon, self._grid_lat
        # mask off output
        self._grid_mask = np.ones_like(self._grid_lon, dtype=bool)
        # save grid timesteps
        self._output_grid = grid

    def _create_sample_kernel(
        self, sample_variables, kernel_name="sample_kernel", prefix="var"
    ):
        """Create the parcels kernel for sampling fields during advection."""

        # make sure the fieldset has C code names assigned, etc.
        self.fieldset.check_complete()

        # string for the kernel itself
        f_str = f"def {kernel_name}(particle, fieldset, time):\n"
        for v in sample_variables:
            f_str += f"\tparticle.{prefix}_{v} = fieldset.{v}.eval(time, particle.depth, particle.lat, particle.lon, applyConversion=False)\n"
        else:
            f_str += "\tpass"

        # create the kernel
        return parcels.Kernel(
            self.fieldset,
            self.particleclass.getPType(),
            funcname="sample_kernel",
            funcvars=["particle", "fieldset", "time"],
            funccode=f_str,
        )

    def _compile(self, kernel):
        """Compile a kernel and tell it to load the resulting shared library."""

        parcels_dir = os.path.join(
            parcels.tools.global_statics.get_package_dir(), "include"
        )
        kernel.compile(
            compiler=parcels.compilation.codecompiler.GNUCompiler(incdirs=[parcels_dir])
        )
        kernel.load_lib()

    def set_output_compression(self, complevel=None):
        """Enable compression on variables in the output NetCDF file.

        This enables zlib compression on the output file, which can
        significantly improve filesize at a small expense to
        computation time.

        Args:
            complevel (Optional[int]): If specified as a value
                from 1-9, this overrides the default compression level
                (4 for the netCDF4 library).
        """

        if complevel is not None:
            if not isinstance(complevel, int) or not 1 <= complevel <= 9:
                raise ValueError(
                    "if specified, complevel must be an integer in the range 1-9"
                )

            self._output_variable_kwargs["complevel"] = complevel

        self._output_variable_kwargs["zlib"] = True

    def set_particle_grid(self, field):
        """Set the grid for the sampling particles by choosing a field.

        By default, particles are seeded on the gridpoints of the
        first field in the Parcels FieldSet (usually U velocity). To
        use another grid, such as a tracer grid, pass the relevant
        field name to this function. This field name should be in
        Parcels-space, i.e. the keys in the ``variables`` dictionary.

        Note:
            Because we get the particle grid from the Parcels gridset,
            and changing halos alters the underlying grids, this needs
            to be called before :func:`~make_zonally_periodic` or
            :func:`~make_meridionally_periodic`.

        Args:
            field (str): The name of the field whose grid to use for particles.

        """

        if self._is_zonally_periodic or self._is_meridionally_periodic:
            raise Exception("grid must be set before making domain periodic")

        if not hasattr(self.fieldset, field):
            raise ValueError(f"{field} is not a valid field name")

        self._set_grid(getattr(self.fieldset, field).grid)
        self._output_field = field

    def make_zonally_periodic(self, width=None):
        """Mark the domain as zonally periodic.

        This will add a halo to the eastern and western edges of the
        domain, so that they may cross over during advection without
        being marked out of bounds. If a particle ends up within the
        halo after advection, it is reset to the valid portion of the
        domain.

        If the domain has already been marked as zonally periodic,
        nothing happens.

        Due to the method of resetting particles that end up in the
        halo, this is incompatible with curvilinear grids.

        Args:
            width (Optional[int]): The width of the halo,
                defaults to 5 (per parcels). This needs to be less
                than half the number of points in the grid in the x
                direction. This may need to be adjusted for small
                domains, or if particles are still escaping the halo.

        Note:
            This causes the kernel to be recompiled to add another stage
            which resets particles that end up in the halo to the main
            domain.

            If the kernel has already been recompiled for meridional periodicity,
            it is again reset to include periodicity in both
            directions.

        """

        # the method of resetting particles won't work on a curvilinear grid
        if self._curvilinear:
            raise Exception("curvilinear grids can not be periodic")

        # make sure we can't do this twice
        if self._is_zonally_periodic:
            return

        # add constants that are accessible within the kernel denoting the
        # edges of the halo region
        self.fieldset.add_constant("halo_west", self._output_grid.lon[0])
        self.fieldset.add_constant("halo_east", self._output_grid.lon[-1])

        if width is None:
            self.fieldset.add_periodic_halo(zonal=True)
        else:
            self.fieldset.add_periodic_halo(zonal=True, halosize=width)

        # unload the advection-only kernel, and add the periodic-reset kernel
        self.kernel.remove_lib()

        if self._is_meridionally_periodic:
            k = _doubly_periodic_BC
        else:
            k = _zonally_periodic_BC

        periodic_kernel = parcels.Kernel(
            self.fieldset, self.particleclass.getPType(), k
        )

        self.kernel = parcels.AdvectionRK4 + periodic_kernel + self.sample_kernel
        self._compile(self.kernel)

        self._is_zonally_periodic = True

    def make_meridionally_periodic(self, width=None):
        """Mark the domain as meridionally periodic.

        This will add a halo to the northern and southern edges of the
        domain, so that they may cross over during advection without
        being marked out of bounds. If a particle ends up within the
        halo after advection, it is reset to the valid portion of the
        domain.

        If the domain has already been marked as meridionally periodic,
        nothing happens.

        Due to the method of resetting particles that end up in the
        halo, this is incompatible with curvilinear grids.

        Args:
            width (Optional[int]): The width of the halo,
                defaults to 5 (per parcels). This needs to be less
                than half the number of points in the grid in the y
                direction. This may need to be adjusted for small
                domains, or if particles are still escaping the halo.

        Note:
            This causes the kernel to be recompiled to add another stage
            which resets particles that end up in the halo to the main
            domain.

            If the kernel has already been recompiled for zonal periodicity,
            it is again reset to include periodicity in both
            directions.

        """

        # the method of resetting particles won't work on a curvilinear grid
        if self._curvilinear:
            raise Exception("curvilinear grids can not be periodic")

        # make sure we can't do this twice
        if self._is_meridionally_periodic:
            return

        # add constants that are accessible within the kernel denoting the
        # edges of the halo region
        self.fieldset.add_constant("halo_north", self._output_grid.lat[-1])
        self.fieldset.add_constant("halo_south", self._output_grid.lat[0])

        if width is None:
            self.fieldset.add_periodic_halo(meridional=True)
        else:
            self.fieldset.add_periodic_halo(meridional=True, halosize=width)

        # unload the previous kernel, and add the meridionally-periodic kernel
        self.kernel.remove_lib()

        if self._is_zonally_periodic:
            k = _doubly_periodic_BC
        else:
            k = _meridionally_periodic_BC

        periodic_kernel = parcels.Kernel(
            self.fieldset, self.particleclass.getPType(), k
        )

        self.kernel = parcels.AdvectionRK4 + periodic_kernel + self.sample_kernel
        self._compile(self.kernel)

        self._is_meridionally_periodic = True

    def seed_subdomain(
        self, min_lon=None, max_lon=None, min_lat=None, max_lat=None, skip=None
    ):
        """Restrict particle seeding to a subdomain.

        This uses the full set of available data for advection, but
        restricts the particle seeding, and therefore data filtering,
        to specified latitude/longitude.

        Points in the output dataset that fall outside this seeding
        range will not be written, and will thus have a missing value.

        Args:
            min_lon (Optional[float]): The lower bound on
                longitude for which to seed particles. If not specifed,
                seed from the western edge of the domain.
            max_lon (Optional[float]): The upper bound on
                longitude for which to seed particles. If not specifed,
                seed from the easter edge of the domain.
            min_lat (Optional[float]): The lower bound on
                latitude for which to seed particles. If not specifed,
                seed from the southern edge of the domain.
            max_lat (Optional[float]): The upper bound on
                latitude for which to seed particles. If not specifed,
                seed from the northern edge of the domain.
            skip (Optional[int]): The number of gridpoints to skip
                from the edge of the domain.

        """

        lon, lat = self._orig_grid

        # originally, mask selects full domain
        mask = np.ones_like(lon, dtype=bool)

        # restrict longitude
        if min_lon is not None:
            mask &= lon >= min_lon
        if max_lon is not None:
            mask &= lon <= max_lon

        # restrict latitude
        if min_lat is not None:
            mask &= lat >= min_lat
        if max_lat is not None:
            mask &= lat <= max_lat

        if skip is not None:
            mask[:skip, :] = 0  # west
            mask[-skip:, :] = 0  # east
            mask[:, :skip] = 0  # south
            mask[:, -skip:] = 0  # north

        self._grid_mask = mask
        self._grid_lon = lon[mask]
        self._grid_lat = lat[mask]

    def particleset(self, time):
        """Create a ParticleSet initialised at the given time.

        Args:
            time (float): The origin time for forward and backward advection
                on this ParticleSet.

        Returns:
            parcels.particlesets.particlesetsoa.ParticleSetSOA: A new ParticleSet
                containing a single particle at every gridpoint,
                initialised at the specified time.

        """

        # reset the global particle ID counter so we can rely on particle IDs making sense
        self.particleclass.setLastID(0)

        ps = parcels.ParticleSet(
            self.fieldset,
            pclass=self.particleclass,
            lon=self._grid_lon,
            lat=self._grid_lat,
            time=time,
        )
        ps.populate_indices()

        return ps

    def advection_step(self, time, output_time=False):
        """Perform forward-backward advection at a single point in time.

        This routine is responsible for creating a new ParticleSet at
        the given time, and performing the forward and backward
        advection steps in the Lagrangian transformation.

        Args:
            time (float): The point in time at which to calculate filtered data.
            output_time (Optional[bool]): Whether to include "time" as
                a numpy array in the output dictionary, for doing manual analysis.

        Note:
            If ``output_time`` is True, the output object will not be compatible
            with the default filtering workflow, :func:`~filter_step`!

            If ``output_dt`` has not been set on the filtering object,
            it will default to the difference between successive time
            steps in the first grid defined in the parcels
            FieldSet. This may be a concern if using data which has
            been sampled at different frequencies in the input data
            files.

        Returns:
            Dict[str, Tuple[int, dask.array.Array]]: A dictionary of the advection
                data, mapping variable names to a pair. The first element is
                the index of the sampled timestep in the data, and the
                second element is a lazy dask array concatenating the forward
                and backward advection data.

        """

        # seed all particles at gridpoints
        ps = self.particleset(time)
        # execute the sample-only kernel to efficiently grab the initial condition
        ps.kernel = self.init_kernel
        ps.execute(self.init_kernel, runtime=0, dt=self.advection_dt)

        # set up the temporary output file for the initial condition and
        # forward advection
        outfile = self._advection_cache_class(
            ps, self.output_dt, self.sample_variables, **self._advection_cache_kwargs
        )

        # now the forward advection kernel can run
        outfile.set_group("forward")
        ps.kernel = self.kernel
        ps.execute(
            self.kernel,
            runtime=self.window_size,
            dt=self.advection_dt,
            output_file=outfile,
            recovery={
                parcels.ErrorCode.ErrorOutOfBounds: _recovery_kernel_out_of_bounds
            },
        )

        # reseed particles back on the grid, then advect backwards
        # we don't need any initial condition sampling since we've already done it
        outfile.set_group("backward")
        ps = self.particleset(time)
        ps.kernel = self.kernel
        ps.execute(
            self.kernel,
            runtime=self.window_size,
            dt=-self.advection_dt,
            output_file=outfile,
            recovery={
                parcels.ErrorCode.ErrorOutOfBounds: _recovery_kernel_out_of_bounds
            },
        )

        # stitch together and filter all sample variables from the temporary
        # output data
        da_out = {}
        for v in self.sample_variables:
            # load data lazily as dask arrays, for forward and backward segments
            var_array_forward = da.from_array(
                outfile.data("forward")[v], chunks=(None, "auto")
            )[:-1, :]
            var_array_backward = da.from_array(
                outfile.data("backward")[v], chunks=(None, "auto")
            )[:-1, :]

            # get an index into the middle of the array
            time_index_data = var_array_backward.shape[0] - 1

            # construct proper sequence by concatenating data and flipping the backward segment
            # for var_array_forward, skip the initial output for both the sample-only and
            # sample-advection kernels, which have meaningless data
            var_array = da.concatenate(
                (da.flip(var_array_backward[1:, :], axis=0), var_array_forward)
            )

            da_out[v] = (time_index_data, var_array)

        if output_time:
            da_out["time"] = np.concatenate(
                (
                    outfile.data("backward").attrs["time"][1:-1][::-1],
                    outfile.data("forward").attrs["time"][:-1],
                )
            )

        return da_out

    def filter_step(self, advection_data):
        """Perform filtering of a single step of advection data.

        The Lagrangian-transformed data from :func:`~advection_step` is
        high-pass filtered in time, leaving only the signal at the
        origin point (i.e. the filtered forward and backward advection
        data is discarded).

        Note:
            If an inertial filter object hasn't been attached before
            this function is called, one will automatically be created.

        Args:
            advection_data (Dict[str, Tuple[int, dask.array.Array]]): A dictionary of
                particle advection data from a single timestep, returned
                from :func:`~advection_step`.

        Returns:
            Dict[str, dask.array.Array]: A dictionary mapping sampled
                variable names to a 1D dask array containing the
                filtered data at the specified time. This data is not
                lazy, as it has already been computed out of the
                temporary advection data.

        """

        # we need a filter for this step, so create the default filter
        # if necessary
        if self.inertial_filter is None:
            self.inertial_filter = Filter(self.filter_frequency, 1.0 / self.output_dt)

        da_out = {}
        for v, a in advection_data.items():
            # don't try to filter the time axis, just take the middle value
            if v == "time":
                da_out[v] = a[a.size // 2]
                continue

            time_index_data, var_array = a
            da_out[v] = self.inertial_filter.apply_filter(
                var_array, time_index_data, min_window=self._min_window
            )

        return da_out

    def filter(self, *args, **kwargs):
        """Run the filtering process on this experiment.

        Note:
            Instead of `f.filter(...)`, you can call `f(...)` directly.

        This is main method of the filtering workflow. The timesteps
        to filter may either be specified manually, or determined from
        the window size and the timesteps within the input files. In
        this latter case, only timesteps that have the full window
        size on either side are selected.

        Note:
            If `absolute` is True, the times must be the same datatype
            as those the input data. For dates with a calendar, this
            is likely :obj:`!numpy.datetime64` or :obj:`cftime.datetime`.
            For abstract times, this may simply be a number.

        Args:
            times (Optional[List[float]]): A list of timesteps at
                which to run the filtering. If this is omitted, all
                timesteps that are fully covered by the filtering
                window are selected.
            clobber (Optional[bool]): Whether to overwrite any
                existing output file with the same name as this
                experiment. Default behaviour will not clobber an
                existing output file.
            absolute (Optional[bool]): If `times` is provided,
                this argument determines whether to interpret them
                as relative to the first timestep in the input dataset
                (False, default), or as absolute, following the actual
                time dimension in the dataset (True).

        """

        self(*args, **kwargs)

    def create_out(self, clobber=False):
        """Create a netCDF dataset to hold filtered output.

        Here we create a new :obj:`!netCDF4.Dataset` for filtered
        output. For each sampled variable in the input files, a
        corresponding variable in created in the output file, with
        the dimensions of the output grid.

        Args:
            clobber (Optional[bool]): Whether to overwrite any
                existing output file with the same name as this
                experiment. Default behaviour will not clobber an
                existing output file.

        Returns:
            Tuple[:obj:`!netCDF4.Dataset`, str]: A tuple containing a single
                dataset that will hold all filtered output and the
                name of the time dimension in the output file.

        """

        # the output dataset we're creating
        ds = netCDF4.Dataset(self.name + ".nc", "w", clobber=clobber)
        time_dim = None

        # helper function to create the dimensions in the ouput file
        def create_dimension(dims, dim, var):
            # translate from parcels -> file convention
            # and check whether we've already created this dimension
            # (e.g. for a previous variable)
            # we have extra logic to handle the curvilinear case here
            file_dim = dims[dim]
            if file_dim in ds.variables:
                return ds.variables[file_dim].dimensions[
                    1
                    if len(ds.variables[file_dim].dimensions) > 1 and dim == "lon"
                    else 0
                ]

            # get the file containing the dimension data as a DataArray
            v_orig = self._variables.get(var, var)
            if isinstance(self._filenames, xr.Dataset):
                ds_orig = self._filenames[file_dim]
            else:
                if isinstance(self._filenames[var], dict):
                    # time dimension accompanies the data itself, unlike spatial dimensions
                    filename = self._filenames[var][dim if dim != "time" else "data"]
                else:
                    filename = self._filenames[var]

                if isinstance(filename, list):
                    filename = filename[0]

                ds_orig = xr.open_dataset(next(iglob(filename)), decode_times=False)[
                    file_dim
                ]

            # create dimensions if needed
            for d in ds_orig.dims:
                if d not in ds.dimensions:
                    # create a record dimension for time
                    ds.createDimension(d, None if dim == "time" else ds_orig[d].size)

            # create the dimension variable
            ds.createVariable(
                file_dim,
                ds_orig.dtype,
                dimensions=ds_orig.dims,
                **self._output_variable_kwargs,
            )
            # copy data if a spatial variable
            if dim != "time":
                ds.variables[file_dim][:] = ds_orig

            # copy attributes
            attrs = ds_orig.attrs
            if attrs != {}:
                ds.variables[file_dim].setncatts(ds_orig.attrs)

            # return the dimension name, handling the curvilinear grid case
            return ds_orig.dims[1 if len(ds_orig.dims) > 1 and dim == "lon" else 0]

        # get the output field
        v_output = self._variables.get(self._output_field, self._output_field)

        # get the relevant dimensions dictionary
        if self._output_field in self._dimensions:
            dims = self._dimensions[self._output_field]
        else:
            dims = self._dimensions

        # create dimensions in the output file for those on the output field
        out_dims = {}
        for d in ["time", "lat", "lon"]:
            out_dims[d] = create_dimension(dims, d, self._output_field)

        # save time dimension name for correct conversion
        time_dim = out_dims["time"]

        for v in self._sample_variables:
            # translate if required (parcels -> file convention)
            v_orig = self._variables.get(v, v)

            # create the variable in the dataset itself
            ds.createVariable(
                f"var_{v}",
                "float32",
                dimensions=(out_dims["time"], out_dims["lat"], out_dims["lon"]),
                **self._output_variable_kwargs,
            )

        return ds, time_dim

    def _window_times(self, times, absolute):
        """Restrict an array of times to those which have an adequate window,
        optionally converting from absolute to relative first.

        """

        tgrid = self._output_grid.time

        if times is None:
            times = tgrid.copy()

        if absolute:
            times = self._output_grid.time_origin.reltime(times)

        times = np.array(times)
        window_left = times - tgrid[0] >= self.window_size
        window_right = times <= tgrid[-1] - self.window_size
        return times[window_left & window_right]

    def _convert_time(self, t, ds, dim):
        """Convert a relative time value in seconds to the right format.

        This makes use of parcels' TimeConverter to get a relative
        time back to an absolute time. However, if the original time
        units require a calendar, we can't just output this directly
        to the netCDF file, so we need to strip the calendar with
        date2num through xarray's fairly advanced interface.

        """

        t = self.fieldset.time_origin.fulltime(t)

        if "units" not in ds[dim].ncattrs() or "calendar" not in ds[dim].ncattrs():
            return t

        return xr.coding.times.encode_cf_datetime(
            t, ds[dim].units, calendar=ds[dim].calendar
        )[0].item()

    def __call__(self, times=None, absolute=False, clobber=False):
        """Run the filtering process on this experiment."""

        if self.uneven_window:
            raise NotImplementedError("uneven windows aren't supported")

        # either restrict the specified times to period covered by window,
        # or use the full range of times covered by window
        times = self._window_times(times, absolute)
        if len(times) == 0:
            logging.warning(
                "No times are suitable for filtering. There may not be a window-width "
                "of data on either side of any of the specified times."
            )

            # early return to not create an output file
            return

        ds, time_dim = self.create_out(clobber=clobber)

        # create a masked array for output
        out_masked = np.ma.masked_array(
            np.empty_like(self._grid_mask, dtype=np.float32), ~self._grid_mask
        )

        # do the filtering at each timestep
        for idx, time in enumerate(times):
            # returns a dictionary of sample_variable -> dask array
            filtered = self.filter_step(self.advection_step(time, output_time=True))
            for v, a in filtered.items():
                # append time to output file
                if v == "time":
                    ds[time_dim][idx] = self._convert_time(a, ds, time_dim)
                    continue

                out_masked[self._grid_mask] = a
                ds[v][idx, ...] = out_masked

        ds.close()


def ParticleFactory(variables, name="SamplingParticle", BaseClass=parcels.JITParticle):
    """Create a Particle class that samples the specified variables.

    The variables that should be sampled will be prepended by ``var_`` as
    class attributes, in case there are any namespace clashes with existing
    variables on the base class.

    Args:
        variables (List[str]): A list of variable names which should be sampled.
        name (str): The name of the generated particle class.
        BaseClass (Type[:obj:`!parcels.particle._Particle`]): The base particles class upon
            which to append the required variables.

    Returns:
        Type[:obj:`!parcels.particle._Particle`]: The new particle class

    """

    var_dict = {v: parcels.Variable(v) for v in variables}

    newclass = type(name, (BaseClass,), var_dict)
    return newclass


def _recovery_kernel_out_of_bounds(particle, fieldset, time):
    """Recovery kernel for particle advection, to delete out-of-bounds particles."""

    particle.state = parcels.OperationCode.Delete


def _zonally_periodic_BC(particle, fieldset, time):
    if particle.lon < fieldset.halo_west:
        particle.lon += fieldset.halo_east - fieldset.halo_west
    elif particle.lon > fieldset.halo_east:
        particle.lon -= fieldset.halo_east - fieldset.halo_west


def _meridionally_periodic_BC(particle, fieldset, time):
    if particle.lat < fieldset.halo_south:
        particle.lat += fieldset.halo_north - fieldset.halo_south
    elif particle.lat > fieldset.halo_north:
        particle.lat -= fieldset.halo_north - fieldset.halo_south


def _doubly_periodic_BC(particle, fieldset, time):
    # because the kernel is run through code generation, we can't simply
    # call the above kernels, so we unfortunately have to reproduce them
    # in full here
    if particle.lon < fieldset.halo_west:
        particle.lon += fieldset.halo_east - fieldset.halo_west
    elif particle.lon > fieldset.halo_east:
        particle.lon -= fieldset.halo_east - fieldset.halo_west

    if particle.lat < fieldset.halo_south:
        particle.lat += fieldset.halo_north - fieldset.halo_south
    elif particle.lat > fieldset.halo_north:
        particle.lat -= fieldset.halo_north - fieldset.halo_south
