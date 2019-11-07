"""The main lagrangian-filtering module.

This module contains the crucial datastructure for
lagrangian-filtering, `LagrangeFilter`. See project documentation
for examples on how to construct a filtering workflow using this
library.

"""

import dask.array as da
import numpy as np
from datetime import timedelta
from glob import iglob
import parcels
from scipy import signal
import netCDF4
import xarray as xr

from .file import LagrangeParticleFile


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
        filenames (Dict[str, str]): A mapping from data variable names
            to the files containing the data.

            Filenames can contain globs if the data is spread across
            multiple files.
        variables_or_data (Union[Dict[str, str], xarray.Dataset]): Either
            a mapping from canonical variable names to the variable
            names in your data files, or an xarray Dataset containing
            the input data.
        dimensions (Dict[str, str]): A mapping from canonical dimension
            names to the dimension names in your data files.
        sample_variables ([str]): A list of variable names that should be sampled
            into the Lagrangian frame of reference and filtered.
        mesh (:obj:`str`, optional): The OceanParcels mesh type, either "flat"
            or "spherical". "flat" meshes are expected to have dimensions
            in metres, and "spherical" meshes in degrees.
        c_grid (:obj:`bool`, optional): Whether to interpolate velocity
            components on an Arakawa C grid (defaults to no).
        indices (:obj:`Dict[str, [int]]`, optional): An optional dictionary
            specifying the indices to which a certain dimension should
            be restricted.
        uneven_window (:obj:`bool`, optional): Whether to allow different
            lengths for the forward and backward advection phases.
        window_size (:obj:`float`, optional): The nominal length of the both
            the forward and backward advection windows, in seconds. A
            longer window may better capture the low-frequency signal to be
            removed.
        highpass_frequency (:obj:`float`, optional): The 3dB cutoff frequency
            for filtering, below which spectral components will be
            attenuated. This should be an angular frequency, in [rad/s].
        advection_dt (:obj:`datetime.timedelta`, optional): The timestep
            to use for advection. May need to be adjusted depending on the
            resolution/frequency of your data.

    """

    def __init__(
        self,
        name,
        filenames_or_dataset,
        variables,
        dimensions,
        sample_variables,
        mesh="flat",
        c_grid=False,
        indices={},
        uneven_window=False,
        window_size=None,
        highpass_frequency=5e-5,
        advection_dt=timedelta(minutes=5),
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
        self._indices = indices
        # sample variables without the "var_" prefix
        self._sample_variables = sample_variables

        # choose the fieldset constructor depending on the format
        # of the input data
        if isinstance(filenames_or_dataset, xr.Dataset):
            fieldset_constructor = parcels.FieldSet.from_xarray_dataset
        else:
            fieldset_constructor = parcels.FieldSet.from_netcdf

        # for C-grid data, we have to change the interpolation method
        fieldset_kwargs = {}
        if c_grid:
            interp_method = {}
            for v in variables:
                if v in ["U", "V", "W"]:
                    interp_method[v] = "cgrid_velocity"
                else:
                    interp_method[v] = "cgrid_tracer"

            fieldset_kwargs["interp_method"] = interp_method

        # construct the OceanParcels FieldSet to use for particle advection
        self.fieldset = fieldset_constructor(
            filenames_or_dataset,
            variables,
            dimensions,
            indices=indices,
            mesh=mesh,
            **fieldset_kwargs,
        )
        # save the lon/lat on which to seed particles
        # this is saved here because if the grid is later made periodic, the
        # underlying grids will be modified, and we'll seed particles in the halos
        if self.fieldset.gridset.grids[0].gtype in [
            parcels.GridCode.CurvilinearZGrid,
            parcels.GridCode.CurvilinearSGrid,
        ]:
            self._curvilinear = True
            self._grid_lon = self.fieldset.gridset.grids[0].lon
            self._grid_lat = self.fieldset.gridset.grids[0].lat
        else:
            self._curvilinear = False
            self._grid_lon, self._grid_lat = np.meshgrid(
                self.fieldset.gridset.grids[0].lon, self.fieldset.gridset.grids[0].lat
            )

        # starts off non-periodic
        self._is_zonally_periodic = False
        self._is_meridionally_periodic = False

        # guess the output timestep
        times = self.fieldset.gridset.grids[0].time
        self.output_dt = times[1] - times[0]

        # create the filter - use a 4th order Butterworth for the moment
        # make sure to convert angular frequency back to linear for passing to the
        # filter constructor
        fs = 1.0 / self.output_dt
        self.inertial_filter = signal.butter(
            4, highpass_frequency / (2 * np.pi), "highpass", fs=fs
        )

        # timestep for advection
        self.advection_dt = advection_dt

        # the sample variable attribute has 'var_' prepended to map to
        # variables on particles
        self.sample_variables = ["var_" + v for v in sample_variables]
        # create the particle class and kernel for sampling
        # map sampled variables to fields
        self.particleclass = ParticleFactory(sample_variables)
        self._create_sample_kernel(sample_variables)
        self.kernel = parcels.AdvectionRK4 + self.sample_kernel

        # compile kernels
        self._compile(self.sample_kernel)
        self._compile(self.kernel)

    def _create_sample_kernel(self, sample_variables):
        """Create the parcels kernel for sampling fields during advection."""

        # make sure the fieldset has C code names assigned, etc.
        self.fieldset.check_complete()

        # string for the kernel itself
        f_str = "def sample_kernel(particle, fieldset, time):\n"
        for v in sample_variables:
            f_str += f"\tparticle.var_{v} = fieldset.{v}[time, particle.depth, particle.lat, particle.lon]\n"
        else:
            f_str += "\tpass"

        # create the kernel
        self.sample_kernel = parcels.Kernel(
            self.fieldset,
            self.particleclass.getPType(),
            funcname="sample_kernel",
            funcvars=["particle", "fieldset", "time"],
            funccode=f_str,
        )

    def _compile(self, kernel):
        """Compile a kernel and tell it to load the resulting shared library."""

        kernel.compile(compiler=parcels.compiler.GNUCompiler())
        kernel.load_lib()

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
            width (:obj:`int`, optional): The width of the halo,
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
        self.fieldset.add_constant("halo_west", self.fieldset.gridset.grids[0].lon[0])
        self.fieldset.add_constant("halo_east", self.fieldset.gridset.grids[0].lon[-1])

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
            width (:obj:`int`, optional): The width of the halo,
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
        self.fieldset.add_constant("halo_north", self.fieldset.gridset.grids[0].lat[-1])
        self.fieldset.add_constant("halo_south", self.fieldset.gridset.grids[0].lat[0])

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

    def particleset(self, time):
        """Create a ParticleSet initialised at the given time.

        Args:
            time (float): The origin time for forward and backward advection
                on this ParticleSet.

        Returns:
            parcels.ParticleSet: A new ParticleSet containing a single particle
                at every gridpoint, initialised at the specified time.

        """

        # reset the global particle ID counter so we can rely on particle IDs making sense
        parcels.particle.lastID = 0

        return parcels.ParticleSet(
            self.fieldset,
            pclass=self.particleclass,
            lon=self._grid_lon,
            lat=self._grid_lat,
            time=time,
        )

    def advection_step(self, time, output_time=False):
        """Perform forward-backward advection at a single point in time.

        This routine is responsible for creating a new ParticleSet at
        the given time, and performing the forward and backward
        advection steps in the Lagrangian transformation.

        Args:
            time (float): The point in time at which to calculate filtered data.
            output_time (:obj:`bool`, optional): Whether to include "time" as
                a numpy array in the output dictionary, for doing manual analysis.

        Note:
            If ``output_time`` is True, the output object will not be compatible
            with the default filtering workflow, :func:`~filter_step`!

        Returns:
            Dict[str, (int, dask.array)]: A dictionary of the advection
                data, mapping variable names to a pair. The first element is
                the index of the sampled timestep in the data, and the
                second element is a lazy dask array concatenating the forward
                and backward advection data.

        """

        # seed all particles at gridpoints
        ps = self.particleset(time)
        # execute the sample-only kernel to efficiently grab the initial condition
        ps.kernel = self.sample_kernel
        ps.execute(self.sample_kernel, runtime=0, dt=self.advection_dt)

        # set up the temporary output file for the initial condition and
        # forward advection
        outfile = LagrangeParticleFile(ps, self.output_dt, self.sample_variables)

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

        Args:
            advection_data (Dict[str, (int, dask.array)]): A dictionary of
                particle advection data from a single timestep, returned
                from :func:`~advection_step`.

        Returns:
            Dict[str, dask.array]: A dictionary mapping sampled
                variable names to a 1D dask array containing the
                filtered data at the specified time. This data is not
                lazy, as it has already been computed out of the
                temporary advection data.

        """

        da_out = {}
        for v, a in advection_data.items():
            time_index_data, var_array = a

            def filter_select(x):
                return signal.filtfilt(*self.inertial_filter, x)[..., time_index_data]

            # apply scipy filter as a ufunc
            # mapping an array to scalar over the first axis, automatically vectorize execution
            # and allow rechunking (since we have a chunk boundary across the first axis)
            filtered = da.apply_gufunc(
                filter_select,
                "(i)->()",
                var_array,
                axis=0,
                output_dtypes=var_array.dtype,
                allow_rechunk=True,
            )

            da_out[v] = filtered.compute()

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
            is likely :obj:`np.datetime64` or :obj:`cftime.datetime`.
            For abstract times, this may simply be a number.

        Args:
            times (:obj:`[float]`, optional): A list of timesteps at
                which to run the filtering. If this is omitted, all
                timesteps that are fully covered by the filtering
                window are selected.
            clobber (:obj:`bool`, optional): Whether to overwrite any
                existing output file with the same name as this
                experiment. Default behaviour will not clobber an
                existing output file.

            absolute (:obj:`bool`, optional): If `times` is provided,
                this argument determines whether to interpret them
                as relative to the first timestep in the input dataset
                (False, default), or as absolute, following the actual
                time dimension in the dataset (True).

        """

        self(*args, **kwargs)

    def create_out(self, clobber=False):
        """Create a netCDF dataset to hold filtered output.

        Here we create a new ``netCDF4.Dataset`` for filtered
        output. For each sampled variable in the input files, a
        corresponding variable in created in the output file, with
        the same dimensions.

        Returns:
            netCDF4.Dataset: A single dataset that will hold all
                filtered output.

        """

        # index dictionary referring to variables in the source files
        indices = {self._dimensions[v]: ind for v, ind in self._indices.items()}

        # the output dataset we're creating
        ds = netCDF4.Dataset(self.name + ".nc", "w", clobber=clobber)

        # helper function to create the dimensions in the ouput file
        def create_dimension(dims, dim, var):
            # translate from parcels -> file convention
            # and check whether we've already created this dimension
            # (e.g. for a previous variable)
            file_dim = dims[dim]
            if file_dim in ds.variables:
                return ds.variables[file_dim].dimensions[
                    1
                    if len(ds.variables[file_dim].dimensions) > 1 and dim == "lon"
                    else 0
                ]

            # get the file containing the dimension data
            v_orig = self._variables.get(var, var)
            if isinstance(self._filenames, xr.Dataset):
                ds_orig = self._filenames[file_dim]
            else:
                if isinstance(self._filenames[var], dict):
                    filename = self._filenames[var][dim]
                else:
                    filename = self._filenames[var]

                if isinstance(filename, list):
                    filename = filename[0]

                ds_orig = xr.open_dataset(next(iglob(filename)))[file_dim]

            # create dimensions if needed
            for d in ds_orig.dims:
                if d not in ds.dimensions:
                    ds.createDimension(d, ds_orig[d].size)

            # create the dimension variable
            ds.createVariable(file_dim, ds_orig.dtype, dimensions=ds_orig.dims)
            ds.variables[file_dim][:] = ds_orig

            # curvilinear grid case
            return ds_orig.dims[1 if len(ds_orig.dims) > 1 and dim == "lon" else 0]

        # create a time dimension if dimensions are uniform across all variables
        if "time" in self._dimensions:
            dim_time = self._dimensions["time"]
            ds.createDimension(dim_time)
        else:
            dim_time = None

        for v in self._sample_variables:
            # translate if required (parcels -> file convention)
            v_orig = self._variables.get(v, v)

            # open all the relevant files for this variable
            if isinstance(self._filenames, xr.Dataset):
                ds_orig = self._filenames[v_orig]
            else:
                if isinstance(self._filenames[v], dict):
                    # variable -> dictionary (for separate coordinate files)
                    filename = self._filenames[v]["data"]
                else:
                    # otherwise, we just have a plain variable -> file mapping
                    filename = self._filenames[v]

                # globs can give us a list, but we only need the first item
                # to get the metadata
                if isinstance(filename, list):
                    filename = filename[0]

                ds_orig = xr.open_dataset(next(iglob(filename)))[v_orig]

            # select only the relevant indices
            # in particular, squeeze to drop z dimension if we index it out
            local_indices = {k: v for k, v in indices.items() if k in ds_orig}
            ds_orig = ds_orig.isel(**local_indices).squeeze()

            # are dimensions defined specifically for this variable?
            if v in self._dimensions:
                dims = self._dimensions[v]
            else:
                dims = self._dimensions

            # create time dimension if required (i.e. not already in the
            # output file we've created)
            out_dims = {}
            if dim_time is None:
                out_dims["time"] = dims["time"]
                if dims["time"] not in ds.dimensions:
                    ds.createDimension(dims["time"])
            else:
                out_dims["time"] = dim_time

            # for each non-time dimension, create it if it doesn't already exist
            # in the output file
            for d in ["lat", "lon"]:
                out_dims[d] = create_dimension(dims, d, v)

            # create the variable in the dataset itself
            ds.createVariable(
                "var_" + v,
                "float32",
                dimensions=(out_dims["time"], out_dims["lat"], out_dims["lon"]),
            )

        return ds

    def _window_times(self, times, absolute):
        """Restrict an array of times to those which have an adequate window,
        optionally converting from absolute to relative first.

        """

        tgrid = self.fieldset.gridset.grids[0].time

        if times is None:
            times = tgrid.copy()

        if absolute:
            times = self.fieldset.gridset.grids[0].time_origin.reltime(times)

        times = np.array(times)
        window_left = times - tgrid[0] >= self.window_size
        window_right = times <= tgrid[-1] - self.window_size
        return times[window_left & window_right]

    def __call__(self, times=None, absolute=False, clobber=False):
        """Run the filtering process on this experiment."""

        if self.uneven_window:
            raise NotImplementedError("uneven windows aren't supported")

        # either restrict the specified times to period covered by window,
        # or use the full range of times covered by window
        times = self._window_times(times, absolute)

        ds = self.create_out(clobber=clobber)

        # do the filtering at each timestep
        for idx, time in enumerate(times):
            # returns a dictionary of sample_variable -> dask array
            filtered = self.filter_step(self.advection_step(time))
            for v, a in filtered.items():
                ds[v][idx, ...] = a

        ds.close()


def ParticleFactory(variables, name="SamplingParticle", BaseClass=parcels.JITParticle):
    """Create a Particle class that samples the specified variables.

    The variables that should be sampled will be prepended by ``var_`` as
    class attributes, in case there are any namespace clashes with existing
    variables on the base class.

    Args:
        variables ([str]): A list of variable names which should be sampled.
        name (str): The name of the generated particle class.
        BaseClass (Type[parcels.particle._Particle]): The base particles class upon
            which to append the required variables.

    Returns:
        Type[parcels.particle._Particle]: The new particle class

    """

    var_dict = {"var_" + v: parcels.Variable("var_" + v) for v in variables}

    newclass = type(name, (BaseClass,), var_dict)
    return newclass


def _recovery_kernel_out_of_bounds(particle, fieldset, time):
    """Recovery kernel for particle advection, to delete out-of-bounds particles."""

    particle.state = parcels.ErrorCode.Delete


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
