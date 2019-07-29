import dask.array as da
import numpy as np
from datetime import timedelta
import parcels
from scipy import signal

from .file import LagrangeParticleFile


def ParticleFactory(variables, name="SamplingParticle", BaseClass=parcels.JITParticle):
    """Create a Particle class that samples the specified variables.

    variables is a dictionary mapping variable names to a field from which
    the initial value of the variable should be sampled. The variable
    attributes on the Particle class are prepended by 'var_'."""

    var_dict = {"var_" + v: parcels.Variable("var_" + v) for v, f in variables.items()}

    newclass = type(name, (BaseClass,), var_dict)
    return newclass


def recovery_kernel_out_of_bounds(particle, fieldset, time):
    """Recovery kernel for particle advection, to delete out-of-bounds particles."""

    particle.state = parcels.ErrorCode.Delete


class LagrangeFilter(object):
    """The main Lagrangian filter class, holds all the required state.
    """

    def __init__(
        self,
        name,
        filenames,
        variables,
        dimensions,
        sample_variables,
        mesh="flat",
        indices=None,
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

        # construct the OceanParcels FieldSet to use for particle advection
        self.fieldset = parcels.FieldSet.from_netcdf(
            filenames, variables, dimensions, indices=indices, mesh=mesh
        )

        # guess the output timestep
        times = self.fieldset.gridset.grids[0].time
        self.output_dt = times[1] - times[0]

        # create the filter - use a 4th order Butterworth for the moment
        fs = 1.0 / self.output_dt
        self.inertial_filter = signal.butter(4, highpass_frequency, "highpass", fs=fs)

        # timestep for advection
        self.advection_dt = advection_dt

        # the sample variable attribute has 'var_' prepended to map to
        # variables on particles
        self.sample_variables = ["var_" + v for v in sample_variables]
        # create the particle class and kernel for sampling
        # map sampled variables to fields
        self.particleclass = ParticleFactory(
            {v: getattr(self.fieldset, v) for v in sample_variables}
        )
        self.create_sample_kernel(sample_variables)
        self.kernel = parcels.AdvectionRK4 + self.sample_kernel

        # compile kernels
        self.sample_kernel.compile(compiler=parcels.compiler.GNUCompiler())
        self.sample_kernel.load_lib()

        self.kernel.compile(compiler=parcels.compiler.GNUCompiler())
        self.kernel.load_lib()

    def create_sample_kernel(self, sample_variables):
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

    def particleset(self, time):
        """Create a particleset initialised at the given time."""

        # we make the assumption that the grid is rectilinear for the moment
        lon, lat = np.meshgrid(
            self.fieldset.gridset.grids[0].lon, self.fieldset.gridset.grids[0].lat
        )

        # reset the global particle ID counter so we can rely on particle IDs making sense
        parcels.particle.lastID = 0

        return parcels.ParticleSet(
            self.fieldset, pclass=self.particleclass, lon=lon, lat=lat, time=time
        )

    def filter_step(self, time_index, time):
        """Perform forward-backward advection at a single timestep."""

        # seed all particles at gridpoints
        ps = self.particleset(time)
        # set up the temporary output file for the initial condition and
        # forward advection
        outfile_forward = LagrangeParticleFile(
            ps, self.output_dt, self.sample_variables
        )
        # execute the sample-only kernel to efficiently grab the initial condition
        ps.kernel = self.sample_kernel
        ps.execute(self.sample_kernel, runtime=0, dt=self.advection_dt)
        # if sampled data is on the same grid as e.g. velocity data, but velocities aren't
        # sampled, parcels will incorrectly think they're already loaded
        # reset all the chunk loading state for velocities so the advection kernel
        # works correctly
        self.fieldset.U.grid.load_chunk = []
        self.fieldset.V.grid.load_chunk = []

        # now the forward advection kernel can run
        ps.kernel = self.kernel
        ps.execute(
            self.kernel,
            runtime=self.window_size,
            dt=self.advection_dt,
            output_file=outfile_forward,
            recovery={
                parcels.ErrorCode.ErrorOutOfBounds: recovery_kernel_out_of_bounds
            },
        )

        # reseed particles back on the grid, then advect backwards
        # we don't need any initial condition sampling since we've already done it
        ps = self.particleset(time)
        outfile_backward = LagrangeParticleFile(
            ps, self.output_dt, self.sample_variables
        )
        ps.kernel = self.kernel
        ps.execute(
            self.kernel,
            runtime=self.window_size,
            dt=-self.advection_dt,
            output_file=outfile_backward,
            recovery={
                parcels.ErrorCode.ErrorOutOfBounds: recovery_kernel_out_of_bounds
            },
        )

        da_out = {}

        # stitch together and filter all sample variables from the temporary
        # output data
        for v in self.sample_variables:
            # load data lazily as dask arrays, for forward and backward segments
            var_array_forward = da.from_array(
                outfile_forward.h5file[v], chunks=(None, "auto")
            )
            var_array_backward = da.from_array(
                outfile_backward.h5file[v], chunks=(None, "auto")
            )

            # get an index into the middle of the array
            time_index_data = var_array_backward.shape[0]

            # construct proper sequence by concatenating data and flipping the backward segment
            # for var_array_forward, skip the initial output for both the sample-only and
            # sample-advection kernels, which have meaningless data
            var_array = da.concatenate(
                (da.flip(var_array_backward[1:, :], axis=0), var_array_forward)
            )

            def filter_select(x):
                return signal.filtfilt(*self.inertial_filter, x)[time_index_data]

            # apply scipy filter as a ufunc
            # mapping an array to scalar over the first axis, automatically vectorize execution
            # and allow rechunking (since we have a chunk boundary across the first axis)
            filtered = da.apply_gufunc(
                filter_select,
                "(i)->()",
                var_array,
                axis=0,
                vectorize=True,
                output_dtypes=var_array.dtype,
                allow_rechunk=True,
            )

            da_out[v] = filtered

        return da_out

    def __call__(self, times=None):
        """Run the filtering process on this experiment."""

        # run over the full range of valid time indices unless specified otherwise
        if times is None:
            times = self.fieldset.gridset.grids[0].time

            if self.uneven_window:
                raise NotImplementedError("uneven windows aren't supported")

        # restrict to period covered by window
        times = np.array(times)
        window_left = times - times[0] >= self.window_size
        window_right = times <= times[-1] - self.window_size
        times = times[window_left & window_right]

        da_out = {v: [] for v in self.sample_variables}

        # do the filtering at each timestep
        for idx, time in enumerate(times):
            # returns a dictionary of sample_variable -> dask array
            filtered = self.filter_step(idx, time)
            for v, a in filtered.items():
                da_out[v].append(a)

        # dump all to disk
        da.to_hdf5(self.name + ".h5", {v: da.stack(a) for v, a in da_out.items()})
