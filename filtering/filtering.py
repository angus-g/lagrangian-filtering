import numpy as np
from datetime import timedelta
import parcels


def ParticleFactory(variables, name="SamplingParticle", BaseClass=parcels.JITParticle):
    """Create a Particle class that samples the specified variables.

    variables is a dictionary mapping variable names to a field from which
    the initial value of the variable should be sampled. The variable
    attributes on the Particle class are prepended by 'var_'."""

    var_dict = {
        "var_" + v: parcels.Variable("var_" + v, initial=f)
        for v, f in variables.items()
    }

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

        # construct the OceanParcels FieldSet
        # to use for particle advection
        self.fieldset = parcels.FieldSet.from_netcdf(
            filenames, variables, dimensions, mesh=mesh, deferred_load=False
        )

        # guess the output timestep
        times = self.fieldset.gridset.grids[0].time
        self.output_dt = times[1] - times[0]

        # timestep for advection
        self.advection_dt = advection_dt

        # map sampled variables to fields
        self.sample_fields = {v: getattr(self.fieldset, v) for v in sample_variables}
        # create the particle class and kernel for sampling
        self.particleclass = ParticleFactory(self.sample_fields)
        self.create_sample_kernel()
        self.kernel = parcels.AdvectionRK4 + self.sample_kernel

    def create_sample_kernel(self):
        """Create the parcels kernel for sampling fields during advection."""

        # make sure the fieldset has C code names assigned, etc.
        self.fieldset.check_complete()

        # string for the kernel itself
        f_str = "def sample_kernel(particle, fieldset, time):\n"
        for v in self.sample_fields.keys():
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

        return parcels.ParticleSet(
            self.fieldset, pclass=self.particleclass, lon=lon, lat=lat, time=time
        )

    def filter_step(self, time_index, time):
        """Perform forward-backward advection at a single timestep."""

        # seed all particles at gridpoints and advect forwards
        ps = self.particleset(time)
        outfile_forward = ps.ParticleFile(
            "{}_{}_forward".format(self.name, time_index), outputdt=self.output_dt
        )
        ps.execute(
            self.kernel,
            runtime=self.window_size,
            dt=self.advection_dt,
            output_file=outfile_forward,
            recovery={
                parcels.ErrorCode.ErrorOutOfBounds: recovery_kernel_out_of_bounds
            },
        )

        # reseed particles, but advect backwards (using negative dt)
        ps = self.particleset(time)
        outfile_backward = ps.ParticleFile(
            "{}_{}_backward".format(self.name, time_index), outputdt=self.output_dt
        )
        ps.execute(
            self.kernel,
            runtime=self.window_size,
            dt=-self.advection_dt,
            output_file=outfile_backward,
            recovery={
                parcels.ErrorCode.ErrorOutOfBounds: recovery_kernel_out_of_bounds
            },
        )

        # filter data from output files and write back this timestep
        # first, tell the output files to not automatically export to netcdf
        outfile_forward.to_export = False
        outfile_backward.to_export = False

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

        # do the filtering at each timestep
        for idx, time in enumerate(times):
            self.filter_step(idx, time)
