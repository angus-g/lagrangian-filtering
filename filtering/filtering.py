import numpy as np
from datetime import timedelta
import h5py
import parcels
import tempfile


class LagrangeParticleFile(object):
    """A specialised ParticleFile class for efficient input/output of
    temporary particle data.

    All variables that are marked "to_write" are written to a temporary HDF5 file.
    """

    def __init__(self, particleset, outputdt=np.infty, variables=None):
        self.outputdt = outputdt

        self.n = len(particleset)

        self._tempfile = tempfile.NamedTemporaryFile(dir=".", suffix=".h5")
        self.h5file = h5py.File(self._tempfile)

        self.var_datasets = {}
        for v in particleset.ptype.variables:
            # this variable isn't marked for output to file -- honour that
            if not v.to_write:
                continue

            # there's an explicit list of variables for us to write, so
            # filter based on that (e.g. outputting only sample_variables)
            if variables is not None and v.name not in variables:
                continue

            self.var_datasets[v.name] = self.h5file.create_dataset(
                v.name, (0, self.n), maxshape=(None, self.n), dtype=v.dtype
            )

    def write(self, particleset, time, deleted_only=False):
        """Write particle data in the particleset at time to this ParticleFile's temporary dataset."""

        # don't write out deleted particles
        if deleted_only:
            return

        for v, d in self.var_datasets.items():
            # first, resize all datasets to add another entry in the time dimension
            d.resize(d.shape[0] + 1, axis=0)

            # allocate enough space for all particles' data
            arr = np.empty(self.n)
            # special case: set ids to -1 ahead of time so we can tell when particle data is deleted
            if v == "id":
                arr[:] = -1

            for p in particleset:
                arr[p.id] = getattr(p, v)

            d[-1, :] = arr


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

        # construct the OceanParcels FieldSet to use for particle advection
        self.fieldset = parcels.FieldSet.from_netcdf(
            filenames, variables, dimensions, mesh=mesh
        )

        # guess the output timestep
        times = self.fieldset.gridset.grids[0].time
        self.output_dt = times[1] - times[0]

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

        # seed all particles at gridpoints and advect forwards
        ps = self.particleset(time)
        outfile_forward = LagrangeParticleFile(
            ps, self.output_dt, self.sample_variables
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
        outfile_backward = LagrangeParticleFile(
            ps, self.output_dt, self.sample_variables
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
