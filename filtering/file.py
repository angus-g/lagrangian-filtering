"""ParticleFile implementations for receiving particle advection data from OceanParcels.

Parcels defines the `ParticleFile` class, which has a ``write`` method
to write a `ParticleSet` to disk. The frequency at which this is
called is determined by the ``outputdt`` property.

"""

import tempfile

import h5py
import numpy as np
from parcels import ErrorCode


class LagrangeParticleFile(object):
    """A ParticleFile for on-disk caching in a temporary HDF5 file.

    A temporary HDF5 file is used to store advection data. Data is
    stored in 2D, with all particles contiguous as they appear in
    their particle set. The `time` dimension is extendable, and is
    appended for each write operation. This means that the number of
    writes does not need to be known ahead-of-time.

    Important:
        Before calling the particle kernel, a group must be created in the
        file by calling :func:`~filtering.file.LagrangeParticleFile.set_group`.

    Example:
        Create an instance of the class, and run forward advection on
        a `ParticleSet`::

            f = LagrangeParticleFile(ps, output_dt)
            f.set_group("forward")
            ps.execute(kernel, dt=advection_dt, output_file=f)

    Note:
        Advection data is stored to a `NamedTemporaryFile` that is
        scoped with the same lifetime as this `ParticleFile`. This
        should ensure that upon successful completion, the temporary
        files are cleaned up, yet they will remain if an error occurs
        that causes an exception.

    Args:
        particleset (parcels.particleset.ParticleSet): The particle set for which this file
            should cache advection data on disk. It's assumed the number of particles
            contained within the set does not change after initialisation.
        outputdt (Optional[float]): The frequency at which advection data
            should be saved. If not specified, or infinite, the data will be saved
            at the first timestep only.
        variables (Optional[parcels.particle.Variable]): An explicit list subset of
            particletype variables to output. If not specified, all variables
            belonging to the particletype that are ``to_write`` are written.

    """

    def __init__(self, particleset, outputdt=np.infty, variables=None):
        self.outputdt = outputdt

        self.n = len(particleset)

        self._tempfile = tempfile.NamedTemporaryFile(dir=".", suffix=".h5")
        self.h5file = h5py.File(self._tempfile, "w")

        # using upstream parcels, it'll try to read this attribute
        # when advection is taking longer than 10 seconds
        self.tempwritedir_base = self._tempfile.name

        # variable -> dtype map for creating datasets
        self._vars = {}

        for v in particleset.ptype.variables:
            # this variable isn't marked for output to file -- honour that
            if not v.to_write:
                continue

            # there's an explicit list of variables for us to write, so
            # filter based on that (e.g. outputting only sample_variables)
            if variables is not None and v.name not in variables:
                continue

            self._vars[v.name] = v.dtype

    def set_group(self, group):
        """Set the group for subsequent write operations.

        This will create the group, and datasets for all variables
        that will be written by this object, if they do not already
        exist. Otherwise, the group will simply be selected without
        overwriting existing data.

        Args:
            group (str): The name of the group.

        """

        self._group = self.h5file.require_group(group)
        if "time" not in self._group.attrs:
            # initialise time attribute
            self._group.attrs["time"] = []
        self._var_datasets = {}
        for v, t in self._vars.items():
            self._var_datasets[v] = self._group.require_dataset(
                v, shape=(0, self.n), maxshape=(None, self.n), dtype=t
            )

    def data(self, group):
        """Return a group from the HDF5 object.

        Each variable saved from particle advection is available as a
        `Dataset` within the group, as well the ``time`` attribute.

        Args:
            group (str): The name of the group to retrieve.

        Returns:
            :class:`h5py.Group <Group>`: The group from the underlying HDF5 file.
                If the group hasn't been initialised with
                :func:`~filtering.file.LagrangeParticleFile.set_group`,
                it will be empty.

        """

        return self.h5file.require_group(group)

    def write(self, particleset, time, deleted_only=False):
        """Write data from a particle set.

        This is intended to be called from a particle execution
        kernel. The frequency of writes is determined by the
        ``outputdt`` attribute on the class.

        Particles which have been deleted (due to becoming out of
        bounds, for example) are masked with NaN.

        Args:
            particleset (parcels.particleset.ParticleSet): Particle set with data
                to write to disk.
            time (float): Timestamp into particle execution at which this
                write was called.
            deleted_only (Optional[bool]): Whether to only write deleted
                particles (does not do anything, only present for compatibility
                with the call signature on the parcels version of the class).

        """

        # don't write out deleted particles
        if deleted_only is not False:
            return

        self._group.attrs["time"] = np.append(self._group.attrs["time"], time)

        # indices of particles still alive
        idx = particleset.particle_data["id"]

        for v, d in self._var_datasets.items():
            # first, resize all datasets to add another entry in the time dimension
            # then we can just pull the array for this variable out of the particleset
            d.resize(d.shape[0] + 1, axis=0)
            # data defaults to nans, and we only fill in the living particles
            d[-1, :] = np.nan
            d[-1, idx] = particleset.particle_data[v]
