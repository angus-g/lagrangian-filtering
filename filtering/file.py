"""ParticleFile implementations for receiving particle advection data from OceanParcels.

Parcels defines the `ParticleFile` class, which has a ``write`` method
to write a `ParticleSet` to disk. The frequency at which this is
called is determined by the ``outputdt`` property.

"""

import tempfile

import h5py
import numpy as np
from parcels import ErrorCode


class Placeholder(object):
    """A placeholder type for an advection array that hasn't yet been allocated."""

    def __init__(self, obj):
        self.value = obj


class BaseParticleCache(object):
    def __init__(self, particleset, outputdt):
        self.outputdt = outputdt
        self.n = len(particleset)
        self._variables = particleset._collection.ptype.variables


class LagrangeParticleFile(BaseParticleCache):
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
        particleset (parcels.particlesets.particlesetsoa.ParticleSetSOA): The particle set
            for which this file should cache advection data on
            disk. It's assumed the number of particles contained
            within the set does not change after initialisation.
        outputdt (Optional[float]): The frequency at which advection data
            should be saved. If not specified, or infinite, the data will be saved
            at the first timestep only.
        variables (Optional[List[parcels.particle.Variable]]): An explicit list subset of
            particletype variables to output. If not specified, all variables
            belonging to the particletype that are ``to_write`` are written.
        output_dir (Optional[str]): The directory in which to place the temporary
            output file.

    """

    def __init__(
        self,
        particleset,
        outputdt=np.infty,
        variables=None,
        write_once_variables=None,
        output_dir=".",
    ):
        super().__init__(particleset, outputdt)

        self._tempfile = tempfile.NamedTemporaryFile(dir=output_dir, suffix=".h5")
        self.h5file = h5py.File(self._tempfile, "w")

        # using upstream parcels, it'll try to read this attribute
        # when advection is taking longer than 10 seconds
        self.tempwritedir_base = self._tempfile.name

        # variable -> dtype map for creating datasets
        self._vars = {}

        variables = variables or []
        write_once_variables = write_once_variables or []

        for v in self._variables:
            # this variable isn't marked for output to file -- honour that
            if not v.to_write:
                continue

            # there's an explicit list of variables for us to write, so
            # filter based on that (e.g. outputting only sample_variables)
            if v.name in variables:
                self._vars[v.name] = v.dtype

            elif v.name in write_once_variables:
                self._vars[v.name] = Placeholder(v.dtype)

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
            if isinstance(t, Placeholder):
                self._var_datasets[v] = self._group.require_dataset(
                    v, shape=(self.n,), dtype=t.value
                )
            else:
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
            :class:`h5py.Group`: The group from the underlying HDF5 file.
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
            particleset (parcels.particlesets.particlesetsoa.ParticleSetSOA): Particle
                set with data to write to disk.
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
        idx = particleset.id

        for v, d in self._var_datasets.items():
            if isinstance(d, Placeholder):
                tmp = np.empty(d.value.shape)
                tmp[:] = np.nan
                tmp[idx] = getattr(particleset, v)

                d.value[:] = tmp

                self._var_datasets[v] = d.value

            elif d.ndim == 2:
                # data defaults to nans, and we only fill in the living particles
                tmp = np.empty(d.shape[1])
                tmp[:] = np.nan
                tmp[idx] = getattr(particleset, v)

                # resize all datasets to add another entry in the time dimension
                # then we can just pull the array for this variable out of the particleset
                d.resize(d.shape[0] + 1, axis=0)
                d[-1, :] = tmp


class LagrangeParticleArray(BaseParticleCache):
    """A ParticleFile for in-memory caching of advected data.

    For smaller spatial extents, or sufficient memory, it is easier to
    work with in-memory arrays to cache advection data.

    Important:
        This requires a bit more management than
        :class:`~filtering.file.LagrangeParticleFile`: after forward
        advection, reverse the data with
        :func:`~filtering.file.LagrangeParticleArray.reverse_data`
        then skip the first output of backward advection (the sampling
        of initial particle positions) with
        :func:`~filtering.file.LagrangeParticleArray.set_skip`.

    Args:
        particleset (parcels.particlesets.particlesetsoa.ParticleSetSOA): The particle set
            for which advection data will be cached. We use this to get
            the names and types of sampled variables.
        outputdt (Optional[float]): The frequency at which advection data
            should be saved. If not specified, or infinite, the data will
            be saved at the first timestep only.
        variables (Optional[List[str]]): An explicit
            subset of variables to output. If not specified, all
            variables belonging to the particleset's particletype that
            are ``to_write`` are written.
        write_once_variables (Optional[List[str]]): An explicit subset of
            variables to write only on the first write call.

    """

    def __init__(
        self, particleset, outputdt=np.infty, variables=None, write_once_variables=None
    ):
        super().__init__(particleset, outputdt)

        self.skip = 0

        # parcels will read this attribute for printing a message
        self.tempwritedir_base = "<nonexistent>"

        # dictionary containing cached arrays for each variable
        self._vars = {}

        # default to empty lists
        variables = variables or []
        write_once_variables = write_once_variables or []

        for v in self._variables:
            if not v.to_write:
                continue

            # explicit list of variables to write, so filter based on that
            if v.name in variables:
                self._vars[v.name] = np.empty((self.n, 0), dtype=v.dtype)

            elif v.name in write_once_variables:
                self._vars[v.name] = Placeholder(v.dtype)

    def set_skip(self, n):
        """Skip a number of subsequent output steps.

        This is particularly useful to ignore the first advection output,
        which is the values of particles before the kernel is called, and
        often contains junk unless an explicit zero-time sampling kernel
        is used.

        Args:
            n (int): The number of output steps to skip.

        """

        self.skip = n

    def reverse_data(self):
        """Reverse all cached advection data.

        This can be used before and after a backward advection to
        make sure the data is correctly ordered.

        """

        for v, d in self._vars.items():
            self._vars[v] = d[:, ::-1]

    def write(self, particleset, time, deleted_only=False):
        """Write data from a particle set.

        This is intended to be called from a particle execution
        kernel. The frequency of writes is determined by the
        ``outputdt`` attribute on the class.

        Particles which have been deleted (due to becoming out of
        bounds, for example) are masked with NaN.

        Args:
            particleset (parcels.particlesets.particlesetsoa.ParticleSetSOA): Particle
                set with data to write to disk.
            time (float): Timestamp into particle execution at which this
                write was called.
            deleted_only (Optional[bool]): Whether to only write deleted
                particles (does not do anything, only present for compatibility
                with the call signature on the parcels version of the class).

        """

        # don't write out deleted particles
        if deleted_only is not False:
            return

        if self.skip > 0:
            self.skip -= 1
            return

        # indices of particles still alive
        idx = particleset.id

        for v, d in self._vars.items():
            if isinstance(d, Placeholder):
                # write-once variables
                arr = np.empty((self.n,), dtype=d.value)
                arr[:] = np.nan
                arr[idx] = getattr(particleset, v)

                self._vars[v] = arr

            elif d.ndim == 2:
                # next slice of data
                next_d = np.empty((self.n, 1), dtype=d.dtype)
                next_d[:] = np.nan
                next_d[idx, 0] = getattr(particleset, v)

                self._vars[v] = np.hstack((d, next_d))
