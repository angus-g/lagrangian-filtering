"""An implementation of dask-based distributed array filtering.

This module contains classes and functions for setting up and
and applying filtering over fairly large datasets, by distributing
the advection and filtering steps over chunks of a distributed array.

"""

import dask
import dask.array as da
import numpy as np
import parcels

from filtering.file import LagrangeParticleArray
from filtering.filtering import LagrangeFilter

# filter module, needs to contain metadata about how to
# construct parcels on a single worker, etc.


class DistributedFilter(object):
    """The main metadata class for distributed filtering.

    Args:
        load (Callable[[], xarray.Dataset]): A function that, when
            called, will load the input data into a (lazy) xarray Dataset.
        variables: (Dict[str, str]): A mapping from canonical variable names
            to the variable names in your data files.
        dimensions (Union[Dict[str, str], Dict[str, Dict[str, str]]]): A
            mapping from canonical dimension names to the dimension
            names in your data files.
        sample_variables (List[str]): A list of variable names that should be
            sampled and filtered.
        **kwargs (Optional): Additional arguments passed to
            :class:`~filtering.filtering.LagrangeFilter`

"""

    def __init__(self, load, variables, dimensions, sample_variables, **kwargs):
        self.load = load
        self.variables = variables
        self.dimensions = dimensions
        self.sample_variables = sample_variables
        self.filter_kwargs = kwargs

    def create_seeds(self, chunks):
        """Create the chunked array of seed positions.

        This will automatically load the lon/lat data from the
        input data and use that for particle positions. The chunks
        should be a two-element tuple with chunk sizes.

        Args:
            chunks (Tuple[int, int]): The chunk sizes in the X and Y
                dimensions, respectively.

        Returns:
            dask.array.Array: The seed locations as a chunked dask array.

        """

        seed_variable = "U"
        if self.particle_grid is not None:
            seed_variable = self.particle_grid

        data = self.load()
        lon = data[self.dimensions[seed_variable]["lon"]]
        lat = data[self.dimensions[seed_variable]["lat"]]

        return da.from_array(
            np.dstack(np.meshgrid(lon, lat)), chunks + (None,), name="seeds"
        )

    def create_filter(self, data):
        """Create an instance of :class:`~filtering.filtering.LagrangeFilter`

        Because parcels is extremely stateful, we can't share particle class
        or FieldSet data between workers. As part of a worker's spin-up on a
        given chunk, it needs to initialise these things.

        Args:
            data (xarray.Dataset): An already-loaded Dataset, use for the
                FieldSet construction.

        Returns:
            filtering.filtering.LagrangeFilter: The filtering state object.

        """

        # XXX: cache kernel compilation
        f = LagrangeFilter(
            "xarray_meta",
            data,
            self.variables,
            self.dimensions,
            self.sample_variables,
            deferred_load=True,
            **self.filter_kwargs,
        )

        return f

    def particleset(self, f, time, block):
        """Seed particles for a given block at a particular time.

        Args:
            f (filtering.filtering.LagrangeFilter): The filter state,
                initialised by :func:`~create_filter`.
            time (float): The time at which the particles should be
                initailised (i.e. the centre of their window).
            block (numpy.ndarray): A stacked array of seed locations.

        Returns:
            parcels.particleset.ParticleSet: A new ParticleSet containing
                a single particle at every gridpoint, initialised at the
                specified time.

        """

        # reset particle IDs so we can use them for indexing
        # into the output array
        f.particleclass.setLastID(0)
        return parcels.ParticleSet(
            f.fieldset,
            pclass=f.particleclass,
            lon=block[..., 0],
            lat=block[..., 1],
            time=time,
        )

    def advect_block(self, f, time, block):
        """Perform the advection step on a single chunk.

        Args:
            f (filtering.filtering.LagrangeFilter): The filter state,
                initialised by :func:`~create_filter`.
            time (float): The time at which the particles should be
                initailised (i.e. the centre of their window).
            block (numpy.ndarray): A stacked array of seed locations.

        Returns:
            filtering.file.LagrangeParticleArray: The cached advection
                sampling data.

        """

        ps = self.particleset(f, time, block)
        outarray = LagrangeParticleArray(ps, f.output_dt, f.sample_variables)

        # create advection coroutine
        adv = f._advect(ps, outarray)
        next(adv)  # forward advection

        # reset loaded chunks and set up particleset for backward advection
        for g in f.fieldset.gridset.grids:
            g.load_chunk[:] = 0
        outarray.reverse_data()
        outarray.set_skip(1)
        adv.send(self.particleset(f, time, block))

        # re-reverse to put data the right way around
        outarray.reverse_data()
        return outarray

    def _process_block(self, time, block, block_info=None):
        """Process a single seed chunk (in a dask worker).

        This function is called from :func:`~filtered`, and thus expects
        the :func:`~dask.array.map_blocks` calling convention.

        """

        f = self.create_filter(self.load())

        # run advection single-threaded for this worker, otherwise
        # data loading locks or communicates too much
        with dask.config.set(scheduler="single-threaded"):
            adv_data = self.advect_block(f, time, block)

        return filter_block(
            f, block, block_info[1]["array-location"][:2], adv_data._vars
        )

    def filtered(self, time, chunks):
        """Perform the filtering workflow at a given time.

        Args:
            time (float): The point in time at which to calculate filtered data.
            chunks (Tuple[int, int]): The chunk sizes in the X and Y
                dimensions, respectively.

        Returns:
            dask.array.Array: A dask array containing the filtered result for
                every input chunk.

        """

        seeds = self.create_seeds(chunks)

        # process each chunk of seeds at time: expect an object (a
        # singular array containing an xarray dataset) drop the x/y
        # stack axis
        return da.map_blocks(
            self._process_block,
            time,
            seeds.blocks,
            dtype=object,
            drop_axis=-1,
            chunks=(1, 1),
        )
