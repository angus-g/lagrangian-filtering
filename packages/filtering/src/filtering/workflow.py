"""Filtering windows composed from independent directional trajectories.

"""

import tempfile
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from datetime import timedelta

import dask.array as da
import numpy as np
import optype.numpy as onp
import xarray as xr
from numpy.typing import ArrayLike, NDArray
from tracking import Flow, Seeds, Tracker

from ._types import Date, DateArray, FilterLike, PathLike, Timespan
from .filter import Filter


def _seconds(
    value: Timespan,
    name: str,
    *,
    allow_zero: bool = False
):
    """Unified conversion from timedelta to number of seconds.

    """

    if isinstance(value, timedelta):
        value = value.total_seconds()

    if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}"
        )
    return value


@dataclass
class WindowSamples:
    """Lazy chronological samples; use as a context manager for disk caching.

    Disk-backed arrays must be consumed before close(). In-memory windows need
    no cleanup. centre_index points at the single seed-time sample.

    """

    times: NDArray[np.float64]
    centre_index: int
    samples: dict
    status: da.Array
    _temporary: tempfile.TemporaryDirectory | None = None

    def close(self):
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class WindowFilter:
    """A thin filtering workflow over the lagrangian-tracking package.

    Note:
        Durations (window size, sample/advection dt) may be specified as a
        scalar seconds value, or a datetime.timedelta.

    Args:
        flow: A tracking Flow or Tracker for handling input data and advection.
        seeds: Particle seed locations, constructed by the flow grid.
        window: Either the forward/backward filtering window length, or a two-
            element tuple with (backward, forward) durations.
        sample_dt: How often to sample output data along the particle trajectories.
        advection_dt: Timestep for particle advection.
        cutoff_frequency: If not none, construct a default frequency-based
            filter with he highpass cutoff frequency [Hz].
        reducer: If not none, a pre-constructed filter reduction object that
            supports the `apply_filter` method. Must be provided if cutoff
            frequency isn't.
        sample_variables: List of output variable names among the sampled Fields
            registered in the Flow. These are sampled along particle trajectories.
            If not specified, defaults to all tracer fields present on the Flow.
        minimum_window: Discard trajectories with incomplete windows that are
            shorter than this duration.
        cache_dir: Temporary directory for disk-backed caching of particle
            trajectories, if there would be more data than exceeds available
            physical memory.

    """

    def __init__(
        self,
        flow: Flow | Tracker,
        seeds: Seeds,
        *,
        window: Timespan | tuple[Timespan, Timespan] | list[Timespan],
        sample_dt: Timespan,
        advection_dt: Timespan,
        cutoff_frequency: float | onp.ToFloat1D | None = None,
        reducer: FilterLike | None = None,
        sample_variables: Iterable[str] | None = None,
        minimum_window: Timespan | None = None,
        cache_dir: PathLike | None = None,
    ) -> None:
        if isinstance(flow, Tracker):
            flow = Flow(flow)

        if not isinstance(flow, Flow):
            raise TypeError("flow must be a Flow or Tracker")
        if not isinstance(seeds, Seeds):
            raise TypeError("seeds must be constructed by the flow grid")

        self.flow = flow
        # copy so that the workflow isn't tied to the external seed object:
        # we use the snapshot at the time the workflow is constructed
        self.seeds = Seeds(
            seeds.x.copy(),
            seeds.y.copy(),
            seeds.status.copy(),
            layout=seeds.layout,
        )

        if self.seeds.x.size == 0:
            raise ValueError("at least one seed is required")

        self.sample_dt = _seconds(sample_dt, "sample_dt")
        self.advection_dt = _seconds(advection_dt, "advection_dt")

        sides = window if isinstance(window, (tuple, list)) else (window, window)
        if len(sides) != 2:
            raise ValueError("window must be a duration or (backward, forward) pair")
        self.backward, self.forward = [
            _seconds(s, "window", allow_zero=True) for s in sides
        ]

        counts = np.array([self.backward, self.forward]) / self.sample_dt
        if not np.allclose(counts, np.rint(counts), rtol=0, atol=1.0e-9):
            raise ValueError("each window duration must be a multiple of sample_dt")
        self.nback, self.nforward = np.rint(counts).astype(int)

        self.variables = tuple(
            flow.fields if sample_variables is None else sample_variables
        )
        if not self.variables or len(set(self.variables)) != len(self.variables):
            raise ValueError("sample_variables must be nonempty and unique")

        unknown = set(self.variables) - flow.fields.keys()
        if unknown:
            raise ValueError(f"unknown sampled fields: {sorted(unknown)}")

        # redundant check is for type narrowing, retain
        if reducer is None:
            if cutoff_frequency is None:
                raise ValueError("provide exactly one of cutoff_frequency or reducer")

            self.reducer = Filter(cutoff_frequency, 1 / self.sample_dt)
        else:
            if cutoff_frequency is not None:
                raise ValueError("provide exactly one of cutoff_frequency or reducer")
            self.reducer = reducer

        if not callable(getattr(self.reducer, "apply_filter", None)):
            raise TypeError("reducer must implement apply_filter")

        self.minimum_samples = None
        if minimum_window is not None:
            duration = _seconds(minimum_window, "minimum_window", allow_zero=True)
            if duration > self.backward + self.forward:
                raise ValueError("minimum_window exceeds the full window duration")
            self.minimum_samples = int(np.ceil(duration / self.sample_dt)) + 1

        self.cache_dir = cache_dir

    def _coverage(self):
        return self.flow.coverage(self.variables)

    def valid_times(self, times: float | Date | DateArray | None = None) -> NDArray[np.float64]:
        """Candidate centre times with complete input coverage, in seconds.

        Args:
            times: If specified, an array of centre times instead of defaulting
                to the time dimension of the Flow.

        """

        values = self.flow.times if times is None else self.flow.relative_times(times)
        values = np.atleast_1d(values)

        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("centre times must be a finite one-dimensional sequence")

        lower, upper = self._coverage()

        return values[
            (values - self.backward >= lower) & (values + self.forward <= upper)
        ]

    def on_seeds(self, result: xr.Dataset) -> xr.Dataset:
        """Restore particle-indexed resuls to their original seed grid.

        Supports filter_at() and run() results, provided the original
        grid was spatially-based (e.g. created with Grid.seed_grid()).

        Args:
            result: A result dataset from an application of `filter_at()` or `run()`.

        Returns:
            The result dataset with dimensions and coordinates matching the
            original seed grid for the particles.

        """

        layout = self.seeds.layout
        if layout is None:
            raise ValueError(
                "These seeds have no recorded grid layout; "
                "use seed_grid() or supply a SeedLayout."
            )

        if "particle" not in result.dims or "particle" not in result.coords:
            raise ValueError("Result must have a particle dimension and IDs")

        ids = np.asarray(result.coords["particle"].values)
        indices = np.asarray(layout.flat_indices)
        size = int(np.prod(layout.shape))

        if (
            indices.shape != (self.seeds.x.size,)
            or indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(indices >= size)
            or np.unique(indices).size != indices.size
        ):
            raise ValueError("Invalid or duplicated seed-grid indices")

        if (
            ids.ndim != 1
            or ids.dtype.kind not in "iu"
            or np.any(ids < 0)
            or np.any(ids >= indices.size)
            or np.unique(ids).size != ids.size
        ):
            raise ValueError("Invalid or duplicated particle IDs")

        if any(dim in result.dims for dim in layout.dims):
            raise ValueError("Seed-grid dimensions already occur in result")

        # Replace particle IDs with flattened grid positions, then restore
        # the complete grid ordering and insert missing locations
        flat = result.assign_coords(
            particle=("particle", indices[ids])
        ).reindex(particle=np.arange(size))

        def spatial(variable):
            if "particle" not in variable.dims:
                return variable.variable

            other_dims = tuple(
                dim for dim in variable.dims if dim != "particle"
            )
            ordered = variable.transpose(*other_dims, "particle")

            return xr.Variable(
                other_dims + layout.dims,
                ordered.data.reshape(
                    ordered.shape[:-1] + layout.shape
                ),
                attrs=dict(variable.attrs),
            )

        # Replace the flattened seed coordinates with the complete physical
        # coordinates stored in the layout
        replaced_coords = {
            "particle", "seed_x", "seed_y", "seed_lon", "seed_lat",
        }
        coords = {
            name: spatial(coordinate)
            for name, coordinate in flat.coords.items()
            if name not in replaced_coords
        }
        coords.update(layout.coords)

        return xr.Dataset(
            {
                name: spatial(variable)
                for name, variable in flat.data_vars.items()
            },
            coords=coords,
            attrs=dict(result.attrs),
        )

    def sample_window(self, time: float) -> WindowSamples:
        """Run two independent trajectories and join once at the seed time.

        Args:
            time: The centre time from which to advect forwards and backwards.

        Returns:
            A WindowSamples context manager over the advection results.

        """

        centre = self.flow.relative_times(time)
        if centre.ndim != 0 or not np.isfinite(centre):
            raise ValueError("time must be a finite scalar")
        centre = float(centre)

        if not self.valid_times(centre).size:
            raise ValueError("filtering window extends outside input time coverage")

        temporary = (
            tempfile.TemporaryDirectory(prefix="filter-window-", dir=self.cache_dir)
            if self.cache_dir is not None
            else None
        )
        directory = temporary.name if temporary else None
        try:
            forward = self.flow.advect(
                self.seeds,
                times=centre + np.arange(self.nforward + 1) * self.sample_dt,
                dt=self.advection_dt,
                fields=self.variables,
                output_dir=directory,
            )
            backward = self.flow.advect(
                self.seeds,
                times=centre - np.arange(self.nback + 1) * self.sample_dt,
                dt=self.advection_dt,
                fields=self.variables,
                output_dir=directory,
            )
            fs, bs = forward.lazy_samples(), backward.lazy_samples()
            samples = {
                name: da.concatenate([bs[name][1:][::-1], fs[name]], axis=0)
                for name in self.variables
            }
            status = da.concatenate(
                [backward.lazy_status()[1:][::-1], forward.lazy_status()], axis=0
            )
            times = np.concatenate([backward.times[1:][::-1], forward.times])
            return WindowSamples(times, int(self.nback), samples, status, temporary)
        except BaseException:
            if temporary is not None:
                temporary.cleanup()
            raise

    def filter_at(self, time: float) -> xr.Dataset:
        """Compute the filtered trajectories at a single centre time.

        Args:
            time: The centre time from which to advect and filter.

        Returns:
            An xarray Dataset over all particles.

        """

        with self.sample_window(time) as window:
            values = {}
            for name, samples in window.samples.items():
                result = self.reducer.apply_filter(
                    samples, window.centre_index, min_window=self.minimum_samples
                )
                if isinstance(result, da.Array):
                    result = result.compute()
                result = np.asarray(result)
                if result.shape != self.seeds.x.shape:
                    raise ValueError("filter reducer must return one value per seed")
                attrs = {
                    k: v
                    for k, v in self.flow.field_attrs.get(name, {}).items()
                    if k in {"units", "long_name"}
                }
                values[name] = (("time", "particle"), result[None], attrs)

            centre_time = window.times[window.centre_index]

        grid = self.flow.grid
        x, y = self.seeds.x, self.seeds.y
        valid = np.isfinite(x) & np.isfinite(y)

        if not grid.periodic_x:
            valid &= (x >= 0) & (x <= grid.nx)
        if not grid.periodic_y:
            valid &= (y >= 0) & (y <= grid.ny)

        px, py = np.full(x.size, np.nan), np.full(x.size, np.nan)
        px[valid], py[valid] = grid.coordinates(grid.logical_seeds(x[valid], y[valid]))
        names = ("seed_lon", "seed_lat") if grid.spherical else ("seed_x", "seed_y")
        units = ("degrees_east", "degrees_north") if grid.spherical else ("m", "m")
        coords = {
            "time": self.flow.absolute_times(np.array([centre_time])),
            "particle": np.arange(x.size),
            names[0]: ("particle", px, {"units": units[0]}),
            names[1]: ("particle", py, {"units": units[1]}),
        }
        ds = xr.Dataset(
            values,
            coords=coords,
            attrs={
                "backward_window_seconds": self.backward,
                "forward_window_seconds": self.forward,
                "sample_interval_seconds": self.sample_dt,
                "maximum_advection_step_seconds": self.advection_dt,
            },
        )
        if self.flow.origin is None:
            ds.time.attrs["units"] = "s"
        return ds

    def iter_results(self, times: ArrayLike | None = None) -> Generator[xr.Dataset]:
        """Yield one centre-time result at a time for streaming output.

        Args:
            times: Array of filter times, otherwise defaults to all valid window
                centres in the Flow.

        Yields:
            Single xarray Datasets for each centre time.

        """

        times = self.valid_times() if times is None else self.flow.relative_times(times)
        times = np.asarray(times)
        if times.ndim != 1 or not np.isfinite(times).all():
            raise ValueError("centre times must be a finite one-dimensional sequence")

        for time in times:
            yield self.filter_at(time)

    def run(self, times: ArrayLike | None = None) -> xr.Dataset:
        """Run the filtering workflow and collect into a single dataset.

        Note:
            If the resultant dataset (sample_variables x valid_times x seed_particles)
            is too large to fit in memory, consider using iter_results() and writing
            intermediate results to disk.

        Args:
            times: Array of filter times, otherwise defaults to all valid window
                centres in the Flow.

        Returns::
            Single xarray Dataset containing the filter results.

        """

        results = list(self.iter_results(times))
        if not results:
            raise ValueError("no centre times have a complete filtering window")

        return xr.concat(
            results, dim="time", coords="minimal", compat="equals", join="exact"
        )

    __call__ = run
