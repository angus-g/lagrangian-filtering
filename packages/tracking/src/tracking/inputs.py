"""Explicit input descriptions and ownership of opened xarray datasets."""

from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from glob import glob
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ._types import (
    BasisString,
    Date,
    DateArray,
    PathLike,
    StaggeringString,
    Timespan,
)
from .fields import Field, seconds_since
from .grid import Grid, Seeds
from .tracker import Tracker, Trajectory


@dataclass(frozen=True)
class Variable:
    """A variable from a Dataset, filename, glob, or sequence of filenames.

    ``dims`` lists (time, y, x). ``select`` holds positional xarray isel
    selections, including the vertical level and any offset/halo slices.
    Numeric, non-calendar time coordinates are multiplied by ``time_scale``
    to obtain seconds. Decoded date coordinates share the Flow's origin.
    """

    source: object
    name: str
    dims: tuple[str, str, str]
    location: StaggeringString = "node"
    select: dict = field(default_factory=dict)

    time_scale: float = 1.0
    "Conversion factor from time coordinates to seconds"


@dataclass(frozen=True)
class GridSpec:
    """Corner geometry in a Dataset or file, independent of field locations.

    Two-dimensional coordinates must have matching dimension order. ``select``
    applies to whichever coordinate/mask dimensions contain each named index.
    A wet mask, if named, must resolve to the canonical cell shape and (y, x)
    order. Use a prebuilt Grid for geometry requiring additional preprocessing.
    """

    source: object
    x: str
    y: str
    spherical: bool = False
    wet: str | None = None
    select: dict = field(default_factory=dict)
    periodic_x: bool = False
    periodic_y: bool = False
    radius: float = 6_371_000.0


class Flow:
    """Prepared grid, velocity, and sampled fields; owns only files it opened.

    Construct with open_flow(), or wrap an array-based Tracker directly.
    Close after trajectories have finished reading input frames. Trajectory
    outputs are independent of the input files once advect returns.

    Args:
        tracker: A pre-constructed RK4 tracker with velocity fields.
        origin: If specified, the calendar origin for relative
            time computation.
        field_attrs: Optional mapping of any attributes that should be
            available on the flow.
        _resources: Cleanup context manager.

    """

    def __init__(
        self,
        tracker: Tracker,
        *,
        origin: Date | np.datetime64 | None = None,
        field_attrs: Mapping[str, Any] | None = None,
        _resources: ExitStack | None = None,
    ) -> None:
        self.tracker = tracker
        self.origin = origin
        self.field_attrs = dict(field_attrs or {})
        self._resources = _resources or ExitStack()
        self.closed = False

    @property
    def grid(self) -> Grid:
        """The Grid associated with the Tracker.

        """

        return self.tracker.grid

    @property
    def fields(self) -> dict[str, Field]:
        """The mapped Fields present on the Tracker.

        """

        return self.tracker.fields

    @property
    def times(self) -> NDArray[np.float64]:
        """The times (as seconds since a shared origin)
        represented by the Tracker.

        """

        return self.tracker.u.times.copy()

    def coverage(self, fields: Iterable[str] | None = None) -> tuple[float, float]:
        """Range of times spanned by Tracker velocities and any other fields.

        Args:
            fields: Optional list of fields on the tracker to include, defaults
                to all extra fields.

        Returns:
            2-tuple of relative timestamps: the smallest timestamp covered by all
            fields; and the largest timestamp covered by all fields.

        """

        names = self.fields.keys() if fields is None else fields
        selected = [self.tracker.u, self.tracker.v, *(self.fields[name] for name in names)]
        return max(f.times[0] for f in selected), min(f.times[-1] for f in selected)

    def relative_times(self, times: float | Date | DateArray) -> NDArray[np.float64]:
        """Convert the provided times to relative seconds since this Flow's origin.

        Args:
            times: An array of seconds to be passed through, else an array of
                dated values to be converted to relative seconds.

        Returns:
            An array of relative seconds to the Flow's origin.

        """

        array = np.asarray(times)

        if array.dtype.kind in "fiu":
            return array.astype(float)

        if self.origin is None:
            raise ValueError("dated times require a flow with a calendar origin")

        return seconds_since(array, origin=self.origin)

    def absolute_times(self, seconds: NDArray[np.float64]) -> DateArray:
        """Convert the provided relative seconds to absolute timestamps.

        Args:
            seconds: An array of relative seconds.

        Returns:
            If the Flow has no origin, timestamps are absolute seconds.
            Otherwise, an absolute timestamp with the same calendar
            as the Flow origin.

        """

        array = np.asarray(seconds, dtype=float)

        if self.origin is None:
            return array.copy()

        if isinstance(self.origin, np.datetime64):
            return self.origin + np.rint(array * 1.e6).astype("timedelta64[ms]")

        return np.array([
            self.origin + timedelta(seconds=float(t))
            for t in array.ravel()
        ]).reshape(array.shape)

    def advect(
        self,
        seeds: Seeds,
        *,
        times: DateArray,
        dt: Timespan,
        fields: Iterable[str] | None = None,
        save_positions: bool = False,
        output_dir: PathLike | None = None
    ) -> Trajectory:
        """Perform advection with the Tracker.

        Args:
            seeds: Seeds object containing initial particle locations.
            times: Array of sampling times along particle trajectories.
            dt: Advection timestep.
            fields: Optional list of variable names to sample.
            save_positions: Whether to save the positions of the particles
                on the trajectories (or just the sampled values).
            output_dir: If provided, use a file-backed cache for the particle
                trajectories.

        Returns:
            Particle tracking trajectories.

        """

        if self.closed:
            raise RuntimeError("flow is closed")

        return self.tracker.advect(
            seeds,
            times=self.relative_times(times),
            dt=dt,
            fields=fields,
            save_positions=save_positions,
            output_dir=output_dir,
        )

    def close(self) -> None:
        """Clean up resources (such as filehandles) used by the Flow.

        """

        if not self.closed:
            self._resources.close()
            self.closed = True

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("flow is closed")

        return self

    def __exit__(self, *args) -> None:
        self.close()


def _resolve_grid(grid: Grid | GridSpec, dataset: Callable[[Any], xr.Dataset]) -> Grid:
    if isinstance(grid, Grid):
        return grid

    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a Grid or GridSpec")

    ds = dataset(grid.source)
    names = [grid.x, grid.y] + ([grid.wet] if grid.wet is not None else [])
    dims = set().union(*(set(ds[name].dims) for name in names))

    if set(grid.select) - dims:
        raise ValueError("grid selection contains unknown dimensions")

    arrays = [
        ds[name].isel({
            k: val
            for k, val in grid.select.items()
            if k in ds[name].dims
        })
        for name in names
    ]

    if arrays[0].ndim == 2 and arrays[0].dims != arrays[1].dims:
        raise ValueError("2-D grid coordinates must have matching dimension order")

    if grid.wet is not None:
        mask = arrays[2].values
        if not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
            raise ValueError("wet mask must contain only finite zero/one values")
    else:
        mask = None

    return Grid(
        arrays[0].values,
        arrays[1].values,
        wet=mask,
        spherical=grid.spherical,
        radius=grid.radius,
        periodic_x=grid.periodic_x,
        periodic_y=grid.periodic_y,
    )


def _resolve_fields(
    u: Variable,
    v: Variable,
    fields: Mapping[str, Variable] | None,
    dataset: Callable[[Any], xr.Dataset],
    origin: Date | None,
) -> tuple[Field, Field, dict[str, Field], Date | None, dict[str, dict[str, Any]]]:
    converted: dict[int, Field] = {}  # field cache
    attrs: dict[int, dict[str, Any]] = {}

    def selected(spec: Variable) -> xr.DataArray:
        if len(spec.dims) != 3 or len(set(spec.dims)) != 3:
            raise ValueError(
                "variables need distinct (time, y, x) dimensions"
            )

        array = dataset(spec.source)[spec.name].isel(spec.select)

        if set(array.dims) != set(spec.dims):
            raise ValueError(
                f"select extra dimensions of {spec.name}, "
                "including the vertical level"
            )

        return array.transpose(*spec.dims)

    uarray = selected(u)
    udates = uarray[u.dims[0]].values

    if not len(udates):
        raise ValueError("U has no selected time records")

    dated = udates.dtype.kind not in "fiu"
    if dated and origin is None:
        origin = udates[0]
    elif not dated and origin is not None:
        raise ValueError(
            "numeric time inputs don't need a time origin"
        )

    def convert(spec: Variable, array: xr.DataArray | None = None) -> Field:
        key = id(spec)

        if key in converted:
            return converted[key]

        if array is None:
            array = selected(spec)

        dates = array[spec.dims[0]].values
        field_dated = dates.dtype.kind not in "fiu"

        if field_dated != dated:
            raise ValueError(
                "cannot mix calendar and numeric time axes"
            )

        if not np.isfinite(spec.time_scale) or spec.time_scale <= 0:
            raise ValueError("time_scale must be positive")

        if dated:
            if spec.time_scale != 1:
                raise ValueError("time_scale only applies to numeric time")
            times = seconds_since(dates, origin=origin)
        else:
            times = np.asarray(dates, dtype=float) * spec.time_scale

        field = Field.from_xarray(
            array,
            times=times,
            time_dim=spec.dims[0],
            y_dim=spec.dims[1],
            x_dim=spec.dims[2],
            location=spec.location,
        )

        converted[key] = field
        attrs[key] = dict(array.attrs)
        return field

    uf = convert(u, uarray)  # skip reloading, since we already had to process
    vf = convert(v)

    if not np.array_equal(uf.times, vf.times):
        raise ValueError("U and V must have matching timestamps")

    sampled = {
        name: convert(spec)
        for name, spec in (fields or {}).items()
    }

    field_attrs = {
        name: attrs[id(spec)]
        for name, spec in (fields or {}).items()
    }

    return uf, vf, sampled, origin, field_attrs


def open_flow(
    *,
    grid: Grid | GridSpec,
    u: Variable,
    v: Variable,
    basis: BasisString,
    fields: Mapping[str, Variable] | None = None,
    origin: Date | None = None,
    open_kwargs: Mapping[str, Any] | None = None,
    combine: Literal["by_coords", "nested"] = "by_coords",
    concat_dim: str | None = None,
):
    """Open explicit GridSpec/Variable descriptions into a reusable Flow.

    No variable-name, staggering, vector-basis or topology inference. Files
    shared by several descriptions are opened once. External xarray datasets
    remain caller-owned. Globs/lists default to coordinate-based combination
    with exact spatial joins. For preordered, compatible files, passe
    ``combine="nested"`` and ``concat_dim`` to concatenate without coordinate
    alignment. Setup failure closes everything opened so far.

    The default origin for decoded dates is the first selected U timestamp.
    U and V must have matching timestamps after conversion. The optional
    fields mapping gives public names; use fields={"U": u} to sample velocity.
    """

    kwargs = dict(open_kwargs or {})
    forbidden = {"decode_cf", "decode_times", "mask_and_scale"}
    if any(kwargs.get(name) is False for name in forbidden):
        raise ValueError("open_flow requires CF time decoding and mask/scale processing")
    if combine not in {"by_coords", "nested"}:
        raise ValueError("combine must be by_coords or nested")
    if combine == "nested" and concat_dim is None:
        raise ValueError("nested combination requires concat_dim")
    if combine == "by_coords" and concat_dim is not None:
        raise ValueError("concat_dim only applies to nested combination")

    with ExitStack() as stack:
        datasets = {}

        def dataset(source: str | Path | Iterable[str | Path]) -> xr.Dataset:
            if isinstance(source, xr.Dataset):
                return source

            items = [source] if isinstance(source, (str, Path)) else list(source)
            paths = [
                str(Path(path).resolve())
                for item in items
                for path in sorted(glob(str(item)))
            ]

            if not paths:
                raise FileNotFoundError(f"no input files match {source!s}")

            key = tuple(sorted(set(paths)))

            if key not in datasets:
                if len(key) == 1:
                    ds = xr.open_dataset(key[0], **kwargs)
                else:
                    multi_kwargs: dict[str, Any] = {
                        "combine": combine,
                        "data_vars": "minimal",
                        "coords": "minimal",
                        "compat": "override" if combine == "nested" else "no_conflicts",
                        "join": "override" if combine == "nested" else "exact",
                    }
                    if concat_dim is not None:
                        multi_kwargs["concat_dim"] = concat_dim

                    ds = xr.open_mfdataset(
                        list(key),
                        **multi_kwargs,
                        **kwargs,
                    )

                stack.callback(ds.close)
                datasets[key] = ds

            return datasets[key]

        grid = _resolve_grid(grid, dataset)
        uf, vf, sampled, origin, field_attrs = _resolve_fields(u, v, fields, dataset, origin)
        tracker = Tracker(grid, uf, vf, basis=basis, fields=sampled)

        return Flow(
            tracker,
            origin=origin,
            field_attrs=field_attrs,
            _resources=stack.pop_all(),
        )
