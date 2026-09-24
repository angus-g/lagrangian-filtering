"""Directional trajectory sampling, independent of filtering windows."""

import json
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _kernels as kernels
from ._types import BasisString, PathLike, Timespan
from .fields import LOCATIONS, Field
from .grid import Grid, Seeds


class _ArraySource:
    """Dask array protocol without a copy method or eager memmap pickling."""

    def __init__(self, array: NDArray) -> None:
        self.shape, self.ndim, self.dtype = array.shape, array.ndim, array.dtype
        self.path = str(array.filename) if isinstance(array, np.memmap) else None
        self.array = None if self.path else array

    def __getitem__(self, key: int) -> NDArray:
        if self.array is not None:
            return np.asarray(self.array[key])

        if self.path is not None:
            array = np.load(self.path, mmap_mode="r")
            return np.asarray(array[key])

        raise RuntimeError("ArraySource without array or path backing")


def lazy_array(values):
    """Expose an array to Dask without copying a complete memory map."""
    import dask.array as da
    return da.from_array(
        _ArraySource(values),
        chunks=(len(values), "auto"),
        asarray=False,
        name=False,
        meta=np.empty((0, 0), dtype=values.dtype),
    )


@dataclass
class Trajectory:
    times: NDArray[np.float64]
    samples: dict
    status: NDArray[np.int8]
    directory: Path | None = None
    positions: NDArray[np.float64] | None = None

    def lazy_samples(self):
        return {name: lazy_array(values) for name, values in self.samples.items()}

    def lazy_status(self):
        return lazy_array(self.status)


class TrajectoryWithPositions(Trajectory):
    positions: NDArray[np.float64]


class Tracker:
    """Horizontal RK4 tracking on an explicit Grid.

    Note:
        U and V currently share a time axis; sampled fields may have their own.

    Args:
        grid: A Grid with the spatial description of the domain.
        u: U-velocity Field.
        v: V-velocity Field.
        basis: String describing velocity basis, one of ("geographic",
            "grid_aligned", "face_normal").
        fields: Optional dictionary mapping variable names to Fields
            that should be sampled on advected particles.

    """

    def __init__(
        self,
        grid: Grid,
        u: Field,
        v: Field,
        *,
        basis: BasisString,
        fields: dict[str, Field] | None = None
    ) -> None:
        bases = {
            "geographic": kernels.GridBasis.GEOGRAPHIC,
            "grid_aligned": kernels.GridBasis.GRID_ALIGNED,
            "face_normal": kernels.GridBasis.FACE_NORMAL,
        }
        if basis not in bases:
            raise ValueError("basis must be geographic, grid_aligned, or face_normal")
        if not np.array_equal(u.times, v.times):
            raise ValueError("U and V must share the same time axis")
        expected = ("x_face", "y_face") if basis == "face_normal" else ("node", "node")
        if (u.location, v.location) != expected:
            raise ValueError(f"{basis} requires U/V locations {expected}")

        self.grid, self.u, self.v = grid, u, v
        self.basis = bases[basis]

        self.fields = dict(fields or {})
        for field in [u, v, *self.fields.values()]:
            if field.shape is None:
                field.frame(0)

            if field.shape != grid.shape(field.location):
                raise ValueError(f"{field.location} field shape {field.shape} != {grid.shape(field.location)}")

    @overload
    def advect(
        self,
        seeds: Seeds,
        *,
        times: ArrayLike,
        dt: Timespan,
        fields: Iterable[str] | None = None,
        save_positions: Literal[False] = False,
        output_dir: PathLike | None = None,
    ) -> Trajectory:
        ...

    @overload
    def advect(
        self,
        seeds: Seeds,
        *,
        times: ArrayLike,
        dt: Timespan,
        fields: Iterable[str] | None = None,
        save_positions: Literal[True] = True,
        output_dir: PathLike | None = None,
    ) -> TrajectoryWithPositions:
        ...

    def advect(self, seeds, *, times, dt, fields=None, save_positions=False, output_dir=None):
        """Start at times[0] and sample along increasing OR decreasing times.

        Times are shared-origin seconds. ``dt`` is a positive maximum step;
        its sign is inferred from the time sequence. Input seeds are not
        modified. Disk outputs, if requested, belong to the caller.

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

        times = np.asarray(times, dtype=float)
        if (
            times.ndim != 1
            or not times.size
            or not np.isfinite(times).all()
            or not (
                np.all(np.diff(times) > 0)
                or np.all(np.diff(times) < 0)
            )
        ):
            raise ValueError("times must be finite and strictly increasing or decreasing")

        return self._execute(
            seeds,
            times=times,
            initial_index=0,
            sweeps=(range(1, times.size),),
            dt=dt,
            fields=fields,
            save_positions=save_positions,
            output_dir=output_dir,
        )

    def _execute(
        self,
        seeds: Seeds,
        *,
        times: NDArray[np.float64],
        initial_index: int,
        sweeps: Iterable[Iterable[int]],
        dt: Timespan,
        fields: Iterable[str] | None,
        save_positions: bool,
        output_dir: PathLike | None,
    ) -> Trajectory:
        dt = dt.total_seconds() if isinstance(dt, timedelta) else dt
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")

        selected = self.fields
        if fields is not None:
            selected = {name: self.fields[name] for name in fields}

        for field in [self.u, self.v, *selected.values()]:
            field.bracket(times[0])
            field.bracket(times[-1])

        if not isinstance(seeds, Seeds):
            raise TypeError("use grid.logical_seeds or grid.locate to construct seeds")

        initial = self.grid.logical_seeds(seeds.x, seeds.y)

        if initial.status.shape != np.shape(seeds.status):
            raise ValueError("seed status shape mismatch")
        if not np.all(np.isin(seeds.status, kernels.ParticleStatus)):  # ty: ignore[invalid-argument-type]
            raise ValueError("unknown particle status")

        initial.status = np.where(
            seeds.status == kernels.ParticleStatus.ACTIVE,
            initial.status,
            seeds.status,
        ).astype(np.int8)
        n = initial.x.size

        directory = None
        if output_dir is not None:
            directory = Path(tempfile.mkdtemp(prefix="trajectories-", dir=output_dir))

        def allocate(name, shape, dtype):
            if directory is None:
                return np.empty(shape, dtype=dtype)

            return np.lib.format.open_memmap(
                directory / (name + ".npy"),
                mode="w+",
                dtype=dtype,
                shape=shape,
            )

        samples = {
            name: allocate(f"field-{k}", (times.size, n), np.float64)
            for k, name in enumerate(selected)
        }
        status = allocate("status", (times.size, n), np.int8)
        positions = None
        if save_positions:
            positions = allocate("logical-positions", (times.size, n, 2), np.float64)

        def record(index, state):
            status[index] = state.status
            if positions is not None:
                positions[index, :, 0] = np.where(
                    state.status == kernels.ParticleStatus.ACTIVE,
                    state.x,
                    np.nan,
                )
                positions[index, :, 1] = np.where(
                    state.status == kernels.ParticleStatus.ACTIVE,
                    state.y,
                    np.nan,
                )

            for name, field in selected.items():
                a, b, alpha = field.pair(times[index])
                samples[name][index] = kernels.sample_values(
                    state.x,
                    state.y,
                    state.status,
                    a,
                    b,
                    alpha,
                    LOCATIONS[field.location],
                    self.grid.wet,
                    self.grid.periodic_x,
                    self.grid.periodic_y,
                )

        record(initial_index, initial)
        for indices in sweeps:
            state = Seeds(initial.x.copy(), initial.y.copy(), initial.status.copy())
            time = float(times[initial_index])
            for index in indices:
                self._advance(state, time, times[index], dt)
                record(index, state)
                time = times[index]

        for array in [*samples.values(), status, positions]:
            if isinstance(array, np.memmap):
                array.flush()

        if directory is not None:
            metadata = {
                "times": times.tolist(),
                "initial_index": initial_index,
                "fields": {name: f"field-{k}.npy" for k, name in enumerate(selected)},
                "status_codes": {
                    "active": 1,
                    "outside": 2,
                    "dry": 3,
                    "missing_velocity": 4,
                    "step_failed": 5,
                },
            }
            (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        if positions is not None:
            return TrajectoryWithPositions(times.copy(), samples, status, directory, positions)
        else:
            return Trajectory(times.copy(), samples, status, directory)

    def _advance(self, state: Seeds, start: float, end: float, dt: float):
        g = self.grid
        times = self.u.times
        inner = times[(times > min(start, end)) & (times < max(start, end))]

        if end < start:
            inner = inner[::-1]

        for stop in np.append(inner, end):
            k = self.u.bracket(start + (stop - start) * .5)
            kernels.advance(
                state.x,
                state.y,
                state.status,
                start,
                stop,
                dt,
                g.x,
                g.y,
                g.wet,
                g.periodic_x,
                g.periodic_y,
                g.spherical, g.radius,
                self.u.frame(k),
                self.u.frame(k + 1),
                self.v.frame(k),
                self.v.frame(k + 1),
                times[k],
                times[k + 1],
                self.basis,
                g.xlength,
                g.ylength,
            )
            start = stop
