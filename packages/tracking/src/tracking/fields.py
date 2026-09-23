"""Two-frame caching with explicit time axes, placement, and array order."""

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Self, overload

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray

from ._kernels import CellStaggering
from ._types import Date, StaggeringString

LOCATIONS = {
    "node": CellStaggering.NODE,
    "cell": CellStaggering.CELL,
    "x_face": CellStaggering.X_FACE,
    "y_face": CellStaggering.Y_FACE,
}


@overload
def seconds_since(
    values: NDArray[np.datetime64],
    *,
    origin: np.datetime64,
) -> NDArray[np.float64]:
    ...


@overload
def seconds_since(
    values: NDArray[np.object_],
    *,
    origin: Date,
) -> NDArray[np.float64]:
    ...


def seconds_since(values, *, origin):
    """Convert decoded datetime64/Python/cftime dates to seconds from a given origin.

    Before calling this routine, decode CF units and calendar attributes to explicit
    datetime objects with xarray.

    Note:
        Every field must use the. same origin and calendar. For numeric coordinates,
        convert units explicitly before constructing Field.

    Args:
        values: Array of numpy datetime64, datetime.datetime or cftime.datetime
            objects.
        origin: Object with the same type as values array specifying the common
            origin time.

    Returns:
        An array where timestamps in the original array are converted to seconds
        since the origin.

    """

    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.datetime64):
        out = (values - np.datetime64(origin)) / np.timedelta64(1, "s")
    elif values.dtype == object:
        out = np.array([
            (value - origin).total_seconds()
            for value in values.ravel()
        ]).reshape(values.shape)
    else:
        raise ValueError("expected decoded dates; numeric times must already be seconds")

    out = np.asarray(out, dtype=float)
    if not np.isfinite(out).all():
        raise ValueError("time coordinates must be finite")

    return out


class Field:
    """A field with numeric times in seconds relative to a shared origin.

    The field will cache the most recent two frames.

    Args:
        times: Array of times represented by this field, as seconds
            relative to a shared origin.
        read_frame: Callable that takes a time index and returns a
            numpy array representing the field value at that time.
        location: Cell staggering of the Field data, one of ("node",
            "cell", "x_face", "y_face").
        shape: If specified, explicit shape of the Field. Otherwise
            determined upon the first read call.
    """

    def __init__(
        self,
        times: ArrayLike,
        read_frame: Callable[[int], NDArray[np.float64]],
        *,
        location: StaggeringString = "node",
        shape: Iterable[int] | None = None
    ) -> None:
        self.times = np.array(times, dtype=float)

        if (
            self.times.ndim != 1
            or self.times.size < 2
            or not np.isfinite(self.times).all()
            or np.any(np.diff(self.times) <= 0)
        ):
            raise ValueError("times must contain at least two finite, strictly increasing seconds")
        if location not in LOCATIONS:
            raise ValueError(f"unknown field location: {location}")

        self.location = location
        self._read_frame = read_frame
        self._cache = OrderedDict()
        self.shape = tuple(shape) if shape is not None else None

    @classmethod
    def from_array(
        cls,
        times: ArrayLike,
        values: NDArray[np.float64],
        *,
        location: StaggeringString = "node",
    ) -> Self:
        """Construct a Field from a 3D array of values.

        Args:
            times: Array of times represented by this field, as seconds
                relative to a shared origin.
            values: 3D array of field values.
            location: Cell staggering of the Field data, one of ("node",
                "cell", "x_face", "y_face").

        """

        values = np.asanyarray(values)
        times = np.asarray(times)
        if values.ndim != 3 or values.shape[0] != len(times):
            raise ValueError("values must have shape (time, y, x)")

        return cls(times, lambda index: values[index], location=location, shape=values.shape[1:])

    @classmethod
    def from_xarray(
        cls,
        array: xr.DataArray,
        *,
        times: ArrayLike,
        time_dim: str,
        y_dim: str,
        x_dim: str,
        location: StaggeringString = "node",
        indices: Mapping[Any, Any] | None = None,
    ) -> Self:
        """Construct a Field from an xarray DataArray.

        Note:
            Make sure the array has been decoded with xarray (i.e. using
            the default mask_and_scale=True). The times array should already
            be decoded into seconds since a shared origin, but values within
            the array should have masking and scaling applied and not
            pending as attributes.

            Additionally, use the `indices` argument to subset the array to
            remove any offsets, halos or extraneous dimensions.

        Args:
            array: DataArray containing Field data.
            times: Array of times represented by this field, as seconds
                relative to a shared origin.
            time_dim: Name of the dimension representing time within the
                DataArray.
            y_dim: Name of the Y-dimension within the DataArray.
            x_dim: Name of the X-dimension within the DataArray.
            location: Cell staggering of the Field data, one of ("node",
                "cell", "x_face", "y_face").
            indices: Optional mapping passed to `isel` to select the
                actual hyperslab that will be provided as Field data.

        """

        times = np.asarray(times)

        array = array.isel(indices or {})
        if set(array.dims) != {time_dim, y_dim, x_dim}:
            raise ValueError("select all non-time/y/x dimensions, including the vertical level")

        array = array.transpose(time_dim, y_dim, x_dim)
        if array.sizes[time_dim] != len(times):
            raise ValueError("time coordinate length mismatch after selection")

        return cls(
            times,
            lambda index: array.isel({time_dim: index}).values,
            location=location,
            shape=array.shape[1:],
        )

    def frame(self, index: int) -> NDArray[np.float64]:
        """Return the 2D frame slice corresponding to a time index.

        The frame will be served from the cache (most recent two
        entries) if present, otherwise the frame read callback will
        be used.

        Args:
            index: Time index of the frame.

        Returns:
            The 2D frame slice at the time index.

        """

        if index not in self._cache:
            raw = np.asanyarray(self._read_frame(index))
            if raw.ndim != 2:
                raise ValueError("a field frame must be two-dimensional")
            if raw.dtype.kind not in "fiu":
                raise ValueError("field values must be real numbers")

            dtype = np.float32 if raw.dtype == np.float32 else np.float64

            array = np.ascontiguousarray(np.ma.asarray(raw, dtype=dtype).filled(np.nan))

            if self.shape is None:
                self.shape = array.shape
            if array.shape != self.shape:
                raise ValueError("field frame shape changed")

            self._cache[index] = array
            while len(self._cache) > 2:
                self._cache.popitem(last=False)

        self._cache.move_to_end(index)
        return self._cache[index]

    def bracket(self, time: float) -> int:
        """Get the left-hand time index of the bracket containing time.

        Args:
            time: A value in seconds since the origin of the Field.

        Returns:
            Integer index for the frame to the left of the specified time.

        """

        if not np.isfinite(time) or time < self.times[0] or time > self.times[-1]:
            raise ValueError("requested time outside field coverage")

        return min(
            max(
                int(np.searchsorted(self.times, time, side="right")) - 1,
                0
            ),
            len(self.times) - 2
        )

    def pair(self, time: float) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Get the frame data for linear interpolation at time.

        Args:
            time: A value in seconds since the origin of the Field.

        Returns:
            A 3-tuple of the left and right frames bracketing time, and the
            normalised position of the time within the bracket.

        """

        k = self.bracket(time)

        return (
            self.frame(k),
            self.frame(k + 1),
            (time - self.times[k]) / (self.times[k + 1] - self.times[k]),
        )
