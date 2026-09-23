"""Inertial filter objects.

This definition allows for the definition of inertial filters. These
may be as simple as constant frequency, or may vary depending on
latitude or even arbitrary conditions like vorticity.

"""

from typing import Literal

import dask.array as da
import numpy as np
import optype.numpy as onp
import sosfilt
from scipy import fftpack, signal


class Filter:
    """The base class for inertial filters.

    This holds the filter state, and provides an interface for
    applying the filter to advected particle data.

    Args:
        frequency: The low-pass or high-pass cutoff
            frequency of the filter in [/s], or a pair denoting the band to pass.
        fs : The sampling frequency of the data over which the
            filter is applied in [s].
        **kwargs: Additional arguments are passed to the
            :func:`~create_filter` method.

    """

    def __init__(self, frequency: float | onp.ToFloat1D, fs: float, **kwargs) -> None:
        self._filter = Filter.create_filter(frequency, fs, **kwargs)

    @staticmethod
    def create_filter(
        frequencies: float | onp.ToFloat1D,
        fs: float,
        order: int = 4,
        filter_type: Literal["highpass", "bandpass", "lowpass"] = "highpass"
    ) -> onp.Array2D[np.float64]:
        """Create a filter.

        This creates an analogue Butterworth filter with the given
        frequency and sampling parameters.

        Args:
            frequencies: The high-pass angular cutoff frequency of the filter
                in, or band-pass frequencies as a two-element array [/s].
            fs : The sampling frequency of the data in [s].
            order: The filter order, default 4.
            filter_type (Optional[str]): The type of filter, one of ("highpass",
                "bandpass", "lowpass"), defaults to "highpass".

        Returns:
            The filter in second-order sections ("sos") format.

        """

        return signal.butter(order, frequencies, filter_type, fs=fs, output="sos")

    @staticmethod
    def pad_window(x: onp.Array2D[np.float64], centre_index: int, min_window: int) -> None:
        """Perform minimum window padding of an array.

        Note:
            This performs in-place modification of ``x``.

        Args:
            x: An array of (time x particle) of particle data.
            centre_index : The index of the seeding time of the particles, to
                identify the forward and backward advection data.
            min_window: The minimum window size; particles with at least this
                many non-NaN datapoints are padded with the last valid value in
                each direction.

        """

        xn = np.isnan(x)
        # particles which fit minimum window requirement
        # that is, the number of non-NaN values is greater
        # than the minimum window size
        is_valid = np.count_nonzero(~xn, axis=-1)[:, None] >= min_window
        # pad value indices -- the index, per-particle, of the last
        # non-NaN value in each direction (forward and backward)
        pl = np.argmax(~xn, axis=-1)
        pr = x.shape[1] - np.argmax(np.flip(~xn, axis=-1), axis=-1) - 1
        # pad values -- get the value associated with the above index
        vl = x[np.arange(x.shape[0]), pl]
        vr = x[np.arange(x.shape[0]), pr]
        # do padding -- for valid particles according to the minimum
        # window size, set the backward NaN values to vl, and the
        # forward NaN values to vr
        x[:, :centre_index] = np.where(
            xn[:, :centre_index] & is_valid, vl[:, None], x[:, :centre_index]
        )
        x[:, centre_index + 1:] = np.where(
            xn[:, centre_index + 1:] & is_valid,
            vr[:, None],
            x[:, centre_index + 1:],
        )

    def apply_filter(
        self,
        data: onp.Array2D[np.float64] | da.Array,
        time_index: int,
        min_window: int | None = None
    ) -> onp.Array1D[np.float64] | da.Array:
        """Apply the filter to an array of data.

        Args:
            data: A dask array of (time x particle) of advected particle data.
            time_index: The index along the time dimension corresponding
                to the central point, to extract after filtering.
            min_window: A minimum window size for considering
                particles valid for filtering.

        Returns:
            An array of (particle) of the filtered particle data, restricted
            to the specified time index.

        """

        def filter_select(x):
            if min_window is not None:
                Filter.pad_window(x, time_index, min_window)

            return signal.sosfiltfilt(self._filter, x)[..., time_index]

        if isinstance(data, np.ndarray):
            return filter_select(data)

        # apply scipy filter as a ufunc
        # mapping an array to scalar over the first axis, automatically vectorize execution
        # and allow rechunking (since we have a chunk boundary across the first axis)
        filtered = da.apply_gufunc(
            filter_select,
            "(i)->()",
            data.rechunk((-1, "auto")),
            axis=0,
            output_dtypes=data.dtype,
        )

        return filtered.compute()


class FrequencySpaceFilter(Filter):
    """A filter defined and applied in frequency space.

    This may be used, for example, to implement a sharp cutoff filter,
    without the possible imprecision of representing the cutoff as a
    time-domain sinc function.

    Args:
        frequency: The high-pass cutoff frequency of the filter
            in [/s].
        fs: The sampling frequency of the daat over which the
            filter is applied in [s].

    """

    def __init__(self, frequency: float, fs: float) -> None:
        self._frequency = frequency
        self._spacing = 1.0 / fs

    def apply_filter(
        self,
        data: onp.Array2D[np.float64] | da.Array,
        time_index: int,
        min_window: int | None = None
    ) -> onp.Array1D[np.float64]:
        """Apply the filter to an array of data.

        Args:
            data: An array of (time x particle) of advected particle data.
                This can be a dask array of lazily-loaded temporary data.
            time_index: The index along the time dimension corresponding
                to the central point, to extract after filtering.
            min_window: A minimum window size for considering
                particles valid for filtering.

        Returns:
            An array of (particle) of the filtered particle data, restricted
            to the specified time index.

        """

        # we can't apply FFT to time-chunked data
        if isinstance(data, da.Array):
            data = data.compute()

        if min_window is not None:
            Filter.pad_window(data, time_index, min_window)

        window_len = data.shape[0]
        # step high-pass filter
        freq_filter = fftpack.rfftfreq(window_len, self._spacing) > self._frequency
        # forward transform
        filtered = fftpack.rfft(data, axis=0) * freq_filter[:, None]
        return fftpack.irfft(filtered, axis=0)[time_index, ...]


class SpatialFilter(Filter):
    """A filter with a programmable, variable frequency.

    Example:
        A Coriolis parameter-dependent filter:

        .. code-block:: python

            Ω = 7.2921e-5
            f = 2*Ω*np.sin(np.deg2rad(ff.seed_lat))
            filt = SpatialFilter(f, 1.0 / ff.output_dt)

    Args:
        frequencies: An array with the same number
            of elements as seeded particles, containing the cutoff
            frequency to be used for each particle, in [/s].
        fs: The sampling frequency of the data over which the
            filter is applied in [s].
        **kwargs: Additional arguments are passed to the
            :func:`~create_filter` method.

    """

    def __init__(self, frequencies: float | onp.ToFloat1D, fs: float, **kwargs) -> None:
        self._filter = SpatialFilter.create_filter(frequencies, fs, **kwargs)

    @staticmethod
    def create_filter(
        frequencies: float | onp.ToFloat1D,
        fs: float,
        order: int = 4,
        filter_type: Literal["highpass", "lowpass", "bandpass"] = "highpass"
    ) -> onp.Array2D[np.float64]:
        """Create a series of filters.

        This creates an analogue Butterworth filter with the given
        array of frequencies and sampling parameters.

        Args:
            frequencies: The high-pass cutoff frequencies of the filters
                in [/s].
            fs: The sampling frequency of the data in [s].
            order: The filter order, default 4.
            filter_type: The type of filter, one of ("highpass",
                "lowpass"), defaults to "highpass". Note that bandpass spatial filters
                aren't supported.

        """

        if filter_type == "bandpass":
            raise NotImplementedError("bandpass spatial filters are unsupported")

        return sosfilt.butter(order, frequencies, filter_type, fs=fs, output="sos")

    def apply_filter(
        self,
        data: onp.Array2D[np.float64] | da.Array,
        time_index: int,
        min_window: int | None = None
    ) -> da.Array:
        """Apply the filter to an array of data.

        Args:
            data: A dask array of (time x particle) of advected particle data.
            time_index: The index along the time dimension corresponding
                to the central point, to extract after filtering.
            min_window: A minimum window size for considering
                particles valid for filtering.

        Returns:
            An array of (particle) of the filtered particle data, restricted
            to the specified time index.

        """

        def filter_select(filt, x):
            if min_window is not None:
                Filter.pad_window(x, time_index, min_window)

            return sosfilt.sosfiltfilt(filt, x)[..., time_index]

        if isinstance(data, np.ndarray):
            return filter_select(self._filter, data)

        # we have to make sure the chunking of filter matches that of data
        data = data.rechunk((-1, "auto"))
        filt = da.from_array(self._filter, chunks=(data.chunksize[1], None, None))

        filtered = da.apply_gufunc(
            filter_select,
            "(s,n),(i)->()",
            filt,
            data,
            axes=[(1, 2), (0,), ()],
            output_dtypes=data.dtype,
        )

        return filtered.compute()
