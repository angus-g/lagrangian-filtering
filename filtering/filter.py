"""Inertial filter objects.

This definition allows for the definition of inertial filters. These
may be as simple as constant frequency, or may vary depending on
latitude or even arbitrary conditions like vorticity.

"""

import dask.array as da
import numpy as np
from scipy import fftpack, signal


class Filter(object):
    """The base class for inertial filters.

    This holds the filter state, and provides an interface for
    applying the filter to advected particle data.

    Args:
        frequency (float): The high-pass cutoff frequency of the filter.
        fs (float): The sampling frequency of the data over which the
            filter is applied.

    """

    def __init__(self, frequency, fs):
        self._filter = Filter.create_filter(frequency, fs)
        self._min_window = False

    @staticmethod
    def create_filter(frequency, fs):
        """Create a filter.

        This creates an analogue Butterworth filter with the given
        frequency and sampling parameters.

        """

        return signal.butter(4, frequency, "highpass", fs=fs)

    def apply_filter(self, data, time_index):
        """Apply the filter to an array of data.

        Args:
            data: An array of (time x particle) of advected particle data.
                This can be a dask array of lazily-loaded temporary data.
            time_index (int): The index along the time dimension corresponding
                to the central point, to extract after filtering.

        Returns:
            An array of (particle) of the filtered particle data, restricted
            to the specified time index.

        """

        def filter_select(x):
            ti = time_index

            if self._min_window:
                xn = np.isnan(x)
                # particles which fit minimum window requirement
                xnn = np.count_nonzero(xn, axis=-1)[:, None] <= ti
                # pad value indices
                pl = np.argmax(~xn, axis=-1)
                pr = x.shape[1] - np.argmax(np.flip(~xn, axis=-1), axis=-1) - 1
                # pad values
                vl = x[np.arange(x.shape[0]), pl]
                vr = x[np.arange(x.shape[0]), pr]
                # do padding
                x[:, :ti] = np.where(xn[:, :ti] & xnn, vl[:, None], x[:, :ti])
                x[:, ti + 1 :] = np.where(
                    xn[:, ti + 1 :] & xnn, vr[:, None], x[:, ti + 1 :]
                )

            return signal.filtfilt(*self._filter, x)[..., ti]

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


class SpatialFilter(Filter):
    """A filter with a programmable, variable frequency.

    Example:
        A Coriolis parameter-dependent filter:

            Ω = 7.2921e-5
            f = lambda lon, lat: 2*Ω*np.sin(np.deg2rad(lat))
            filt = SpatialFilter(f, fs, fieldset)

    Args:
        frequency_func: A function taking ``lon`` and ``lat`` args, returning
            the inertial cutoff frequency at that location.
        fs (float): The sampling frequency of the data over which the
            filter is applied.
        fieldset (:obj:`parcels.FieldSet`): An initialised
            ``FieldSet`` with loaded grid information that can be passed
            to ``frequency_func``.

    """

    def __init__(self, frequency_func, fs, lon, lat):
        # if lon and lat are 1d, broadcast them together
        if lon.ndim == 1 and lat.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)
        elif lon.ndim != lat.ndim:
            raise Exception("grid dimensions don't match")

        self._filter = np.array(
            [
                Filter.create_filter(frequency_func(*coord), fs)
                for coord in zip(lon, lat)
            ]
        )

    def apply_filter(self, data, time_index):
        """Apply the filter to an array of data."""

        def filter_select(filt, x):
            return signal.filtfilt(*filt, x)[..., time_index]

        filtered = da.apply_gufunc(
            filter_select,
            "()->(i)->()",
            self._filter,
            data,
            axis=0,
            output_dtypes=data.dtype,
            allow_rechunk=True,
        )

        return filtered.compute()


class FrequencySpaceFilter(Filter):
    """A filter defined and applied in frequency space.

    This may be used, for example, to implement a sharp cutoff filter,
    without the possible imprecision of representing the cutoff as a
    time-domain sinc function.

    Args:
        frequency (float): The high-pass cutoff frequency of the filter.
        fs (float): The sampling frequency of the daat over which the
            filter is applied.

    """

    def __init__(self, frequency, fs):
        self._frequency = frequency
        self._spacing = 1.0 / fs

    def apply_filter(self, data, time_index):
        """Apply the filter to an array of data."""

        # we can't apply FFT to time-chunked data
        if isinstance(data, da.Array):
            data = data.compute()

        window_len = data.shape[0]
        # step high-pass filter
        freq_filter = fftpack.rfftfreq(window_len, self._spacing) > self._frequency
        # forward transform
        filtered = fftpack.rfft(data, axis=0) * freq_filter[:, None]
        return fftpack.irfft(filtered, axis=0)[time_index, ...]
