"""The spatially-varying filter.

"""

from .filter import Filter
import dask.array as da
import sosfilt


class SpatialFilter(Filter):
    """A filter with a programmable, variable frequency.

    Example:
        A Coriolis parameter-dependent filter:

        .. code-block:: python

            Ω = 7.2921e-5
            f = 2*Ω*np.sin(np.deg2rad(ff.seed_lat))
            filt = SpatialFilter(f, 1.0 / ff.output_dt)

    Args:
        frequencies (numpy.ndarray): An array with the same number
            of elements as seeded particles, containing the cutoff
            frequency to be used for each particle.
            cutoff frequency at that location.
        fs (float): The sampling frequency of the data over which the
            filter is applied.

    """

    def __init__(self, frequencies, fs):
        self._filter = SpatialFilter.create_filter(frequencies, fs)

    @staticmethod
    def create_filter(frequencies, fs):
        """Create a series of filters.

        This creates an analogue Butterworth filter with the given
        array of frequencies and sampling parameters.

        Args:
            frequencies (numpy.ndarray): The high-pass angular cutoff frequencies of the filters.
            fs (float): The sampling frequency of the data.

        """

        return sosfilt.butter(4, frequencies, "highpass", fs=fs, output="sos")

    def apply_filter(self, data, time_index, static_data=None, min_window=None):
        """Apply the filter to an array of data."""

        def filter_select(filt, x):
            if min_window is not None:
                Filter.pad_window(x, time_index, min_window)

            return sosfilt.sosfiltfilt(filt, x)[..., time_index]

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
