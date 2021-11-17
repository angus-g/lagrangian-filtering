"""A filter that can vary cutoff frequencies depending on the input data.

"""

from .filter import Filter
import dask.array as da
import sosfilt


class DataDependentFilter(Filter):
    """A filter with a variable, data-dependent cutoff frequnecy.

    Args:
        func (Callable[[dict[str, numpy.ndarray]], numpy.ndarray]): A
            function which takes a dictionary of static data, and
            returns the desired cutoff frequency at each point.
        fs (float): The sampling frequency of the data over which the
            filter is applied.

    """

    def __init__(self, func, fs):
        self.func = func
        self.fs = fs

    def create_filter(self, static_data):
        """Create a series of filters.

        This method creates an analogue Butterworth filter at each
        point by transforming the incoming sampled static_data with
        the previously-supplied transformation function and sampling
        frequency.

        Args:
            static_data (dict[str, numpy.ndarray]): The dictionary mapping
                variable names to the sampled static data.
        """

        frequencies = self.func(static_data)

        return sosfilt.butter(4, frequencies, "highpass", fs=self.fs, output="sos")

    def apply_filter(self, data, time_index, static_data=None, min_window=None):
        """Apply the filter to an array of data."""

        def filter_select(filt, x):
            if min_window is not None:
                Filter.pad_window(x, time_index, min_window)

            return sosfilt.sosfiltfilt(filt, x)[..., time_index]

        data = data.rechunk((-1, "auto"))

        filt = self.create_filter(static_data)

        filtered = da.apply_gufunc(
            filter_select,
            "(s,n),(i)->()",
            filt,
            data,
            axes=[(1, 2), (0,), ()],
            output_dtypes=data.dtype,
        )

        return filtered.compute()
