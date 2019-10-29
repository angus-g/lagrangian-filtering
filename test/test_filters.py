import pytest

import dask.array as da
import numpy as np
from scipy import signal
import xarray as xr

import filtering


def test_spatial_filter():
    """Test creation and frequency response of a latitude-dependent filter."""

    lats = np.array([1, 2])

    # construct filter
    f = lambda lon, lat: lat * 0.1
    filt = filtering.filter.SpatialFilter(f, 1, np.array([0]), lats)

    # compute frequency response of filter
    for lat, filter_obj in zip(lats, filt._filter):
        w, h = signal.freqz(*filter_obj)
        assert np.all(abs(h)[w < f(0, lat)] < 0.1)


def test_frequency_filter(leewave_data, enable_threads):
    """Test creation and application of frequency-space step filter."""

    f = filtering.LagrangeFilter(
        "frequency_filter",
        leewave_data,
        {"U": "U", "V": "V"},
        {"lon": "x", "lat": "y", "time": "t"},
        ["U"],
        window_size=3 * 24 * 3600,
    )
    f.make_zonally_periodic()
    f.make_meridionally_periodic()

    # attach filter to object
    filt = filtering.filter.FrequencySpaceFilter(1e-4, 3600)
    f.inertial_filter = filt

    adv = f.advection_step(7 * 24 * 3600)
    U_filt = f.filter_step(adv)["var_U"].reshape(leewave_data.y.size, -1)

    assert np.all((leewave_data.U_orig.data - U_filt[0, :]) ** 2 < 3e-8)
