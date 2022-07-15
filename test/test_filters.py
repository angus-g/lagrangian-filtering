import pytest

import dask.array as da
import numpy as np
from scipy import signal
import xarray as xr

import filtering


@pytest.fixture
def lats_grid():
    lons = np.array([0])
    lats = np.array([1, 2])

    lons, lats = np.meshgrid(lons, lats)

    return lats


def test_frequency_filter(leewave_data):
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


def test_spatial_filter(lats_grid):
    """Test creation and frequency response of a latitude-dependent filter."""

    f = lats_grid * 0.1
    filt = filtering.filter.SpatialFilter(f.flatten(), 1)

    for freq, filter_obj in zip(f, filt._filter):
        w, h = signal.sosfreqz(filter_obj)
        assert np.all(abs(h)[w < freq] < 0.1)


@pytest.mark.parametrize("order", [3, 4])
@pytest.mark.parametrize(
    "filter_type,freq",
    [("highpass", 1e-4), ("bandpass", (1e-4, 1e-2)), ("lowpass", 1e-2)],
)
def test_create_filter(order, filter_type, freq):
    """Test parameters for filter creation."""

    filt = filtering.filter.Filter(freq, 1, order=order, filter_type=filter_type)


@pytest.mark.parametrize("order", [3, 4])
@pytest.mark.parametrize("filter_type", ["highpass", "lowpass"])
def test_create_spatial_filter(lats_grid, order, filter_type):
    """Test parameters for spatial filter creation."""

    f = lats_grid * 0.1
    filt = filtering.filter.SpatialFilter(
        f.flatten(), 1, order=order, filter_type=filter_type
    )


def test_create_bandpass_spatial_filter(lats_grid):
    """Expect a failure for creating a bandpass spatial filter."""

    f = lats_grid * (0.1, 0.2)
    with pytest.raises(NotImplementedError):
        filt = filtering.filter.SpatialFilter(f.flatten(), 1, filter_type="bandpass")
