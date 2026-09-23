import filtering
import numpy as np
import pytest
import tracking
from scipy import signal


@pytest.fixture
def lats_grid():
    lons = np.array([0])
    lats = np.array([1, 2])

    lons, lats = np.meshgrid(lons, lats)

    return lats


def test_frequency_filter(leewave_data):
    """Test creation and application of frequency-space step filter."""

    t = leewave_data["t"].values

    g = tracking.Grid(
        leewave_data.x,
        leewave_data.y,
        periodic_x=True,
        periodic_y=True,
    )
    u = tracking.Field.from_xarray(
        leewave_data["U"],
        times=t,
        time_dim="t",
        y_dim="y",
        x_dim="x",
    )
    v = tracking.Field.from_xarray(
        leewave_data["V"],
        times=leewave_data["t"].values,
        time_dim="t",
        y_dim="y",
        x_dim="x",
    )
    tracker = tracking.Tracker(
        g,
        u,
        v,
        basis="geographic",
        fields={"U": u}
    )

    time_spacing = t[1] - t[0]

    # attach filter to object
    filt = filtering.filter.FrequencySpaceFilter(1e-4, 3600)
    workflow = filtering.WindowFilter(
        tracker,
        g.seed_grid(location="node"),
        window=3 * 24 * 3600,
        sample_dt=time_spacing,
        advection_dt=5 * 60,
        reducer=filt,
    )

    result = workflow.on_seeds(
        workflow.filter_at(7 * 24 * 3600)
    )

    assert np.all((leewave_data.U_orig.data[:-1] - result["U"][0, :]) ** 2 < 3e-8)


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

    _ = filtering.filter.Filter(freq, 1, order=order, filter_type=filter_type)


@pytest.mark.parametrize("order", [3, 4])
@pytest.mark.parametrize("filter_type", ["highpass", "lowpass"])
def test_create_spatial_filter(lats_grid, order, filter_type):
    """Test parameters for spatial filter creation."""

    f = lats_grid * 0.1
    _ = filtering.filter.SpatialFilter(
        f.flatten(), 1, order=order, filter_type=filter_type
    )


def test_create_bandpass_spatial_filter(lats_grid):
    """Expect a failure for creating a bandpass spatial filter."""

    f = lats_grid * (0.1, 0.2)
    with pytest.raises(NotImplementedError):
        _ = filtering.filter.SpatialFilter(f.flatten(), 1, filter_type="bandpass")
