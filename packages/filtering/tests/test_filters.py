import dask.array as da
import filtering
import numpy as np
import pytest
import sosfilt
import tracking
from scipy import fftpack, signal


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


def test_minimum_window_pads_each_particle_along_time():
    samples = np.arange(25.)[:, None] + np.array([0., 100., 200.])
    samples[:4, 1] = np.nan
    samples[20:, 1] = np.nan
    samples[:, 2] = np.nan
    samples[11:14, 2] = [211., 212., 213.]

    filtering.filter.Filter.pad_window(samples, centre_index=12, min_window=16)

    np.testing.assert_array_equal(samples[:, 0], np.arange(25.))
    np.testing.assert_array_equal(samples[:4, 1], np.full(4, 104.))
    np.testing.assert_array_equal(samples[20:, 1], np.full(5, 119.))
    assert np.isnan(samples[:11, 2]).all()
    np.testing.assert_array_equal(samples[11:14, 2], [211., 212., 213.])
    assert np.isnan(samples[14:, 2]).all()


@pytest.mark.parametrize("kind", ["butterworth", "frequency", "spatial"])
@pytest.mark.parametrize("backing", ["numpy", "dask"])
@pytest.mark.parametrize("minimum_window", [None, 35])
def test_filter_array_axes_and_minimum_window(kind, backing, minimum_window):
    times = np.arange(49.)
    centre = 24
    samples = np.column_stack([
        np.sin(.7 * times) + .02 * times,
        np.cos(.5 * times) + .03 * times,
        np.sin(.3 * times) - .01 * times,
    ])
    expected_input = samples.copy()

    if minimum_window is not None:
        samples[:4, 1] = np.nan
        samples[44:, 1] = np.nan
        samples[:, 2] = np.nan
        samples[centre - 1:centre + 2, 2] = expected_input[centre - 1:centre + 2, 2]

        expected_input[:4, 1] = expected_input[4, 1]
        expected_input[44:, 1] = expected_input[43, 1]
        expected_input[:, 2] = samples[:, 2]

    if kind == "butterworth":
        reducer = filtering.filter.Filter(.08, 1.)
        expected = signal.sosfiltfilt(reducer._filter, expected_input, axis=0)[centre]
    elif kind == "frequency":
        reducer = filtering.filter.FrequencySpaceFilter(.08, 1.)
        passed = fftpack.rfftfreq(len(times), 1.) > .08
        expected = fftpack.irfft(
            fftpack.rfft(expected_input, axis=0) * passed[:, None], axis=0
        )[centre]
    else:
        reducer = filtering.filter.SpatialFilter([.08, .1, .12], 1.)
        expected = sosfilt.sosfiltfilt(reducer._filter, expected_input.T)[:, centre]

    data = samples.copy()
    if backing == "dask":
        data = da.from_array(data, chunks=(len(times), 2))

    actual = reducer.apply_filter(data, centre, min_window=minimum_window)
    actual = np.asarray(actual)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
