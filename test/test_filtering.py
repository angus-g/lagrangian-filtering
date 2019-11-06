import pytest

import numpy as np
from scipy import signal
import xarray as xr

import filtering


def velocity_series(nt, U0, f):
    """Construct a 1D velocity timeseries."""

    t = np.arange(nt) + 1
    t0 = nt // 2 + 1  # middle time index
    u = U0 + (U0 / 2) * np.sin(2 * np.pi * f * (t - t0))

    return t, u


def velocity_dataset(nt, w, curvilinear=False):
    U0 = 100 / 24
    t, u = velocity_series(nt, U0, w)

    # convert hours to seconds
    u /= 3600
    t *= 3600

    # note:
    # velocity data is offset from times unless times
    # start at zero -- this shouldn't be an absolute
    # requirement for tests (regular datasets don't
    # necessarily begin at t=0...)
    t -= 3600

    x = np.array([0, 500, 1000])
    y = np.array([0, 500, 1000])

    # broadcast velocity to right shape
    u_full = np.empty((nt, y.size, x.size))
    u_full[:] = u[:, None, None]

    dataset_vars = {
        "u": (["time", "y", "x"], u_full),
        "v": (["time", "y", "x"], np.zeros_like(u_full)),
    }
    dataset_coords = {"x": x, "y": y, "time": t}

    if curvilinear:
        xc, yc = np.meshgrid(x, y)
        dataset_coords["x_curv"] = (["y", "x"], xc)
        dataset_coords["y_curv"] = (["y", "x"], yc)

    # create dataset
    d = xr.Dataset(dataset_vars, coords=dataset_coords)

    return d, t, u


def test_sanity():
    """Sanity check of filtering.

    Set up a mean velocity field with an oscillating component,
    then filter out the mean.
    """

    # construct sample times (hrs) and velocity field (m/hr)
    U0 = 100 / 24
    w = 1 / 6  # tidal frequency
    nt = 37
    _, u = velocity_series(nt, U0, w)
    assert u[nt // 2] == pytest.approx(U0)

    # construct filter
    f = signal.butter(4, w / 2, "highpass")
    fu = signal.filtfilt(*f, u)
    assert fu[nt // 2] == pytest.approx(0.0, abs=1e-2)


def test_sanity_advection():
    """Sanity check of advection.

    Using a uniform velocity field, the particles should have the same
    value regardless of where in the domain they go.
    """

    nt = 37
    w = 1 / 6
    d, t, u = velocity_dataset(nt, w)

    f = filtering.LagrangeFilter(
        "advection_test",
        d,
        {"U": "u", "V": "v"},
        {"lon": "x", "lat": "y", "time": "time"},
        sample_variables=["U"],
        mesh="flat",
        window_size=18 * 3600,
        highpass_frequency=(w / 2) / 3600,
        advection_dt=60,
    )

    transformed = f.advection_step(t[nt // 2], output_time=True)
    t_trans = transformed["time"]
    u_trans = transformed["var_U"][1][:, 4].compute()

    assert np.allclose(u, u_trans, rtol=1e-1)
    assert np.array_equal(t, t_trans)


def test_sanity_advection_from_file(tmp_path):
    """Sanity check of advection, with data loaded from a file."""

    nt = 37
    w = 1 / 6
    d, t, u = velocity_dataset(nt, w)
    p = tmp_path / "data.nc"
    d.to_netcdf(p)

    f = filtering.LagrangeFilter(
        "advection_test",
        {"U": str(p), "V": str(p)},
        {"U": "u", "V": "v"},
        {"lon": "x", "lat": "y", "time": "time"},
        sample_variables=["U"],
        mesh="flat",
        window_size=18 * 3600,
        highpass_frequency=(w / 2) / 3600,
        advection_dt=60,
    )

    transformed = f.advection_step(t[nt // 2], output_time=True)
    t_trans = transformed["time"]
    u_trans = transformed["var_U"][1][:, 4].compute()

    assert np.allclose(u, u_trans, rtol=1e-1)
    assert np.array_equal(t, t_trans)


def test_curvilinear_advection():
    """Sanity check of advection on a curvilinear grid."""

    nt = 37
    w = 1 / 5
    d, t, u = velocity_dataset(nt, w, curvilinear=True)

    f = filtering.LagrangeFilter(
        "curvilinear_test",
        d,
        {"U": "u", "V": "v"},
        {"lon": "x_curv", "lat": "y_curv", "time": "time"},
        sample_variables=["U"],
        mesh="flat",
        window_size=18 * 3600,
        highpass_frequency=(w / 2) / 3600,
        advection_dt=60,
    )

    transformed = f.advection_step(t[nt // 2], output_time=True)
    t_trans = transformed["time"]
    u_trans = transformed["var_U"][1][:, 4].compute()

    assert np.allclose(u, u_trans, rtol=1e-1)
    assert np.array_equal(t, t_trans)


def test_zonally_periodic_advection():
    """Sanity check of advection in a zonally periodic domain.

    Because the flow in this test is purely zonal, and we set up a
    zonally-periodic domain, we expect that all particles remain
    alive.
    """

    nt = 37
    w = 1 / 6
    d, t, u = velocity_dataset(nt, w)

    f = filtering.LagrangeFilter(
        "periodic_test",
        d,
        {"U": "u", "V": "v"},
        {"lon": "x", "lat": "y", "time": "time"},
        sample_variables=["U"],
        mesh="flat",
        window_size=18 * 3600,
        highpass_frequency=(w / 2) / 3600,
        advection_dt=60,
    )
    f.make_zonally_periodic(width=3)

    transformed = f.advection_step(t[nt // 2])
    u_trans = transformed["var_U"][1].compute()

    assert not np.any(np.isnan(u_trans))


def test_meridionally_periodic_advection():
    """Sanity check of advection in a meridionally periodic domain.

    Because the flow in this test is purely meridional, and we set up a
    meridionally-periodic domain, we expect that all particles remain
    alive.
    """

    nt = 37
    w = 1 / 6
    d, t, u = velocity_dataset(nt, w)

    f = filtering.LagrangeFilter(
        "periodic_test",
        d,
        {"U": "v", "V": "u"},
        {"lon": "x", "lat": "y", "time": "time"},
        sample_variables=["V"],
        mesh="flat",
        window_size=18 * 3600,
        highpass_frequency=(w / 2) / 3600,
        advection_dt=60,
    )
    f.make_meridionally_periodic(width=3)

    transformed = f.advection_step(t[nt // 2])
    v_trans = transformed["var_V"][1].compute()

    assert not np.any(np.isnan(v_trans))


def test_doubly_periodic_advection():
    """Sanity check of advection in a doubly periodic domain.

    The flow in this test is diagonal, but we set up a doubly-periodic
    domain, so we expect that all particles remain alive.
    """

    nt = 37
    w = 1 / 6
    d, t, u = velocity_dataset(nt, w)

    f = filtering.LagrangeFilter(
        "periodic_test",
        d,
        {"U": "u", "V": "u"},
        {"lon": "x", "lat": "y", "time": "time"},
        sample_variables=["V"],
        mesh="flat",
        window_size=18 * 3600,
        highpass_frequency=(w / 2) / 3600,
        advection_dt=60,
    )
    f.make_zonally_periodic(width=3)
    f.make_meridionally_periodic(width=3)

    transformed = f.advection_step(t[nt // 2])
    v_trans = transformed["var_V"][1].compute()

    assert not np.any(np.isnan(v_trans))


def test_sanity_filtering_from_dataset():
    """Sanity check of filtering using the library.

    As with the :func:`~test_sanity` test, this sets up a mean
    velocity field (in 2D) with an oscillating component. Because the
    velocity field is uniform in time, the Lagrangian timeseries
    should be the same as the 1D timeseries.
    """

    nt = 37
    w = 1 / 6
    d, t, _ = velocity_dataset(nt, w)

    f = filtering.LagrangeFilter(
        "sanity_test",
        d,
        {"U": "u", "V": "v"},
        {"lon": "x", "lat": "y", "time": "time"},
        sample_variables=["U"],
        mesh="flat",
        window_size=18 * 3600,
        highpass_frequency=(w / 2) / 3600,
        advection_dt=30 * 60,
    )

    # filter from the middle of the series
    filtered = f.filter_step(f.advection_step(t[nt // 2]))["var_U"]
    # we expect a lot of parcels to hit the edge and die
    # but some should stay alive
    filtered = filtered[~np.isnan(filtered)]
    assert filtered.size > 0
    value = filtered.item(0)
    assert value == pytest.approx(0.0, abs=1e-3)


def test_absolute_times():
    """Test decoding of absolute times"""

    nt = 37
    w = 1 / 6
    d, t, _ = velocity_dataset(nt, w)

    t = t.copy()

    # offset absolute and relative times
    d["time"] += 1800

    f = filtering.LagrangeFilter(
        "absolute_times",
        d,
        {"U": "u", "V": "v"},
        {"lon": "x", "lat": "y", "time": "time"},
        sample_variables=[],
        window_size=0,
    )

    assert np.all(f._window_times(d["time"], True) == t)
