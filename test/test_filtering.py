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


def velocity_dataset(nt, w):
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

    # create dataset
    d = xr.Dataset(
        {
            "u": (["time", "y", "x"], u_full),
            "v": (["time", "y", "x"], np.zeros_like(u_full)),
        },
        coords={"x": x, "y": y, "time": t},
    )

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


def test_sanity_advection(tmp_path):
    """Sanity check of advection.

    Using a uniform velocity field, the particles should have the same
    value regardless of where in the domain they go.
    """

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

    transformed = f.advection_step(t[nt // 2])
    t_trans = np.concatenate(
        (
            transformed.data("backward").attrs["time"][1:-1][::-1],
            transformed.data("forward").attrs["time"][:-1],
        )
    )

    u_trans = np.concatenate(
        (
            transformed.data("backward")["var_U"][1:-1, 4][::-1],
            transformed.data("forward")["var_U"][:-1, 4],
        )
    )

    assert np.allclose(u, u_trans, rtol=1e-1)
    assert np.array_equal(t, t_trans)


def test_periodic_advection(tmp_path):
    """Sanity check of advection in a periodic domain.

    Because the flow in this test is purely zonal, and we set up a
    zonally-periodic domain, we expect that all particles remain
    alive.
    """

    nt = 37
    w = 1 / 6
    d, t, u = velocity_dataset(nt, w)
    p = tmp_path / "data.nc"
    d.to_netcdf(p)

    f = filtering.LagrangeFilter(
        "periodic_test",
        {"U": str(p), "V": str(p)},
        {"U": "u", "V": "v"},
        {"lon": "x", "lat": "y", "time": "time"},
        sample_variables=["U"],
        mesh="flat",
        window_size=18 * 3600,
        highpass_frequency=(w / 2) / 3600,
        advection_dt=60,
    )

    transformed = f.advection_step(t[nt // 2])
    u_trans = np.concatenate(
        (
            transformed.data("backward")["var_U"][1:-1, 4][::-1],
            transformed.data("forward")["var_U"][:-1, 4],
        )
    )

    assert not np.any(np.isnan(u_trans))


def test_sanity_filtering_from_dataset(tmp_path):

    """Sanity check of filtering using the library.

    As with the :func:`~test_sanity` test, this sets up a mean
    velocity field (in 2D) with an oscillating component. Because the
    velocity field is uniform in time, the Lagrangian timeseries
    should be the same as the 1D timeseries.
    """

    nt = 37
    w = 1 / 6
    d, t, _ = velocity_dataset(nt, w)
    p = tmp_path / "data.nc"
    d.to_netcdf(p)

    f = filtering.LagrangeFilter(
        "sanity_test",
        {"U": str(p), "V": str(p)},
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
