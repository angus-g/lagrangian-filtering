import filtering
import numpy as np
import pytest
import tracking
import xarray as xr
from scipy import signal


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


@pytest.fixture
def make_workflow():
    def make(
        dataset,
        *,
        window,
        cutoff_frequency=None,
        reducer=None,
        coords=("x", "y"),
        dims=("time", "y", "x"),
        velocities=("u", "v"),
        sample_variables=None,
        basis="geographic",
        grid_kwargs=None,
        seeds=None,
        sample_dt=None,
        advection_dt=30 * 60,
    ):
        x_coord, y_coord = coords
        time_dim, y_dim, x_dim = dims
        u_name, v_name = velocities

        grid = tracking.Grid(
            dataset[x_coord],
            dataset[y_coord],
            **(grid_kwargs or {}),
        )

        if sample_variables is None:
            sample_variables = {"U": u_name}

        names = {u_name, v_name, *sample_variables.values()}
        fields = {
            name: tracking.Field.from_xarray(
                dataset[name],
                times=dataset[time_dim].values,
                time_dim=time_dim,
                y_dim=y_dim,
                x_dim=x_dim,
            )
            for name in names
        }

        tracker = tracking.Tracker(
            grid,
            fields[u_name],
            fields[v_name],
            basis=basis,
            fields={
                name: fields[source]
                for name, source in sample_variables.items()
            },
        )

        if seeds is None:
            seeds = grid.seed_grid()
        elif callable(seeds):
            seeds = seeds(grid)

        if sample_dt is None:
            times = np.asarray(dataset[time_dim].values, dtype=float)
            spacing = np.diff(times)
            if (
                spacing.size == 0
                or np.any(spacing <= 0)
                or not np.allclose(spacing, spacing[0])
            ):
                raise ValueError("Specify sample_dt for irregular test input")

            sample_dt = float(spacing[0])

        return filtering.WindowFilter(
            tracker,
            seeds,
            window=window,
            sample_dt=sample_dt,
            advection_dt=advection_dt,
            cutoff_frequency=cutoff_frequency,
            reducer=reducer,
        )

    return make


def rotated_velocity_dataset(nt, frequency, angle, basis):
    d, times, speed = velocity_dataset(nt, frequency)
    theta = np.deg2rad(angle)
    c, s = np.cos(theta), np.sin(theta)

    x, y = np.meshgrid(d.x.values, d.y.values)
    xc = (x.min() + x.max()) / 2
    yc = (y.min() + y.max()) / 2

    # Rotate about the centre, preserving the centre seed's position.
    d = d.assign_coords(
        x_curv=(("y", "x"), xc + c * (x - xc) - s * (y - yc)),
        y_curv=(("y", "x"), yc + s * (x - xc) + c * (y - yc)),
    )

    if basis == "geographic":
        u, v = d.u, d.v
        d = d.assign(
            u=c * u - s * v,
            v=s * u + c * v,
        )

    # A spatially varying tracer, expressed in the original coordinates.
    d["q"] = (
        ("time", "y", "x"),
        np.broadcast_to(x, d.u.shape).copy(),
    )

    return d, times, speed


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
    f = signal.butter(4, w / 2, "highpass", output="sos")
    fu = signal.sosfiltfilt(f, u)
    assert fu[nt // 2] == pytest.approx(0.0, abs=1e-2)


@pytest.mark.parametrize("curvilinear", [False, True])
def test_sanity_advection(curvilinear, make_workflow, null_reducer):
    """Sanity check of advection.

    Using a uniform velocity field, the particles should have the same
    value regardless of where in the domain they go.
    """

    nt = 37
    w = 1 / 6
    d, t_orig, u_orig = velocity_dataset(nt, w, curvilinear=curvilinear)

    x_dim = "x_curv" if curvilinear else "x"
    y_dim = "y_curv" if curvilinear else "y"

    workflow = make_workflow(
        d,
        window=18 * 3600,
        reducer=null_reducer,
        coords=(x_dim, y_dim),
        seeds=lambda g: g.logical_seeds([1], [1]),
        advection_dt=60,
    )

    with workflow.sample_window(t_orig[nt // 2]) as window:
        u_trans = window.samples["U"][:, 0].compute()
        assert np.allclose(u_orig, u_trans, rtol=1e-1)
        assert np.array_equal(t_orig, window.times)


def test_sanity_advection_from_file(tmp_path, null_reducer):
    """Sanity check of advection, with data loaded from a file."""

    nt = 37
    w = 1 / 6
    d, t_orig, u_orig = velocity_dataset(nt, w)
    p = tmp_path / "data.nc"
    d.to_netcdf(p)

    u = tracking.Variable(p, "u", dims=("time", "y", "x"))
    v = tracking.Variable(p, "v", dims=("time", "y", "x"))

    with tracking.open_flow(
        grid=tracking.GridSpec(p, x="x", y="y"),
        u=u,
        v=v,
        basis="geographic",
        fields={"U": u},
    ) as flow:
        workflow = filtering.WindowFilter(
            flow,
            flow.grid.logical_seeds([1], [1]),
            window=18 * 3600,
            sample_dt=float(t_orig[1] - t_orig[0]),
            advection_dt=60,
            reducer=null_reducer,
        )

        with workflow.sample_window(t_orig[nt // 2]) as window:
            u_trans = window.samples["U"][:, 0].compute()
            assert np.allclose(u_orig, u_trans, rtol=1e-1)
            assert np.array_equal(t_orig, window.times)


@pytest.mark.parametrize("periodic_x,periodic_y", [(True, False), (False, True), (True, True)])
def test_periodic_advection(periodic_x, periodic_y, make_workflow, null_reducer):
    """Sanity check of advection in a zonally periodic domain.

    Because the flow in this test is purely zonal, and we set up a
    zonally-periodic domain, we expect that all particles remain
    alive.
    """

    nt = 37
    w = 1 / 6
    d, t_orig, _ = velocity_dataset(nt, w)

    workflow = make_workflow(
        d,
        window=18 * 3600,
        reducer=null_reducer,
        advection_dt=60,
        grid_kwargs={
            "periodic_x": periodic_x,
            "periodic_y": periodic_y,
        },
    )

    with workflow.sample_window(t_orig[nt // 2]) as window:
        u_trans = window.samples["U"][1].compute()
        assert not np.any(np.isnan(u_trans))


@pytest.mark.parametrize("angle", [0, 30, 75])
@pytest.mark.parametrize("basis", ["geographic", "grid_aligned"])
def test_rotated_advection(angle, basis, null_reducer):
    nt = 37
    d, times, speed = rotated_velocity_dataset(
        nt, frequency=1 / 6, angle=angle, basis=basis,
    )

    grid = tracking.Grid(d.x_curv, d.y_curv)

    fields = {
        name: tracking.Field.from_xarray(
            d[name],
            times=times,
            time_dim="time",
            y_dim="y",
            x_dim="x",
        )
        for name in ("u", "v", "q")
    }

    tracker = tracking.Tracker(
        grid,
        fields["u"],
        fields["v"],
        basis=basis,
        fields={
            "U": fields["u"],
            "V": fields["v"],
            "q": fields["q"],
        },
    )

    workflow = filtering.WindowFilter(
        tracker,
        grid.logical_seeds([1], [1]),
        window=18 * 3600,
        sample_dt=float(times[1] - times[0]),
        advection_dt=60,
        reducer=null_reducer,
    )

    centre = nt // 2

    # Exact integral at input timestamps for the piecewise-linear
    # temporal velocity reconstruction used by tracking.
    displacement = np.concatenate([
        [0.0],
        np.cumsum(
            0.5 * (speed[:-1] + speed[1:]) * np.diff(times)
        ),
    ])
    expected_q = (
        float(d.x.values[1])
        + displacement
        - displacement[centre]
    )

    theta = np.deg2rad(angle)
    expected_u = speed * (
        np.cos(theta) if basis == "geographic" else 1.0
    )
    expected_v = speed * (
        np.sin(theta) if basis == "geographic" else 0.0
    )

    with workflow.sample_window(times[centre]) as window:
        np.testing.assert_array_equal(window.times, times)

        for name, expected in {
            "U": expected_u,
            "V": expected_v,
            "q": expected_q,
        }.items():
            actual = window.samples[name][:, 0].compute()
            np.testing.assert_allclose(
                actual, expected, rtol=1e-6, atol=1e-10,
            )


def test_sanity_filtering_from_dataset(make_workflow):
    """Sanity check of filtering using the library.

    As with the :func:`~test_sanity` test, this sets up a mean
    velocity field (in 2D) with an oscillating component. Because the
    velocity field is uniform in time, the Lagrangian timeseries
    should be the same as the 1D timeseries.
    """

    nt = 37
    w = 1 / 6
    d, t_orig, _ = velocity_dataset(nt, w)

    workflow = make_workflow(
        d,
        window=18 * 3600,
        cutoff_frequency=(w / 2) / 3600 / (2 * np.pi),
    )

    result = workflow.filter_at(t_orig[nt // 2])
    filtered = result["U"].isel(time=0).values
    valid = filtered[np.isfinite(filtered)]

    assert valid.size > 0
    np.testing.assert_allclose(valid, 0.0, atol=1e-3)


def test_filtering_output_times(make_workflow):
    """Test that input times are copied to the output dataset."""

    nt = 37
    w = 1 / 6
    d, _, _ = velocity_dataset(nt, w)

    workflow = make_workflow(
        d,
        window=9 * 3600,
        cutoff_frequency=(w / 2) / 3600 / (2 * np.pi),
    )

    result = workflow.run()
    xr.testing.assert_allclose(result.time, d.time[9:-9])


def test_filtering_output_times_with_calendar(make_workflow):
    """Test that input times from a file with a calendar
    are correctly copied to the output file."""

    nt = 37
    w = 1 / 6
    d, _, _ = velocity_dataset(nt, w)

    # modify the dataset to have a "days since ..." calendar
    d["time"] = d["time"] / 3600 / 24
    d.time.attrs.update(
        units="days since 1900-01-01 00:00:00",
        calendar="noleap",
    )
    d = xr.decode_cf(d)

    u = tracking.Variable(d, "u", dims=("time", "y", "x"))
    v = tracking.Variable(d, "v", dims=("time", "y", "x"))

    with tracking.open_flow(
        grid=tracking.GridSpec(d, x="x", y="y"),
        u=u,
        v=v,
        basis="geographic",
        fields={"U": u},
    ) as flow:
        # test the calendar passes through filtering
        workflow = filtering.WindowFilter(
            flow,
            flow.grid.seed_grid(),
            window=9 * 3600,
            sample_dt=3600,
            advection_dt=30 * 60,
            cutoff_frequency=(w / 2) / 3600 / (2 * np.pi),
        )

        result = workflow.run()
        # check the calendar attributes were propagated
        assert result.time.attrs == d.time.attrs

        # check the time values themselves
        xr.testing.assert_allclose(result.time, d.time[9:-9])


def test_masked_filtering(make_workflow):
    """Test running the full filtering workflow, seeding only on a subdomain."""

    nt = 37
    w = 1 / 6
    d, t, _ = velocity_dataset(nt, w)

    def seed_func(g):
        return g.seed_grid(
            location="node",
            bounds=(500, 500, 500, 500),
        )

    workflow = make_workflow(
        d,
        window=18 * 3600,
        cutoff_frequency=(w / 2) / 3600 / (2 * np.pi),
        seeds=seed_func,
    )

    result = workflow.filter_at(t[nt // 2])
    assert result.particle.size > 0
