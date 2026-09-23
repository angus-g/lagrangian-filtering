import numpy as np
import pytest
from numba import get_num_threads, set_num_threads
from tracking import (
    Field,
    Grid,
    ParticleStatus,
    Tracker,
    seconds_since,
)


@pytest.fixture(autouse=True)
def threads():
    old = get_num_threads()
    set_num_threads(min(2, old))
    yield
    set_num_threads(old)


def constant(grid, value, location="node", times=(-10., 0., 10.)):
    return Field.from_array(
        times,
        np.full((len(times), *grid.shape(location)), value, dtype=float),
        location=location
    )


@pytest.mark.parametrize("basis", ["geographic", "grid_aligned", "face_normal"])
@pytest.mark.parametrize("direction", [1, -1])
def test_constant_advection_exact_times(basis, direction):
    g = Grid(np.arange(11) * 2., np.arange(9) * 3.)

    uloc, vloc = (
        ("x_face", "y_face")
        if basis == "face_normal"
        else ("node", "node")
    )

    u = constant(g, .4, uloc)
    v = constant(g, -.3, vloc)

    # Sample a linear space/time function; catches position/time misalignment.
    times = np.array([-10., 0., 10.])
    q = Field.from_array(
        times,
        2 * g.x[None] + 3 * g.y[None] + 5 * times[:, None, None],
    )

    tr = Tracker(g, u, v, basis=basis, fields={"q": q})

    outtimes = np.array([0., .7, 1.9, 4.])
    outtimes *= direction

    result = tr.advect(
        g.logical_seeds([4.], [4.]),
        times=outtimes,
        dt=.37,
        save_positions=True,
    )

    expected_x = 8 + .4 * outtimes
    expected_y = 12 - .3 * outtimes

    assert result.positions is not None

    np.testing.assert_allclose(
        result.positions[:, 0, 0] * 2,
        expected_x,
        atol=1.e-12,
    )
    np.testing.assert_allclose(
        result.positions[:, 0, 1] * 3,
        expected_y,
        atol=1.e-12,
    )
    np.testing.assert_allclose(
        result.samples["q"][:, 0],
        2 * expected_x + 3 * expected_y + 5 * outtimes,
        atol=1.e-11,
    )
    assert (result.status == ParticleStatus.ACTIVE).all()


def test_rotated_grid_aligned_velocity():
    xx, yy = np.meshgrid(np.arange(9.), np.arange(9.))
    a = .6
    g = Grid(
        xx * np.cos(a) - yy * np.sin(a),
        xx * np.sin(a) + yy * np.cos(a),
    )
    tr = Tracker(g, constant(g, .2), constant(g, .1), basis="grid_aligned")

    result = tr.advect(
        g.logical_seeds([4], [4]),
        times=[0, 2],
        dt=.1,
        save_positions=True,
    )

    assert result.positions is not None
    np.testing.assert_allclose(result.positions[-1, 0], [4.4, 4.2], atol=1.e-12)


def test_spherical_cgrid_across_dateline():
    # Constant geographic eastward speed at fixed latitude has known dlon/dt.
    g = Grid(
        [178., 179., -180., -179., -178.],
        [-1., 0., 1.],
        spherical=True,
    )
    times = [0., 100000.]
    u = constant(g, 1., "x_face", times)
    v = constant(g, 0., "y_face", times)

    tr = Tracker(g, u, v, basis="face_normal")
    seeds = g.locate([-179.5], [.5])
    np.testing.assert_allclose(seeds.x, [2.5], atol=1.e-10)

    r = tr.advect(
        seeds,
        times=[0., 10000.],
        dt=1800.,
        save_positions=True,
    )
    speed = 1 / (g.radius * np.cos(np.deg2rad(.5)) * np.pi / 180)

    assert r.positions is not None
    np.testing.assert_allclose(
        r.positions[:, 0, 0],
        2.5 + r.times * speed,
        atol=1.e-10,
    )


def test_curvilinear_geographic_constant_velocity():
    x, y = np.meshgrid(np.arange(9.), np.arange(9.))
    g = Grid(x + .03 * x * y, y + .01 * x * y)
    seeds = g.logical_seeds([3.2], [3.6])
    x0, y0 = g.coordinates(seeds)

    tr = Tracker(g, constant(g, .2), constant(g, .1), basis="geographic")
    r = tr.advect(seeds, times=[0, 3], dt=.05, save_positions=True)

    assert r.positions is not None
    end = g.logical_seeds(r.positions[-1, :, 0], r.positions[-1, :, 1])
    xx, yy = g.coordinates(end)

    np.testing.assert_allclose(xx, x0 + .6, atol=1.e-10)
    np.testing.assert_allclose(yy, y0 + .3, atol=1.e-10)

    back = g.locate(x0, y0)
    np.testing.assert_allclose(back.x, seeds.x, atol=1.e-10)
    np.testing.assert_allclose(back.y, seeds.y, atol=1.e-10)


def test_time_knots_are_respected():
    g = Grid(np.arange(21.), [0., 1., 2.])
    times = [-2., 0., 1., 3.]
    values = np.broadcast_to(np.array([0., 0., 2., 0.])[:, None, None], (4, 3, 21))
    tr = Tracker(g, Field.from_array(times, values), constant(g, 0., times=times), basis="geographic")

    r = tr.advect(g.logical_seeds([5.], [.5]), times=[-2, 0, 3], dt=10, save_positions=True)
    np.testing.assert_allclose(r.positions[:, 0, 0], [5., 5., 8.], atol=1.e-12)


def test_independent_direction_status_and_column_identity():
    g = Grid([0., 1., 2.], [0., 1.])
    u, v = constant(g, 1.), constant(g, 0.)
    tr = Tracker(g, u, v, basis="geographic", fields={"U": u})
    r = tr.advect(
        g.logical_seeds([.25, 1.75], [.5, .5]),
        times=[0., 1.],
        dt=.1,
    )

    # Indices are (time, particle): second particle goes OOB
    assert r.status[1, 1] == ParticleStatus.OUTSIDE
    assert r.samples["U"][1, 0] == 1.
    assert r.samples["U"][0, 1] == 1.
    assert np.isnan(r.samples["U"][1, 1])
    assert (r.status[0] == ParticleStatus.ACTIVE).all()


def test_periodic_large_step_and_multiple_crossings():
    g = Grid(np.arange(5.), [0., 1.], periodic_x=True)
    tr = Tracker(g, constant(g, 10.), constant(g, 0.), basis="geographic")
    r = tr.advect(
        g.logical_seeds([.2], [.5]),
        times=[0, 2],
        dt=2,
        save_positions=True,
    )
    np.testing.assert_allclose(r.positions[:, 0, 0], .2, atol=1.e-12)
    assert (r.status == ParticleStatus.ACTIVE).all()


def test_no_jump_across_dry_cell():
    g = Grid(np.arange(6.), [0., 1.], wet=[[True, True, False, True, True]])
    tr = Tracker(g, constant(g, 10.), constant(g, 0.), basis="geographic")
    r = tr.advect(
        g.logical_seeds([.2], [.5]),
        times=[0, .4],
        dt=.4,
    )

    assert r.status[-1, 0] == ParticleStatus.DRY


def test_missing_sample_does_not_kill_trajectory():
    g = Grid(np.arange(5.), [0., 1.])
    u = constant(g, .1)
    v = constant(g, 0.)
    q = constant(g, np.nan)
    tr = Tracker(g, u, v, basis="geographic", fields={"q": q})
    r = tr.advect(
        g.logical_seeds([1.2], [.5]),
        times=[0, 1],
        dt=.1,
    )

    assert (r.status == ParticleStatus.ACTIVE).all()
    assert np.isnan(r.samples["q"]).all()


def test_missing_velocity_sets_status():
    g = Grid([0., 1.], [0., 1.])
    tr = Tracker(g, constant(g, np.nan), constant(g, 0.), basis="geographic")
    r = tr.advect(
        g.logical_seeds([.5], [.5]),
        times=[0, 1],
        dt=.1,
    )

    assert r.status[-1, 0] == ParticleStatus.MISSING


def rk4_rotation_error(dt):
    n = round(1 / dt)
    z = (
        1
        + 1j * dt
        + (1j * dt) ** 2 / 2
        + (1j * dt) ** 3 / 6
        + (1j * dt) ** 4 / 24
    ) ** n

    return abs(z - np.exp(1j))


def test_rk4_convergence_in_rotation():
    g = Grid(np.linspace(-2, 2, 9), np.linspace(-2, 2, 9))
    u = Field.from_array([-1., 2.], np.stack([-g.y, -g.y]))
    v = Field.from_array([-1., 2.], np.stack([g.x, g.x]))

    tr = Tracker(g, u, v, basis="geographic")
    seed = g.locate([1.], [0.])

    for dt in [.1, .05]:
        r = tr.advect(seed, times=[0, 1], dt=dt, save_positions=True)
        end = g.logical_seeds(r.positions[-1, :, 0], r.positions[-1, :, 1])
        xx, yy = g.coordinates(end)
        error = np.hypot(xx[0] - np.cos(1), yy[0] - np.sin(1))

        np.testing.assert_allclose(error, rk4_rotation_error(dt), rtol=1e-6)


def test_xarray_level_selection_and_two_frame_cache():
    import xarray as xr
    values = np.arange(2 * 3 * 4 * 5.).reshape(2, 3, 4, 5)
    a = xr.DataArray(values, dims=("z", "record", "j", "i"))
    f = Field.from_xarray(
        a,
        times=[0, 1, 2],
        time_dim="record",
        y_dim="j",
        x_dim="i",
        indices={"z": 1},
    )

    np.testing.assert_array_equal(f.frame(1), values[1, 1])

    f.frame(0)
    f.frame(2)

    assert len(f._cache) == 2
    with pytest.raises(ValueError, match="vertical level"):
        Field.from_xarray(a, times=[0, 1, 2], time_dim="record", y_dim="j", x_dim="i")


def test_bad_input_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        Field.from_array([0, 0], np.zeros((2, 2, 2)))

    with pytest.raises(ValueError, match="polar"):
        Grid([0, 1], [89, 90], spherical=True)

    g = Grid([0, 1], [0, 1])
    with pytest.raises(ValueError, match="locations"):
        Tracker(g, constant(g, 0), constant(g, 0), basis="face_normal")

    tr = Tracker(g, constant(g, 0), constant(g, 0), basis="geographic")
    with pytest.raises(ValueError, match="outside field coverage"):
        tr.advect(g.logical_seeds([.5], [.5]), times=[0, 20], dt=.1)


def test_distorted_cgrid_reconstructs_uniform_physical_flow():
    x, y = np.meshgrid(np.arange(10.), np.arange(8.))
    g = Grid(x + .04 * x * y, y + .02 * x * y)

    # Exact normal projections of the physical vector (.2, -.1).
    edge_dx, edge_dy = np.diff(g.x, axis=0), np.diff(g.y, axis=0)
    un = (.2 * edge_dy + .1 * edge_dx) / g.xlength

    edge_dx, edge_dy = np.diff(g.x, axis=1), np.diff(g.y, axis=1)
    vn = (-.2 * edge_dy - .1 * edge_dx) / g.ylength

    u = Field.from_array([-5., 5.], np.stack([un, un]), location="x_face")
    v = Field.from_array([-5., 5.], np.stack([vn, vn]), location="y_face")

    tr = Tracker(g, u, v, basis="face_normal")
    seeds = g.logical_seeds([3.2, 4.8], [3.6, 4.2])
    xx0, yy0 = g.coordinates(seeds)

    r = tr.advect(seeds, times=[0, 2], dt=.05, save_positions=True)
    for idx, time in enumerate(r.times):
        end = g.logical_seeds(r.positions[idx, :, 0], r.positions[idx, :, 1])
        xx, yy = g.coordinates(end)
        np.testing.assert_allclose(xx, xx0 + .2 * time, atol=1.e-10)
        np.testing.assert_allclose(yy, yy0 - .1 * time, atol=1.e-10)


def test_decoded_cf_noleap_and_common_origin():
    import xarray as xr
    raw = xr.Dataset(coords={"time": ("time", [0., 1., 2.],
                                      {"units": "days since 2001-02-28", "calendar": "noleap"})})
    dates = xr.decode_cf(raw).time.values
    assert dates[1].month == 3 and dates[1].day == 1
    np.testing.assert_array_equal(seconds_since(dates, origin=dates[0]), [0, 86400, 172800])
    np.testing.assert_array_equal(seconds_since(dates[1:], origin=dates[0]), [86400, 172800])
    dates64 = np.array(["2001-02-28", "2001-03-01"], dtype="datetime64[D]")
    np.testing.assert_array_equal(seconds_since(dates64, origin=dates64[0]), [0, 86400])


def test_dask_memmap_does_not_copy_entire_output(tmp_path, monkeypatch):
    g = Grid([0., 1.], [0., 1.])
    u = constant(g, 0.)
    tr = Tracker(g, u, u, basis="geographic", fields={"U": u})

    r = tr.advect(
        g.logical_seeds([.5], [.5]),
        times=[0, 1],
        dt=.1,
        output_dir=tmp_path,
    )

    def forbidden_copy(*args, **kwargs):
        raise AssertionError("eager copy of disk-backed output")

    monkeypatch.setattr(np.memmap, "copy", forbidden_copy)
    result = r.lazy_samples()["U"]
    np.testing.assert_array_equal(result.compute(), [[0.], [0.]])


@pytest.mark.parametrize("location", ["node", "cell", "x_face", "y_face"])
def test_field_placement_and_independent_sample_time_axis(location):
    g = Grid([0., 1., 2.], [0., 1., 2.])
    j, i = np.indices(g.shape(location))

    values = np.stack([i + 2 * j, i + 2 * j + 4.])
    q = Field.from_array([-1., 1.], values, location=location)

    tr = Tracker(g, constant(g, 0.), constant(g, 0.), basis="geographic", fields={"q": q})

    r = tr.advect(
        g.logical_seeds([.25], [.75]),
        times=[-.5, 0, .5],
        dt=.1,
    )

    spatial = {"node": 1.75, "cell": 0., "x_face": .25, "y_face": 1.5}[location]
    np.testing.assert_allclose(
        r.samples["q"][:, 0],
        spatial + np.array([1., 2., 3.]),
    )


def test_exact_node_ignores_missing_zero_weight_neighbours():
    g = Grid([0., 1.], [0., 1.])

    q = np.full((2, 2, 2), np.nan)
    q[0, 0, 0] = 42.
    f = Field.from_array([0., 1.], q)

    zero = constant(g, 0.)
    tr = Tracker(g, zero, zero, basis="geographic", fields={"q": f})

    r = tr.advect(
        g.logical_seeds([0.], [0.]),
        times=[0],
        dt=.1,
    )

    assert r.samples["q"][0, 0] == 42.
