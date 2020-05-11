import pytest

from datetime import timedelta
import numpy as np
import xarray as xr

import filtering

# Parcels' dimension dictionaries for each type of Arakawa grid
a_grid = {
    "U": {"lon": "xt", "lat": "yt", "time": "time"},
    "V": {"lon": "xt", "lat": "yt", "time": "time"},
    "T": {"lon": "xt", "lat": "yt", "time": "time"},
}
b_grid = {
    "U": {"lon": "xu", "lat": "yu", "time": "time"},
    "V": {"lon": "xu", "lat": "yu", "time": "time"},
    "T": {"lon": "xt", "lat": "yt", "time": "time"},
}
c_grid = {
    "U": {"lon": "xu", "lat": "yu", "time": "time"},
    "V": {"lon": "xu", "lat": "yu", "time": "time"},
    "T": {"lon": "xt", "lat": "yt", "time": "time"},
}


def parcels_arg_tuple(dimensions):
    """Return Parcels triplet for given dimensions dictionary"""

    dims = {
        "xt": np.array([25, 75, 125]),
        "xu": np.array([50, 100, 150]),
        "yt": np.array([30, 40, 50, 60]),
        "yu": np.array([35, 45, 55, 65]),
    }

    shape = (2, 4, 3)

    parcels_vars = ["U", "V", "T"]
    dataset_vars = {
        var: (["time", dimensions[var]["lat"], dimensions[var]["lon"]], np.empty(shape))
        for var in parcels_vars
    }
    dims["time"] = np.array([123, 234])

    dataset = xr.Dataset(dataset_vars, coords=dims)
    variables = {var: var for var in parcels_vars}

    return dataset, variables, dimensions


@pytest.fixture(
    scope="module",
    params=[(a_grid, False), (b_grid, False), (c_grid, True)],
    ids=["A-grid", "B-grid", "C-grid"],
)
def dimensions_grid(request):
    """Fixture for getting all three types of Arakawa grids"""

    return request.param


@pytest.fixture(scope="module")
def parcels_args(dimensions_grid):
    """Fixture for returning Parcels' args, parameterised for each type of Arakawa grid"""

    dimensions, c_grid = dimensions_grid
    arg_tuple = parcels_arg_tuple(dimensions)
    return arg_tuple + (c_grid,)


def test_particleset_default_particle_locations(parcels_args, nocompile_LagrangeFilter):
    """Test that particles end up at the U-grid gridpoints by default"""

    dataset, variables, dimensions, c_grid = parcels_args

    f = nocompile_LagrangeFilter(
        "test", dataset, variables, dimensions, sample_variables=["T"], c_grid=c_grid
    )

    # first field should be U velocity
    assert f.fieldset.get_fields()[0].name == "U"

    # expect particles on the U grid by default
    out_lon, out_lat = np.meshgrid(
        dataset[dataset.U.dims[-1]], dataset[dataset.U.dims[-2]]
    )

    # create particles and check their location
    ps = f.particleset(0)
    assert np.array_equal(ps.lon, out_lon.flatten())
    assert np.array_equal(ps.lat, out_lat.flatten())


def test_particleset_periodic_particle_locations(
    parcels_args, nocompile_LagrangeFilter
):
    """Test that particle locations aren't changed by a periodic grid."""

    dataset, variables, dimensions, c_grid = parcels_args

    f = nocompile_LagrangeFilter(
        "test", dataset, variables, dimensions, sample_variables=["T"], c_grid=c_grid
    )

    # save the size of the previous grid
    prev_lon_size = f.fieldset.gridset.grids[0].lon.size
    f.make_zonally_periodic(width=1)
    # make sure the underlying grid changed size
    assert prev_lon_size != f.fieldset.gridset.grids[0].lon.size

    out_lon, out_lat = np.meshgrid(
        dataset[dataset.U.dims[-1]], dataset[dataset.U.dims[-2]]
    )

    # create particles and check their location
    ps = f.particleset(0)
    assert np.array_equal(ps.lon, out_lon.flatten())
    assert np.array_equal(ps.lat, out_lat.flatten())


@pytest.mark.parametrize("direction", ["zonal", "meridional"])
def test_set_grid_after_periodic(parcels_args, direction, nocompile_LagrangeFilter):
    """Test that we can't change the particle grid after making the domain periodic."""

    dataset, variables, dimensions, c_grid = parcels_args

    f = nocompile_LagrangeFilter(
        "test", dataset, variables, dimensions, sample_variables=["T"], c_grid=c_grid
    )

    if direction == "zonal":
        f.make_zonally_periodic(width=1)
    elif direction == "meridional":
        f.make_meridionally_periodic(width=1)
    else:
        raise ValueError("unexpected direction")

    with pytest.raises(Exception, match="grid must be set"):
        f.set_particle_grid("T")


@pytest.mark.parametrize("direction", ["zonal", "meridional"])
def test_periodic_after_set_grid(parcels_args, direction, nocompile_LagrangeFilter):
    """Test that the underlying grid respects the set grid when made periodic."""

    dataset, variables, dimensions, c_grid = parcels_args

    f = nocompile_LagrangeFilter(
        "test", dataset, variables, dimensions, sample_variables=["T"], c_grid=c_grid
    )
    f.set_particle_grid("T")

    x = dataset[dataset.T.dims[-1]]
    y = dataset[dataset.T.dims[-2]]

    if direction == "zonal":
        f.make_zonally_periodic(width=1)
        assert f.fieldset.halo_west == x[0]
        assert f.fieldset.halo_east == x[-1]
    elif direction == "meridional":
        f.make_meridionally_periodic(width=1)
        assert f.fieldset.halo_north == y[-1]
        assert f.fieldset.halo_south == y[0]
    else:
        raise ValueError("unexpected direction")


def test_set_grid_wrong_name(parcels_args, nocompile_LagrangeFilter):
    """Test that we raise an error if we try to set the grid from an invalid field."""

    dataset, variables, dimensions, c_grid = parcels_args

    f = nocompile_LagrangeFilter(
        "test", dataset, variables, dimensions, sample_variables=["T"], c_grid=c_grid
    )

    with pytest.raises(ValueError, match="INVALID is not"):
        f.set_particle_grid("INVALID")


def test_particleset_set_grid_particle_locations(
    parcels_args, nocompile_LagrangeFilter
):
    """Test that set_grid changes where particles are spawned."""

    dataset, variables, dimensions, c_grid = parcels_args

    f = nocompile_LagrangeFilter(
        "test", dataset, variables, dimensions, sample_variables=["T"], c_grid=c_grid
    )
    f.set_particle_grid("T")

    out_lon, out_lat = np.meshgrid(
        dataset[dataset.T.dims[-1]], dataset[dataset.T.dims[-2]]
    )

    # create particles and check their location
    ps = f.particleset(0)
    assert np.array_equal(ps.lon, out_lon.flatten())
    assert np.array_equal(ps.lat, out_lat.flatten())


def test_minwindow_size(nocompile_LagrangeFilter):
    """Test that the minimum window size is set correctly."""

    dataset, variables, dimensions = parcels_arg_tuple(a_grid)
    # set output dt to be one day in seconds
    dataset["time"] = np.array([0, 3600 * 24])

    f = nocompile_LagrangeFilter(
        "test",
        dataset,
        variables,
        dimensions,
        sample_variables=["T"],
        minimum_window=timedelta(days=3).total_seconds(),
    )

    assert f._min_window == pytest.approx(3)
