import pytest

import numpy as np
import os
from pathlib import Path
import xarray as xr
from netCDF4 import Dataset

import filtering


@pytest.fixture
def simple_dataset():
    """Simple xarray dataset for filtering tests."""

    coords = {"lon": np.arange(5), "lat": np.arange(4), "time": np.arange(3)}

    d = xr.Dataset(
        {
            "U": (["time", "lat", "lon"], np.empty((3, 4, 5))),
            "V": (["time", "lat", "lon"], np.empty((3, 4, 5))),
        },
        coords=coords,
    )

    return d


def test_single_file(tmp_path, simple_dataset, nocompile_LagrangeFilter):
    """Test creation of output file from a single input file."""

    # because filtering puts files in the current directory, we need to change
    # to the test directory
    os.chdir(tmp_path)

    # set up path for input file
    p = "test.nc"
    simple_dataset.to_netcdf(p)

    # create class
    f = nocompile_LagrangeFilter(
        "single_file",
        {"U": p, "V": p},
        {"U": "U", "V": "V"},
        {k: k for k in ["lon", "lat", "time"]},
        sample_variables=["U"],
    )

    # create output file
    ds, _ = f.create_out()
    ds.close()

    # check that we actually made the right file
    out = Path("single_file.nc")
    assert out.exists()

    # check dimensions and sizes, and variables
    d = xr.open_dataset(out)
    assert d.dims == {"lon": 5, "lat": 4, "time": 0}
    assert "var_U" in d.variables
    assert "var_V" not in d.variables


def test_xarray_input(tmp_path, simple_dataset, nocompile_LagrangeFilter):
    """Test creation of output file from an xarray dataset."""

    # because filtering puts files in the current directory, we need to change
    # to the test directory
    os.chdir(tmp_path)

    # create class
    f = nocompile_LagrangeFilter(
        "xarray_input",
        simple_dataset,
        {"U": "U", "V": "V"},
        {k: k for k in ["lon", "lat", "time"]},
        sample_variables=["U"],
    )

    # create output file
    ds, _ = f.create_out()
    ds.close()

    # check that we actually made the right file
    out = Path("xarray_input.nc")
    assert out.exists()

    # check dimensions and sizes, and variables
    d = xr.open_dataset(out)
    assert d.dims == {"lon": 5, "lat": 4, "time": 0}
    assert "var_U" in d.variables
    assert "var_V" not in d.variables


def test_time_dim(tmp_path, simple_dataset, nocompile_LagrangeFilter):
    """Test the correct time dimension is returned from the input file."""

    os.chdir(tmp_path)

    f = nocompile_LagrangeFilter(
        "time_dim",
        simple_dataset,
        {"U": "U", "V": "V"},
        {k: k for k in ["lon", "lat", "time"]},
        sample_variables=["U"],
    )

    ds, time_dim = f.create_out()
    ds.close()

    assert time_dim == "time"


def test_clobber(tmp_path, simple_dataset, nocompile_LagrangeFilter):
    """Test whether existing output files are clobbered."""

    os.chdir(tmp_path)
    out_path = tmp_path / "clobbering.nc"

    # write an input file
    p = "test.nc"
    simple_dataset.to_netcdf(p)

    # create filter
    f = nocompile_LagrangeFilter(
        "clobbering",
        {"U": p, "V": p},
        {"U": "U", "V": "V"},
        {k: k for k in ["lon", "lat", "time"]},
        sample_variables=["U"],
    )

    # first time, file should create correctly
    with f.create_out()[0] as d:
        pass
    assert out_path.exists()

    # second time, we should fail on clobbering
    with pytest.raises(OSError):
        f.create_out()

    # but, we should be able to open the file with the clobber flag
    assert out_path.exists()
    with f.create_out(clobber=True)[0] as d:
        pass
    assert out_path.exists()


def test_multiple_files(tmp_path, simple_dataset, nocompile_LagrangeFilter):
    """Test creation of output file from multiple input files."""

    os.chdir(tmp_path)
    pu = "test_u.nc"
    pv = "test_v.nc"

    simple_dataset.U.to_netcdf(pu)
    simple_dataset.V.to_netcdf(pv)

    f = nocompile_LagrangeFilter(
        "multiple_files",
        {"U": pu, "V": pv},
        {"U": "U", "V": "V"},
        {k: k for k in ["lon", "lat", "time"]},
        sample_variables=["U"],
    )
    ds, _ = f.create_out()
    ds.close()

    out = Path("multiple_files.nc")
    assert out.exists()

    d = xr.open_dataset(out)
    assert d.dims == {"lon": 5, "lat": 4, "time": 0}
    assert "var_U" in d.variables
    assert "var_V" not in d.variables


def test_dimension_files(tmp_path, nocompile_LagrangeFilter):
    """Test creation of output file where input variables pull dimensions
    from different files.

    """

    os.chdir(tmp_path)

    pvel = "test_vel.nc"
    pdata = "test_data.nc"

    # create a velocity dataset with depth data
    coords = {
        "lon": np.arange(5),
        "lat": np.arange(4),
        "depth": np.arange(3),
        "time": np.arange(3),
    }
    d = xr.Dataset(
        {
            "U": (["time", "depth", "lat", "lon"], np.empty((3, 3, 4, 5))),
            "V": (["time", "depth", "lat", "lon"], np.empty((3, 3, 4, 5))),
        },
        coords=coords,
    )
    d.to_netcdf(pvel)

    # create a data dataset without depth data
    del coords["depth"]
    d = xr.Dataset(
        {
            "UBAR": (["time", "lat", "lon"], np.empty((3, 4, 5))),
            "VBAR": (["time", "lat", "lon"], np.empty((3, 4, 5))),
        },
        coords=coords,
    )
    d.to_netcdf(pdata)

    # filename dicts
    fd = {d: pvel for d in ["lon", "lat", "depth", "data"]}
    dd = fd.copy()
    dd["data"] = pdata

    f = nocompile_LagrangeFilter(
        "dimension_files",
        {"U": fd, "V": fd, "UBAR": dd, "VBAR": dd},
        {"U": "U", "V": "V", "UBAR": "UBAR", "VBAR": "VBAR"},
        {k: k for k in ["lon", "lat", "depth", "time"]},
        sample_variables=["UBAR", "VBAR"],
        indices={"depth": [0]},
    )
    ds, _ = f.create_out()
    ds.close()

    out = Path("dimension_files.nc")
    assert out.exists()

    d = xr.open_dataset(out)
    assert "var_UBAR" in d.variables
    assert "var_VBAR" in d.variables

    assert d["var_UBAR"].dims == ("time", "lat", "lon")


def test_dims_indices_dicts(tmp_path, nocompile_LagrangeFilter):
    """Test creation of output file where dimensions and indices are specified in
    per-variable dictionaries, instead of globally."""

    os.chdir(tmp_path)
    p = "test.nc"

    coords = {
        "X": np.arange(5),
        "Xp": np.arange(5) + 0.5,
        "Y": np.arange(4),
        "Yp": np.arange(4) + 0.5,
        "Z": np.arange(3),
        "Zm": np.arange(1),
        "time": np.arange(3),
    }

    d = xr.Dataset(
        {
            "UVEL": (["time", "Z", "Y", "X"], np.empty((3, 3, 4, 5))),
            "VVEL": (["time", "Z", "Y", "X"], np.empty((3, 3, 4, 5))),
            "UBAR": (["time", "Zm", "Y", "Xp"], np.empty((3, 1, 4, 5))),
            "VBAR": (["time", "Zm", "Yp", "X"], np.empty((3, 1, 4, 5))),
        },
        coords=coords,
    )
    d.to_netcdf(p)

    dims = {
        "U": {"lon": "X", "lat": "Y", "time": "time", "depth": "Z"},
        "V": {"lon": "X", "lat": "Y", "time": "time", "depth": "Z"},
        "UBAR": {"lon": "Xp", "lat": "Y", "time": "time", "depth": "Zm"},
        "VBAR": {"lon": "X", "lat": "Yp", "time": "time", "depth": "Zm"},
    }
    indices = {
        "U": {"depth": [2]},
        "V": {"depth": [2]},
        "UBAR": {"depth": [0]},
        "VBAR": {"depth": [0]},
    }

    f = nocompile_LagrangeFilter(
        "dims_indices_dicts",
        {v: p for v in ["U", "V", "UBAR", "VBAR"]},
        {"U": "UVEL", "V": "VVEL", "UBAR": "UBAR", "VBAR": "VBAR"},
        dims,
        sample_variables=["UBAR", "VBAR"],
        indices=indices,
    )
    ds, _ = f.create_out()
    ds.close()

    out = Path("dims_indices_dicts.nc")
    assert out.exists()


def test_other_data(tmp_path, simple_dataset, nocompile_LagrangeFilter):
    """Test creation of output file where a non-velocity variable is sampled."""

    os.chdir(tmp_path)
    p = "test.nc"
    simple_dataset["P"] = (["time", "lat", "lon"], np.empty((3, 4, 5)))
    simple_dataset.to_netcdf(p)

    f = nocompile_LagrangeFilter(
        "other_data",
        {"U": p, "V": p, "P": p},
        {"U": "U", "V": "V", "P": "P"},
        {k: k for k in ["lon", "lat", "time"]},
        sample_variables=["P"],
    )
    ds, _ = f.create_out()
    ds.close()

    out = Path("other_data.nc")
    assert out.exists()

    d = xr.open_dataset(out)
    assert "var_P" in d.variables


def test_staggered(tmp_path, nocompile_LagrangeFilter):
    """Test creation of output file where velocity is staggered."""

    os.chdir(tmp_path)
    p = "test.nc"
    coords = {
        "xu": np.arange(5) + 0.5,
        "xt": np.arange(5),
        "yv": np.arange(4) + 0.5,
        "yt": np.arange(4),
        "time": np.arange(3),
    }
    d = xr.Dataset(
        {
            "U": (["time", "yt", "xu"], np.empty((3, 4, 5))),
            "V": (["time", "yv", "xt"], np.empty((3, 4, 5))),
            "P": (["time", "yt", "xt"], np.empty((3, 4, 5))),
        },
        coords=coords,
    )
    d.to_netcdf(p)

    # variables
    v = ["U", "V", "P"]

    f = nocompile_LagrangeFilter(
        "staggered",
        {k: p for k in v},
        {k: k for k in v},
        {
            "U": {"lon": "xu", "lat": "yt", "time": "time"},
            "V": {"lon": "xt", "lat": "yv", "time": "time"},
            "P": {"lon": "xt", "lat": "yt", "time": "time"},
        },
        sample_variables=v,
    )
    ds, _ = f.create_out()
    ds.close()

    out = Path("staggered.nc")
    assert out.exists()

    d = xr.open_dataset(out)
    for n in v:
        assert f"var_{n}" in d.variables


def test_curvilinear(tmp_path, nocompile_LagrangeFilter):
    """Test creation of output file where grid is curvilinear."""

    os.chdir(tmp_path)
    p = "test.nc"
    coords = {"xi": np.arange(5), "eta": np.arange(4), "time": np.arange(3)}
    d = xr.Dataset(
        {
            "U": (["time", "eta", "xi"], np.empty((3, 4, 5))),
            "V": (["time", "eta", "xi"], np.empty((3, 4, 5))),
            "lat": (["eta", "xi"], np.ones((4, 5))),
            "lon": (["eta", "xi"], 2 * np.ones((4, 5))),
        },
        coords=coords,
    )
    d.to_netcdf(p)

    f = nocompile_LagrangeFilter(
        "curvilinear",
        {"U": p, "V": p},
        {"U": "U", "V": "V"},
        {k: k for k in ["lon", "lat", "time"]},
        sample_variables=["U"],
    )
    ds, _ = f.create_out()
    ds.close()

    out = Path("curvilinear.nc")
    assert out.exists()

    d = xr.open_dataset(out)
    assert "var_U" in d.variables
    assert d["var_U"].dims == ("time", "eta", "xi")
    assert "lat" in d.variables
    assert d["lat"].dims == ("eta", "xi")


def test_valid_complevel(simple_dataset, nocompile_LagrangeFilter):
    """Test setting compression levels for output."""

    f = nocompile_LagrangeFilter(
        "valid_complevel",
        simple_dataset,
        {"U": "U", "V": "V"},
        {k: k for k in ["lon", "lat", "time"]},
        sample_variables=["U"],
    )

    with pytest.raises(ValueError, match="complevel must be an integer"):
        f.set_output_compression("str")

    # should not have enabled compression
    assert f._output_variable_kwargs == {}

    with pytest.raises(ValueError, match="complevel must be an integer"):
        f.set_output_compression(42)

    # should not have enabled compression
    assert f._output_variable_kwargs == {}

    f.set_output_compression(5)
    assert f._output_variable_kwargs == {"zlib": True, "complevel": 5}


def test_compression_setting(tmp_path, simple_dataset, nocompile_LagrangeFilter):
    """Test whether compression is enabled if it's requested."""

    os.chdir(tmp_path)

    f = nocompile_LagrangeFilter(
        "compression",
        simple_dataset,
        {"U": "U", "V": "V"},
        {k: k for k in ["lon", "lat", "time"]},
        sample_variables=["U"],
    )

    f.set_output_compression()
    ds, _ = f.create_out()
    ds.close()

    out = Path("compression.nc")
    assert out.exists()

    # use the lower level netCDF4 library to check the filters directly
    d = Dataset(out)
    assert "var_U" in d.variables
    assert d.variables["var_U"].filters()["zlib"] == True
