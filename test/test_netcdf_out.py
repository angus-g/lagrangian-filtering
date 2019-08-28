import pytest

import numpy as np
import os
from pathlib import Path
import xarray as xr

import filtering


def test_single_file(tmp_path):
    """Test creation of output file from a single input file."""

    # because filtering puts files in the current directory, we need to change
    # to the test directory
    os.chdir(tmp_path)

    # set up path for input file
    p = "test.nc"

    coords = {"lon": np.arange(5), "lat": np.arange(4), "time": np.arange(3)}

    d = xr.Dataset(
        {
            "U": (["time", "lat", "lon"], np.empty((3, 4, 5))),
            "V": (["time", "lat", "lon"], np.empty((3, 4, 5))),
        },
        coords=coords,
    )
    d.to_netcdf(p)

    # create class
    f = filtering.LagrangeFilter(
        "single_file",
        {"U": p, "V": p},
        {"U": "U", "V": "V"},
        {k: k for k in ["lon", "lat", "time"]},
        sample_variables=["U"],
    )

    # create output file
    f.create_out().close()

    # check that we actually made the right file
    out = Path("single_file.nc")
    assert out.exists()

    # check dimensions and sizes, and variables
    d = xr.open_dataset(out)
    assert d.dims == {"lon": 5, "lat": 4, "time": 0}
    assert "var_U" in d.variables
    assert "var_V" not in d.variables


def test_multiple_files(tmp_path):
    """Test creation of output file from multiple input files."""

    os.chdir(tmp_path)
    pu = "test_u.nc"
    pv = "test_v.nc"
    coords = {"lon": np.arange(5), "lat": np.arange(4), "time": np.arange(3)}

    du = xr.Dataset({"U": (["time", "lat", "lon"], np.empty((3, 4, 5)))})
    du.to_netcdf(pu)
    dv = xr.Dataset({"V": (["time", "lat", "lon"], np.empty((3, 4, 5)))})
    dv.to_netcdf(pv)

    f = filtering.LagrangeFilter(
        "multiple_files",
        {"U": pu, "V": pv},
        {"U": "U", "V": "V"},
        {k: k for k in ["lon", "lat", "time"]},
        sample_variables=["U"],
    )
    f.create_out().close()

    out = Path("multiple_files.nc")
    assert out.exists()

    d = xr.open_dataset(out)
    assert d.dims == {"lon": 5, "lat": 4, "time": 0}
    assert "var_U" in d.variables
    assert "var_V" not in d.variables


def test_other_data(tmp_path):
    """Test creation of output file where a non-velocity variable is sampled."""

    os.chdir(tmp_path)
    p = "test.nc"
    coords = {"lon": np.arange(5), "lat": np.arange(4), "time": np.arange(3)}
    d = xr.Dataset(
        {
            "U": (["time", "lat", "lon"], np.empty((3, 4, 5))),
            "V": (["time", "lat", "lon"], np.empty((3, 4, 5))),
            "P": (["time", "lat", "lon"], np.empty((3, 4, 5))),
        },
        coords=coords,
    )
    d.to_netcdf(p)

    f = filtering.LagrangeFilter(
        "other_data",
        {"U": p, "V": p, "P": p},
        {"U": "U", "V": "V", "P": "P"},
        {k: k for k in ["lon", "lat", "time"]},
        sample_variables=["P"],
    )
    f.create_out().close()

    out = Path("other_data.nc")
    assert out.exists()

    d = xr.open_dataset(out)
    assert "var_P" in d.variables


def test_staggered(tmp_path):
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

    f = filtering.LagrangeFilter(
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
    f.create_out().close()

    out = Path("staggered.nc")
    assert out.exists()

    d = xr.open_dataset(out)
    for n in v:
        assert f"var_{n}" in d.variables
