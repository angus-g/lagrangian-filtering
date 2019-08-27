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
        {"lon": "lon", "lat": "lat", "time": "time"},
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
