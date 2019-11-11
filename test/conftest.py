import pytest

from findiff import FinDiff
import numpy as np
import os
import xarray as xr


@pytest.fixture(scope="function")
def enable_threads():
    num_threads_old = None
    if "OMP_NUM_THREADS" in os.environ:
        num_threads_old = os.environ["OMP_NUM_THREADS"]
        del os.environ["OMP_NUM_THREADS"]

    yield True

    if num_threads_old is not None:
        os.environ["OMP_NUM_THREADS"] = num_threads_old


@pytest.fixture(scope="session")
def leewave_data():
    """Session-wide fixture containing lee wave data with an overlaid moving eddy and mean flow.

    The eddy is advected by the mean flow hourly for two weeks.

    """

    # lee wave velocity dataset
    d = xr.open_dataset("test/data/lee_wave.nc")

    # mean flow velocity
    U = 0.2
    # eddy size
    es = 10e3

    # grid
    dx = 200
    dy = 1000
    dt = 3600
    x = np.arange(0, 350e3 + 1, dx)
    y = np.arange(0, 100e3 + 1, dy)
    t = np.arange(0, 2 * 7 * 24 * 3600 + 1, dt)
    T, Y, X = np.meshgrid(t, y, x, indexing="ij")

    # finite difference operators
    d_dx = FinDiff(2, dx)
    d_dy = FinDiff(1, dy)

    # eddy centre through advection
    xc = U * T
    # eddy field
    psit1 = 0.05 * es * np.exp(-((X - xc) ** 2 + (Y - 50e3) ** 2) / es ** 2)
    psit2 = 0.05 * es * np.exp(-((X - (xc + x[-1])) ** 2 + (Y - 50e3) ** 2) / es ** 2)
    psit = psit1 + psit2
    VM = -d_dx(psit)
    UM = d_dy(psit)

    Utot = U + UM + d.U.data[None, None, :]
    Vtot = VM + d.V.data[None, None, :]

    return xr.Dataset(
        {
            "U": (["t", "y", "x"], Utot),
            "V": (["t", "y", "x"], Vtot),
            "U_orig": (["x"], d.U),
            "V_orig": (["x"], d.V),
        },
        coords={"x": x, "y": y, "t": t},
    )
