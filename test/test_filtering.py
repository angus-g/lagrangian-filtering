import pytest

import numpy as np
from scipy import signal
import xarray as xr

import filtering


def test_sanity():
    """Sanity check of filtering.

    Set up a mean velocity field with an oscillating component,
    then filter out the mean.
    """

    # construct sample times (hrs) and velocity field (m/hr)
    t = np.arange(25) + 1  # 12 hours on either side of the centre point
    U0 = 100.0 / 24
    u = U0 + (U0 / 2) * np.sin(2 * np.pi / 6 * (t - 13))

    assert u[12] == pytest.approx(U0)

    # construct filter
    f = signal.butter(4, 1 / 3, "highpass")
    fu = signal.filtfilt(*f, u)
    assert fu[12] == pytest.approx(0.0, abs=1e-4)
