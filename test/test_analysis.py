import pytest

import numpy as np

import filtering


def test_power_spectrum(leewave_data):
    """Test computation of the power spectrum for velocity in
    the leewave dataset."""

    f = filtering.LagrangeFilter(
        "power_spectrum",
        leewave_data,
        {"U": "U", "V": "V"},
        {"lon": "x", "lat": "y", "time": "t"},
        ["U"],
        window_size=3 * 24 * 3600,
    )
    f.make_zonally_periodic()
    f.make_meridionally_periodic()

    spectrum = filtering.analysis.power_spectrum(f, 7 * 24 * 3600)

    assert "var_U" in spectrum
    assert np.all(np.isreal(spectrum["var_U"]))
