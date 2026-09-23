import filtering
import numpy as np
import tracking


def test_power_spectrum(leewave_data, null_reducer):
    """Test computation of the power spectrum for velocity in
    the leewave dataset."""

    g = tracking.Grid(
        leewave_data.x,
        leewave_data.y,
        periodic_x=True,
        periodic_y=True
    )
    u = tracking.Field.from_xarray(
        leewave_data.U,
        times=leewave_data.t,
        time_dim="t",
        y_dim="y",
        x_dim="x",
    )
    v = tracking.Field.from_xarray(
        leewave_data.V,
        times=leewave_data.t,
        time_dim="t",
        y_dim="y",
        x_dim="x",
    )
    tracker = tracking.Tracker(g, u, v, basis="grid_aligned", fields={"U": u})
    w = filtering.WindowFilter(
        tracker,
        g.seed_grid(location="node"),
        window=3 * 24 * 3600,  # 3 days on either side
        sample_dt=float(leewave_data.t[1] - leewave_data.t[0]),
        advection_dt=300,  # 5 minute
        reducer=null_reducer,
    )

    spectrum = filtering.analysis.power_spectrum(w, 7 * 24 * 3600)

    assert "U" in spectrum
    assert np.all(np.isreal(spectrum["U"]))
