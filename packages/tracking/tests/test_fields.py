import datetime

import cftime
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from tracking.fields import seconds_since


def test_seconds_since_datetime64():
    values = np.array(
        [
            "2000-01-01T00:00:00.000",
            "2000-01-01T00:01:30.500",
            "1999-12-31T23:59:59.750",
        ],
        dtype="datetime64[ms]",
    )

    result = seconds_since(values, origin=np.datetime64("2000-01-01"))

    assert_allclose(result, [0.0, 90.5, -0.25])
    assert result.dtype == np.float64


def test_seconds_since_python_datetimes_preserves_shape():
    origin = datetime.datetime(2000, 1, 1)  # noqa: DTZ001
    values = np.array(
        [
            [origin, origin + datetime.timedelta(seconds=1)],
            [
                origin - datetime.timedelta(seconds=2.5),
                origin + datetime.timedelta(days=1),
            ],
        ],
        dtype=object,
    )

    result = seconds_since(values, origin=origin)

    assert_array_equal(
        result,
        [
            [0.0, 1.0],
            [-2.5, 86_400.0],
        ],
    )
    assert result.shape == values.shape
    assert result.dtype == np.float64


def test_seconds_since_cftime_dates():
    origin = cftime.Datetime360Day(2000, 1, 1)
    values = np.array(
        [
            origin,
            cftime.Datetime360Day(2000, 2, 1),
            cftime.Datetime360Day(1999, 12, 30),
        ],
        dtype=object,
    )

    result = seconds_since(values, origin=origin)

    assert_array_equal(result, [0.0, 30 * 86_400.0, -86_400.0])


@pytest.mark.parametrize(
    "values",
    [
        np.array([0, 1, 2]),
        np.array([0.0, 1.0, 2.0]),
    ],
)
def test_seconds_since_rejects_numeric_values(values):
    with pytest.raises(
        ValueError,
        match="numeric times must already be seconds",
    ):
        seconds_since(values, origin=np.datetime64("2000-01-01"))


def test_seconds_since_rejects_nat():
    values = np.array(
        ["2000-01-01", "NaT"],
        dtype="datetime64[D]",
    )

    with pytest.raises(ValueError, match="time coordinates must be finite"):
        seconds_since(values, origin=np.datetime64("2000-01-01"))
