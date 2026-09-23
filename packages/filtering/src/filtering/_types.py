import datetime
import os
from typing import Protocol

import cftime
import dask.array as da
import numpy as np
from numpy.typing import NDArray

Date = datetime.datetime | cftime.datetime
DateArray = NDArray[np.float64] | NDArray[np.datetime64] | NDArray[np.object_]
PathLike = str | os.PathLike[str]
Timespan = float | datetime.timedelta


class FilterLike(Protocol):
    def apply_filter(
        self,
        data: np.ndarray | da.Array,
        time_index: int,
        min_window: int | None = None
    ) -> np.ndarray | da.Array:
        ...
