import datetime
import os
from typing import Protocol

import cftime
import dask.array as da
import numpy as np
import optype.numpy as onp
from numpy.typing import NDArray

Date = datetime.datetime | cftime.datetime
DateArray = NDArray[np.float64] | NDArray[np.datetime64] | NDArray[np.object_]
PathLike = str | os.PathLike[str]
Timespan = float | datetime.timedelta


class FilterLike(Protocol):
    def apply_filter(
        self,
        data: onp.Array2D[np.float64] | da.Array,
        time_index: int,
        min_window: int | None = None
    ) -> onp.Array1D[np.float64] | da.Array:
        ...
