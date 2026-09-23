import datetime
import os
from typing import Literal

import cftime
import numpy as np
from numpy.typing import NDArray

BasisString = Literal["geographic", "grid_aligned", "face_normal"]
Date = datetime.datetime | cftime.datetime
DateArray = NDArray[np.float64] | NDArray[np.datetime64] | NDArray[np.object_]
PathLike = str | os.PathLike[str]
StaggeringString = Literal["node", "cell", "x_face", "y_face"]
Timespan = float | datetime.timedelta
