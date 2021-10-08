# lagrangian-filtering
Temporal filtering of data in a Lagrangian frame of reference. Visit the
[documentation](https://lagrangian-filtering.readthedocs.io/en/latest/),
or continue reading below.

## Overview
This provides a library and a post-processing analysis tool for the
removal of sub-inertial frequencies from data in a Lagrangian frame of
reference. This may be useful, for example, for distinguishing
stationary internal waves. At a high level, the algorithm looks like:

1. particle advection using [OceanParcels](http://oceanparcels.org)
2. sampling of data (e.g. velocity, density) along particle tracks
3. temporal filtering
4. writing filtered data back to disk

## Citing
The description and analysis of the Lagrangian filtering method has
been published in JAMES, and can be cited as:

> Shakespeare, C. J., Gibson, A. H., Hogg, A. M., Bachman, S. D., Keating, S. R., & Velzeboer, N. (2021). A new open source implementation of Lagrangian filtering: A method to identify internal waves in high-resolution simulations. *Journal of Advances in Modeling Earth Systems*, 13, e2021MS002616. https://doi.org/10.1029/2021MS002616

## Installation
### Using Conda
If you use [Conda](https://conda.io) to manage Python packages, you
may simply run `conda install -c angus-g -c conda-forge
lagrangian-filtering` to install this package and its required
dependencies. You may want to constrain this to its own conda
environment instead, with `conda create -n filtering -c angus-g -c
conda-forge lagrangian-filtering`. The environment can be activated
with `conda activate filtering` and deactivated with `conda
deactivate`.

### Using pip or developing
If you don't use Conda, or are looking to develop this package, it is
easier to get started using Python's `pip` package
manager. Optionally, the package can be installed inside a
`virtualenv` virtual environment, for cleaner separation from your
native Python environment. There are very few dependencies, but a
custom branch of OceanParcels is necessary to get acceptable
performance at the moment, and the GCC compiler with OpenMP support is
also required to compile the runtime kernels. During the development
phase, I recommend installing this as a "development package", meaning
that changes to your local checkout are instantly reflected in your
Python environment.

1. Clone this repository `git clone https://github.com/angus-g/lagrangian-filtering`
2. Change to the directory `cd lagrangian-filtering`
3. (Optional) Create the virtualenv `virtualenv env` and activate it `source env/bin/activate`
4. Install the development version of the package `pip install -e .`
5. Install the dependencies required for parcels `pip install -r requirements.txt`

#### Upgrading
With the package installed in development mode, updating is as easy as
`git pull` (or making local changes) in the `lagrangian-filtering`
directory. If dependencies (particularly parcels) need to be updated,
run `pip install --upgrade --upgrade-strategy eager .` to force installation
of the new versions.

## Usage
For the moment, it's easiest to set up the filtering workflow in a script or
a jupyter notebook. An example looks like:

```python
import filtering
from datetime import timedelta

filenames = {
	"U": "/data/data_wave_U.nc", "V": "/data/data_wave_V.nc"
}
variables = {"U": "U", "V": "V"}
dimensions = {"lon": "X", "lat", "Y", "time": "T"}

f = filtering.LagrangeFilter(
	"waves", filenames, variables, dimensions,
	sample_variables=["U"], mesh="flat",
	window_size=timedelta(days=2).total_seconds()
)

f()
```

This uses velocity data from the two specified files. Zonal velocity
data from the `U` variable will be sampled and filtered, with a
filtering window for 2 days on either side of each sample (i.e. a
4-day window for filtering).

### 3D input data
For 3D input data (i.e. with a Z dimension), there are a couple of
options. Running the filtering as the example above will load only the
first Z slice (likely the surface). For more precise control of the Z
dimension, modify the `dimensions` dictionary, and specify the level
to load in the `indices` dictionary. For example, to load only the
21st Z slice:

```python
dimensions = {"lon": "X", "lat": "Y", "time": "T", "depth", "Z"}
indices = {"depth": [20]}

f = filtering.LagrangeFilter(
	... # other parameters as normal
	indices=indices,
	... # any other keyword parameters
)
```

Beware that specifyng the `depth` dimension without restricting its
indices will load data through full depth, however particles will only
be seeded at the top. This would be a huge expense for no gain!
