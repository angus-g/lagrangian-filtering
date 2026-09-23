# Lagrangian tracking — numerical prototype

A standalone horizontal tracking package being developed alongside
`lagrangian-filtering`. It does not import Parcels or the `filtering` package.
New code can use `open_flow` with `GridSpec` and `Variable` descriptions, followed
by `flow.advect(seeds, times=..., dt=...)` in either direction. Filtering windows
belong to `filtering.WindowFilter`. See [the composition guide](../TRACKING_WORKFLOW.md)
and [the complete file-input example](../examples/filter_with_tracking.py).

This implementation establishes the numerical and data interface. It is
tested on synthetic grids, **not yet validated against production MOM5, MOM6,
or ROMS trajectories**. In particular, global tripolar connectivity is not
implemented. Do not treat this as a production replacement yet.

## Install and run

From the repository root:

```sh
python -m pip install -e './tracking[test,io,filtering]'
NUMBA_NUM_THREADS=4 python tracking/examples/synthetic.py
cd tracking
NUMBA_NUM_THREADS=2 python -m pytest -q
```

Numba compiles the numerical functions on first use and caches them. Python
handles I/O and scheduling. A parallel loop advances independent particles
against shared read-only velocity frames. Set `NUMBA_NUM_THREADS` to the CPUs
allocated to each PBS process; the library does not change global thread limits.

## Grid and field contract

`Grid(x, y)` accepts planar coordinates in metres. Use
`Grid(lon, lat, spherical=True)` for degrees, with an optional `radius` in metres.
Coordinates may be 1-D axes or 2-D arrays of **cell corners**. Grid cells must be
convex, positively oriented, and nondegenerate. The mesh is static.

For a grid of `ny` by `nx` cells, fields have these spatial shapes:

| Location | Shape `(y, x)` | Interpolation |
| --- | --- | --- |
| `node` | `(ny+1, nx+1)` | Bilinear within each cell |
| `cell` | `(ny, nx)` | Piecewise constant |
| `x_face` | `(ny, nx+1)` | Linear between west/east faces; constant along a face |
| `y_face` | `(ny+1, nx)` | Linear between south/north faces; constant along a face |

On a B-grid, U and V occupy the nodes. On a C-grid they occupy `x_face` and
`y_face`, respectively. Array offsets, halos, missing boundary faces, and model
indexing conventions must be resolved before constructing fields. There is no
automatic model-name or dimension-name inference. A ROMS psi-point grid and a
MOM6 corner-point grid feed the same interface once their arrays are aligned.

`Tracker(..., basis=...)` makes velocity orientation explicit:

* `geographic`: colocated components along east/north (planar x/y on a flat mesh).
* `grid_aligned`: colocated components along local positive grid directions;
  the bilinear coordinate mapping defines the basis. Use orthogonal grids for
  the usual interpretation as orthogonal vector components.
* `face_normal`: C-grid components normal to faces, positive toward increasing
  logical indices. Face speed times face length gives a 2-D flux per unit
  depth; its reconstruction and the local Jacobian determine logical motion.

Do not label geographic components on staggered faces as `face_normal`.
Likewise, arbitrary nonorthogonal grid-tangent components need conversion to
normal components first. Transport variables must be converted to velocity by
the caller; this is not a layer-volume or thickness-conserving 3-D scheme.

Spherical cells use bilinear **unwrapped longitude/latitude** geometry, with
spherical metric factors and face-length quadrature. This is an approximation
of model geometry, not a reconstruction from the model's exact metric arrays.
It handles longitude discontinuities away from poles. Latitude magnitudes
at or above 89.9 degrees are rejected; northern folds, polar caps, and multi-tile
grids are not supported. Supplying a folded grid without its connectivity is
not supported even if it passes local cell checks.

## Input and calendars

`Field.from_array(times, values, location=...)` expects `(time, y, x)` arrays.
`Field(times, read_frame, location=..., shape=...)` accepts a callback returning
one `(y, x)` frame, and caches at most two frames. Frames must be real numbers;
missing values must already be NaNs or NumPy masks. Fill-value sentinels such as
`-1e20` are not inferred by the numerical core.

`Field.from_xarray` preserves lazy loading, selects the vertical level before
reading, and transposes to the explicit order. Open data with normal CF decoding
and mask/scale processing enabled. For example, using the supplied MOM5 naming:

```python
import xarray as xr
from lagrangian_tracking import Field, seconds_since

ds = xr.open_dataset("ocean_daily_3d_u.nc")
origin = ds.time.values[0]  # reuse this same origin for V and sampled fields
times = seconds_since(ds.time.values, origin=origin)
u = Field.from_xarray(
    ds.u, times=times, time_dim="time", y_dim="yu_ocean", x_dim="xu_ocean",
    indices={"st_ocean": 20}, location="node",
)
```

Keep the dataset open until tracking finishes. The external `geolon_c/geolat_c`
grid must be supplied separately. For this B-grid, they are velocity nodes;
the unpadded arrays define cells between those nodes. Closing a periodic seam
requires an additional node column and matching field values. Selecting or
padding tracer fields to match those cells is the caller's responsibility.

`seconds_since` supports decoded NumPy dates, Python datetimes and cftime
dates, including NOLEAP. All fields must share the same origin/calendar. Numeric
time axes must be converted explicitly to seconds. Every field needs at least
two strictly increasing time records. Velocity components share a time axis;
sampled fields may have separate axes and must cover all requested samples.

Time reconstruction is linear between the recorded times. For daily means,
that means treating each recorded mean as the reconstructed value at its
timestamp; averaging bounds are not used to infer subdaily velocities.

## Compatibility with the first prototype

```python
from lagrangian_tracking import AdvectionAdapter, Tracker

tracker = Tracker(grid, u, v, basis="grid_aligned", fields={"U": u, "V": v})
seeds = grid.locate(seed_lon, seed_lat)
adapter = AdvectionAdapter(
    tracker, seeds,
    window_size=3*86400, output_dt=86400, advection_dt=300,
    output_dir="/path/to/existing/scratch-directory",
)
data = adapter.advection_step(time_in_seconds, output_time=True)
# data['var_U'] == (centre_index, dask_array[time, particle])
# data['time'] contains chronological seconds, with the centre exactly once.
```

For an existing `LagrangeFilter` object `ff`, use its flattened seed positions in
the same order, its relative-time origin, its sampling interval, and its window:

```python
ff.advection_step = adapter.advection_step
```

The existing `filter_step` can consume the result. This replaces the advection
operation only: the legacy constructor, output-grid metadata and calendar
conversion still depend on Parcels. Removing those is a later integration step.
The adapter takes a snapshot of its own configuration; changing `ff` afterwards
does not update it. Windows must be integer multiples of the sampling interval.

The compatibility `tracker.sample(...)` accepts arbitrary increasing sample times
including the seed time, and can optionally return logical positions. It runs
forward and backward independently from the seed, stopping exactly at output
times and velocity input knots. `dt` is a positive maximum RK4 step. Large
logical displacements trigger step subdivision; this is **not** an adaptive
error-tolerance integrator. Convergence should be checked by reducing `dt`.

Particles retain their original output columns. Status codes distinguish active,
outside-domain, dry-cell, missing-velocity, and failed-step particles. After
failure in a direction, its samples are NaN. A missing sampled scalar does not
kill an otherwise valid trajectory. Samples preserve the field's native values
and units; sampling U/V does not rotate their components automatically.

`wet` is an explicit cell mask. Stage/final segments crossing a dry cell stop
the particle; the code does not extrapolate through land, reflect particles, or
implement beaching. A bad RK stage is handled conservatively as failure.
Missing face velocities are not silently replaced by zero. Exact positions on
an internal face use the higher-index cell, except the outermost boundary.

`periodic_x`/`periodic_y` wrap logical positions at **every RK stage**. The input
must include the closing node edge and geometrically consistent seams with
matching field values. Only simple translational seams are supported; vector
rotation or index reversal across a fold is not implemented.

Without `output_dir`, all samples reside in memory. With it, each call creates a
unique directory of `.npy` memory maps and metadata; Dask reads slices from disk
without copying the full sample array. The caller must remove these directories
after all consumers finish. Restart/resume is not implemented. Input frames and
particle state remain in memory; physical-coordinate seed lookup additionally
builds a spatial index. Use `grid.logical_seeds` for known grid locations to
avoid that lookup. Float32 input frames are retained as float32, while geometry,
particle positions and sample outputs use float64.

## Next implementation work

* Validate real MOM5 and ROMS subsets against the existing workflow, checking
  coordinate orientation, seed locations and array offsets explicitly.
* Add model input helpers, supplied metric support, and explicit fold connectivity.
* Improve tracer interpolation beyond the initial cellwise constant option.
* Migrate legacy callers to the new Parcels-free WindowFilter composition API.
* Benchmark realistic particle counts and memory consumption on a compute node.
