"""Kernels for reference coordinate conversion and interpolation.

Kernels are JIT-compiled with numba.
"""

import math
from enum import IntEnum, auto
from typing import overload

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


class ParticleStatus(IntEnum):
    """Kernel status.

    ACTIVE: A normal, wet cell.
    OUTSIDE: A cell outside the domain.
    DRY: A dry cell within the domain.
    MISSING: TODO
    STEP_FAILED: TODO
    """

    ACTIVE = auto()
    OUTSIDE = auto()
    DRY = auto()
    MISSING = auto()
    STEP_FAILED = auto()


class CellStaggering(IntEnum):
    """Spatial staggering of a point within a cell.

    NODE: A node (q-point).
    CELL: A center (h-point).
    X_FACE: Face in X direction (u-point).
    Y_FACE: Face in Y direction (v-point).

    """

    NODE = auto()
    CELL = auto()
    X_FACE = auto()
    Y_FACE = auto()


class GridBasis(IntEnum):
    """Alignment basis of grid.

    GEOGRAPHIC: Velocities are in geographic X/Y directions.
    GRID_ALIGNED: Velocities are along grid X/Y directions.
    FACE_NORMAL: Velocities are normal to X/Y faces.

    """

    GEOGRAPHIC = auto()
    GRID_ALIGNED = auto()
    FACE_NORMAL = auto()


@njit(cache=True)
def cell_at(
    x: float,
    y: float,
    wet: NDArray[np.bool],
    periodic_x: bool,
    periodic_y: bool
) -> tuple[int, int, float, float, ParticleStatus]:
    """Convert continuous logical coordinates to discrete cell indices.

    Args:
        x: X logical coordinate (continuous in [0, nx)).
        y: Y logical coordinate (cordinuous in [0, ny)).
        wet: Mask array of wet cells.
        periodic_x: Whether domain is x-periodic.
        periodic_y: Whether domain is y-periodic.

    """

    ny, nx = wet.shape

    if not math.isfinite(x) or not math.isfinite(y):
        return 0, 0, 0.0, 0.0, ParticleStatus.OUTSIDE

    if periodic_x:
        x %= nx
    if periodic_y:
        y %= ny

    if x < 0 or x > nx or y < 0 or y > ny:
        return 0, 0, 0.0, 0.0, ParticleStatus.OUTSIDE

    i, j = min(int(x), nx - 1), min(int(y), ny - 1)
    status = ParticleStatus.ACTIVE if wet[j, i] else ParticleStatus.DRY

    return i, j, x - i, y - j, status


@overload
def bilinear_mapping(
    a: NDArray[np.float64],
    i: int,
    j: int,
    s: float,
    r: float,
) -> tuple[float, float, float]:
    ...


@overload
def bilinear_mapping(
    a: NDArray[np.float64],
    i: NDArray[np.integer],
    j: NDArray[np.integer],
    s: NDArray[np.float64],
    r: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    ...


@overload
def bilinear_mapping(
    a: NDArray[np.float64],
    i: NDArray[np.integer],
    j: NDArray[np.integer],
    s: float,
    r: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    ...


@njit(cache=True)
def bilinear_mapping(a, i, j, s, r):
    """Forward map from logical coordinates to interpolated value
    from corners.

    `i`, `j`, `s` and `r` may be all scalars to perform interpolation
    at a single point, or all arrays with the same shape. The output
    shape will match that of the inputs.

    Args:
        a: Data array, values located at corners.
        i: Cell index in x direction.
        j: Cell index in y direction.
        s: Reference coordinate in i direction ∈ [0,1].
        r: Reference coordinate in j direction ∈ [0,1].

    Returns:
        3-tuple of interpolated value, x and y derivatives

    """

    a00, a10 = a[j, i], a[j, i + 1]
    a01, a11 = a[j + 1, i], a[j + 1, i + 1]
    value = (
        (1 - s) * (1 - r) * a00
        + s * (1 - r) * a10
        + (1 - s) * r * a01
        + s * r * a11
    )
    ds = (1 - r) * (a10 - a00) + r * (a11 - a01)
    dr = (1 - s) * (a01 - a00) + s * (a11 - a10)
    return value, ds, dr


@njit(cache=True)
def jacobian(
    gx: NDArray[np.float64],
    gy: NDArray[np.float64],
    i: int,
    j: int,
    s: float,
    r: float,
    spherical: bool,
    radius: float,
) -> tuple[float, float, float, float]:
    """Calculate grid Jacobian components.

    Calculates dx/ds, dx/dr, dy/ds, dy/dr, handling
    the conversion from degrees to metres on spherical
    grids.

    Args:
        gx: Grid X-coordinates.
        gy: Grid Y-coordinates.
        i: Cell index in x direction.
        j: Cell index in y direction.
        s: Reference coordinate in i direction ∈ [0,1].
        r: Reference coordinate in j direction ∈ [0,1].
        spherical: Whether grid is spherical (coordinates in degrees).
        radius: Radius of the sphere.

    Returns:
        4-tuple of Jacobian components (dx/ds, dx/dr, dy/ds, dy/dr).

    """

    _, xs, xr = bilinear_mapping(gx, i, j, s, r)
    latitude, ys, yr = bilinear_mapping(gy, i, j, s, r)
    if spherical:
        scale = radius * math.pi / 180.0
        xs *= scale * math.cos(latitude * math.pi / 180.0)
        xr *= scale * math.cos(latitude * math.pi / 180.0)
        ys *= scale
        yr *= scale
    return xs, xr, ys, yr


@njit(cache=True)
def weighted(a: float, b: float, weight: float) -> float:
    """1D linear interpolation.

    Interpolates between a and b with weight ∈ [0,1]. In the case when
    the weight is exactly at an extremum and the opposite value is NaN,
    this will still return a value (useful for interpolation next to
    missing data).

    Args:
        a: Left-hand value.
        b: Right-hand value.
        weight: Interpolation weight.

    Returns:
        The interpolated value between a and b.

    """

    if weight == 0.0:
        return a
    if weight == 1.0:
        return b
    return a * (1.0 - weight) + b * weight


@njit(cache=True)
def interpolate(
    a: NDArray[np.float64],
    location: CellStaggering,
    i: int,
    j: int,
    s: float,
    r: float,
) -> float:
    match location:
        case CellStaggering.CELL:
            return a[j, i]
        case CellStaggering.X_FACE:
            return weighted(a[j, i], a[j, i + 1], s)
        case CellStaggering.Y_FACE:
            return weighted(a[j, i], a[j + 1, i], r)

    low = weighted(a[j, i], a[j, i + 1], s)
    high = weighted(a[j + 1, i], a[j + 1, i + 1], s)
    return weighted(low, high, r)


@njit(cache=True)
def tendency(
    x: float,
    y: float,
    t: float,
    gx: NDArray[np.float64],
    gy: NDArray[np.float64],
    wet: NDArray[np.bool],
    px: bool,
    py: bool,
    spherical: bool,
    radius: float,
    u0: NDArray[np.float64],
    u1: NDArray[np.float64],
    v0: NDArray[np.float64],
    v1: NDArray[np.float64],
    t0: float,
    t1: float,
    basis: GridBasis,
    xlength: NDArray[np.float64],
    ylength: NDArray[np.float64],
) -> tuple[float, float, ParticleStatus]:
    """Evaluate particle position tendency at given position and time.

    Velocity is spatially interpolated in the cell, and linearly
    interpolated in time. Velocity components are transformed into
    logical coordinate space by the local grid Jacobian.

    Args:
        x: Particle X-coordinate.
        y: Particle Y-coordinate.
        t: Time at which to evaluate.
        gx: Grid X-coordinates.
        gy: Grid Y-coordinates.
        wet: Ocean mask array.
        px: Whether grid is x-periodic.
        py: Whether grid is y-periodic.
        spherical: Whether grid is spherical.
        radius: Radius of spherical grid.
        u0: U-velocity array at time t0.
        u1: U-velocity array at time t1.
        v0: V-velocity array at time t0.
        v1: V-velocity array at time t1.
        t0: Sampling time at start of interval.
        t1: Sampling time at end of interval.
        basis: Velocity basis for grid.
        xlength: Length of U-faces of cells.
        ylength: Length of V-faces of cells.

    Returns:
        3-tuple of dx/dt, dy/dt and particle status.

    """

    i, j, s, r, status = cell_at(x, y, wet, px, py)
    if status != ParticleStatus.ACTIVE:
        return 0.0, 0.0, status

    # Normalised position along trajectory
    alpha = (t - t0) / (t1 - t0)
    xs, xr, ys, yr = jacobian(gx, gy, i, j, s, r, spherical, radius)
    det = xs * yr - xr * ys

    if basis == GridBasis.FACE_NORMAL:
        # Construct flux at each face of the cell, then interpolate
        # at the target location per the local Jacobian
        west = weighted(u0[j, i], u1[j, i], alpha) * xlength[j, i]
        east = weighted(u0[j, i + 1], u1[j, i + 1], alpha) * xlength[j, i + 1]
        south = weighted(v0[j, i], v1[j, i], alpha) * ylength[j, i]
        north = weighted(v0[j + 1, i], v1[j + 1, i], alpha) * ylength[j + 1, i]
        dx = weighted(west, east, s) / det
        dy = weighted(south, north, r) / det
    else:
        u = weighted(
            interpolate(u0, CellStaggering.NODE, i, j, s, r),
            interpolate(u1, CellStaggering.NODE, i, j, s, r),
            alpha
        )
        v = weighted(
            interpolate(v0, CellStaggering.NODE, i, j, s, r),
            interpolate(v1, CellStaggering.NODE, i, j, s, r),
            alpha
        )

        if basis == GridBasis.GRID_ALIGNED:
            # Components along the local positive grid directions, we
            # only need to scale to logical grid coordinates.
            dx = u / math.hypot(xs, ys)
            dy = v / math.hypot(xr, yr)
        else:
            # Components are along geographic directions and need
            # transformation to local coordinates.
            dx = (yr * u - xr * v) / det
            dy = (xs * v - ys * u) / det

    if not math.isfinite(dx) or not math.isfinite(dy):
        return 0.0, 0.0, ParticleStatus.MISSING

    return dx, dy, ParticleStatus.ACTIVE


@njit(cache=True)
def segment_status(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    wet: NDArray[np.bool],
    px: bool,
    py: bool,
) -> ParticleStatus:
    """Check every cell traversed by a straight logical-coordinate segment.

    The segment is split at integer grid-cell boundaries, and one point in
    each traversed cell is checked.

    Args:
        x0: Start x-coordinate.
        y0: Start y-coordinate.
        x1: End x-coordinate.
        y1: End y-coordinate.
        wet: Ocean mask array.
        px: Whether domain is x-periodic.
        py: Whether domain is y-periodic.

    Returns:
        The first non-ACTIVE status along the trajectory if one
        is encountered. If the trajectory only passes through ACTIVE
        cells, return ACTIVE.

    """

    BOUNDARY_EPS = 1.e-10
    ITERATION_PADDING = 5

    # Check the endpoint first, because it won't lie on the
    # interior of a split of the segment.
    _, _, _, _, status = cell_at(x1, y1, wet, px, py)
    if status != ParticleStatus.ACTIVE:
        return status

    dx, dy = x1 - x0, y1 - y0
    t = 0.0

    # The number of crossed cells may be slightly greater than
    # the L0-norm of the segment: add a few iterations of
    # padding rather than computing precisely.
    for _ in range(int(abs(dx) + abs(dy)) + ITERATION_PADDING):
        # Current point along the segment
        x, y = x0 + t * dx, y0 + t * dy

        # Solve for the next integer coordinates
        tx, ty = np.inf, np.inf
        if dx > 0:
            tx = (math.floor(x + BOUNDARY_EPS) + 1 - x0) / dx
        elif dx < 0:
            tx = (math.ceil(x - BOUNDARY_EPS) - 1 - x0) / dx
        if dy > 0:
            ty = (math.floor(y + BOUNDARY_EPS) + 1 - y0) / dy
        elif dy < 0:
            ty = (math.ceil(y - BOUNDARY_EPS) - 1 - y0) / dy

        end = min(1.0, tx, ty)
        mid = (t + end) * 0.5
        # Search the midpoint of the current position and the next
        # integer coordinates, to land within a cell
        _, _, _, _, status = cell_at(x0 + mid * dx, y0 + mid * dy, wet, px, py)

        if status != ParticleStatus.ACTIVE:
            return status

        if end >= 1.0:
            return ParticleStatus.ACTIVE

        t = end

    return ParticleStatus.STEP_FAILED


@njit(cache=True, parallel=True)
def advance(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    status: NDArray[np.int8],
    start: float,
    end: float,
    max_dt: float,
    gx: NDArray[np.float64],
    gy: NDArray[np.float64],
    wet: NDArray[np.bool],
    periodic_x: bool,
    periodic_y: bool,
    spherical: bool,
    radius: float,
    u0: NDArray[np.float64],
    u1: NDArray[np.float64],
    v0: NDArray[np.float64],
    v1: NDArray[np.float64],
    t0: float,
    t1: float,
    basis: GridBasis,
    xlength: NDArray[np.float64],
    ylength: NDArray[np.float64],
) -> None:
    """Take one RK4 timestep across all particles, simultaneously.

    Args:
        x: X-coordinates of particles.
        y: Y-coordinates of particles.
        status: Cell status of particles (active, grounded, etc.)
        start: Start time of this timestep.
        end: End time of this timestep.
        max_dt: Maximum advective timestep.
        gx: Grid X-coordinates.
        gy: Grid Y-coordinates.
        wet: Ocean mask array.
        periodic_x: Whether grid is x-periodic.
        periodic_y: Whether grid is y-periodic.
        spherical: Whether grid is spherical.
        radius: Spherical grid radius.
        u0: U-velocity array at time t0.
        u1: U-velocity array at time t1.
        v0: V-velocity array at time t0.
        v1: V-velocity array at time t1.
        t0: Sampling time at start of interval.
        t1: Sampling time at end of interval.
        basis: Velocity basis for grid.
        xlength: Length of U-faces of cells.
        ylength: Length of V-faces of cells.

    """

    # Determine whether advection is forward or backward in time.
    direction = 1.0 if end > start else -1.0

    for p in prange(x.size):  # ty: ignore[not-iterable]
        if status[p] != ParticleStatus.ACTIVE:
            continue

        t = start
        xp, yp = x[p], y[p]
        # Tendencies for each RK stage
        kx = np.empty(4)
        ky = np.empty(4)

        while direction * (end - t) > 0:
            # Initially, try to take the largest timestep (either to the
            # specified advection end, or the maximum advective timestep).
            h = direction * min(max_dt, abs(end - t))

            # Iteratively try timesteps to ensure we don't exceed a CFL
            # condition in local coordinates: displacement must not
            # exceed 0.5 for a value of h.
            # This is a control on cell traversal, not truncation error; dt
            # remains user-selected.
            accepted = False
            for attempt in range(40):
                valid = ParticleStatus.ACTIVE
                too_far = False  # whether we moved more than half a cell

                # RK4 kernel for this combination of (t,h)
                for stage in range(4):
                    fraction = [0.0, 0.5, 0.5, 1.0][stage]

                    xx = xp
                    yy = yp
                    # for subsequent stages, add the sub-stage
                    # contribution of the previous one
                    if stage != 0:
                        xx += h * fraction * kx[stage - 1]
                        yy += h * fraction * ky[stage - 1]

                    # substage is too far from original point
                    if max(abs(xx - xp), abs(yy - yp)) > 0.5:
                        too_far = True
                        break

                    valid = segment_status(xp, yp, xx, yy, wet, periodic_x, periodic_y)
                    if valid != ParticleStatus.ACTIVE:
                        break

                    kx[stage], ky[stage], valid = tendency(
                        xx,
                        yy,
                        t + fraction * h,
                        gx,
                        gy,
                        wet,
                        periodic_x,
                        periodic_y,
                        spherical,
                        radius,
                        u0,
                        u1,
                        v0,
                        v1,
                        t0,
                        t1,
                        basis,
                        xlength,
                        ylength,
                    )

                    if valid != ParticleStatus.ACTIVE:
                        break

                # after RK4 stages, check that we didn't break out of a substage
                if valid != ParticleStatus.ACTIVE:
                    status[p] = valid
                    break

                # compute the final position from all substages
                if not too_far:
                    xn = xp + h * (kx[0] + 2 * kx[1] + 2 * kx[2] + kx[3]) / 6
                    yn = yp + h * (ky[0] + 2 * ky[1] + 2 * ky[2] + ky[3]) / 6
                    too_far = max(abs(xn - xp), abs(yn - yp)) > 0.5

                # reduce step length and retry
                if too_far:
                    h *= 0.5
                    continue

                valid = segment_status(xp, yp, xn, yn, wet, periodic_x, periodic_y)
                if valid != ParticleStatus.ACTIVE:
                    status[p] = valid
                    break

                # prevent infinite loop if we split too finely
                if t + h == t:
                    status[p] = ParticleStatus.STEP_FAILED
                    break

                # wrap logical coordinates
                xp, yp = xn, yn
                if periodic_x:
                    xp %= wet.shape[1]
                if periodic_y:
                    yp %= wet.shape[0]

                # update current time by the step length we actually took
                t = end if abs(h) >= abs(end - t) else t + h
                accepted = True
                break

            if not accepted:
                if status[p] == ParticleStatus.ACTIVE:
                    status[p] = ParticleStatus.STEP_FAILED
                break

        x[p], y[p] = xp, yp


@njit(cache=True, parallel=True)
def sample_values(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    status: NDArray[np.int8],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    alpha: float,
    location: CellStaggering,
    wet: NDArray[np.bool],
    px: bool,
    py: bool,
) -> NDArray[np.float64]:
    """Field sampling kernel: linearly interpolate a field
    at all particle locations.

    Args:
        x: X-coordinates of particles.
        y: Y-coordinates of particles.
        status: Cell status of particles (active, grounded, etc.)
        a: Tracer field at start of interval.
        b: Tracer field at end of interval.
        alpha: Normalised time in sampling interval.
        location: Cell staggering of data.
        wet: Ocean mask array.
        px: Whether grid is x-periodic.
        py: Whether grid is y-periodic.

    Returns:
        An array of sampled values (NaN for invalid particles).

    """

    values = np.full(x.size, np.nan)

    for p in prange(x.size):  # ty: ignore[not-iterable]
        if status[p] != ParticleStatus.ACTIVE:
            continue

        i, j, s, r, code = cell_at(x[p], y[p], wet, px, py)

        if code == ParticleStatus.ACTIVE:
            values[p] = weighted(
                interpolate(a, location, i, j, s, r),
                interpolate(b, location, i, j, s, r),
                alpha
            )

    return values
