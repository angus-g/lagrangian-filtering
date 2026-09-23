"""Explicit quadrilateral geometry, independent of a model's array indexing."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from ._kernels import ParticleStatus, bilinear_mapping
from ._types import StaggeringString

R_EARTH = 6_371_000.0  # Earth's radius in [m]


@dataclass(frozen=True)
class SeedLayout:
    """Original physical layout of flattened seeds."""

    dims: tuple[str, str]
    shape: tuple[int, int]
    flat_indices: NDArray[np.integer]
    coords: dict


@dataclass
class Seeds:
    """Flat logical coordinates and status, copied before each direction runs."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    status: NDArray[np.int8]
    layout: SeedLayout | None = None


class Grid:
    """A logically rectangular mesh whose arrays describe cell corners.

    The x and y axes may be 1-D arrays, or equally shaped 2-D corner coordinates. For
    spherical grids they are longitude/latitude in degrees. Periodic axes
    require the closing row/column of nodes, plus consistent field values.
    Only positively oriented, nondegenerate cells away from poles are allowed.

    Args:
        x: 1D or 2D X-coordinates.
        y: 1D or 2D Y-coordinates.
        spherical: Whether the grid is spherical (coordinates in degrees) or not
            (coordinates in metres).
        radius: Radius of spherical grid.
        wet: Ocean mask array.
        periodic_x: Whether the grid is x-periodic.
        periodic_y: Whether the grid is y-periodic.

    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        spherical: bool = False,
        radius: float = R_EARTH,
        wet: ArrayLike | None = None,
        periodic_x: bool = False,
        periodic_y: bool = False,
    ) -> None:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim == y.ndim == 1:
            x, y = np.meshgrid(x, y)
        if x.ndim != 2 or x.shape != y.shape or min(x.shape) < 2:
            raise ValueError(
                "corner coordinates must have matching 2-D shapes >= (2, 2)"
            )

        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("corner coordinates must be finite")

        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be positive")

        self.spherical, self.radius = spherical, float(radius)
        if spherical:
            if np.any(np.abs(y) >= 89.9):
                raise ValueError("polar cells and tripolar folds are not supported yet")
            x = np.rad2deg(np.unwrap(np.unwrap(np.deg2rad(x), axis=1), axis=0))
            if np.any(np.abs(np.diff(x, axis=1)) >= 180):
                raise ValueError("spherical cells must span less than 180 degrees")

        self.x, self.y = np.ascontiguousarray(x), np.ascontiguousarray(y)
        self.ny, self.nx = x.shape[0] - 1, x.shape[1] - 1

        self.periodic_x, self.periodic_y = periodic_x, periodic_y
        for periodic, edge_dx, edge_dy in [
            (periodic_x, x[:, -1] - x[:, 0], y[:, -1] - y[:, 0]),
            (periodic_y, x[-1] - x[0], y[-1] - y[0]),
        ]:
            if periodic:
                if not np.allclose(
                    edge_dx, edge_dx[0], rtol=1.0e-10, atol=1.0e-10
                ) or not np.allclose(edge_dy, edge_dy[0], rtol=1.0e-10, atol=1.0e-10):
                    raise ValueError(
                        "periodic seams must be translations; folds are unsupported"
                    )
                if spherical and not np.allclose(edge_dy, 0.0, rtol=0, atol=1.0e-10):
                    raise ValueError("spherical periodic seams must preserve latitude")

        self.wet = (
            np.ones((self.ny, self.nx), dtype=bool)
            if wet is None
            else np.array(wet, dtype=bool)
        )
        if self.wet.shape != (self.ny, self.nx):
            raise ValueError("wet mask must have one value per cell")
        self.wet = np.ascontiguousarray(self.wet)
        j, i = np.indices(self.wet.shape)

        # check determinant of all cells
        for s, r in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
            _, xs, xr = bilinear_mapping(x, i, j, s, r)
            _, ys, yr = bilinear_mapping(y, i, j, s, r)
            det = xs * yr - xr * ys
            scale = np.hypot(xs, ys) * np.hypot(xr, yr)
            if np.any(det <= 1.0e-12 * scale) or np.any(scale == 0):
                raise ValueError(
                    "cells must be convex, nondegenerate and positively oriented"
                )

        self.xlength = self._edge_lengths(x[:-1], y[:-1], x[1:], y[1:])
        self.ylength = self._edge_lengths(x[:, :-1], y[:, :-1], x[:, 1:], y[:, 1:])

        self._tree = None

    def _edge_lengths(
        self,
        x0: NDArray[np.float64],
        y0: NDArray[np.float64],
        x1: NDArray[np.float64],
        y1: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Integrate physical edge lengths along the same coordinate mapping
        used by the Jacobian.

        For latitude, this uses three-point Gaussian quadrature.

        Args:
            x0: Left (west) X-coordinates.
            x1: Right (east) X-coordinates.
            y0: Bottom (south) Y-coordinates.
            y1: Top (north) Y-coordinates.

        Returns:
            Array of edge lengths.

        """

        dx, dy = x1 - x0, y1 - y0
        if not self.spherical:
            return np.ascontiguousarray(np.hypot(dx, dy))

        out = np.zeros_like(dx)
        for q, w in zip(
            [0.5 - np.sqrt(15) / 10, 0.5, 0.5 + np.sqrt(15) / 10],
            [5 / 18, 4 / 9, 5 / 18],
        ):
            out += w * np.hypot(dx * np.cos(np.deg2rad(y0 + q * dy)), dy)

        return np.ascontiguousarray(out * self.radius * np.pi / 180)

    def shape(self, location: StaggeringString) -> tuple[int, int]:
        """Get the array shape for a given staggering location.

        Args:
            location: String value of staggering location, one of
                ("node", "cell", "x_face", "y_face").

        """

        return {
            "node": (self.ny + 1, self.nx + 1),
            "cell": (self.ny, self.nx),
            "x_face": (self.ny, self.nx + 1),
            "y_face": (self.ny + 1, self.nx),
        }[location]

    def logical_seeds(self, x: ArrayLike, y: ArrayLike) -> Seeds:
        """Seed values on grid in logical coordinates.

        The seed coord arrays contain values between 0 and the number of
        points in that direction.

        Args:
            x: Logical X-coordinate of seeds.
            y: Logical Y-coordinate of seeds.

        Returns:
            Constructed seed object for valid locations specified
            by the coordinate arrays.
        """

        x, y = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        )
        x, y = x.ravel().copy(), y.ravel().copy()
        if self.periodic_x:
            x %= self.nx
        if self.periodic_y:
            y %= self.ny
        valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & (x >= 0)
            & (x <= self.nx)
            & (y >= 0)
            & (y <= self.ny)
        )
        status = np.full(x.size, ParticleStatus.OUTSIDE, dtype=np.int8)
        p = np.flatnonzero(valid)
        i = np.minimum(x[p].astype(int), self.nx - 1)
        j = np.minimum(y[p].astype(int), self.ny - 1)
        status[p] = np.where(self.wet[j, i], ParticleStatus.ACTIVE, ParticleStatus.DRY)
        return Seeds(x, y, status)

    def seed_grid(
        self,
        *,
        location: StaggeringString = "cell",
        stride: int = 1,
        bounds: tuple[float, float, float, float] | None = None,
        wet_only: bool = True
    ) -> Seeds:
        """Seed known grid locations without an expensive physical lookup.

        bounds=(xmin, xmax, ymin, ymax) uses physical coordinates; longitude
        bounds may cross the dateline (xmin > xmax). Periodic closing nodes
        are excluded to avoid duplicate particles. Returned order is row-major.

        Args:
            location: Cell staggering of seed locations, one of ("node",
                "cell", "x_face", "y_face").
            stride: Stride of original grid points on which to seed, default
                to every grid location.
            bounds: If present, a tuple of (xmin, xmax, ymin, ymax) in physical
                coordinates of a subregion to which to restrict seeds.
            wet_only: Whether to seed on all points, or restrict to non-masked
                points by the ocean mask.
        """

        if not isinstance(stride, (int, np.integer)) or stride < 1:
            raise ValueError("stride must be a positive integer")
        if location not in {"node", "cell", "x_face", "y_face"}:
            raise ValueError("unknown seed location")

        shift_x = 0.5 if location in {"cell", "y_face"} else 0.0
        shift_y = 0.5 if location in {"cell", "x_face"} else 0.0

        nx = self.nx + int(shift_x == 0 and not self.periodic_x)
        ny = self.ny + int(shift_y == 0 and not self.periodic_y)

        x, y = np.meshgrid(
            np.arange(0, nx, stride) + shift_x, np.arange(0, ny, stride) + shift_y
        )
        seeds = self.logical_seeds(x, y)
        if wet_only:
            keep = seeds.status == ParticleStatus.ACTIVE
        else:
            keep = np.ones(seeds.x.size, dtype=bool)

        px, py = self.coordinates(seeds)
        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            if self.spherical:
                width = xmax - xmin
                if abs(width) < 360:
                    keep &= (px - xmin) % 360 <= width % 360
            else:
                if xmax < xmin:
                    raise ValueError("planar bounds require xmin <= xmax")

                keep &= (px >= xmin) & (px <= xmax)
            if ymax < ymin:
                raise ValueError("bounds require ymin <= ymax")

            keep &= (py >= ymin) & (py <= ymax)

        dims = ("j", "i")
        coords = {"i": x[0, :].copy(), "j": y[:, 0].copy()}

        # coordinate attributes for the output dataset
        if self.spherical:
            coords.update(
                {
                    "lon": (
                        dims,
                        px.reshape(x.shape),
                        {"standard_name": "longitude", "units": "degrees_east"},
                    ),
                    "lat": (
                        dims,
                        py.reshape(y.shape),
                        {"standard_name": "latitude", "units": "degrees_north"},
                    ),
                }
            )
        else:
            coords.update(
                {
                    "x": (dims, px.reshape(x.shape), {"units": "m"}),
                    "y": (dims, py.reshape(y.shape), {"units": "m"}),
                }
            )

        layout = SeedLayout(
            dims=dims,
            shape=x.shape,
            flat_indices=np.flatnonzero(keep),
            coords=coords,
        )

        return Seeds(seeds.x[keep], seeds.y[keep], seeds.status[keep], layout)

    def coordinates(self, seeds: Seeds) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Map logical seeds to physical coordinates (longitude is unwrapped).

        Args:
            seeds: A structure containing the seeds to map.

        Returns:
            Tuple of physical X- and Y-coordinates for the seeds.

        """

        x, y = np.broadcast_arrays(seeds.x, seeds.y)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("logical coordinates must be finite")

        if self.periodic_x:
            x = x % self.nx
        if self.periodic_y:
            y = y % self.ny

        if np.any((x < 0) | (x > self.nx) | (y < 0) | (y > self.ny)):
            raise ValueError("logical coordinates outside grid")

        i = np.minimum(x.astype(int), self.nx - 1)
        j = np.minimum(y.astype(int), self.ny - 1)
        return (
            bilinear_mapping(self.x, i, j, x - i, y - j)[0],
            bilinear_mapping(self.y, i, j, x - i, y - j)[0],
        )

    def _points(self, x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
        if not self.spherical:
            return np.column_stack((np.ravel(x), np.ravel(y)))
        lon, lat = np.deg2rad(x).ravel(), np.deg2rad(y).ravel()
        return np.column_stack(
            (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat))
        )

    def locate(self, x: ArrayLike, y: ArrayLike) -> Seeds:
        """Locate physical seeds, using a spatial index and inverse bilinear map.

        The nearest 16 cells are tried first. Unresolved points use a
        conservative distance bound to include elongated cells as well.
        For large regular seed sets, logical_seeds avoids this search entirely.
        """
        x, y = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        )
        x, y = x.ravel(), y.ravel()
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("seed coordinates must be finite")

        jj, ii = np.indices(self.wet.shape)

        # Construct k-d tree for cells surrounding gridpoints
        if self._tree is None:
            cx = bilinear_mapping(self.x, ii, jj, 0.5, 0.5)[0]
            cy = bilinear_mapping(self.y, ii, jj, 0.5, 0.5)[0]
            self._centres = (cx.ravel(), cy.ravel())
            self._tree = cKDTree(self._points(cx, cy))
            rx = np.maximum.reduce(
                [
                    abs(self.x[:-1, :-1] - cx),
                    abs(self.x[1:, :-1] - cx),
                    abs(self.x[:-1, 1:] - cx),
                    abs(self.x[1:, 1:] - cx),
                ]
            )
            ry = np.maximum.reduce(
                [
                    abs(self.y[:-1, :-1] - cy),
                    abs(self.y[1:, :-1] - cy),
                    abs(self.y[:-1, 1:] - cy),
                    abs(self.y[1:, 1:] - cy),
                ]
            )
            self._search_radius = float(
                np.max(np.deg2rad(rx + ry) if self.spherical else np.hypot(rx, ry))
            ) * (1 + 1.0e-10)

        outx, outy = np.full(x.size, np.nan), np.full(y.size, np.nan)

        # Bound query memory regardless of the total number of particles.
        for first in range(0, x.size, 4096):
            last = min(first + 4096, x.size)
            points = self._points(x[first:last], y[first:last])
            _, candidates = self._tree.query(points, k=min(16, self.nx * self.ny))
            candidates = candidates.reshape(last - first, -1)
            for local, cand in enumerate(candidates):
                # Try directly computing inverse of reference cell
                # before using k-d tree query.
                p = first + local
                found = self._inverse(x[p], y[p], cand)

                if found is None:
                    cand = self._tree.query_ball_point(
                        points[local], self._search_radius
                    )
                    found = self._inverse(x[p], y[p], np.asarray(cand, dtype=int))

                if found is not None:
                    outx[p], outy[p] = found

        return self.logical_seeds(outx, outy)

    def _inverse(
        self,
        x: float,
        y: float,
        candidates: NDArray[np.integer],
    ) -> tuple[float, float] | None:
        if len(candidates) == 0:
            return None

        i, j = candidates % self.nx, candidates // self.nx
        target_x = x
        if self.spherical:
            centres = self._centres[0][candidates]
            target_x = x + 360 * np.round((centres - x) / 360)

        s, r = np.full(len(i), 0.5), np.full(len(i), 0.5)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for _ in range(15):
                xx, xs, xr = bilinear_mapping(self.x, i, j, s, r)
                yy, ys, yr = bilinear_mapping(self.y, i, j, s, r)
                ex, ey = xx - target_x, yy - y
                det = xs * yr - xr * ys
                s -= (yr * ex - xr * ey) / det
                r -= (xs * ey - ys * ex) / det
            xx = bilinear_mapping(self.x, i, j, s, r)[0]
            yy = bilinear_mapping(self.y, i, j, s, r)[0]

        tolerance = 1.0e-8 * max(1.0, abs(y), float(np.max(np.abs(target_x))))
        ok = (
            (s >= -1.0e-9)
            & (s <= 1 + 1.0e-9)
            & (r >= -1.0e-9)
            & (r <= 1 + 1.0e-9)
            & (abs(xx - target_x) <= tolerance)
            & (abs(yy - y) <= tolerance)
        )

        matches = np.flatnonzero(ok)
        if matches.size:
            k = matches[0]
            return i[k] + np.clip(s[k], 0, 1), j[k] + np.clip(r[k], 0, 1)

        return None
