"""Interior volumetric perfusion field for the head viewer.

Builds a regular 3-D voxel grid bounded by the head mesh, selects voxels
inside the skull via VTK's enclosed-points test, and builds a sparse
(N_interior_voxels, 16) weight matrix so that ``W @ bfi_vec`` produces a
smooth interior cloud driven by the 16 sensor anchors.

Each anchor's kernel is a 3-D Gaussian whose centre is pushed INWARDS
along the surface normal by ``depth_mm`` (default 8 mm, matching the DCS
photon-banana centroid for an ~850 nm source/detector on scalp).  That
way the volume "lights up" a few mm below the skin directly under each
sensor, like real cerebral perfusion would.

Public API:
    build_voxel_grid(mesh, spacing_mm=4.0, margin_mm=6.0) -> pv.ImageData
    select_interior(grid, mesh) -> np.ndarray[bool, n_points]
    compute_volume_weight_matrix(voxel_points, anchors, anchor_normals,
        depth_mm=8.0, sigma_mm=14.0, radius_mm=38.0) -> csr_matrix
    VolumeField.build(mesh, anchors, anchor_normals, ...)
    VolumeField.update(bfi_vec, clip=(0.0, 1.5))
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pyvista as pv
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def _make_image_grid(dims: tuple[int, int, int],
                     spacing: tuple[float, float, float],
                     origin: tuple[float, float, float]) -> pv.ImageData:
    """Construct a ``pv.ImageData`` robustly across PyVista versions.

    Newer (>= 0.41) supports keyword ``dimensions``/``spacing``/``origin``.
    Older versions require attribute assignment.  Fall all the way back to
    the deprecated ``UniformGrid`` if ``ImageData`` isn't present.
    """
    try:
        return pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)
    except TypeError:
        g = pv.ImageData()
        g.dimensions = dims
        g.spacing = spacing
        g.origin = origin
        return g
    except AttributeError:
        g = pv.UniformGrid()
        g.dimensions = dims
        g.spacing = spacing
        g.origin = origin
        return g


def build_voxel_grid(mesh: pv.PolyData,
                     spacing_mm: float = 4.0,
                     margin_mm: float = 6.0) -> pv.ImageData:
    """Axis-aligned voxel grid that tightly bounds the head plus a margin."""
    b = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    origin = (b[0] - margin_mm, b[2] - margin_mm, b[4] - margin_mm)
    size = (b[1] - b[0] + 2.0 * margin_mm,
            b[3] - b[2] + 2.0 * margin_mm,
            b[5] - b[4] + 2.0 * margin_mm)
    dims = (
        int(math.ceil(size[0] / spacing_mm)) + 1,
        int(math.ceil(size[1] / spacing_mm)) + 1,
        int(math.ceil(size[2] / spacing_mm)) + 1,
    )
    return _make_image_grid(dims, (spacing_mm,) * 3, origin)


def select_interior(grid: pv.ImageData, mesh: pv.PolyData) -> np.ndarray:
    """Boolean mask of voxels inside the mesh (length == grid.n_points)."""
    enclosed = grid.select_enclosed_points(mesh, tolerance=1e-4,
                                           check_surface=False)
    # VTK attaches the mask as "SelectedPoints" (uint8); normalise to bool.
    sel = np.asarray(enclosed["SelectedPoints"])
    return sel.astype(bool)


def compute_volume_weight_matrix(voxel_points: np.ndarray,
                                 anchors: np.ndarray,
                                 anchor_normals: np.ndarray,
                                 depth_mm: float = 8.0,
                                 sigma_mm: float = 14.0,
                                 radius_mm: float = 38.0) -> csr_matrix:
    """Sparse (n_voxels, 16) Gaussian weights, L1-normalised per row.

    Each column j is a 3-D Gaussian with centre at
    ``anchors[j] - depth_mm * anchor_normals[j]`` (i.e. pushed a few mm
    INSIDE the skull) and isotropic std ``sigma_mm``.  Only voxels within
    ``radius_mm`` (~3 sigma) of a centre get a non-zero weight.
    """
    if anchors.shape != (16, 3) or anchor_normals.shape != (16, 3):
        raise ValueError("anchors and anchor_normals must be (16, 3)")
    n_vox = voxel_points.shape[0]
    inset = anchors - depth_mm * anchor_normals

    tree = cKDTree(voxel_points)
    inv_two_sigma_sq = 1.0 / (2.0 * sigma_mm * sigma_mm)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for j, c in enumerate(inset):
        nearby = tree.query_ball_point(c, r=radius_mm)
        if not nearby:
            continue
        d2 = np.sum((voxel_points[nearby] - c) ** 2, axis=1)
        w = np.exp(-d2 * inv_two_sigma_sq)
        rows.extend(nearby)
        cols.extend([j] * len(nearby))
        data.extend(w.tolist())

    W = csr_matrix((data, (rows, cols)), shape=(n_vox, 16))

    # L1-normalise rows so each voxel is a convex combination of sensors.
    row_sums = np.asarray(W.sum(axis=1)).flatten()
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(row_sums > 0, 1.0 / row_sums, 0.0)
    for row in range(n_vox):
        if inv[row] == 0.0:
            continue
        W.data[W.indptr[row]:W.indptr[row + 1]] *= inv[row]

    return W


class VolumeField:
    """Wraps the voxel grid + interior mask + sparse weight matrix.

    One-time cost (build): voxel construction + interior enclose-test +
    sparse matrix build.  For a 50^3 grid on the sphere fallback, this is
    around 1-2 s end-to-end on a modern CPU.

    Per-tick cost (update): 16-vector @ sparse(N_interior, 16) matvec plus
    a scatter into the full grid scalar array — sub-millisecond.
    """

    def __init__(self, grid: pv.ImageData, interior_mask: np.ndarray,
                 W: csr_matrix) -> None:
        self.grid = grid
        self.interior_mask = interior_mask
        self.W = W
        self._interior_indices = np.flatnonzero(interior_mask)
        self._full = np.zeros(grid.n_points, dtype=np.float32)
        # Register the scalar array so the volume actor can find it from
        # frame 1 (an unregistered name would silently break rendering).
        self.grid["bfi"] = self._full.copy()
        # Pre-compute interior voxel positions + per-voxel spatial phases
        # for the cheap 3-sine turbulence modulation.  Wavelengths (mm):
        # 40 / 35 / 32 — deliberately incommensurate so the beat pattern
        # doesn't look periodic.
        pts = np.asarray(grid.points)[interior_mask]
        self._phase_x = (pts[:, 0] / 40.0).astype(np.float32)
        self._phase_y = (pts[:, 1] / 35.0).astype(np.float32)
        self._phase_z = (pts[:, 2] / 32.0).astype(np.float32)

    @classmethod
    def build(cls, mesh: pv.PolyData,
              anchors: np.ndarray,
              anchor_normals: np.ndarray,
              spacing_mm: float = 4.0,
              depth_mm: float = 8.0,
              sigma_mm: float = 14.0,
              radius_mm: float = 38.0) -> "VolumeField":
        grid = build_voxel_grid(mesh, spacing_mm=spacing_mm)
        mask = select_interior(grid, mesh)
        interior_pts = np.asarray(grid.points)[mask]
        W = compute_volume_weight_matrix(
            interior_pts, anchors, anchor_normals,
            depth_mm=depth_mm, sigma_mm=sigma_mm, radius_mm=radius_mm,
        )
        obj = cls(grid=grid, interior_mask=mask, W=W)
        logger.info(
            "Volume: %d voxels total, %d inside (%.1f%%), Wv.nnz=%d",
            grid.n_points, int(mask.sum()),
            100.0 * mask.sum() / max(1, grid.n_points), W.nnz,
        )
        return obj

    def update(self, bfi_vec: np.ndarray,
               t_abs: float = 0.0,
               turbulence: float = 0.35,
               clip: tuple[float, float] = (0.0, 1.3)) -> None:
        """Recompute interior voxel scalars from the 16 BFI values.

        ``t_abs`` is absolute time in seconds and drives a 3-axis drifting
        sine modulation that makes the cloud visibly flow.  ``turbulence``
        controls the modulation amplitude (0 = static, 0.5 = dramatic).
        """
        raw = self.W @ bfi_vec
        vals = np.asarray(raw, dtype=np.float32).ravel()

        if turbulence > 0.0:
            # Three drifting sines along different spatial axes.  Temporal
            # frequencies 1.1 / 0.8 / 1.4 rad/s (roughly 0.13–0.22 Hz beat).
            mod = (1.0
                   + turbulence * 0.40 * np.sin(self._phase_x + 1.1 * t_abs)
                   + turbulence * 0.35 * np.sin(self._phase_y + 0.8 * t_abs + 1.0)
                   + turbulence * 0.30 * np.sin(self._phase_z + 1.4 * t_abs + 2.3))
            # Keep modulation strictly positive so it never inverts signal.
            np.clip(mod, 0.25, 1.75, out=mod)
            vals *= mod.astype(np.float32)

        np.clip(vals, clip[0], clip[1], out=vals)
        self._full[:] = 0.0
        self._full[self._interior_indices] = vals
        # Re-assign (rather than modify in place) so PyVista / VTK see the
        # data change and drop any cached render state.
        self.grid["bfi"] = self._full
        # Defensive: some volume mappers cache aggressively even after scalar
        # replacement — calling Modified() on the dataset is cheap and
        # guarantees the next render pulls fresh data.
        self.grid.Modified()
