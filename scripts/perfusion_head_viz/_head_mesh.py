"""Head mesh, sensor anchors, and vertex-weight matrix.

Coordinate convention (right-handed, head-centred, units = mm):
    x : left (-) / right (+)
    y : back (-) / front (+)
    z : down (-) / up (+)

Forehead is therefore at large +y and large +z.

Public API:
    load_head_mesh(asset_path: Path | None) -> pv.PolyData
    compute_sensor_anchors(mesh, flip_rows=False, flip_cols=False)
        -> (np.ndarray[16, 3], np.ndarray[16, 3])   # (anchors, outward_normals)
    compute_weight_matrix(mesh, anchors, sigma_mm=12.0, radius_mm=30.0)
        -> scipy.sparse.csr_matrix (n_vertices, 16)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pyvista as pv
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# Ellipsoid fallback half-axes (mm), roughly an adult head.
FALLBACK_X = 85.0
FALLBACK_Y = 105.0
FALLBACK_Z = 115.0

# Forehead region for the ellipsoid fallback: upper-front octant thresholds.
FOREHEAD_MIN_Y = 30.0
FOREHEAD_MIN_Z = 30.0


def load_head_mesh(asset_path: Optional[Path] = None,
                   target_vertices: int = 10000) -> pv.PolyData:
    """Load a head mesh, falling back to a scaled sphere if the asset is missing.

    The returned mesh is guaranteed to be triangulated and decimated to
    roughly ``target_vertices`` points.
    """
    mesh: Optional[pv.PolyData] = None
    if asset_path is not None and asset_path.is_file():
        try:
            mesh = pv.read(str(asset_path))
            logger.info("Loaded head mesh from %s (%d points)", asset_path,
                        mesh.n_points)
        except Exception as exc:
            logger.warning("Failed to read %s: %s — using fallback", asset_path, exc)
            mesh = None

    if mesh is None:
        # Scaled sphere fallback — avoid pv.ParametricEllipsoid (pyvista#800
        # has a long-standing axis-swap bug).
        mesh = pv.Sphere(radius=1.0, theta_resolution=48, phi_resolution=48)
        mesh.points = mesh.points * np.array(
            [FALLBACK_X, FALLBACK_Y, FALLBACK_Z], dtype=np.float32
        )
        logger.info("Using scaled-sphere head fallback (%.0f × %.0f × %.0f mm)",
                    FALLBACK_X, FALLBACK_Y, FALLBACK_Z)

    # Triangulate first — decimate silently no-ops on non-triangulated input.
    mesh = mesh.triangulate()

    if mesh.n_points > target_vertices and mesh.n_cells > 0:
        # Decimation factor: fraction to REMOVE (not keep).
        reduction = max(0.0, min(0.95,
                                 1.0 - (target_vertices / mesh.n_points)))
        if reduction > 0.01:
            try:
                mesh = mesh.decimate(reduction)
                logger.info("Decimated mesh to %d points (reduction=%.2f)",
                            mesh.n_points, reduction)
            except Exception as exc:
                logger.warning("decimate(%.2f) failed: %s — keeping original",
                               reduction, exc)

    return mesh


def _ideal_sensor_grid(flip_rows: bool = False,
                       flip_cols: bool = False) -> np.ndarray:
    """Return 16 idealised (x, y, z) sensor positions in head-local mm.

    Column index (cam_id % 4) runs 0..3 medial (near midline) → lateral
    (near temple). Row index (cam_id // 4) runs 0 (top / near hairline)
    → 1 (bottom / near brow). The ``flip_rows`` / ``flip_cols`` flags
    reverse each mapping without requiring code changes elsewhere.
    """
    # Distances from the sagittal midline (|x| in mm).  4 columns spanning
    # roughly 25–55 mm off-midline cover the prefrontal region per side.
    col_dists = np.array([25.0, 35.0, 45.0, 55.0])  # medial → lateral
    if flip_cols:
        col_dists = col_dists[::-1]

    # Vertical positions for top / bottom rows (mm above bregma-like centre).
    row_z = np.array([62.0, 48.0])  # row 0 top, row 1 bottom
    if flip_rows:
        row_z = row_z[::-1]

    module_y = 75.0  # front of head (mm forward of origin)

    positions = np.zeros((16, 3), dtype=np.float64)
    for side_idx, x_sign in enumerate((-1.0, +1.0)):  # 0 = left, 1 = right
        for row in range(2):
            for col in range(4):
                flat = side_idx * 8 + row * 4 + col
                positions[flat] = (x_sign * col_dists[col],
                                   module_y,
                                   row_z[row])
    return positions


def compute_sensor_anchors(mesh: pv.PolyData,
                           flip_rows: bool = False,
                           flip_cols: bool = False
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Ray-cast the idealised sensor grid onto the mesh front surface.

    Returns ``(anchors, normals)`` — both (16, 3).  ``anchors`` are vertex
    positions on the forehead; ``normals`` are unit outward-pointing surface
    normals at those anchors (useful for pushing a 3-D kernel INWARDS into
    the head).  Falls back to the idealised position, radially projected to
    the nearest mesh vertex, when the ray misses the mesh.
    """
    ideal = _ideal_sensor_grid(flip_rows=flip_rows, flip_cols=flip_cols)
    anchors = np.zeros_like(ideal)
    tree = cKDTree(mesh.points)

    # Pre-compute per-vertex normals on the mesh (idempotent: compute_normals
    # stores them under the "Normals" point-data array).  Default orientation
    # is outward for closed meshes produced by pv.Sphere / most OBJ exports.
    if "Normals" not in mesh.point_data:
        mesh.compute_normals(point_normals=True, cell_normals=False,
                             inplace=True)
    vert_normals = np.asarray(mesh.point_normals, dtype=np.float64)

    normals = np.zeros_like(ideal)

    for i, p in enumerate(ideal):
        # Ray shoots from far in front (+y) straight through to far behind (-y).
        origin = np.array([p[0], p[1] + 400.0, p[2]], dtype=np.float64)
        end = np.array([p[0], p[1] - 400.0, p[2]], dtype=np.float64)
        try:
            hit_points, _ = mesh.ray_trace(origin, end, first_point=True)
            hit_points = np.asarray(hit_points)
        except Exception:
            hit_points = np.empty((0,), dtype=np.float64)
        if hit_points.size == 3:
            anchors[i] = hit_points.reshape(3)
        else:
            # Fallback: nearest mesh vertex to the idealised point.
            _, idx = tree.query(p)
            anchors[i] = mesh.points[idx]
        # Normal at the anchor: nearest vertex normal, renormalised.
        _, idx_n = tree.query(anchors[i])
        n = vert_normals[idx_n]
        nn = float(np.linalg.norm(n))
        if nn > 1e-9:
            n = n / nn
        else:
            # Degenerate fallback — radial direction from head centre.
            c = np.asarray(mesh.center, dtype=np.float64)
            r = anchors[i] - c
            rn = float(np.linalg.norm(r))
            n = r / rn if rn > 1e-9 else np.array([0.0, 1.0, 0.0])
        normals[i] = n
    return anchors, normals


def compute_weight_matrix(mesh: pv.PolyData,
                          anchors: np.ndarray,
                          sigma_mm: float = 12.0,
                          radius_mm: float = 30.0) -> csr_matrix:
    """Build a sparse (n_vertices, 16) Gaussian weight matrix.

    Rows are L1-normalised so ``W @ bfi_vec`` gives a convex combination at
    every in-range vertex. Vertices further than ``radius_mm`` from every
    anchor have all-zero rows (they're outside the forehead region).
    """
    n_verts = mesh.n_points
    if anchors.shape != (16, 3):
        raise ValueError(f"anchors shape {anchors.shape} must be (16, 3)")

    tree = cKDTree(mesh.points)
    # For each anchor, find all vertices within radius.
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    inv_two_sigma_sq = 1.0 / (2.0 * sigma_mm * sigma_mm)

    for j, anchor in enumerate(anchors):
        nearby = tree.query_ball_point(anchor, r=radius_mm)
        if not nearby:
            continue
        verts = mesh.points[nearby]
        d2 = np.sum((verts - anchor) ** 2, axis=1)
        w = np.exp(-d2 * inv_two_sigma_sq)
        rows.extend(nearby)
        cols.extend([j] * len(nearby))
        data.extend(w.tolist())

    W = csr_matrix((data, (rows, cols)), shape=(n_verts, 16))
    # L1-normalise rows (skip zero rows).
    row_sums = np.asarray(W.sum(axis=1)).flatten()
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(row_sums > 0, 1.0 / row_sums, 0.0)
    # Multiply each row by its inverse sum.  Use the fact that csr allows
    # in-place scaling per row via the indptr structure.
    for row in range(n_verts):
        if inv[row] == 0.0:
            continue
        W.data[W.indptr[row]:W.indptr[row + 1]] *= inv[row]

    return W


def flat_anchor_index(side: str, cam_id: int) -> int:
    """Map (side, cam_id_in_side) -> 0..15 flat index used by the weight matrix."""
    if side == "left":
        return 0 + cam_id
    if side == "right":
        return 8 + cam_id
    raise ValueError(f"Unknown side: {side!r}")
