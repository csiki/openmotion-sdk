# Perfusion Head Visualization — Implementation Plan

**Status:** draft v2
**Pairs with:** `SPEC.md`
**Target file:** `scripts/perfusion_head_viz/head_viewer.py` (~ 350 lines)
**Supporting file:** `scripts/perfusion_head_viz/_dual_stream.py` (~ 180 lines, extracted from `dual_live_viewer.py`)
**Asset path:** `scripts/perfusion_head_viz/assets/head.obj` (optional; parametric ellipsoid fallback)

---

## 1. Dependencies

Add to `requirements.txt` or install ad-hoc:

```
pyvista>=0.43
pyvistaqt>=0.11
scipy>=1.10
```

`PyQt6`, `numpy`, `matplotlib` already present. VTK is pulled in as a pyvista dependency.

**Verification note:** PyVista bundles VTK wheels for Windows; no separate VTK install required.

**IMPORTANT: Qt binding selection.** `pyvistaqt` uses `qtpy` to abstract the Qt binding. We must set the binding *before importing pyvistaqt or qtpy*:

```python
import os
os.environ["QT_API"] = "pyqt6"   # must come BEFORE any pyvistaqt / qtpy import

from pyvistaqt import QtInteractor   # now safe
```

Without this, `pyvistaqt` may default to PyQt5 and either fail to import or run in a parallel Qt event loop separate from our `QApplication`.

## 2. Module Layout

```
scripts/perfusion_head_viz/
├── SPEC.md
├── IMPLEMENTATION.md
├── _dual_stream.py        # extracted SideStream / RollingBuffer / ParserThread
├── head_viewer.py         # main entry point, PyVista scene, UI
├── _bfi.py                # BFI / BVI math, baseline normaliser
├── _head_mesh.py          # load head.obj OR generate ellipsoid, define sensor anchors
└── assets/
    └── head.obj           # optional; script falls back to ellipsoid if missing
```

### 2.1 Why extract `_dual_stream.py`?

`dual_live_viewer.py` and `head_viewer.py` share 90% of their plumbing. Extracting once into a shared helper avoids duplication and lets both scripts evolve in lockstep. The extraction is mechanical (cut-and-paste + imports); diff risk low.

## 3. Step-by-Step Build Order

Each step is independently runnable / testable. Stop after any step if the result is good enough.

### Step 1 — Extract shared streaming helper (30 min)

- [ ] Create `_dual_stream.py` with `SideStream`, `RollingBuffer`, `ParserThread`, `STREAM_EXPECTED_SIZE` moved out of `dual_live_viewer.py`.
- [ ] Update `dual_live_viewer.py` to `from _dual_stream import ...`.
- [ ] Verify `python scripts/dual_live_viewer.py --left-mask 0x01 --right-mask 0x00` still works identically.

### Step 2 — BFI/BVI math module (30 min)

- [ ] `_bfi.py` with:
  - `compute_bfi(hist: np.ndarray) -> tuple[float, float]` returning `(bfi, mean)`.
  - `BviEstimator(window_sec=1.0, rate_hz=40)` with `push(ts, bfi)` and `.value` property (rolling std-dev).
  - `BaselineNormalizer(window_sec=30.0)` with `push(bfi)` → `normalized_bfi` in the rough range `[0, 2]`.
- [ ] Unit test: feed a synthetic histogram with known contrast `K=0.3`, assert `BFI ≈ 11.1` within 1%.

### Step 3 — Head mesh + sensor anchors (1 h)

- [ ] `_head_mesh.py`:
  - `load_head_mesh() -> pyvista.PolyData`:
    - Try `assets/head.obj` via `pv.read(...)`.
    - On failure: generate a scaled sphere as fallback.
      ```python
      mesh = pv.Sphere(radius=1.0, theta_resolution=48, phi_resolution=48)
      mesh.points *= np.array([85.0, 105.0, 115.0])  # x, y, z half-axes in mm
      ```
      **Do not use `pv.ParametricEllipsoid`** — it has a known axis-swap bug (pyvista/pyvista#800) and constructor-arg compatibility varies across versions.
    - Ensure triangulated: `mesh = mesh.triangulate()` before decimation (required; `decimate` silently no-ops on non-triangulated polydata).
    - Decimate: `mesh = mesh.decimate(0.5)` to target ≤ 10k vertices.
  - `compute_sensor_anchors(mesh) -> np.ndarray[16, 3]`:
    - Define sensor grid in *forehead-local* coordinates: two 2×4 rectangular patches, 40 mm wide × 20 mm tall, centred ±25 mm from midline, vertical centre at a parametric "forehead height" constant (≈ +60 mm above the origin for our scaled-sphere fallback; real anatomical mesh would use the known Fp1/Fp2 landmarks).
    - For *each* idealised sensor point `p`, shoot a ray from `p + [0, 0, 300]` (well in front) toward `p - [0, 0, 300]` (through the head). Take the first intersection with `mesh.ray_trace(origin, end, first_point=True)`.
    - `first_point=True` returns `(points, ind)` where `points` has shape `(3,)` on hit or `(0, 3)` on miss. Fallback on miss: project `p` radially outward to nearest vertex.
  - **Ellipsoid fallback has no anatomical forehead** — "forehead" is defined purely geometrically as the set of vertices satisfying `y > 30 mm AND z > 40 mm` (upper-front octant of the ellipsoid). The weight-matrix mask should use this condition rather than a real forehead segmentation.
- [ ] `compute_weight_matrix(mesh, anchors, sigma_mm=12.0, radius_mm=30.0) -> scipy.sparse.csr_matrix`:
  - For each vertex within `radius_mm` of any anchor, Gaussian weights (normalised so rows sum to 1).
  - Sparse matrix of shape `(n_vertices, 16)`.
- [ ] **Test harness**: render mesh with fake `BFI = [1,2,…,16]` to visually confirm the 16 anchors land on the forehead with the expected left/right split.

### Step 4 — PyVista scene skeleton (1 h)

- [ ] `head_viewer.py` builds a `QMainWindow` with `pyvistaqt.QtInteractor` as central widget.
- [ ] Add head mesh with base colour `#303540`, smooth shading.
- [ ] Add 16 sphere glyphs at anchors (all same size/colour for now).
- [ ] Lighting: `plotter.remove_all_lights()`, add two `pv.Light` (key + rim) plus ambient.
- [ ] Background: `plotter.set_background("#07101F")`.
- [ ] Run: manual smoke test — head appears, is rotatable.

### Step 5 — Live-data wiring (45 min)

- [ ] Reuse `SideStream`s from `_dual_stream.py` but swap the parser hook: compute `BFI`, push into `RollingBuffer` where values are now `BFI` (not `mean_bin`).
- [ ] Add `BviEstimator` and `BaselineNormalizer` per camera.
- [ ] Start/Stop buttons call into `SideStream.start(...)` / `.stop(...)`.
- [ ] `QTimer` at 50 ms ticks a `refresh()` method.

### Step 6 — Colour the mesh (1 h)

- [ ] In `refresh()`:
  - Pull latest `(bfi, bvi)` per camera via `RollingBuffer.snapshot()`.
  - Normalise `bfi` via per-camera baseline (clip to `[0, 2]`).
  - `field = W @ bfi_vec` → dense length-n_vertices vector.
  - `plotter.mesh.cell_data["perfusion"] = field` or point_data as appropriate.
  - Colour map: build once using `matplotlib.colors.LinearSegmentedColormap.from_list("perfusion", [(0.0, "#07101F"), (0.4, "#3A2E5A"), (0.7, "#E77438"), (1.0, "#FFE58A")])`. Apply with `plotter.update_scalars(field)`.
  - Alpha mask: vertices outside `radius_mm` of any sensor get a custom LUT alpha of 0.
- [ ] Update glyph radius and colour from the same per-camera `bfi, bvi`.

### Step 7 — Cardiac pulse (30 min)

- [ ] `_bfi.py`: add `HeartRateEstimator` — `scipy.signal.welch` on last 8 s of mean-BFI, find peak in 0.8–2.5 Hz.
- [ ] In `refresh()`: if HR SNR > threshold, set `plotter.camera.exposure` (or a global scalar multiplier) to `1 + 0.15 * sin(2π·hr·t)`.
- [ ] Toggleable with `P` key.

### Step 8 — HUD overlay (45 min)

- [ ] `plotter.add_text(..., position="upper_left")` for numeric HUD. Updated on every refresh (cheap).
- [ ] Small 2×8 colour legend in upper-right built as a second small PyVista actor or as 16 `add_mesh(small_plane)` actors — simpler: 16 text cells with background colour, positioned via `plotter.add_text` absolute coords. If too finicky, fall back to a side-panel `QWidget` with 16 coloured labels.

### Step 9 — Keybindings + UX polish (30 min)

- [ ] `plotter.add_key_event("r", lambda: plotter.reset_camera())`.
- [ ] Same for H/G/P toggles (HUD, glyphs, pulse).
- [ ] Lock zoom / pan bounds so trackball can't lose the head.
- [ ] Window title shows session id + current HR.

### Step 10 — Final smoke + tuning (30 min)

- [ ] End-to-end run with bench console (safety interlock tripped is fine — trigger still fires).
- [ ] Tune σ, colour stops, glyph sizes by eye.
- [ ] Record a 30 s screen capture for the user.

**Total: ~6 hours (one working day).** Steps 1–5 give you a correct but boring "16 coloured balls on a grey head." Steps 6–8 are where it becomes cool.

## 4. Key Code Sketches

### 4.1 BFI from histogram

```python
def compute_bfi(hist: np.ndarray) -> tuple[float, float]:
    total = hist.sum()
    if total <= 0:
        return 0.0, 0.0
    probs = hist.astype(np.float64) / total
    mean = float(np.dot(BIN_INDEX, probs))
    var = float(np.dot((BIN_INDEX - mean) ** 2, probs))
    if mean <= 1e-6 or var <= 0:
        return 0.0, mean
    k = np.sqrt(var) / mean
    return 1.0 / (k * k), mean
```

### 4.2 PyVista live-update pattern

```python
# once (during init)
head_mesh["perfusion"] = np.zeros(head_mesh.n_points, dtype=np.float32)
self.plotter.add_mesh(head_mesh, scalars="perfusion", cmap=perfusion_cmap,
                      clim=[0, 2], name="head", smooth_shading=True)

# every tick (50 ms)
field = W @ bfi_vec                    # (n_verts,) dense result
head_mesh["perfusion"] = field          # in-place scalar swap
self.plotter.render()                   # explicit redraw
```

**Note on `update_scalars`**: the `Plotter.update_scalars(...)` method is deprecated
since PyVista 0.43 (see https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.update_scalars).
The new idiom is: *assign to the mesh's scalar array and call `render()`*. The assignment
is a shallow swap into `mesh.point_data["perfusion"]`, so it stays O(n_points) and keeps
the same VTK actor — no re-creation cost.

Do not call `add_mesh` on every tick (it rebuilds the actor and thrashes).

### 4.3 Anchor computation (simplified)

```python
def compute_sensor_anchors(mesh):
    # Forehead-local coordinates (mm): y = up, x = left(-)/right(+), z = front(+)
    grid_x = np.array([-45, -30, -15, -5]) # left module columns, from lateral to medial
    grid_y = np.array([70, 55])            # top row, bottom row
    left_pts  = np.array([[-x, y, 80] for y in grid_y for x in grid_x[::-1]])
    right_pts = np.array([[ x, y, 80] for y in grid_y for x in grid_x[::-1]])
    ideal = np.vstack([left_pts, right_pts])        # 16 × 3

    # Ray-cast each ideal point onto the mesh surface from directly in front
    anchors = []
    for p in ideal:
        origin = p + np.array([0, 0, 200])    # far in front
        end = p - np.array([0, 0, 200])       # through the head
        hit, _ = mesh.ray_trace(origin, end, first_point=True)
        anchors.append(hit if hit.size else p)
    return np.array(anchors)
```

## 5. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `pyvistaqt` picks PyQt5 instead of PyQt6 | High if not handled | Set `os.environ["QT_API"] = "pyqt6"` *before* any `pyvistaqt` / `qtpy` import. Covered in §1. |
| `pyvistaqt` event-loop conflicts with `PyQt6` asyncio | Low | Keep everything synchronous; no `qasync`. Reuse same `QApplication` throughout. |
| Mesh `.obj` hard to source CC0 | Medium | Ship with scaled-sphere fallback (works, just less pretty). Not `ParametricEllipsoid` (axis-swap bug pyvista#800). |
| `decimate` silently no-ops on non-triangulated mesh | Medium | Always call `mesh.triangulate()` before `decimate(...)`. |
| Ray-casting misses (concave head or miss altogether) | Low | Fall back to radial projection onto nearest vertex. |
| Sensor grid orientation wrong | Medium | Expose `--flip-rows` / `--flip-cols` CLI flags to rotate the mapping without code change. |
| PyVista scalar-update slow on large mesh | Low | Decimated mesh ≤ 10k verts; in-place assign + `plotter.render()` benchmarks at 200+ Hz. |
| BFI uncalibrated → boring colour range | High | `BaselineNormalizer` auto-rescales to session's own range (30 s rolling). |
| Safety interlock trips trigger | N/A | Visualisation doesn't gate on safety; data will flow regardless. Script works on bench. |
| Ellipsoid fallback has no forehead geometry | Medium | Forehead region defined purely geometrically (`y > 30 mm AND z > 40 mm`) with sensor anchors placed by ray-cast from in front. Documented as a limitation. |

## 6. Deferred / Future Work (Not v1)

- Hemisphere-sync bar (phase coherence L vs R) as a bottom panel.
- Cardiac-locked averaged pulsation overlay.
- Historical scrubber for recorded sessions.
- Integration with the bloodflow app (separate scope; that app has its own QML plot stack).
- Real cortical projection (requires co-registered anatomical model — major effort).

## 7. Changelog

- **v1** — initial draft.
- **v2** — after cross-check with SPEC: clarified that `_dual_stream.py` is shared with `dual_live_viewer.py`, added `--flip-rows/--flip-cols` risk mitigation to match SPEC Open Question 1, flagged that `BaselineNormalizer` is the answer to SPEC Open Question 3.
- **v3** — feasibility review against actual PyVista / `omotion` APIs: replaced deprecated `update_scalars` with in-place scalar assignment + `render()`; flagged `QT_API=pyqt6` requirement; swapped `ParametricEllipsoid` fallback for scaled `pv.Sphere` (avoids pyvista#800 axis-swap bug); added `triangulate()` step before `decimate()`; spelled out ellipsoid-fallback forehead mask rule. All `omotion` API references (`MOTIONInterface.acquire_motion_interface`, `sensor.uart.histo.start_streaming`, `parse_histogram_stream` callback signature, etc.) verified against source and unchanged.
