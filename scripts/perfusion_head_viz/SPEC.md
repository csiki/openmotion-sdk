# Perfusion Head Visualization — Specification

**Status:** draft v2
**Owner:** Vik
**Script name:** `scripts/perfusion_head_viz/head_viewer.py`

---

## 1. Goal

Live 3D visualization of dual-module prefrontal speckle measurements. A rotatable head is rendered in a Qt window; the forehead surface glows red/orange where blood flow is high and cool/dim where it is low. Each heartbeat softly pulses the whole visualization. It must be:

- **Airtight**: no hand-waving. Every visual element corresponds to a known data field.
- **Simple**: ≤ 400 lines of Python.
- **Cool**: non-technical viewer says "oh, that's alive."
- **Honest**: nothing implies data we do not have (flow direction, depth, vector fields).

## 2. Hardware & Data Assumptions

- Two sensor modules, one over **left prefrontal**, one over **right prefrontal** (standard headset placement, roughly Fp1/Fp2 area of 10-20 system).
- Each module has 8 cameras in a physical 2×4 grid. `cam_id // 4` = row (0 top, 1 bottom), `cam_id % 4` = column (0 medial, 3 lateral). *(Assumption to confirm with user; flippable with one constant.)*
- Per-frame data available from `parse_histogram_stream`: 1024-bin photon-count histogram, `row_sum`, `temperature_c`, `frame_id`, `timestamp_s`. 40 Hz per camera.
- **16 scalar time series** total (left × 8 + right × 8). **No directional information**, no depth.

## 3. Derived Quantities (Per Camera, Per Frame)

| Symbol | Definition | Meaning |
|---|---|---|
| `mean` | `Σ i·h[i] / Σ h[i]` | Average pixel intensity (illumination) |
| `var`  | `Σ (i−mean)²·h[i] / Σ h[i]` | Pixel intensity variance |
| `K`    | `√var / mean` | **Speckle contrast** |
| `BFI`  | `1 / K²` | Canonical **blood-flow index** (higher = more flow) |
| `BVI`  | rolling std-dev of `BFI` over last 1 s | **Blood volume (pulsatility) index** |

These are simplified (no dark-frame correction, no per-camera calibration) but give a correct relative perfusion map at the granularity this visualization needs. Dark-interval frames (every 600th per parser metadata) are excluded.

## 4. Visual Design

### 4.1 Scene

- **Background**: dark navy (`#07101F`) for emissive glow to pop.
- **Head mesh**: generic adult head `.obj` if available, decimated to ~8k vertices. Otherwise a scaled sphere fallback (x=85, y=105, z=115 mm half-axes) — *not* `pv.ParametricEllipsoid` (known axis-swap bug pyvista#800). Matte dark-grey base material so illuminated forehead reads cleanly against it.
- **Camera**: starts at three-quarter front-top view (head tilted down ~10°, yaw 30°). Trackball interactor. Locked zoom limits so user can't accidentally lose the head.
- **Lighting**: one soft key light from front-upper-left, one rim light from back-right, plus ambient at 10%.

### 4.2 Forehead Perfusion Field (primary element)

- Precomputed mapping: for each mesh vertex `v` on the forehead region, a 16-element weight vector `W[v]` such that `W[v,j] = exp(-d(v, sensor_j)² / (2σ²))`, renormalised so `Σ_j W[v,j] = 1`. σ ≈ 12 mm (bigger than a single-camera footprint; gives smooth blending between neighbours without blurring left/right hemispheres together).
- Each UI tick: `field[v] = Σ_j W[v,j] · BFI_j`. Pushed to mesh as active scalar.
- **Colormap**: custom 3-stop — transparent-cool-blue → orange → bright yellow. Low perfusion stays dark (blends into skull), high perfusion glows warm. Transparency in the cold range prevents the forehead from looking permanently painted.
- Forehead mask: vertices further than 30 mm from any sensor get `alpha = 0` (no painting on the cheeks / scalp).

### 4.3 Sensor Glyphs (secondary element, always visible)

- 16 small spheres rendered at the anchor vertices.
- **Radius** = `2 mm + 4 mm · normalize(BVI_j)`  → pulsatility is readable.
- **Color** = same colormap as the field, evaluated at `BFI_j`.
- **Outline**: thin white ring so glyphs remain visible even when field is dim.

### 4.4 Cardiac Pulse (tertiary, optional)

- Detect heart rate from dominant 0.8–2.5 Hz FFT peak of mean-BFI across all cameras, over last 8 s rolling.
- Modulate global scene emissive multiplier: `1.0 + 0.15 · sin(2π·HR·t)` — a subtle ~±15% breathing at the detected heart rate.
- Disabled if detected peak SNR too low (e.g., at boot, or with dark cameras).

### 4.5 HUD Overlay

Top-left, monospaced 12pt:
- `HR: 72 bpm` (or `---` if not locked)
- `BFI L: 4.2  R: 4.4` (mean across each module's active cameras)
- `Frames: 1234 L / 1235 R` (live frame counters)
- `Status: streaming | stopped`

Top-right: a small 2×8 legend panel showing all 16 cameras' current values as colored cells — a quick "which camera is misbehaving" debug aid.

Bottom-left: **Start / Stop / Pause** buttons.

### 4.6 Explicitly Rejected

- **Streamlines / arrows / "flow particles"**: we have no velocity direction. Omitted entirely.
- **Depth / volumetric rendering**: we have no depth. Omitted.
- **Brain cortex rendering**: we're measuring scalp perfusion. A brain model would falsely imply cortical specificity. Omitted.

## 5. Data Pipeline (Reused From `dual_live_viewer.py`)

`head_viewer.py` inherits the same per-side plumbing:

1. `MOTIONInterface.acquire_motion_interface()` → console + both sensors.
2. For each side: `program_fpga(mask) → configure_registers(mask) → enable_camera_fsin_ext() → enable_camera(mask)`.
3. Per side: `queue.Queue`, `ParserThread` wrapping `parse_histogram_stream`, `sensor.uart.histo.start_streaming`.
4. Console `start_trigger()` once.
5. Per-camera rolling buffer of `(timestamp_s, BFI, BVI)` with `maxlen ≈ 10 s * 50 Hz = 500`.

The `SideStream` / `RollingBuffer` / `ParserThread` classes from `dual_live_viewer.py` are shared — extract them to a small `_dual_stream.py` helper module imported by both scripts. Parser hook replaced to compute `K` and `BFI` inline.

## 6. Controls

- **Start** — boot pipeline, spin parsers, start trigger. Button toggles to Stop.
- **Stop** — reverse.
- **Pause** — freeze the visualization but keep streaming in the background (useful for pointing at something).
- **R** key — reset camera view.
- **H** key — toggle HUD.
- **G** key — toggle glyphs.
- **P** key — toggle cardiac pulse.
- **Mouse** — trackball rotate, scroll zoom.

## 7. Performance Targets

- ≥ 20 FPS visualisation update on Intel integrated graphics at 1920×1080.
- UI updates decoupled from 40 Hz data rate: UI polls the rolling buffer's latest values at 20 Hz.
- Memory: < 150 MB RSS including mesh.
- Startup (from click-Run to rendered head): < 4 s excluding FPGA programming.

## 8. Out-of-Scope (v1)

- Data recording / playback (use existing capture tools).
- Multi-session history.
- Cortical projection.
- Anything involving networked access.
- Custom head meshes per user.

## 9. Open Questions

1. **Camera grid orientation on the physical module.** Is `cam_id=0` top-medial, or bottom-lateral, or some other mapping? Affects where sensor anchors sit on the mesh. *Default assumption: `cam_id=0` = top-medial (closest to midline, closest to hairline). Implementation exposes `--flip-rows` and `--flip-cols` CLI flags so the mapping can be rotated without code changes.*
2. **Head mesh license.** Need a CC0 / permissive head `.obj`. Candidates: OSF brain-template repos, Sketchfab CC0, or a scaled-sphere fallback as a dead-simple default. *Default: scaled-sphere fallback baked into the script (see §4.1); user can drop in a real `.obj` at `assets/head.obj` to upgrade.*
3. **BFI normalization.** Absolute values depend on exposure / calibration. *Implementation uses a per-camera `BaselineNormalizer` with a 30 s rolling window, mapping into a relative `[0, 2]` range — colour scale stays meaningful without hardware calibration.*

## 10. Changelog

- **v1** — initial draft.
- **v2** — cross-checked with implementation plan: reflected CLI flip flags, baseline normaliser, and explicit rejection of flow-arrow / cortex / depth elements (§4.6). Confirmed shared streaming helper (`_dual_stream.py`) with `dual_live_viewer.py`.
- **v3** — feasibility review: swapped `ParametricEllipsoid` fallback for scaled `pv.Sphere` (§4.1) after confirming the `ParametricEllipsoid` axis-swap bug. No behaviour-visible changes for the user; implementation plan holds all other mitigations (QT_API, triangulate-before-decimate, deprecated update_scalars replacement).
