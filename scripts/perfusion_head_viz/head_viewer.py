"""Perfusion Head Visualization — live 3D view of dual-module prefrontal BFI.

See SPEC.md / IMPLEMENTATION.md in this directory for design and rationale.

Usage:
    python scripts/perfusion_head_viz/head_viewer.py
        [--left-mask 0xFF] [--right-mask 0xFF]
        [--cams-per-side N]  (N in 1..8, medial-first preset)
        [--quick]            (shortcut: N=2 for fast FPGA bring-up)
        [--skip-fpga] [--disable-laser]
        [--flip-rows] [--flip-cols]
        [--no-volume] [--no-particles] [--n-particles 2500]
        [--asset path/to/head.obj]

Hotkeys in-window:
    R   reset camera
    P   toggle cardiac-pulse breathing
    V   cycle view: skin ghost + cloud / cloud only / surface only
"""

from __future__ import annotations

# IMPORTANT: set Qt binding BEFORE any pyvistaqt / qtpy import.  Required
# because pyvistaqt goes through qtpy and will otherwise default to the
# first binding it finds (which may not match our PyQt6 QApplication).
import os
os.environ.setdefault("QT_API", "pyqt6")

import argparse
import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

# Local package imports (script is a package; run with -m or directly).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _bfi import BaselineNormalizer, BviEstimator, HeartRateEstimator, compute_bfi
    from _dual_stream import SideStream
    from _head_mesh import (
        compute_sensor_anchors,
        compute_weight_matrix,
        flat_anchor_index,
        load_head_mesh,
    )
    from _head_volume import VolumeField
    from _particle_cloud import ParticleCloud
else:
    from ._bfi import (  # type: ignore[no-redef]
        BaselineNormalizer, BviEstimator, HeartRateEstimator, compute_bfi,
    )
    from ._dual_stream import SideStream  # type: ignore[no-redef]
    from ._head_mesh import (  # type: ignore[no-redef]
        compute_sensor_anchors,
        compute_weight_matrix,
        flat_anchor_index,
        load_head_mesh,
    )
    from ._head_volume import VolumeField  # type: ignore[no-redef]
    from ._particle_cloud import ParticleCloud  # type: ignore[no-redef]

from omotion import MotionInterface

logger = logging.getLogger(__name__)

PERFUSION_CMAP = LinearSegmentedColormap.from_list(
    "perfusion",
    [
        (0.00, "#303540"),  # matches head base tone (cold / no signal)
        (0.35, "#5A3A6A"),  # deep purple
        (0.70, "#E77438"),  # warm orange
        (1.00, "#FFE58A"),  # bright amber
    ],
)

# NOTE: the particle cloud uses matplotlib's built-in "hot" cmap
# (black → red → orange → yellow → white).  We pass the string name rather
# than a LinearSegmentedColormap object because some PyVista / VTK
# versions silently ignore a custom Colormap when combined with
# style="points_gaussian" + emissive=True, falling back to VTK's default
# HSV LUT (which is BLUE at the low end and caused the "dark blue
# particle" bug).  A string cmap name is routed through get_cmap_safe()
# and always applied to the Gaussian mapper's lookup table.


def parse_args() -> argparse.Namespace:
    def parse_mask(x: str) -> int:
        try:
            return int(x, 0) & 0xFF
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid mask: {x}") from exc

    # Masks picking the N innermost cameras per side (medial-first).  Each
    # side's camera grid is 2 rows x 4 cols; cam_id % 4 goes medial (0) to
    # lateral (3), cam_id // 4 goes top (0) to bottom (1).  We add cams in
    # the order [0, 4, 1, 5, 2, 6, 3, 7] so medial columns are filled first
    # and the set is always top/bottom balanced.
    _CAMS_PER_SIDE_MASK = {
        1: 0x01,
        2: 0x11,
        3: 0x13,
        4: 0x33,
        5: 0x37,
        6: 0x77,
        7: 0x7F,
        8: 0xFF,
    }

    p = argparse.ArgumentParser(description="Perfusion Head Visualizer")
    p.add_argument("--left-mask", type=parse_mask, default=None,
                   help="Per-camera bitmask on LEFT sensor (default 0xFF)")
    p.add_argument("--right-mask", type=parse_mask, default=None,
                   help="Per-camera bitmask on RIGHT sensor (default 0xFF)")
    p.add_argument("--cams-per-side", type=int, default=None,
                   choices=sorted(_CAMS_PER_SIDE_MASK.keys()),
                   help="Preset: N cameras per side, medial-first. Overrides "
                        "--left-mask/--right-mask unless those are given explicitly.")
    p.add_argument("--quick", action="store_true",
                   help="Shortcut for --cams-per-side 2 (fast FPGA bring-up)")
    p.add_argument("--skip-fpga", action="store_true",
                   help="Skip FPGA programming + register config")
    p.add_argument("--disable-laser", action="store_true",
                   help="Skip enable_camera_fsin_ext (cameras free-run)")
    p.add_argument("--flip-rows", action="store_true",
                   help="Swap top/bottom row mapping for the sensor grid")
    p.add_argument("--flip-cols", action="store_true",
                   help="Swap medial/lateral column mapping for the sensor grid")
    p.add_argument("--asset", type=Path, default=None,
                   help="Path to an .obj head mesh (optional; falls back to a sphere)")
    p.add_argument("--no-volume", action="store_true",
                   help="Disable the interior 3-D perfusion cloud (surface only)")
    p.add_argument("--no-particles", action="store_true",
                   help="Disable the drifting particle layer")
    p.add_argument("--n-particles", type=int, default=5000,
                   help="Number of advected particles (default 5000)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # Resolve mask presets.  Explicit --left-mask / --right-mask always win;
    # --quick implies --cams-per-side 2 unless that was also given.
    cps = args.cams_per_side
    if args.quick and cps is None:
        cps = 2
    default_mask = _CAMS_PER_SIDE_MASK[cps] if cps is not None else 0xFF
    if args.left_mask is None:
        args.left_mask = default_mask
    if args.right_mask is None:
        args.right_mask = default_mask
    return args


class PerfusionState:
    """Thread-safe latest-value table for 16 cameras.

    The parser threads call ``push(side, cam_id, hist, ts, temp)``; the UI
    thread reads via ``snapshot()``.  Per-channel estimators (BVI, baseline)
    live inside and are advanced on each push.
    """

    N = 16

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bfi = np.zeros(self.N, dtype=np.float64)
        self._bfi_norm = np.zeros(self.N, dtype=np.float64)
        self._bvi = np.full(self.N, np.nan, dtype=np.float64)
        self._temp = np.full(self.N, np.nan, dtype=np.float64)
        self._last_ts = np.zeros(self.N, dtype=np.float64)
        self._frames = np.zeros(self.N, dtype=np.int64)
        self._have_data = np.zeros(self.N, dtype=bool)
        self._baselines = [BaselineNormalizer() for _ in range(self.N)]
        self._bvis = [BviEstimator() for _ in range(self.N)]
        self._hr = HeartRateEstimator()

    def push(self, side: str, cam_id: int, ts_s: float,
             hist: np.ndarray, temp_c: float) -> None:
        try:
            flat = flat_anchor_index(side, cam_id)
        except ValueError:
            return
        bfi, _mean = compute_bfi(hist)
        with self._lock:
            self._bfi[flat] = bfi
            self._baselines[flat].push(bfi)
            self._bfi_norm[flat] = self._baselines[flat].normalize(bfi)
            self._bvis[flat].push(bfi)
            self._bvi[flat] = self._bvis[flat].value
            self._temp[flat] = temp_c
            self._last_ts[flat] = ts_s
            self._frames[flat] += 1
            self._have_data[flat] = True
            # Feed HR estimator with globally averaged normalised BFI.
            mean_norm = float(np.mean(self._bfi_norm[self._have_data])) \
                if self._have_data.any() else 0.0
            self._hr.push(ts_s, mean_norm)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "bfi": self._bfi.copy(),
                "bfi_norm": self._bfi_norm.copy(),
                "bvi": self._bvi.copy(),
                "temp": self._temp.copy(),
                "frames": self._frames.copy(),
                "have_data": self._have_data.copy(),
                "hr": self._hr.estimate(),
            }


class MainWindow(QMainWindow):
    UI_REFRESH_MS = 50  # 20 Hz

    def __init__(self, interface: MotionInterface,
                 args: argparse.Namespace) -> None:
        super().__init__()
        self._interface = interface
        self._args = args
        self.setWindowTitle("MOTION Perfusion Head Viewer")
        self.resize(1280, 820)

        self._state = PerfusionState()
        self._streaming = False
        self._paused = False
        self._trigger_started = False
        self._t_start: Optional[float] = None
        self._hr_pulse_enabled = True

        # ---- Head mesh + anchors + weight matrix ----
        self._mesh = load_head_mesh(args.asset)
        self._anchors, self._anchor_normals = compute_sensor_anchors(
            self._mesh,
            flip_rows=args.flip_rows,
            flip_cols=args.flip_cols,
        )
        self._W = compute_weight_matrix(self._mesh, self._anchors)
        logger.info("Mesh: %d points, %d cells; weight-matrix nnz=%d",
                    self._mesh.n_points, self._mesh.n_cells, self._W.nnz)

        # Prime scalar on mesh so the actor has valid data from frame 1.
        self._mesh["perfusion"] = np.zeros(self._mesh.n_points, dtype=np.float32)

        # ---- Interior volume field (optional) ----
        self._volume: Optional[VolumeField] = None
        self._volume_actor = None
        self._mesh_actor = None  # filled in by _setup_scene
        if not args.no_volume:
            self._volume = VolumeField.build(
                self._mesh, self._anchors, self._anchor_normals,
                spacing_mm=4.0, depth_mm=8.0, sigma_mm=14.0, radius_mm=38.0,
            )

        # ---- Particle cloud (optional) ----
        self._particles: Optional[ParticleCloud] = None
        self._particle_poly: Optional[pv.PolyData] = None
        self._particle_actor = None
        self._last_tick_t: Optional[float] = None  # for dt in particle update
        if not args.no_particles:
            self._particles = ParticleCloud.build(
                self._anchors, self._anchor_normals,
                n_particles=args.n_particles,
                depth_mm=8.0, sigma_mm=12.0,
                lifetime_range=(2.0, 4.0),
                speed_mm_s=3.0, turbulence_mm_s=8.0,
            )
            self._particle_poly = pv.PolyData(self._particles.positions)
            self._particle_poly["bfi"] = self._particles.bfi

        # View mode: 0 = volume + translucent skin + particles,
        # 1 = volume + particles only (no skin — "pure signal"),
        # 2 = opaque BFI-coloured skin (legacy surface look, no cloud).
        # Falls straight to mode 2 when --no-volume AND --no-particles.
        self._view_mode = 2 if (args.no_volume and args.no_particles) else 0

        # One-shot flag: log which volume render mode VTK picked (GPU / CPU)
        # after the first real render.  GetLastUsedRenderMode() returns 0
        # before any render so we defer.
        self._gpu_mode_logged = False

        # ---- Qt layout ----
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Top button bar
        top = QHBoxLayout()
        self._start_btn = QPushButton("Start")
        self._stop_btn = QPushButton("Stop")
        self._pause_btn = QPushButton("Pause")
        self._stop_btn.setEnabled(False)
        self._pause_btn.setEnabled(False)
        self._start_btn.clicked.connect(self.start_capture)
        self._stop_btn.clicked.connect(self.stop_capture)
        self._pause_btn.clicked.connect(self.toggle_pause)
        for btn in (self._start_btn, self._stop_btn, self._pause_btn):
            btn.setMinimumWidth(100)
            top.addWidget(btn)
        top.addStretch(1)
        layout.addLayout(top)

        # PyVista interactor
        self._plotter = QtInteractor(central)
        layout.addWidget(self._plotter.interactor, stretch=1)

        self._setup_scene()

        # Stream sinks
        left_sensor = interface.left
        right_sensor = interface.right
        self._left = SideStream(
            "left",
            left_sensor if (left_sensor and left_sensor.is_connected()) else None,
            args.left_mask,
            self._make_sample_callback("left"),
            disable_laser=args.disable_laser,
        )
        self._right = SideStream(
            "right",
            right_sensor if (right_sensor and right_sensor.is_connected()) else None,
            args.right_mask,
            self._make_sample_callback("right"),
            disable_laser=args.disable_laser,
        )

        # Timer
        self._timer = QTimer(self)
        self._timer.setInterval(self.UI_REFRESH_MS)
        self._timer.timeout.connect(self._refresh)

        # Shortcuts
        reset_act = QAction("Reset View", self)
        reset_act.setShortcut("R")
        reset_act.triggered.connect(lambda: self._plotter.reset_camera())
        self.addAction(reset_act)
        pulse_act = QAction("Toggle Pulse", self)
        pulse_act.setShortcut("P")
        pulse_act.triggered.connect(self._toggle_pulse)
        self.addAction(pulse_act)
        view_act = QAction("Cycle View Mode", self)
        view_act.setShortcut("V")
        view_act.triggered.connect(self._cycle_view_mode)
        self.addAction(view_act)

        if not self._init_sides():
            self._start_btn.setEnabled(False)
            self.statusBar().showMessage("Init failed — see console")
        else:
            active = [s.side.upper() for s in (self._left, self._right) if s.active]
            self.statusBar().showMessage(
                "Ready. Active: " + (", ".join(active) if active else "none")
            )

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        plotter = self._plotter
        plotter.set_background("#07101F")

        # Interior perfusion cloud — added FIRST so the translucent skin
        # renders over it correctly (front-to-back alpha compositing).
        if self._volume is not None:
            self._volume_actor = plotter.add_volume(
                self._volume.grid,
                scalars="bfi",
                # "hot" (black → red → orange → yellow → white) has ZERO
                # blue component — crucial because the volume is dense and
                # dominates the scene.  magma's low end (#3B0F70, dark navy
                # purple) was reading as "dark blue" during BFI warm-up.
                cmap="hot",
                # Shifted lower bound so baseline noise voxels land at
                # ~20% on "hot" = visible dim red rather than pure black.
                clim=[-0.2, 1.3],
                # 7-stop opacity ramp: ramps up aggressively so the cloud is
                # actually visible at mid-range BFI (~0.6-1.0).
                opacity=[0.0, 0.05, 0.2, 0.45, 0.7, 0.9, 0.98],
                # Smaller = denser cloud.  ~14 mm gives visible mass without
                # saturating the whole head at peak BFI.
                opacity_unit_distance=14.0,
                shade=True,
                ambient=0.15,
                diffuse=0.75,
                specular=0.15,
                show_scalar_bar=False,
                name="perfusion_volume",
            )

        # Advected particle layer — plain antialiased point sprites with a
        # guaranteed-supported cmap path.  (We tried style="points_gaussian"
        # + emissive=True, but at least one PyVista/VTK combo silently
        # dropped the custom LUT and defaulted to VTK's HSV table, which is
        # BLUE at the low end — hence the "all dark blue" bug.)  Plain
        # points go through the standard vtkPolyDataMapper → guaranteed to
        # apply the "hot" cmap exactly as specified.
        #
        # clim=[0.0, 0.6]: chosen so the baseline_glow=0.15 lands ~25% up
        # the "hot" ramp (dim red / orange, clearly visible).  Peak BFI
        # (~1.3 + 0.15 = 1.45) clips to 1.0 = yellow-white.
        if self._particle_poly is not None:
            self._particle_actor = plotter.add_mesh(
                self._particle_poly,
                style="points",
                scalars="bfi",
                cmap="hot",
                clim=[0.0, 0.6],
                render_points_as_spheres=True,
                point_size=4,
                opacity=0.7,
                show_scalar_bar=False,
                name="perfusion_particles",
            )
            logger.info("Particles: plain antialiased point sprites, "
                        "cmap=hot, clim=[0, 0.6]")

        # Head skin surface.  In view modes 0/1 it becomes a translucent
        # ghost; mode 2 restores the opaque BFI-coloured original.
        self._mesh_actor = plotter.add_mesh(
            self._mesh,
            scalars="perfusion",
            cmap=PERFUSION_CMAP,
            clim=[0.0, 1.5],
            smooth_shading=True,
            show_scalar_bar=False,
            ambient=0.15,
            diffuse=0.75,
            specular=0.12,
            specular_power=18,
            name="head",
        )

        # Apply the chosen initial view mode BEFORE we start rendering.
        self._apply_view_mode()

        # Build initial glyph actor.
        self._glyph_base = pv.Sphere(radius=1.0, theta_resolution=14,
                                     phi_resolution=14)
        self._glyph_points = pv.PolyData(self._anchors.astype(np.float32))
        self._glyph_points["bfi"] = np.zeros(16, dtype=np.float32)
        self._glyph_points["radius"] = np.full(16, 3.5, dtype=np.float32)
        self._render_glyphs()

        # HUD text actor.  PyVista's `add_text(position=<str>)` creates a
        # CornerAnnotation whose text-setting API varies between versions
        # (and between string vs. xy positions).  We avoid that fragility by
        # simply re-adding the text with the same ``name`` on every refresh;
        # pyvista replaces the actor in place when a name collision occurs,
        # which is cheap for small text.
        self._hud_text = "initialising\u2026"
        plotter.add_text(
            self._hud_text, position="upper_left",
            font_size=10, color="#CCCCCC", shadow=False, name="hud",
        )

        # Camera: 3/4 front-top view.
        plotter.camera_position = [
            (180.0, 320.0, 120.0),   # camera position
            (0.0, 0.0, 40.0),        # focal point (roughly forehead centre)
            (0.0, 0.0, 1.0),         # up vector
        ]
        plotter.camera.zoom(0.95)

        # Lighting: explicit key + rim (warm from upper-left, cool from
        # lower-right rim).  Do NOT call ``enable_lightkit()`` afterwards —
        # it installs its own 5-light rig and would wash out the mood.
        # Scene ambient comes from each mesh's material ``ambient`` factor
        # (set in ``add_mesh`` above).
        plotter.remove_all_lights()
        key = pv.Light(position=(-150.0, 220.0, 220.0),
                       focal_point=(0, 0, 40),
                       color="white", intensity=0.9)
        rim = pv.Light(position=(180.0, -180.0, 120.0),
                       focal_point=(0, 0, 40),
                       color="#99bbff", intensity=0.5)
        plotter.add_light(key)
        plotter.add_light(rim)

    def _render_glyphs(self) -> None:
        sphere_source = self._glyph_base
        glyphs = self._glyph_points.glyph(
            geom=sphere_source, scale="radius", orient=False,
        )
        # Preserve per-source-point "bfi" scalar on the glyph mesh for colour.
        # Glyph filter duplicates point scalars per glyph copy; we map back
        # through the geometry by relying on the fact that `glyph()` assigns
        # the source-point scalar to every vertex of that copy.
        self._plotter.add_mesh(
            glyphs,
            scalars="bfi",
            cmap=PERFUSION_CMAP,
            clim=[0.0, 1.5],
            show_scalar_bar=False,
            smooth_shading=True,
            ambient=0.35,
            diffuse=0.55,
            specular=0.4,
            specular_power=25,
            name="glyphs",  # `name` replaces the prior actor in-place.
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        logger.info(msg)

    def _init_sides(self) -> bool:
        if not self._left.active and not self._right.active:
            self._log("Neither side is active.")
            return False
        return (self._left.configure(self._args.skip_fpga, self._log)
                and self._right.configure(self._args.skip_fpga, self._log))

    def _make_sample_callback(self, side: str):
        state = self._state

        def _on_sample(cam_id: int, _frame_id: int, ts_s: float,
                       hist: np.ndarray, _row_sum: int, temp_c: float) -> None:
            state.push(side, cam_id, ts_s, hist, temp_c)

        return _on_sample

    def start_capture(self) -> None:
        if self._streaming:
            return
        if not self._left.start(self._log):
            return
        if not self._right.start(self._log):
            self._left.stop(self._log)
            return
        try:
            started = self._interface.console.start_trigger()
        except Exception as e:
            self._log(f"start_trigger raised: {e}")
            started = False
        if not started:
            self._log("start_trigger failed; tearing down.")
            self._left.stop(self._log)
            self._right.stop(self._log)
            return
        self._trigger_started = True
        self._streaming = True
        self._t_start = time.monotonic()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._pause_btn.setEnabled(True)
        self._timer.start()
        self.statusBar().showMessage("Streaming...")

    def stop_capture(self) -> None:
        if not self._streaming:
            return
        self._timer.stop()
        if self._trigger_started:
            try:
                self._interface.console.stop_trigger()
            except Exception as e:
                self._log(f"stop_trigger error: {e}")
            self._trigger_started = False
        self._left.stop(self._log)
        self._right.stop(self._log)
        self._streaming = False
        self._paused = False
        self._pause_btn.setText("Pause")
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._pause_btn.setEnabled(False)
        self.statusBar().showMessage("Stopped")

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        self._pause_btn.setText("Resume" if self._paused else "Pause")

    def _toggle_pulse(self) -> None:
        self._hr_pulse_enabled = not self._hr_pulse_enabled
        self.statusBar().showMessage(
            f"Cardiac pulse: {'ON' if self._hr_pulse_enabled else 'OFF'}"
        )

    # ------------------------------------------------------------------
    # View mode (V key)
    # ------------------------------------------------------------------
    def _cycle_view_mode(self) -> None:
        # If there's no cloud at all (both volume and particles disabled),
        # there's nothing meaningful to cycle.
        if self._volume is None and self._particles is None:
            return
        self._view_mode = (self._view_mode + 1) % 3
        self._apply_view_mode()
        names = {0: "skin ghost + cloud", 1: "cloud only", 2: "surface only"}
        self.statusBar().showMessage(f"View: {names[self._view_mode]}")

    def _apply_view_mode(self) -> None:
        mode = self._view_mode
        # --- Volume & particle actor visibility (both off in surface-only mode) ---
        if self._volume_actor is not None:
            self._volume_actor.SetVisibility(mode != 2)
        if self._particle_actor is not None:
            self._particle_actor.SetVisibility(mode != 2)
        # --- Surface mesh properties ---
        if self._mesh_actor is not None:
            prop = self._mesh_actor.GetProperty()
            mapper = self._mesh_actor.GetMapper()
            if mode == 0:      # both: translucent neutral ghost over cloud
                prop.SetOpacity(0.18)
                mapper.SetScalarVisibility(False)
                prop.SetColor(0.80, 0.82, 0.90)
                prop.SetSpecular(0.25)
                prop.SetSpecularPower(35)
            elif mode == 1:    # cloud only: hide the skin completely
                prop.SetOpacity(0.0)
                mapper.SetScalarVisibility(False)
            else:              # mode 2: original opaque BFI-coloured skin
                prop.SetOpacity(1.0)
                mapper.SetScalarVisibility(True)
                prop.SetSpecular(0.12)
                prop.SetSpecularPower(18)
        self._plotter.render()

    # ------------------------------------------------------------------
    # Render tick
    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        if self._paused:
            return
        snap = self._state.snapshot()
        bfi_norm = snap["bfi_norm"]
        bvi = snap["bvi"]
        hr = snap["hr"]

        # Apply cardiac-pulse breathing to the scalar field.  Pure visual;
        # only enabled when the HR estimator is locked (SNR above threshold).
        pulse_factor = 1.0
        if self._hr_pulse_enabled and hr is not None and hr.ok and self._t_start:
            t = time.monotonic() - self._t_start
            pulse_factor = 1.0 + 0.15 * math.sin(2.0 * math.pi * hr.hz * t)

        # --- Time bookkeeping ---
        now = time.monotonic()
        t_abs = (now - self._t_start) if self._t_start is not None else 0.0
        dt = (now - self._last_tick_t) if self._last_tick_t is not None else 0.0
        # Clamp dt so a stalled frame (e.g. FPGA prog) doesn't teleport
        # particles to another county.
        dt = float(np.clip(dt, 0.0, 0.2))
        self._last_tick_t = now

        # --- Mesh scalars (surface) ---
        # Sparse matrix @ dense vector -> 1-D (n_verts,) ndarray.  Older
        # scipy returns a 2-D np.matrix; ``.ravel()`` handles both.
        pulsed = bfi_norm * pulse_factor
        raw = self._W @ pulsed
        field = np.asarray(raw, dtype=np.float32).ravel()
        np.clip(field, 0.0, 1.5, out=field)
        self._mesh["perfusion"] = field

        # --- Volume scalars (interior cloud) ---
        # Only compute if the volume actor is actually visible to avoid the
        # matvec + scatter when the user is in surface-only mode.
        if self._volume is not None and self._view_mode != 2:
            self._volume.update(pulsed, t_abs=t_abs, turbulence=0.35)

        # --- Particle advection ---
        if (self._particles is not None
                and self._particle_poly is not None
                and self._view_mode != 2):
            self._particles.update(dt, pulsed, t_abs)
            # Swap positions + scalars on the PolyData.  Assigning to .points
            # triggers the mapper to refetch geometry; re-assigning "bfi"
            # refreshes the per-point colour scalar.
            self._particle_poly.points = self._particles.positions
            self._particle_poly["bfi"] = self._particles.bfi
            self._particle_poly.Modified()

        # --- Glyphs ---
        # Radius scales with BVI (pulsatility).  Clamp NaN to 0.
        bvi_clean = np.nan_to_num(bvi, nan=0.0, posinf=0.0, neginf=0.0)
        if bvi_clean.max() > 0.0:
            bvi_scaled = bvi_clean / bvi_clean.max()
        else:
            bvi_scaled = np.zeros_like(bvi_clean)
        radii = 3.0 + 4.0 * bvi_scaled
        self._glyph_points["bfi"] = bfi_norm.astype(np.float32)
        self._glyph_points["radius"] = radii.astype(np.float32)
        self._render_glyphs()

        # --- HUD ---
        left_frames = int(snap["frames"][:8].sum())
        right_frames = int(snap["frames"][8:].sum())
        left_have = snap["have_data"][:8]
        right_have = snap["have_data"][8:]
        left_mean = float(bfi_norm[:8][left_have].mean()) if left_have.any() else 0.0
        right_mean = float(bfi_norm[8:][right_have].mean()) if right_have.any() else 0.0
        hr_str = (f"{hr.bpm:5.1f} bpm  (snr {hr.snr:.1f})"
                  if (hr is not None and hr.ok) else "---")
        status = ("streaming" if self._streaming and not self._paused
                  else "paused" if self._paused else "stopped")
        hud = (
            f"HR:        {hr_str}\n"
            f"BFI L:     {left_mean:.2f}    R: {right_mean:.2f}\n"
            f"Frames:    {left_frames} L  /  {right_frames} R\n"
            f"Status:    {status}"
        )
        if hud != self._hud_text:
            self._plotter.add_text(
                hud, position="upper_left", font_size=10,
                color="#CCCCCC", shadow=False, name="hud",
            )
            self._hud_text = hud

        # Explicit render (in-place scalar assign doesn't auto-redraw).
        self._plotter.render()

        # One-shot: after the first real render, report which path VTK's
        # SmartVolumeMapper actually took.  Values (from vtkSmartVolumeMapper
        # enum, 9.x): 0=Default, 1=RayCastAndTexture, 2=RayCast(CPU),
        # 3=Texture(GPU), 4=GPU ray cast, 5=Undefined, 6=OSPRay.
        if not self._gpu_mode_logged:
            if self._volume_actor is not None:
                try:
                    mapper = self._volume_actor.GetMapper()
                    mode = int(mapper.GetLastUsedRenderMode())
                    mode_names = {
                        0: "Default", 1: "RayCastAndTexture",
                        2: "RayCast (CPU software)", 3: "Texture (GPU)",
                        4: "GPU ray cast", 5: "Undefined", 6: "OSPRay",
                    }
                    logger.info("Volume render path: %s (%d)",
                                mode_names.get(mode, "?"), mode)
                except Exception as exc:
                    logger.debug("Could not query volume render mode: %s", exc)
            if self._particles is not None:
                pb = self._particles.bfi
                logger.info(
                    "Particle scalar range (first tick): min=%.3f mean=%.3f "
                    "max=%.3f (clim=[0, 1.45]) — non-zero values here confirm "
                    "BFI is flowing to the GPU sprites.",
                    float(pb.min()), float(pb.mean()), float(pb.max()),
                )
            self._gpu_mode_logged = True

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_capture()
        try:
            self._plotter.close()
        except Exception:
            pass
        event.accept()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    interface = MotionInterface()
    interface.start(wait=True)
    # Devices enumerate asynchronously via the hotplug monitor, so block
    # until the console + at least one sensor reach CONNECTED before reading.
    interface.wait_for_ready(console=True, sensors=1, timeout=10.0)
    console_ok, left_ok, right_ok = interface.is_device_connected()
    if not console_ok:
        print("[head_viewer] Console not connected.")
        return 1
    if not (left_ok or right_ok):
        print("[head_viewer] No sensor connected.")
        return 1

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(interface, args)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
