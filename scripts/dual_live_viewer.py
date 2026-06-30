"""Dual-sensor live time-series viewer.

Streams histograms from LEFT and RIGHT sensors simultaneously and plots one
scalar-per-frame time series per camera. Two subplots side by side (LEFT,
RIGHT), each with up to 8 colored lines (J1..J8) encoding a 2x4 physical grid
(cam_id//4 = row, cam_id%4 = column).

CLI:
    python scripts/dual_live_viewer.py --left-mask 0xFF --right-mask 0xFF
        [--metric mean_bin|row_sum|max_bin] [--window-sec 30]
        [--skip-fpga] [--disable-laser]
"""

from __future__ import annotations

import argparse
import colorsys
import queue
import sys
import threading
import time
from collections import deque
from typing import Callable, Dict, List, Optional

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from omotion import MotionInterface
from omotion.MotionProcessing import parse_histogram_stream

STREAM_EXPECTED_SIZE = 32833
BIN_INDEX = np.arange(1024, dtype=np.float64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    def parse_mask(x: str) -> int:
        try:
            return int(x, 0) & 0xFF
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid camera mask: {x}") from exc

    p = argparse.ArgumentParser(description="Dual-sensor live time-series viewer")
    p.add_argument("--left-mask", type=parse_mask, default=0xFF,
                   help="Camera bitmask on LEFT sensor (default 0xFF = all 8). Set 0 to disable side.")
    p.add_argument("--right-mask", type=parse_mask, default=0xFF,
                   help="Camera bitmask on RIGHT sensor (default 0xFF). Set 0 to disable side.")
    p.add_argument("--metric", choices=["mean_bin", "row_sum", "max_bin"],
                   default="mean_bin",
                   help="Scalar to plot per frame (default: mean_bin)")
    p.add_argument("--window-sec", type=float, default=30.0,
                   help="Rolling time window on X axis (default: 30 s)")
    p.add_argument("--skip-fpga", action="store_true",
                   help="Skip FPGA programming + register config (assume already done)")
    p.add_argument("--disable-laser", action="store_true",
                   help="Skip enable_camera_fsin_ext (cameras free-run; trigger still fires)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mask_to_cam_positions(mask: int) -> List[int]:
    return [i for i in range(8) if mask & (1 << i)]


def make_palette(side: str) -> Dict[int, tuple]:
    """Return {cam_id: (r,g,b)} for cam_id in 0..7, encoding a 2x4 grid.

    row = cam_id // 4  (0 = J1-J4 top, 1 = J5-J8 bottom)
    col = cam_id %  4  (0..3)
    """
    # LEFT:  hue 0.45 -> 0.69  (green -> teal -> blue -> indigo)
    # RIGHT: hue 0.98 -> 1.22 mod 1  (pink -> red -> orange -> amber)
    base_hue = 0.45 if side == "left" else 0.98
    palette: Dict[int, tuple] = {}
    for cam_id in range(8):
        row = cam_id // 4
        col = cam_id % 4
        hue = (base_hue + col * 0.08) % 1.0
        # Top row: brighter + more saturated.  Bottom row: darker + desaturated.
        if row == 0:
            lightness, saturation = 0.62, 1.00
        else:
            lightness, saturation = 0.30, 0.70
        palette[cam_id] = colorsys.hls_to_rgb(hue, lightness, saturation)
    return palette


def compute_metric(metric: str, hist: np.ndarray, row_sum: int) -> float:
    if metric == "row_sum":
        return float(row_sum)
    if metric == "max_bin":
        return float(int(np.argmax(hist)))
    # mean_bin (default)
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    return float(np.dot(BIN_INDEX, hist.astype(np.float64)) / total)


# ---------------------------------------------------------------------------
# Per-side rolling buffer + parser
# ---------------------------------------------------------------------------

class RollingBuffer:
    """Per-camera rolling (timestamp, value) deques with a thread-safe push."""

    def __init__(self, cam_ids: List[int], maxlen: int) -> None:
        self._lock = threading.Lock()
        self._ts: Dict[int, deque] = {c: deque(maxlen=maxlen) for c in cam_ids}
        self._vals: Dict[int, deque] = {c: deque(maxlen=maxlen) for c in cam_ids}
        self._temp: Dict[int, float] = {c: float("nan") for c in cam_ids}
        self._last_frame: Dict[int, int] = {c: -1 for c in cam_ids}
        self._cam_ids = set(cam_ids)

    def push(self, cam_id: int, ts: float, val: float, temp_c: float,
             frame_id: int) -> None:
        if cam_id not in self._cam_ids:
            return
        with self._lock:
            self._ts[cam_id].append(ts)
            self._vals[cam_id].append(val)
            self._temp[cam_id] = temp_c
            self._last_frame[cam_id] = frame_id

    def snapshot(self) -> Dict[int, tuple]:
        """Return {cam_id: (ts_array, val_array, temp_c, last_frame_id)}."""
        with self._lock:
            out: Dict[int, tuple] = {}
            for cam in self._cam_ids:
                out[cam] = (
                    np.fromiter(self._ts[cam], dtype=np.float64,
                                count=len(self._ts[cam])),
                    np.fromiter(self._vals[cam], dtype=np.float64,
                                count=len(self._vals[cam])),
                    self._temp[cam],
                    self._last_frame[cam],
                )
        return out


class ParserThread(threading.Thread):
    def __init__(self, side: str, q: queue.Queue, buffer: RollingBuffer,
                 metric: str) -> None:
        super().__init__(daemon=True, name=f"DualViewerParser-{side}")
        self._q = q
        self._buffer = buffer
        self._stop_evt = threading.Event()
        self._accum = bytearray()
        self._metric = metric
        self.samples_seen = 0

    def _on_row(self, cam_id: int, frame_id: int, ts_s: float,
                hist: np.ndarray, row_sum: int, temp_c: float) -> None:
        self.samples_seen += 1
        value = compute_metric(self._metric, hist, row_sum)
        self._buffer.push(cam_id, ts_s, value, temp_c, frame_id)

    def run(self) -> None:
        parse_histogram_stream(
            q=self._q,
            stop_evt=self._stop_evt,
            csv_writer=None,
            buffer_accumulator=self._accum,
            on_row_fn=self._on_row,
        )

    def stop(self) -> None:
        self._stop_evt.set()


# ---------------------------------------------------------------------------
# Per-side stream owner
# ---------------------------------------------------------------------------

class SideStream:
    """Owns one sensor's init, start/stop, queue, parser, and rolling buffer."""

    def __init__(self, side: str, sensor, mask: int, metric: str,
                 buffer_maxlen: int, disable_laser: bool) -> None:
        self.side = side
        self.sensor = sensor
        self.mask = mask
        self.metric = metric
        self.disable_laser = disable_laser
        self.cam_ids = mask_to_cam_positions(mask)
        self.buffer = RollingBuffer(self.cam_ids, maxlen=buffer_maxlen)
        self._queue: Optional[queue.Queue] = None
        self._parser: Optional[ParserThread] = None
        self._running = False

    @property
    def active(self) -> bool:
        return self.sensor is not None and self.mask != 0

    def configure(self, skip_fpga: bool, log: Callable[[str], None]) -> bool:
        if not self.active:
            log(f"[{self.side}] skipped (no sensor / mask=0)")
            return True
        if not self.sensor.is_connected():
            log(f"[{self.side}] sensor not connected")
            return False
        if not self.sensor.ping():
            log(f"[{self.side}] ping failed")
            return False
        if skip_fpga:
            return True
        log(f"[{self.side}] programming camera FPGA(s) for mask=0x{self.mask:02X}...")
        if not self.sensor.program_fpga(camera_position=self.mask,
                                        manual_process=False):
            log(f"[{self.side}] FPGA programming failed")
            return False
        log(f"[{self.side}] writing camera sensor registers...")
        if not self.sensor.camera_configure_registers(self.mask):
            log(f"[{self.side}] register configuration failed")
            return False
        return True

    def start(self, log: Callable[[str], None]) -> bool:
        if not self.active or self._running:
            return True
        if not self.disable_laser:
            if not self.sensor.enable_camera_fsin_ext():
                log(f"[{self.side}] enable_camera_fsin_ext failed")
                return False
        if not self.sensor.enable_camera(self.mask):
            log(f"[{self.side}] enable_camera failed")
            return False
        self._queue = queue.Queue()
        self._parser = ParserThread(self.side, self._queue, self.buffer,
                                    self.metric)
        self._parser.start()
        self.sensor.uart.histo.start_streaming(
            self._queue, expected_size=STREAM_EXPECTED_SIZE
        )
        self._running = True
        return True

    def stop(self, log: Callable[[str], None]) -> None:
        if not self._running:
            return
        try:
            self.sensor.disable_camera(self.mask)
        except Exception as e:
            log(f"[{self.side}] disable_camera error: {e}")
        try:
            self.sensor.uart.histo.stop_streaming()
        except Exception as e:
            log(f"[{self.side}] stop_streaming error: {e}")
        if self._parser is not None:
            self._parser.stop()
            self._parser.join(timeout=2.0)
            log(f"[{self.side}] parser saw {self._parser.samples_seen} sample(s)")
            self._parser = None
        self._queue = None
        self._running = False


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------

class DualCanvas(FigureCanvasQTAgg):
    def __init__(self, left: SideStream, right: SideStream, metric: str,
                 window_sec: float) -> None:
        fig = Figure(figsize=(14, 6), dpi=100, constrained_layout=True)
        super().__init__(fig)
        self.fig = fig
        self._left = left
        self._right = right
        self._metric = metric
        self._window_sec = window_sec
        self._t0: Optional[float] = None

        self._ax_left = fig.add_subplot(1, 2, 1)
        self._ax_right = fig.add_subplot(1, 2, 2)

        self._lines: Dict[str, Dict[int, any]] = {"left": {}, "right": {}}
        for side_name, stream, ax in (
            ("left", left, self._ax_left),
            ("right", right, self._ax_right),
        ):
            self._configure_axes(ax, side_name, stream)
            palette = make_palette(side_name)
            for cam_id in stream.cam_ids:
                color = palette[cam_id]
                (line,) = ax.plot([], [], linewidth=1.3, color=color,
                                  label=f"J{cam_id + 1}")
                self._lines[side_name][cam_id] = line
            if stream.cam_ids:
                # Ordered legend so top row appears above bottom row visually.
                handles = [self._lines[side_name][c] for c in
                           sorted(stream.cam_ids, key=lambda c: (c // 4, c % 4))]
                ax.legend(handles=handles, loc="upper right", ncol=2,
                          fontsize=8, framealpha=0.85)

    def _configure_axes(self, ax, side_name: str, stream: SideStream) -> None:
        title = f"{side_name.upper()} — mask=0x{stream.mask:02X}"
        if not stream.active:
            title += "  (not connected)"
        ax.set_title(title)
        ax.set_xlabel("Time (s since start)")
        ax.set_ylabel(self._y_label())
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(0, self._window_sec)

    def _y_label(self) -> str:
        return {
            "mean_bin": "Mean bin",
            "row_sum": "Row sum (pixels)",
            "max_bin": "Peak bin",
        }[self._metric]

    def refresh(self) -> None:
        now = time.monotonic()
        if self._t0 is None:
            # Start wallclock only once a sample has actually arrived so the
            # X axis begins at the first frame, not at Start-click.
            any_data = any(
                len(snap[0]) > 0
                for stream in (self._left, self._right)
                if stream.active
                for snap in stream.buffer.snapshot().values()
            )
            if any_data:
                self._t0 = now
            else:
                return

        elapsed = now - self._t0
        x_lo = max(0.0, elapsed - self._window_sec)
        x_hi = max(self._window_sec, elapsed)

        for side_name, stream, ax in (
            ("left", self._left, self._ax_left),
            ("right", self._right, self._ax_right),
        ):
            if not stream.active:
                continue
            snap = stream.buffer.snapshot()
            y_vals_for_scaling: List[float] = []
            for cam_id in stream.cam_ids:
                ts_arr, val_arr, _temp, _fid = snap[cam_id]
                if ts_arr.size == 0:
                    continue
                # Convert sample timestamps to seconds-since-viewer-start using
                # the first observed timestamp as t0 per camera-stream. We use
                # monotonic wallclock approach: x = elapsed - (latest_ts -
                # latest_sample_time). Simpler: just plot vs "wall seconds since
                # first frame across both sides", using sample ts minus a
                # per-camera bias. Here we treat ts_arr as a monotonically
                # growing "sensor seconds" and anchor it to arrival time of the
                # last sample = now.
                offset = now - self._t0 - (ts_arr[-1] - ts_arr[0])
                x_arr = (ts_arr - ts_arr[0]) + offset
                self._lines[side_name][cam_id].set_data(x_arr, val_arr)
                mask = x_arr >= x_lo
                if mask.any():
                    y_vals_for_scaling.extend(val_arr[mask].tolist())
            ax.set_xlim(x_lo, x_hi)
            if y_vals_for_scaling:
                y_min = min(y_vals_for_scaling)
                y_max = max(y_vals_for_scaling)
                span = max(1e-3, y_max - y_min)
                ax.set_ylim(y_min - 0.05 * span, y_max + 0.05 * span)
        self.draw_idle()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    UI_REFRESH_MS = 50  # 20 Hz

    def __init__(self, interface: MotionInterface,
                 args: argparse.Namespace) -> None:
        super().__init__()
        self._interface = interface
        self._args = args

        # Decide which sides are active based on connectivity + mask.
        left_sensor = interface.left
        right_sensor = interface.right

        buffer_maxlen = int(max(10.0, args.window_sec) * 50)  # 50 sps headroom

        self._left = SideStream(
            "left",
            left_sensor if (left_sensor and left_sensor.is_connected()) else None,
            args.left_mask, args.metric, buffer_maxlen, args.disable_laser,
        )
        self._right = SideStream(
            "right",
            right_sensor if (right_sensor and right_sensor.is_connected()) else None,
            args.right_mask, args.metric, buffer_maxlen, args.disable_laser,
        )

        self._streaming = False
        self._trigger_started = False

        self.setWindowTitle(
            f"MOTION Dual Live Viewer — L=0x{args.left_mask:02X} R=0x{args.right_mask:02X}"
            f"  metric={args.metric}"
        )
        self.resize(1400, 720)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        controls = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_capture)
        self.stop_btn.clicked.connect(self.stop_capture)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.canvas = DualCanvas(self._left, self._right, args.metric,
                                 args.window_sec)
        layout.addWidget(self.canvas, stretch=1)

        self._timer = QTimer(self)
        self._timer.setInterval(self.UI_REFRESH_MS)
        self._timer.timeout.connect(self.canvas.refresh)

        if not self._init_sides():
            self.start_btn.setEnabled(False)
            self.statusBar().showMessage("Init failed — see console")
        else:
            active = [s.side.upper() for s in (self._left, self._right) if s.active]
            self.statusBar().showMessage(
                "Ready. Active: " + (", ".join(active) if active else "none")
            )

    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        print(f"[dual_live_viewer] {msg}")

    def _init_sides(self) -> bool:
        if not self._left.active and not self._right.active:
            self._log("Neither side is active (no sensor or mask=0 on both).")
            return False
        ok_left = self._left.configure(self._args.skip_fpga, self._log)
        ok_right = self._right.configure(self._args.skip_fpga, self._log)
        return ok_left and ok_right

    def start_capture(self) -> None:
        if self._streaming:
            return
        if not self._left.start(self._log):
            return
        if not self._right.start(self._log):
            self._left.stop(self._log)
            return
        if not self._interface.console.start_trigger():
            self._log("start_trigger failed; tearing down")
            self._left.stop(self._log)
            self._right.stop(self._log)
            return
        self._trigger_started = True
        self._streaming = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
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
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("Stopped")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_capture()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    interface = MotionInterface()
    interface.start(wait=True)
    # Devices enumerate asynchronously via the hotplug monitor, so block
    # until the console + at least one sensor reach CONNECTED before reading.
    interface.wait_for_ready(console=True, sensors=1, timeout=10.0)
    console_ok, left_ok, right_ok = interface.is_device_connected()
    if not console_ok:
        print("[dual_live_viewer] Console not connected.")
        return 1
    if not (left_ok or right_ok):
        print("[dual_live_viewer] No sensor connected.")
        return 1
    app = QApplication(sys.argv)
    win = MainWindow(interface, args)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
