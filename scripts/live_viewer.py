"""Live histogram viewer for the OpenMotion SDK.

Uses the real streaming pipeline (fsin_ext + trigger + camera stream) on one
sensor module and one or more cameras (selected via --camera-mask). Histogram
packets are pulled from the USB bulk stream, parsed with
``parse_histogram_packet_structured``, and rendered in a PyQt6 + matplotlib
window. One subplot per enabled camera; the latest frame for each camera is
redrawn on a ~20 Hz UI timer.

Run with:
    set PYTHONPATH=%cd%;%PYTHONPATH%
    python scripts\live_viewer.py --side left --camera-mask 0x01
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from typing import Dict, List

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

STREAM_EXPECTED_SIZE = 32833  # per capture_data.py / SDK constant


def parse_args() -> argparse.Namespace:
    def parse_mask(x: str) -> int:
        try:
            return int(x, 0)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid camera mask: {x}") from exc

    p = argparse.ArgumentParser(description="OpenMotion live histogram viewer")
    p.add_argument("--side", choices=["left", "right"], default="left",
                   help="Which sensor module to view (default: left)")
    p.add_argument("--camera-mask", type=parse_mask, default=0x01,
                   help="Bitmask of cameras to enable (default: 0x01)")
    p.add_argument("--disable-laser", action="store_true",
                   help="Skip enable_camera_fsin_ext; use internal sensor sync. "
                        "Trigger is still started so cameras free-run.")
    p.add_argument("--skip-fpga", action="store_true",
                   help="Skip FPGA programming + register config (assume already done)")
    p.add_argument("--log-y", action="store_true",
                   help="Render histogram Y axis in log scale")
    return p.parse_args()


def mask_to_cam_positions(mask: int) -> List[int]:
    """Return the bit positions set in ``mask`` (0..7)."""
    return [i for i in range(8) if mask & (1 << i)]


class LatestSampleBuffer:
    """Thread-safe per-camera 'latest histogram' slot, with a change flag."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Dict[int, np.ndarray] = {}
        self._frame_ids: Dict[int, int] = {}
        self._temp: Dict[int, float] = {}
        self._dirty: set[int] = set()

    def push(self, cam_id: int, frame_id: int, hist: np.ndarray, temp: float) -> None:
        with self._lock:
            self._latest[cam_id] = hist
            self._frame_ids[cam_id] = frame_id
            self._temp[cam_id] = temp
            self._dirty.add(cam_id)

    def drain(self) -> Dict[int, tuple[np.ndarray, int, float]]:
        with self._lock:
            out = {
                cam: (self._latest[cam], self._frame_ids[cam], self._temp[cam])
                for cam in self._dirty
            }
            self._dirty.clear()
            return out


class ParserThread(threading.Thread):
    """Consume the sensor's histogram stream queue and push samples into buffer."""

    def __init__(self, q: queue.Queue, buffer: LatestSampleBuffer) -> None:
        super().__init__(daemon=True, name="LiveViewerParser")
        self._q = q
        self._buffer = buffer
        self._stop_evt = threading.Event()
        self._accum = bytearray()
        self.samples_seen = 0

    def _on_row(self, cam_id: int, frame_id: int, ts_s: float,
                hist: np.ndarray, row_sum: int, temp_c: float) -> None:
        self.samples_seen += 1
        self._buffer.push(cam_id, frame_id, hist, temp_c)

    def run(self) -> None:
        # csv_writer=None → pure callback mode, no CSV written.
        parse_histogram_stream(
            q=self._q,
            stop_evt=self._stop_evt,
            csv_writer=None,
            buffer_accumulator=self._accum,
            on_row_fn=self._on_row,
        )

    def stop(self) -> None:
        self._stop_evt.set()


class HistogramCanvas(FigureCanvasQTAgg):
    """One subplot per camera. ``update_frame`` redraws a single camera's bars."""

    def __init__(self, cam_positions: List[int], log_y: bool, width: int = 10,
                 height: int = 7, dpi: int = 100) -> None:
        fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        super().__init__(fig)
        self.fig = fig
        self._cam_positions = cam_positions
        self._log_y = log_y

        n = len(cam_positions)
        ncols = 1 if n == 1 else 2 if n <= 4 else 4
        nrows = (n + ncols - 1) // ncols

        self._axes: Dict[int, any] = {}
        self._lines: Dict[int, any] = {}
        self._y_ceiling: Dict[int, float] = {}
        self._titles: Dict[int, any] = {}

        x = np.arange(1024)
        for idx, cam_pos in enumerate(cam_positions):
            ax = fig.add_subplot(nrows, ncols, idx + 1)
            ax.set_xlim(0, 1023)
            ax.set_ylim(1 if log_y else 0, 100)
            if log_y:
                ax.set_yscale("log")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_xlabel("Bin")
            ax.set_ylabel("Count")
            title = ax.set_title(f"Cam J{cam_pos + 1}  (waiting…)")
            (line,) = ax.plot(x, np.zeros(1024), linewidth=0.8)
            # cam_id on the wire is 0-based; the mask bit index equals cam_id.
            self._axes[cam_pos] = ax
            self._lines[cam_pos] = line
            self._titles[cam_pos] = title
            self._y_ceiling[cam_pos] = 100.0

    def update_frame(self, cam_id: int, frame_id: int, hist: np.ndarray,
                     temp_c: float) -> None:
        ax = self._axes.get(cam_id)
        if ax is None:
            return
        self._lines[cam_id].set_ydata(hist)

        current_max = float(hist.max()) if hist.size else 1.0
        ceil = self._y_ceiling[cam_id]
        if current_max > ceil * 1.2 or current_max < ceil * 0.4:
            new_ceil = max(10.0, current_max * 1.2)
            ax.set_ylim(1 if self._log_y else 0, new_ceil)
            self._y_ceiling[cam_id] = new_ceil

        self._titles[cam_id].set_text(
            f"Cam J{cam_id + 1}  frame={frame_id}  T={temp_c:.1f}°C  max={int(current_max)}"
        )


class MainWindow(QMainWindow):
    UI_REFRESH_MS = 50  # 20 Hz redraw

    def __init__(self, interface: MotionInterface, args: argparse.Namespace) -> None:
        super().__init__()
        self._interface = interface
        self._args = args
        self._sensor = getattr(interface, args.side)
        self._cam_positions = mask_to_cam_positions(args.camera_mask)
        self._buffer = LatestSampleBuffer()
        self._queue: queue.Queue | None = None
        self._parser: ParserThread | None = None
        self._streaming = False

        self.setWindowTitle(
            f"MOTION Live Viewer — {args.side.upper()} mask=0x{args.camera_mask:02X}"
        )
        self.resize(1100, 750)

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

        self.canvas = HistogramCanvas(self._cam_positions, log_y=args.log_y)
        layout.addWidget(self.canvas, stretch=1)

        self._timer = QTimer(self)
        self._timer.setInterval(self.UI_REFRESH_MS)
        self._timer.timeout.connect(self._refresh_ui)

        if not self._init_sensor():
            self.start_btn.setEnabled(False)
            self.statusBar().showMessage("Sensor init failed — see console")
        else:
            self.statusBar().showMessage(
                f"{args.side.upper()} ready. Cameras: "
                + ", ".join(f"J{p+1}" for p in self._cam_positions)
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init_sensor(self) -> bool:
        if self._sensor is None or not self._sensor.is_connected():
            print(f"[live_viewer] Sensor '{self._args.side}' not connected.")
            return False

        if not self._sensor.ping():
            print(f"[live_viewer] Ping failed on {self._args.side}.")
            return False

        if not self._args.skip_fpga:
            print("[live_viewer] Programming camera FPGA(s)...")
            if not self._sensor.program_fpga(
                camera_position=self._args.camera_mask, manual_process=False
            ):
                print("[live_viewer] FPGA programming failed.")
                return False

            print("[live_viewer] Writing camera sensor registers...")
            if not self._sensor.camera_configure_registers(self._args.camera_mask):
                print("[live_viewer] Camera register config failed.")
                return False
        return True

    def start_capture(self) -> None:
        if self._streaming:
            return

        if not self._args.disable_laser:
            if not self._sensor.enable_camera_fsin_ext():
                print("[live_viewer] enable_camera_fsin_ext failed.")
                return

        if not self._sensor.enable_camera(self._args.camera_mask):
            print("[live_viewer] enable_camera failed.")
            return

        self._queue = queue.Queue()
        self._parser = ParserThread(self._queue, self._buffer)
        self._parser.start()

        self._sensor.uart.histo.start_streaming(
            self._queue, expected_size=STREAM_EXPECTED_SIZE
        )

        if not self._interface.console.start_trigger():
            print("[live_viewer] start_trigger failed.")
            return

        self._streaming = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._timer.start()
        self.statusBar().showMessage("Streaming...")

    def stop_capture(self) -> None:
        if not self._streaming:
            return
        self._timer.stop()

        try:
            self._interface.console.stop_trigger()
        except Exception as e:
            print(f"[live_viewer] stop_trigger error: {e}")

        try:
            self._sensor.disable_camera(self._args.camera_mask)
        except Exception as e:
            print(f"[live_viewer] disable_camera error: {e}")

        try:
            self._sensor.uart.histo.stop_streaming()
        except Exception as e:
            print(f"[live_viewer] stop_streaming error: {e}")

        if self._parser is not None:
            self._parser.stop()
            self._parser.join(timeout=2.0)
            print(f"[live_viewer] Parser saw {self._parser.samples_seen} sample(s).")
            self._parser = None

        self._queue = None
        self._streaming = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("Stopped")

    # ------------------------------------------------------------------
    # UI tick
    # ------------------------------------------------------------------

    def _refresh_ui(self) -> None:
        updates = self._buffer.drain()
        if not updates:
            return
        for cam_id, (hist, frame_id, temp_c) in updates.items():
            self.canvas.update_frame(cam_id, frame_id, hist, temp_c)
        self.canvas.draw_idle()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        self.stop_capture()
        event.accept()


def main() -> int:
    args = parse_args()

    interface = MotionInterface()
    interface.start(wait=True)
    # Devices enumerate asynchronously via the hotplug monitor, so block
    # until the console + at least one sensor reach CONNECTED before reading.
    interface.wait_for_ready(console=True, sensors=1, timeout=10.0)
    console_ok, left_ok, right_ok = interface.is_device_connected()
    side_ok = left_ok if args.side == "left" else right_ok
    if not console_ok:
        print("[live_viewer] Console not connected.")
        return 1
    if not side_ok:
        print(f"[live_viewer] {args.side.upper()} sensor not connected.")
        return 1

    app = QApplication(sys.argv)
    win = MainWindow(interface, args)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
