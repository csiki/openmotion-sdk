"""Dual-sensor streaming plumbing shared by live-view scripts.

Provides the boilerplate that drives two sensors' histogram streams in
parallel from a single console trigger. Consumers supply a per-sample
callback via ``RollingBuffer``-style sinks or by subclassing ``SideStream``
to override ``_on_sample``.

The public API:

    SideStream(side, sensor, mask, on_sample, disable_laser)
        .configure(skip_fpga, log) -> bool
        .start(log) -> bool
        .stop(log) -> None
        .active -> bool

    ParserThread(side, q, on_row)    # exposed for direct use if needed

Nothing in this module knows about plotting.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

from omotion.MotionProcessing import parse_histogram_stream

STREAM_EXPECTED_SIZE = 32833

# Callback signature: (cam_id, frame_id, ts_s, histogram_np, row_sum, temp_c)
SampleCallback = Callable[[int, int, float, np.ndarray, int, float], None]


def mask_to_cam_positions(mask: int) -> list[int]:
    return [i for i in range(8) if mask & (1 << i)]


class ParserThread(threading.Thread):
    """Drain a histogram queue and invoke a sample callback per frame."""

    def __init__(self, side: str, q: queue.Queue,
                 on_row: SampleCallback) -> None:
        super().__init__(daemon=True, name=f"DualStreamParser-{side}")
        self._q = q
        self._on_row = on_row
        self._stop_evt = threading.Event()
        self._accum = bytearray()
        self.samples_seen = 0

    def _wrapped(self, cam_id: int, frame_id: int, ts_s: float,
                 hist: np.ndarray, row_sum: int, temp_c: float) -> None:
        self.samples_seen += 1
        try:
            self._on_row(cam_id, frame_id, ts_s, hist, row_sum, temp_c)
        except Exception:
            # Do not let a single bad sample bring the parser thread down.
            pass

    def run(self) -> None:
        parse_histogram_stream(
            q=self._q,
            stop_evt=self._stop_evt,
            csv_writer=None,
            buffer_accumulator=self._accum,
            on_row_fn=self._wrapped,
        )

    def stop(self) -> None:
        self._stop_evt.set()


class SideStream:
    """Owns one sensor's configure / start / stop lifecycle and parser."""

    def __init__(self, side: str, sensor, mask: int,
                 on_sample: SampleCallback,
                 disable_laser: bool = False) -> None:
        self.side = side
        self.sensor = sensor
        self.mask = int(mask) & 0xFF
        self.cam_ids = mask_to_cam_positions(self.mask)
        self.disable_laser = disable_laser
        self._on_sample = on_sample
        self._queue: Optional[queue.Queue] = None
        self._parser: Optional[ParserThread] = None
        self._running = False

    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.sensor is not None and self.mask != 0

    # ------------------------------------------------------------------
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
        log(f"[{self.side}] enabling camera power for mask=0x{self.mask:02X}...")
        if not self.sensor.enable_camera_power(self.mask):
            log(f"[{self.side}] enable_camera_power failed")
            return False
        time.sleep(0.5)  # rails + FPGA supply settle (matches ScanWorkflow)
        # Program FPGAs one camera at a time — this is the only tested firmware
        # path (ScanWorkflow.py:1062-1089). mask=0xFF in one shot hangs the
        # firmware and drops the USB endpoint.
        for cam_id in self.cam_ids:
            bit = 1 << cam_id
            log(f"[{self.side}] programming FPGA cam{cam_id} (0x{bit:02X})... up to 60 s")
            if not self.sensor.program_fpga(camera_position=bit,
                                            manual_process=False):
                log(f"[{self.side}] program_fpga(cam{cam_id}) failed")
                return False
            time.sleep(0.1)  # settle after bitstream load
            if not self.sensor.camera_configure_registers(bit):
                log(f"[{self.side}] camera_configure_registers(cam{cam_id}) failed")
                return False
        return True

    # ------------------------------------------------------------------
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
        self._parser = ParserThread(self.side, self._queue, self._on_sample)
        self._parser.start()
        self.sensor.uart.histo.start_streaming(
            self._queue, expected_size=STREAM_EXPECTED_SIZE
        )
        self._running = True
        log(f"[{self.side}] streaming started ({len(self.cam_ids)} cam(s))")
        return True

    # ------------------------------------------------------------------
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
        try:
            self.sensor.disable_camera_power(self.mask)
        except Exception as e:
            log(f"[{self.side}] disable_camera_power error: {e}")
        if self._parser is not None:
            self._parser.stop()
            self._parser.join(timeout=2.0)
            log(f"[{self.side}] parser saw {self._parser.samples_seen} sample(s)")
            self._parser = None
        self._queue = None
        self._running = False
