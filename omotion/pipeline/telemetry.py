"""TelemetryAggregator + TelemetryIngestStage.

Bridges the per-snapshot telemetry channel (events arriving at ~10 Hz on
the runner's telemetry loop) into the per-frame world (FrameBatch.pdc/
tcm/tcl fields the CsvSink raw writer expects). Also exposes a query
API (snapshot_at) for future correction stages.

See spec §3.6.5.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Optional, Deque

import numpy as np

from .batch import FrameBatch, TelemetryEvent


class TelemetryAggregator:
    """Thread-safe ring buffer of recent TelemetryEvents."""

    def __init__(self, max_history: int = 100):
        if max_history < 1:
            raise ValueError(f"max_history must be >= 1, got {max_history}")
        self._max = int(max_history)
        self._history: Deque[TelemetryEvent] = deque(maxlen=self._max)
        self._lock = threading.Lock()

    def update(self, event: TelemetryEvent) -> None:
        with self._lock:
            self._history.append(event)

    def snapshot_at(self, t: float) -> Optional[TelemetryEvent]:
        """Return the most recent event with timestamp_s <= t, or None."""
        with self._lock:
            for ev in reversed(self._history):
                if ev.timestamp_s <= t:
                    return ev
            return None

    def size(self) -> int:
        with self._lock:
            return len(self._history)

    def clear(self) -> None:
        with self._lock:
            self._history.clear()


class TelemetryIngestStage:
    """Per-frame telemetry attachment.

    Reads the most recent TelemetryEvent from the aggregator for each
    frame's timestamp and populates batch.pdc/tcm/tcl. When no aggregator
    is configured (telemetry source not running) the stage is a no-op.
    """
    name = "telemetry_ingest"

    def __init__(self, *, aggregator: Optional[TelemetryAggregator]):
        self._aggregator = aggregator

    def process(self, batch: FrameBatch) -> FrameBatch:
        if self._aggregator is None or batch.timestamp_s.size == 0:
            return batch

        n = batch.timestamp_s.size
        pdc = np.full(n, np.nan, dtype=np.float32)
        tcm = np.zeros(n, dtype=np.int64)
        tcl = np.zeros(n, dtype=np.int64)
        for i in range(n):
            event = self._aggregator.snapshot_at(float(batch.timestamp_s[i]))
            if event is None:
                continue
            # pdc_samples is a list of recent PDC readings (mA). Use the
            # mean if multiple are present so a per-frame scalar best
            # represents the laser power over that frame's exposure.
            if event.pdc_samples:
                pdc[i] = float(np.mean(event.pdc_samples))
            tcm[i] = int(event.tcm)
            tcl[i] = int(event.tcl)

        batch.pdc = pdc
        batch.tcm = tcm
        batch.tcl = tcl
        return batch

    def reset(self) -> None:
        # NOTE: intentionally does NOT clear the aggregator. Telemetry history
        # is owned by the source and must outlive transient pipeline
        # exceptions (Pipeline.reset() runs after any stage raises); clearing
        # here would drop history mid-scan on every recoverable error.
        pass
