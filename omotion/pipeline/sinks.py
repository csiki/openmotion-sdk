"""Sink protocol + ScanMetadata.

Concrete sink implementations (CsvSink, ScanDBSink, QtUiSink) come later
in this PR. This module establishes the protocol so the runner and stages
can refer to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class ScanMetadata:
    """Per-scan metadata handed to every sink at on_scan_start."""
    scan_id:               str
    subject_id:            str
    operator:              str
    started_at_iso:        str
    duration_sec:          int
    left_camera_mask:      int
    right_camera_mask:     int
    reduced_mode:          bool
    write_raw_csv:         bool
    raw_csv_duration_sec:  Optional[float]


@runtime_checkable
class Sink(Protocol):
    """A consumer of pipeline output.

    Channels in this pipeline:
        "raw"          — per-frame, all non-stale frames including warmup
        "live"         — per-frame, best-effort corrected (light + dark)
        "rolling"      — per-frame, rolling-averaged for test/calibration
        "final"        — per-dark-interval, accurately corrected CorrectedBatch
        "diagnostics"  — out-of-band events (DarkIntegrityWarning, etc.)
    """
    channels: set[str]

    def on_scan_start(self, meta: ScanMetadata) -> None: ...

    def consume(self, channel: str, payload: Any) -> None: ...

    def on_complete(self) -> None: ...
