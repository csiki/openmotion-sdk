"""Source protocol + concrete source implementations.

A Source produces an iterator of FrameBatch and carries the ScanMetadata
for the scan. Concrete sources:
  - CsvReplaySource  — replays raw histogram CSVs (Task 20)
  - DbReplaySource   — replays scan-DB session_raw rows (Task 21)
  - LiveUsbSource    — reads from USB on background threads (Task 22, skeleton)
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Iterator, Optional, Protocol, runtime_checkable

import numpy as np

from .batch import FrameBatch
from .sinks import ScanMetadata


@runtime_checkable
class Source(Protocol):
    metadata: ScanMetadata

    def __iter__(self) -> Iterator[FrameBatch]: ...

    def close(self) -> None: ...


class _BaseSource:
    """Shared scaffolding for concrete sources."""

    def __init__(self, *, metadata: ScanMetadata):
        self.metadata = metadata

    def close(self) -> None:
        pass


class CsvReplaySource(_BaseSource):
    """Replays a raw-histogram CSV produced by CsvSink.

    CSV schema (SciencePipeline.md §12 / spec §12):
        cam_id, frame_id, timestamp_s, type, 0..1023, temperature, sum, tcm, tcl, pdc

    Accepts up to two CSVs (one per side); if a side is None, only the
    other side is replayed.
    """

    def __init__(self, *,
                 raw_csv_left: Optional[Path],
                 raw_csv_right: Optional[Path],
                 batch_size_frames: int = 100,
                 metadata: ScanMetadata):
        super().__init__(metadata=metadata)
        self._paths = {"left": raw_csv_left, "right": raw_csv_right}
        self._batch_size = int(batch_size_frames)

    def __iter__(self) -> Iterator[FrameBatch]:
        for side_name, path in self._paths.items():
            if path is None:
                continue
            yield from self._iter_side(side_name, path)

    def _iter_side(self, side_name: str, path: Path) -> Iterator[FrameBatch]:
        side_idx = 0 if side_name == "left" else 1
        rows_buf: list[dict] = []
        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows_buf.append(row)
                if len(rows_buf) >= self._batch_size:
                    yield self._rows_to_batch(side_idx, rows_buf)
                    rows_buf = []
            if rows_buf:
                yield self._rows_to_batch(side_idx, rows_buf)

    def _rows_to_batch(self, side_idx: int, rows: list[dict]) -> FrameBatch:
        n = len(rows)
        cam_ids     = np.array([int(r["cam_id"])        for r in rows], dtype=np.int8)
        frame_ids   = np.array([int(r["frame_id"])      for r in rows], dtype=np.uint8)
        timestamp_s = np.array([float(r["timestamp_s"]) for r in rows], dtype=np.float64)

        raw_hist = np.zeros((n, 2, 8, 1024), dtype=np.uint32)
        temp_arr = np.zeros((n, 2, 8), dtype=np.float32)
        for i, row in enumerate(rows):
            cam = int(row["cam_id"])
            for b in range(1024):
                raw_hist[i, side_idx, cam, b] = int(row[str(b)])
            temp_arr[i, side_idx, cam] = float(row["temperature"])

        return FrameBatch(
            cam_ids=cam_ids,
            frame_ids=frame_ids,
            raw_histograms=raw_hist,
            temperature_c=temp_arr,
            timestamp_s=timestamp_s,
            pdc=None, tcm=None, tcl=None,
        )


class DbReplaySource(_BaseSource):
    """Replays a scan-DB session by reading rows out of session_raw.

    Assumes the table layout used by ScanDBSink (see omotion/ScanDatabase.py).
    Each session_raw row carries a 4096-byte histogram blob (1024 × uint32 LE)
    in the `hist` column.
    """

    def __init__(self, *, db_path: str, session_id: int,
                 batch_size_frames: int = 100,
                 metadata: ScanMetadata):
        super().__init__(metadata=metadata)
        self._db_path = db_path
        self._session_id = int(session_id)
        self._batch_size = int(batch_size_frames)

    def __iter__(self) -> Iterator[FrameBatch]:
        conn = sqlite3.connect(self._db_path)
        try:
            cur = conn.execute(
                "SELECT side, cam_id, frame_id, timestamp_s, hist, "
                "       temp, sum "
                "FROM session_raw WHERE session_id = ? "
                "ORDER BY timestamp_s, side, cam_id",
                (self._session_id,),
            )
            buf: list[tuple] = []
            for row in cur:
                buf.append(tuple(row))
                if len(buf) >= self._batch_size:
                    yield self._rows_to_batch(buf)
                    buf = []
            if buf:
                yield self._rows_to_batch(buf)
        finally:
            conn.close()

    def _rows_to_batch(self, rows: list[tuple]) -> FrameBatch:
        n = len(rows)
        cam_ids     = np.zeros(n, dtype=np.int8)
        frame_ids   = np.zeros(n, dtype=np.uint8)
        timestamp_s = np.zeros(n, dtype=np.float64)
        raw_hist    = np.zeros((n, 2, 8, 1024), dtype=np.uint32)
        temp_arr    = np.zeros((n, 2, 8), dtype=np.float32)

        for i, (side, cam_id, frame_id, t, blob, temp, _row_sum) in enumerate(rows):
            side_idx = 0 if side == "left" else 1
            cam_ids[i]     = int(cam_id)
            frame_ids[i]   = int(frame_id)
            timestamp_s[i] = float(t)
            hist_bytes = bytes(blob)
            hist_arr = np.frombuffer(hist_bytes, dtype=np.uint32, count=1024)
            raw_hist[i, side_idx, int(cam_id)] = hist_arr
            if temp is not None:
                temp_arr[i, side_idx, int(cam_id)] = float(temp)

        return FrameBatch(
            cam_ids=cam_ids, frame_ids=frame_ids,
            raw_histograms=raw_hist, temperature_c=temp_arr,
            timestamp_s=timestamp_s,
            pdc=None, tcm=None, tcl=None,
        )
