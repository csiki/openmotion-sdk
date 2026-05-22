#!/usr/bin/env python3
"""Standalone dark-frame schedule integrity check for raw histogram CSVs.

Given a ``*_raw.csv`` produced by ScanWorkflow, verify that every frame
the firmware schedule expects to be a dark frame actually looks dark in
the data. Catches firmware off-by-one symptoms (where a light frame
lands in a slot the SDK treats as dark, polluting the dark-frame
interpolation) and gross ambient-light contamination of the dark slots.

Standalone by design — only depends on ``csv`` and ``numpy``. No SDK
import, so it can be vendored into post-processing pipelines or run on
machines without the openmotion package installed.

Usage
-----
    python scripts/check_dark_schedule.py path/to/scan_left_maskFF_raw.csv

    # custom thresholds / schedule:
    python scripts/check_dark_schedule.py raw.csv \\
        --pedestal 128 --max-u1-above-pedestal 30 --max-std 20 \\
        --discard-count 9 --dark-interval 600

Exit code
---------
    0 — every scheduled dark frame on every camera passes both bounds
    1 — at least one scheduled dark frame failed (or CSV unreadable)

Notes
-----
- Schedule formula matches ``SciencePipeline._is_dark_frame``: first
  dark at ``discard_count + 1`` (= 10), subsequent at
  ``(absolute_frame - 1) % dark_interval == 0`` for
  absolute_frame > discard+1.
- ``frame_id`` in the raw CSV is an 8-bit firmware counter that runs
  ``1,2,…,255,0,1,…`` and wraps every 256 frames. The SDK normally
  unwraps it via ``FrameIdUnwrapper``; this script does its own
  per-camera unwrap (backward-jump = wrap) so it can apply the schedule
  formula on the un-wrapped ``absolute_frame``. Rows are assumed to be
  in arrival order, which is what the raw writer produces.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from typing import Iterator

import numpy as np

HISTOGRAM_BINS = 1024
HISTO_BIN_INDICES = np.arange(HISTOGRAM_BINS, dtype=np.float64)
HISTO_BIN_SQUARES = HISTO_BIN_INDICES ** 2
# Firmware ships frame_id as a single byte (counts 1..255 then 0,1,…).
FRAME_ID_MODULUS = 256


@dataclass
class Thresholds:
    pedestal: float = 128.0
    max_u1_above_pedestal: float = 30.0
    max_std: float = 20.0
    discard_count: int = 9
    dark_interval: int = 600

    @property
    def max_u1(self) -> float:
        return self.pedestal + self.max_u1_above_pedestal


@dataclass
class FrameStats:
    cam_id: int
    raw_frame_id: int
    absolute_frame: int
    u1: float
    std: float
    fail_reasons: list[str]

    @property
    def ok(self) -> bool:
        return not self.fail_reasons


class FrameIdUnwrapper:
    """Per-camera ``raw_frame_id`` (0..255, wraps) -> monotonic
    ``absolute_frame``. A backward step from the previous raw value
    counts as one wrap; first frame seen is taken as the absolute
    starting point.
    """
    def __init__(self) -> None:
        self._state: dict[int, dict[str, int]] = {}

    def unwrap(self, cam_id: int, raw: int) -> int:
        s = self._state.get(cam_id)
        if s is None:
            self._state[cam_id] = {"prev_raw": raw, "wraps": 0}
            return raw
        if raw < s["prev_raw"]:
            s["wraps"] += 1
        s["prev_raw"] = raw
        return s["wraps"] * FRAME_ID_MODULUS + raw


def is_scheduled_dark_frame(frame_id: int, discard_count: int, dark_interval: int) -> bool:
    """Mirror of ``SciencePipeline._is_dark_frame``."""
    if frame_id == discard_count + 1:
        return True
    return (
        frame_id > discard_count + 1
        and (frame_id - 1) % dark_interval == 0
    )


def compute_u1_std(histogram: np.ndarray) -> tuple[float, float]:
    """Compute the population mean (``u1``) and std of a 1024-bin histogram.

    Returns ``(nan, nan)`` on an empty histogram so the caller can decide
    whether to surface or silently skip.
    """
    total = float(histogram.sum())
    if total <= 0:
        return float("nan"), float("nan")
    u1 = float((HISTO_BIN_INDICES * histogram).sum() / total)
    u2 = float((HISTO_BIN_SQUARES * histogram).sum() / total)
    variance = max(0.0, u2 - u1 * u1)
    return u1, float(np.sqrt(variance))


def parse_row(row: list[str]) -> tuple[int, int, np.ndarray] | None:
    """Pull ``(cam_id, frame_id, histogram)`` out of one raw CSV row.

    The raw CSV layout is::

        cam_id, frame_id, timestamp_s, 0..1023, temperature, sum, tcm, tcl, pdc

    Returns None when the row is too short to contain a full histogram —
    happens occasionally on truncated CSVs (e.g. scan canceled mid-row).
    """
    if len(row) < 3 + HISTOGRAM_BINS:
        return None
    try:
        cam_id = int(row[0])
        frame_id = int(row[1])
    except ValueError:
        return None
    try:
        histogram = np.array(row[3:3 + HISTOGRAM_BINS], dtype=np.int64)
    except ValueError:
        return None
    return cam_id, frame_id, histogram


def iter_dark_frame_stats(
    csv_path: str, thresholds: Thresholds,
) -> Iterator[FrameStats]:
    """Stream the raw CSV and yield a ``FrameStats`` for every row whose
    ``absolute_frame`` (unwrapped from the 8-bit raw counter) is a
    scheduled dark frame."""
    unwrapper = FrameIdUnwrapper()
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return
        if not header or header[0] != "cam_id":
            raise ValueError(
                f"{csv_path}: header does not start with 'cam_id' — "
                "is this really a raw histogram CSV?"
            )

        for row in reader:
            parsed = parse_row(row)
            if parsed is None:
                continue
            cam_id, raw_frame_id, histogram = parsed
            absolute_frame = unwrapper.unwrap(cam_id, raw_frame_id)
            if not is_scheduled_dark_frame(
                absolute_frame, thresholds.discard_count, thresholds.dark_interval,
            ):
                continue
            u1, std = compute_u1_std(histogram)
            fail_reasons: list[str] = []
            if np.isnan(u1):
                fail_reasons.append("empty histogram")
            else:
                if u1 > thresholds.max_u1:
                    fail_reasons.append(
                        f"u1={u1:.2f} > pedestal+{thresholds.max_u1_above_pedestal:.0f}"
                        f"={thresholds.max_u1:.0f}"
                    )
                if std > thresholds.max_std:
                    fail_reasons.append(
                        f"std={std:.2f} > {thresholds.max_std:.0f}"
                    )
            yield FrameStats(
                cam_id=cam_id,
                raw_frame_id=raw_frame_id,
                absolute_frame=absolute_frame,
                u1=u1, std=std, fail_reasons=fail_reasons,
            )


def check_file(csv_path: str, thresholds: Thresholds) -> tuple[int, int, list[FrameStats]]:
    """Returns ``(total_checked, fail_count, failures)``."""
    total = 0
    failures: list[FrameStats] = []
    for fs in iter_dark_frame_stats(csv_path, thresholds):
        total += 1
        if not fs.ok:
            failures.append(fs)
    return total, len(failures), failures


def _format_failure(fs: FrameStats) -> str:
    return (
        f"  cam={fs.cam_id} frame={fs.absolute_frame} (raw={fs.raw_frame_id})  "
        f"u1={fs.u1:>7.2f}  std={fs.std:>6.2f}  -> {'; '.join(fs.fail_reasons)}"
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Verify scheduled dark frames in a raw histogram CSV "
                    "actually look dark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("csv_path", help="Path to the *_raw.csv file to check.")
    p.add_argument(
        "--pedestal", type=float, default=128.0,
        help="Sensor pedestal height in DN (128 for sensor-fw >= 1.5.0, "
             "64 for older firmware).",
    )
    p.add_argument(
        "--max-u1-above-pedestal", type=float, default=30.0,
        help="Max DN above pedestal for a frame's u1 to still count as dark.",
    )
    p.add_argument(
        "--max-std", type=float, default=20.0,
        help="Max histogram std for a frame to still count as dark.",
    )
    p.add_argument(
        "--discard-count", type=int, default=9,
        help="Number of warmup frames the science pipeline discards.",
    )
    p.add_argument(
        "--dark-interval", type=int, default=600,
        help="Frames between scheduled darks (default 600 = 15 s @ 40 fps).",
    )
    p.add_argument(
        "-q", "--quiet", action="store_true",
        help="Only print failures and final summary.",
    )
    p.add_argument(
        "--max-print", type=int, default=20,
        help="Cap the number of failure lines printed (set to 0 for all).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    thresholds = Thresholds(
        pedestal=args.pedestal,
        max_u1_above_pedestal=args.max_u1_above_pedestal,
        max_std=args.max_std,
        discard_count=args.discard_count,
        dark_interval=args.dark_interval,
    )

    try:
        total, fail_count, failures = check_file(args.csv_path, thresholds)
    except FileNotFoundError:
        print(f"ERROR: file not found: {args.csv_path}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(
            f"{args.csv_path}: checked {total} scheduled-dark rows "
            f"(pedestal={thresholds.pedestal:.0f}, "
            f"max_u1={thresholds.max_u1:.0f}, max_std={thresholds.max_std:.0f}, "
            f"discard={thresholds.discard_count}, "
            f"interval={thresholds.dark_interval})"
        )

    if not failures:
        if not args.quiet:
            print("OK: all scheduled dark frames look dark.")
        return 0

    print(f"FAIL: {fail_count}/{total} scheduled dark rows look bright:")
    if args.max_print > 0:
        for fs in failures[:args.max_print]:
            print(_format_failure(fs))
        if len(failures) > args.max_print:
            print(f"  ... (+{len(failures) - args.max_print} more)")
    else:
        for fs in failures:
            print(_format_failure(fs))
    return 1


if __name__ == "__main__":
    sys.exit(main())
