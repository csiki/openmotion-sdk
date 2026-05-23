"""Tests for FrameBatch — the per-batch data carrier that flows through stages."""

import numpy as np
import pytest

from omotion.pipeline.batch import FrameBatch, BatchEvent, IntervalClosed


def test_framebatch_construction_minimum_fields():
    """A FrameBatch can be constructed with only the fields HistogramParseStage populates.
    All later-stage fields default to None."""
    n = 5
    batch = FrameBatch(
        cam_ids=np.zeros(n, dtype=np.int8),
        frame_ids=np.arange(n, dtype=np.uint8),
        raw_histograms=np.zeros((n, 2, 8, 1024), dtype=np.uint32),
        temperature_c=np.zeros((n, 2, 8), dtype=np.float32),
        timestamp_s=np.linspace(0.0, 0.1, n, dtype=np.float64),
        pdc=None,
        tcm=None,
        tcl=None,
    )
    assert batch.raw_histograms.shape == (n, 2, 8, 1024)
    assert batch.mean_raw is None
    assert batch.bfi_live is None
    assert batch.events == []


def test_framebatch_events_list_is_mutable():
    batch = _trivial_batch(n=1)
    batch.events.append(IntervalClosed(corrected_batch=None))
    assert len(batch.events) == 1
    assert isinstance(batch.events[0], IntervalClosed)


def test_telemetry_event_carries_console_telemetry_fields():
    from omotion.pipeline.batch import TelemetryEvent
    ev = TelemetryEvent(
        timestamp_s=12.5, pdc_samples=[1.23, 1.24, 1.22],
        tec_setpoint_c=25.0, tec_actual_c=25.1,
        tec_setpoint_raw=0.612, tec_actual_raw=0.615,
        safety_status=0, tcm=12345, tcl=67890,
    )
    assert ev.timestamp_s == 12.5
    assert ev.pdc_samples == [1.23, 1.24, 1.22]
    assert ev.tec_setpoint_c == 25.0
    assert ev.tec_actual_raw == 0.615
    assert ev.tcm == 12345
    assert ev.tcl == 67890


def test_telemetry_event_is_a_batch_event():
    from omotion.pipeline.batch import TelemetryEvent, BatchEvent
    ev = TelemetryEvent(
        timestamp_s=0.0, pdc_samples=[], tec_setpoint_c=0.0,
        tec_actual_c=0.0, tec_setpoint_raw=0.0, tec_actual_raw=0.0,
        safety_status=0, tcm=0, tcl=0,
    )
    assert isinstance(ev, BatchEvent)


def _trivial_batch(n: int) -> FrameBatch:
    return FrameBatch(
        cam_ids=np.zeros(n, dtype=np.int8),
        frame_ids=np.arange(n, dtype=np.uint8),
        raw_histograms=np.zeros((n, 2, 8, 1024), dtype=np.uint32),
        temperature_c=np.zeros((n, 2, 8), dtype=np.float32),
        timestamp_s=np.zeros(n, dtype=np.float64),
        pdc=None, tcm=None, tcl=None,
    )
