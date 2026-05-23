"""Tests for TelemetryAggregator + TelemetryIngestStage."""

import numpy as np
import pytest
from omotion.pipeline.batch import FrameBatch, TelemetryEvent
from omotion.pipeline.telemetry import TelemetryAggregator, TelemetryIngestStage


def _ev(t, pdc, tcm=0, tcl=0):
    return TelemetryEvent(
        timestamp_s=t, pdc_samples=[pdc],
        tec_setpoint_c=25.0, tec_actual_c=25.0,
        tec_setpoint_raw=0.612, tec_actual_raw=0.615,
        safety_status=0, tcm=tcm, tcl=tcl,
    )


def test_aggregator_starts_empty():
    agg = TelemetryAggregator()
    assert agg.snapshot_at(0.0) is None


def test_aggregator_returns_most_recent_event_before_target_t():
    agg = TelemetryAggregator()
    agg.update(_ev(1.0, 1.10))
    agg.update(_ev(2.0, 1.20))
    agg.update(_ev(3.0, 1.30))
    snap = agg.snapshot_at(2.5)
    assert snap is not None
    assert snap.timestamp_s == 2.0


def test_aggregator_returns_none_when_no_event_before_t():
    agg = TelemetryAggregator()
    agg.update(_ev(5.0, 1.10))
    assert agg.snapshot_at(2.0) is None


def test_aggregator_bounded_capacity():
    agg = TelemetryAggregator(max_history=3)
    for i in range(5):
        agg.update(_ev(float(i), 1.0 + i * 0.01))
    assert agg.size() == 3


def _empty_batch_with_timestamps(timestamps):
    n = len(timestamps)
    return FrameBatch(
        cam_ids=np.zeros(n, dtype=np.int8),
        frame_ids=np.zeros(n, dtype=np.uint8),
        raw_histograms=np.zeros((n, 2, 8, 1024), dtype=np.uint32),
        temperature_c=np.zeros((n, 2, 8), dtype=np.float32),
        timestamp_s=np.array(timestamps, dtype=np.float64),
        pdc=None, tcm=None, tcl=None,
    )


def test_ingest_stage_fills_batch_telemetry_fields_per_frame():
    agg = TelemetryAggregator()
    agg.update(_ev(1.0, 1.10, tcm=10, tcl=100))
    agg.update(_ev(2.0, 1.20, tcm=20, tcl=200))
    stage = TelemetryIngestStage(aggregator=agg)
    batch = _empty_batch_with_timestamps([0.5, 1.5, 2.5])
    stage.process(batch)
    assert batch.pdc is not None
    assert np.isnan(batch.pdc[0])      # no event at/before 0.5
    assert batch.pdc[1] == pytest.approx(1.10)
    assert batch.pdc[2] == pytest.approx(1.20)
    assert batch.tcm[1] == 10
    assert batch.tcl[2] == 200


def test_ingest_stage_is_noop_when_no_aggregator():
    stage = TelemetryIngestStage(aggregator=None)
    batch = _empty_batch_with_timestamps([0.5, 1.5])
    stage.process(batch)
    assert batch.pdc is None
