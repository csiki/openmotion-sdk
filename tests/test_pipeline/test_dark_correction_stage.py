"""DarkCorrectionStage — orchestrator: dual-output realtime + batch."""

import numpy as np
import pytest
from omotion.pipeline.batch import FrameBatch, IntervalClosed
from omotion.pipeline.stages.dark import (
    DarkCorrectionStage, HybridRealtimePredictor, LinearInterpolation,
)


def _batch(n_frames, frame_types, abs_ids, *, mean_raw, std_raw, u2=None):
    """Build a minimal FrameBatch — only side=0 cam=0 populated."""
    n = n_frames
    raw_hist = np.zeros((n, 2, 8, 1024), dtype=np.uint32)
    raw_hist[:, 0, 0, 0] = 1   # marker for the only active camera

    batch = FrameBatch(
        cam_ids=np.zeros(n, dtype=np.int8),
        frame_ids=np.arange(n, dtype=np.uint8),
        raw_histograms=raw_hist,
        temperature_c=np.zeros((n, 2, 8), dtype=np.float32),
        timestamp_s=np.arange(n, dtype=np.float64) * 0.025,
        pdc=None, tcm=None, tcl=None,
        abs_frame_ids=np.array(abs_ids, dtype=np.int64),
        frame_type=np.array(frame_types, dtype="<U8"),
        mean_raw=mean_raw, std_raw=std_raw,
    )
    return batch


def test_no_realtime_emit_before_first_dark():
    """Light frames before the first dark have no baseline to subtract — NaN result."""
    mean = np.full((3, 2, 8), 500.0, dtype=np.float32)
    std  = np.full((3, 2, 8), 10.0,  dtype=np.float32)
    batch = _batch(3, ["light", "light", "light"], [11, 12, 13],
                   mean_raw=mean, std_raw=std)

    stage = DarkCorrectionStage(
        realtime_estimator=HybridRealtimePredictor(),
        batch_estimator=LinearInterpolation(),
    )
    stage.process(batch)
    assert np.isnan(batch.mean_dc_rt[0, 0, 0])


def test_emits_interval_closed_event_when_two_darks_bracket_lights():
    n = 4
    mean = np.array([[100.0], [500.0], [510.0], [105.0]], dtype=np.float32).reshape(4, 1, 1) * np.ones((1, 2, 8))
    std  = np.array([[10.0],  [20.0],  [22.0],  [11.0]], dtype=np.float32).reshape(4, 1, 1) * np.ones((1, 2, 8))
    batch = _batch(n, ["dark", "light", "light", "dark"], [10, 11, 12, 14],
                   mean_raw=mean.astype(np.float32), std_raw=std.astype(np.float32))

    stage = DarkCorrectionStage(
        realtime_estimator=HybridRealtimePredictor(),
        batch_estimator=LinearInterpolation(),
    )
    stage.process(batch)

    events = [e for e in batch.events if isinstance(e, IntervalClosed)]
    assert len(events) > 0


def test_reset_clears_dark_history_and_pending():
    stage = DarkCorrectionStage(
        realtime_estimator=HybridRealtimePredictor(),
        batch_estimator=LinearInterpolation(),
    )
    mean = np.full((1, 2, 8), 100.0, dtype=np.float32)
    std  = np.full((1, 2, 8), 10.0,  dtype=np.float32)
    batch1 = _batch(1, ["dark"], [10], mean_raw=mean, std_raw=std)
    stage.process(batch1)

    stage.reset()

    batch2 = _batch(1, ["light"], [11],
                    mean_raw=np.full((1, 2, 8), 500.0, dtype=np.float32),
                    std_raw=np.full((1, 2, 8), 20.0, dtype=np.float32))
    stage.process(batch2)
    assert np.isnan(batch2.mean_dc_rt[0, 0, 0])
