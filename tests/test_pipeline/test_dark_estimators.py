"""HybridRealtimePredictor — avg-of-3 u1 + linear-extrap std + ZOH fallback."""

import pytest
from omotion.pipeline.stages.dark import (
    DarkHistory, HybridRealtimePredictor,
)


def _hist_with(entries):
    """entries: list of (t, u1, std)"""
    h = DarkHistory(max_darks=10)
    for t, u1, std in entries:
        h.append("left", 0, t=t, u1=u1, std=std)
    return h


def test_zoh_when_only_one_dark():
    h = _hist_with([(0.0, 100.0, 10.0)])
    pred = HybridRealtimePredictor()
    u1, std = pred.predict("left", 0, history=h, target_t=15.0)
    assert u1 == 100.0
    assert std == 10.0


def test_avg_of_2_when_only_two_darks():
    h = _hist_with([(0.0, 100.0, 10.0), (15.0, 200.0, 20.0)])
    pred = HybridRealtimePredictor()
    u1, std = pred.predict("left", 0, history=h, target_t=30.0)
    assert u1 == pytest.approx(150.0)
    assert std == pytest.approx(30.0)


def test_avg_of_3_when_three_or_more_darks():
    h = _hist_with([
        (0.0,  100.0, 10.0),
        (15.0, 110.0, 12.0),
        (30.0, 120.0, 14.0),
        (45.0, 130.0, 16.0),
    ])
    pred = HybridRealtimePredictor()
    u1, std = pred.predict("left", 0, history=h, target_t=60.0)
    assert u1 == pytest.approx(120.0)
    assert std == pytest.approx(18.0)


def test_returns_none_when_history_is_empty():
    h = DarkHistory(max_darks=4)
    pred = HybridRealtimePredictor()
    result = pred.predict("left", 0, history=h, target_t=15.0)
    assert result is None


def test_zoh_std_when_two_darks_have_same_timestamp():
    h = _hist_with([(0.0, 100.0, 10.0), (0.0, 200.0, 20.0)])
    pred = HybridRealtimePredictor()
    u1, std = pred.predict("left", 0, history=h, target_t=15.0)
    assert std == 20.0
