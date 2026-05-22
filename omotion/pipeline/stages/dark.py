"""Dark correction subsystem.

Built up across 5 tasks:
    Task 14: DarkHistory          — ring buffer of recent dark observations
    Task 15: DarkIntegrityGuard   — u1 > pedestal+threshold check
    Task 16: HybridRealtimePredictor — avg-of-3 u1 + linear-extrap std + ZOH
    Task 17: PendingInterval, LinearInterpolation, DarkFrameQuadraticStencil
    Task 18: DarkCorrectionStage  — orchestrator (NOT in this batch)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from ..batch import DarkIntegrityWarning


@dataclass(frozen=True)
class DarkObservation:
    """One dark-frame measurement for one camera."""
    t:   float
    u1:  float
    std: float


class DarkHistory:
    """Per-(side, cam) ring buffer of recent DarkObservations."""

    def __init__(self, max_darks: int = 4):
        if max_darks < 1:
            raise ValueError(f"max_darks must be >= 1, got {max_darks}")
        self._max = int(max_darks)
        self._rings: dict[tuple[str, int], Deque[DarkObservation]] = {}

    def append(self, side: str, cam_id: int, *, t: float, u1: float, std: float) -> None:
        key = (side, int(cam_id))
        ring = self._rings.get(key)
        if ring is None:
            ring = deque(maxlen=self._max)
            self._rings[key] = ring
        ring.append(DarkObservation(t=float(t), u1=float(u1), std=float(std)))

    def recent(self, side: str, cam_id: int, n: int) -> list[DarkObservation]:
        """Return up to the most recent n entries in chronological order."""
        ring = self._rings.get((side, int(cam_id)))
        if ring is None:
            return []
        if n >= len(ring):
            return list(ring)
        return list(ring)[-n:]

    def size(self, side: str, cam_id: int) -> int:
        ring = self._rings.get((side, int(cam_id)))
        return 0 if ring is None else len(ring)

    def is_empty(self, side: str, cam_id: int) -> bool:
        return self.size(side, cam_id) == 0

    def clear(self) -> None:
        self._rings.clear()


class DarkIntegrityGuard:
    """Flag dark frames whose u1 looks suspiciously bright.

    A genuine dark frame should have u1 within ~30 DN of the sensor pedestal.
    Higher u1 suggests the laser wasn't actually off (firmware off-by-one or
    unwrapper alignment quirk). The guard appends a diagnostic event but
    does not drop the frame.

    See docs/SciencePipeline.md §11 (input validation rails).
    """

    def __init__(self, max_above_pedestal: float = 30.0):
        self.max_above_pedestal = float(max_above_pedestal)

    def check(self, *, side: str, cam_id: int, abs_frame_id: int,
              u1: float, pedestal: float, events: list) -> bool:
        """Return True if the dark frame passes; False if it failed (warning emitted)."""
        threshold = pedestal + self.max_above_pedestal
        if u1 > threshold:
            events.append(DarkIntegrityWarning(
                side=side, cam_id=int(cam_id), abs_frame_id=int(abs_frame_id),
                u1=float(u1), pedestal=float(pedestal),
                threshold=float(self.max_above_pedestal),
            ))
            return False
        return True


class HybridRealtimePredictor:
    """Realtime dark-baseline predictor.

    Algorithm (see docs/SciencePipeline.md §7.4.1):
        u1   ← average of last 3 dark observations (truncated; ZOH with 1)
        std  ← linear extrapolation through last 2 darks; ZOH with 1 or
                when both darks share a timestamp

    Returns None when no darks have been observed yet — caller skips
    realtime emission for that frame (warmup window).
    """

    def predict(self, side: str, cam_id: int, *, history: DarkHistory,
                target_t: float) -> Optional[tuple[float, float]]:
        recent = history.recent(side, cam_id, n=3)
        if not recent:
            return None

        u1_pred = sum(o.u1 for o in recent) / len(recent)

        if len(recent) < 2 or history.size(side, cam_id) < 2:
            std_pred = recent[-1].std
        else:
            last2 = history.recent(side, cam_id, n=2)
            a, b = last2
            dt = b.t - a.t
            if dt <= 0:
                std_pred = b.std
            else:
                slope = (b.std - a.std) / dt
                std_pred = b.std + slope * (target_t - b.t)

        return (float(u1_pred), float(std_pred))


@dataclass
class _LightSample:
    abs_frame_id: int
    t: float
    u1: float
    u2: float


@dataclass
class _DarkBoundary:
    obs: DarkObservation
    abs_frame_id: int


@dataclass
class Interval:
    """A closed dark-bounded interval, ready for batch correction."""
    left:         _DarkBoundary
    right:        _DarkBoundary
    light_frames: list[_LightSample]

    @property
    def left_abs(self) -> int:
        return self.left.abs_frame_id

    @property
    def right_abs(self) -> int:
        return self.right.abs_frame_id


@dataclass
class CorrectedFrame:
    """One corrected sample, output of batch correction."""
    abs_frame_id: int
    t:    float
    mean: float
    std:  float


@dataclass
class CorrectedInterval:
    """Output of batch correction."""
    left_abs:  int
    right_abs: int
    frames:    list[CorrectedFrame]


class PendingInterval:
    """Buffers non-dark frames between two bounding darks."""

    def __init__(self):
        self._left:  Optional[_DarkBoundary] = None
        self._right: Optional[_DarkBoundary] = None
        self._light: list[_LightSample] = []

    def set_left_dark(self, obs: DarkObservation, *, abs_frame_id: int) -> None:
        self._left = _DarkBoundary(obs=obs, abs_frame_id=int(abs_frame_id))
        self._light = []
        self._right = None

    def add_light(self, *, abs_frame_id: int, t: float, u1: float, u2: float) -> None:
        self._light.append(_LightSample(
            abs_frame_id=int(abs_frame_id), t=float(t), u1=float(u1), u2=float(u2),
        ))

    def set_right_dark(self, obs: DarkObservation, *, abs_frame_id: int) -> None:
        self._right = _DarkBoundary(obs=obs, abs_frame_id=int(abs_frame_id))

    def is_closed(self) -> bool:
        return self._left is not None and self._right is not None

    def flush(self) -> Interval:
        """Return the closed interval; reset for next interval."""
        assert self.is_closed(), "flush() called on non-closed interval"
        interval = Interval(
            left=self._left,
            right=self._right,
            light_frames=list(self._light),
        )
        self._left = self._right
        self._right = None
        self._light = []
        return interval


class LinearInterpolation:
    """Compute corrected values for a closed dark-bounded interval.

    See docs/SciencePipeline.md §8.1–§8.3.
    """

    def correct_interval(self, interval: Interval) -> CorrectedInterval:
        d_prev = interval.left.obs
        d_next = interval.right.obs
        dt = d_next.t - d_prev.t

        corrected_frames: list[CorrectedFrame] = []
        for lf in interval.light_frames:
            t_frac = (lf.t - d_prev.t) / dt if dt > 0 else 0.0
            baseline_u1  = d_prev.u1  + t_frac * (d_next.u1  - d_prev.u1)
            baseline_std = d_prev.std + t_frac * (d_next.std - d_prev.std)

            mean = lf.u1 - baseline_u1
            raw_var = max(0.0, lf.u2 - lf.u1 ** 2)
            corrected_var = max(0.0, raw_var - baseline_std ** 2)
            std = float(corrected_var ** 0.5)

            corrected_frames.append(CorrectedFrame(
                abs_frame_id=lf.abs_frame_id, t=lf.t, mean=mean, std=std,
            ))

        return CorrectedInterval(
            left_abs=interval.left_abs, right_abs=interval.right_abs,
            frames=corrected_frames,
        )


class DarkFrameQuadraticStencil:
    """4-point quadratic interpolation for the dark frame's own corrected value.

    Stencil:
        v(D) = (-1/6) v(D-2) + (2/3) v(D-1) + (2/3) v(D+1) + (-1/6) v(D+2)

    Fallback chain (see SciencePipeline.md §8.4):
        full        — all four neighbours present
        right_only  — left missing, right ≥2 → (v(+1) + v(+2)) / 2
        simple_avg  — only v(-1) and v(+1) → (v(-1) + v(+1)) / 2
        repeat_right — only v(+1) → v(+1)
    """

    def interpolate_dark_value(self, *,
                               v_minus_2: Optional[float],
                               v_minus_1: Optional[float],
                               v_plus_1: Optional[float],
                               v_plus_2: Optional[float]) -> float:
        if all(v is not None for v in (v_minus_2, v_minus_1, v_plus_1, v_plus_2)):
            return (-1/6) * v_minus_2 + (2/3) * v_minus_1 \
                + (2/3) * v_plus_1 + (-1/6) * v_plus_2

        if v_minus_1 is None and v_plus_1 is not None and v_plus_2 is not None:
            return (v_plus_1 + v_plus_2) / 2

        if v_minus_1 is not None and v_plus_1 is not None:
            return (v_minus_1 + v_plus_1) / 2

        if v_plus_1 is not None:
            return v_plus_1

        raise ValueError("DarkFrameQuadraticStencil needs at least v_plus_1")
