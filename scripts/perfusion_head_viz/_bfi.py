"""BFI / BVI math and streaming estimators.

Computes:
  - ``compute_bfi(histogram)``: canonical speckle-contrast BFI per frame.
  - ``BviEstimator``: rolling std-dev of BFI over a short window (pulsatility).
  - ``BaselineNormalizer``: per-channel percentile-based rescale so that the
    visualisation's colour scale stays meaningful without hardware calibration.
  - ``HeartRateEstimator``: Welch PSD peak in the cardiac band, with SNR gate.

No Qt or rendering code in this module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

_HIST_SIZE = 1024
_BIN_INDEX = np.arange(_HIST_SIZE, dtype=np.float64)
_BIN_INDEX_SQ = _BIN_INDEX * _BIN_INDEX


def compute_bfi(hist: np.ndarray) -> tuple[float, float]:
    """Return ``(bfi, mean_bin)`` from a 1024-bin photon-count histogram.

    BFI = 1 / K² where K = sqrt(var) / mean is the speckle contrast.
    Returns (0.0, 0.0) for degenerate / dark frames.
    """
    total = float(hist.sum())
    if total <= 0.0:
        return 0.0, 0.0
    probs = hist.astype(np.float64) / total
    mean = float(np.dot(_BIN_INDEX, probs))
    if mean <= 1e-6:
        return 0.0, mean
    second_moment = float(np.dot(_BIN_INDEX_SQ, probs))
    var = second_moment - mean * mean
    if var <= 0.0:
        return 0.0, mean
    k = np.sqrt(var) / mean
    if k <= 1e-9:
        return 0.0, mean
    return 1.0 / (k * k), mean


class BviEstimator:
    """Rolling std-dev of BFI — pulsatility over ``window_sec`` seconds.

    Value is NaN until at least ``min_samples`` have been pushed, after which
    it's a real std-dev.
    """

    def __init__(self, window_sec: float = 1.0, rate_hz: float = 40.0,
                 min_samples: int = 5) -> None:
        self._buf: deque[float] = deque(maxlen=max(2, int(window_sec * rate_hz)))
        self._min_samples = min_samples

    def push(self, bfi: float) -> None:
        self._buf.append(float(bfi))

    @property
    def value(self) -> float:
        if len(self._buf) < self._min_samples:
            return float("nan")
        return float(np.std(np.fromiter(self._buf, dtype=np.float64)))


class BaselineNormalizer:
    """Percentile-based baseline → maps BFI into a visualisation-friendly range.

    The visible range [p10, p90] of BFI over the last ``window_sec`` seconds is
    stretched to [0, 1] in the output; anything above p90 continues up to 2.
    Anything below p10 saturates to 0.  Until we have ``min_samples`` of
    history we pass the raw value through (normalised by a bootstrap scale of
    10.0 which is a rough BFI magnitude) so the display isn't blank while the
    window fills.
    """

    def __init__(self, window_sec: float = 30.0, rate_hz: float = 40.0,
                 min_samples: int = 20) -> None:
        self._buf: deque[float] = deque(maxlen=max(16, int(window_sec * rate_hz)))
        self._min_samples = min_samples
        self._bootstrap = 10.0  # rough expected BFI magnitude before window fills

    def push(self, bfi: float) -> None:
        self._buf.append(float(bfi))

    def normalize(self, bfi: float) -> float:
        n = len(self._buf)
        if n < self._min_samples:
            return max(0.0, min(2.0, bfi / self._bootstrap))
        arr = np.fromiter(self._buf, dtype=np.float64, count=n)
        p10, p90 = np.percentile(arr, [10.0, 90.0])
        span = max(1e-6, p90 - p10)
        return float(max(0.0, min(2.0, (bfi - p10) / span)))


@dataclass
class HeartRateEstimate:
    hz: float
    bpm: float
    snr: float       # peak-power / mean-power ratio in the cardiac band
    ok: bool         # True when snr >= snr_threshold


class HeartRateEstimator:
    """Welch-PSD based heart-rate estimator.

    Collects a rolling window of globally averaged BFI samples; when the
    window has at least ``min_seconds`` of data, runs Welch's method and
    picks the dominant peak inside ``[f_lo, f_hi]`` Hz.  Reports the peak's
    SNR so callers can gate downstream visual effects.
    """

    def __init__(self, window_sec: float = 8.0, rate_hz: float = 40.0,
                 f_lo: float = 0.8, f_hi: float = 2.5,
                 snr_threshold: float = 2.0,
                 min_seconds: float = 4.0) -> None:
        self._rate = rate_hz
        self._f_lo = f_lo
        self._f_hi = f_hi
        self._snr_threshold = snr_threshold
        self._min_samples = int(min_seconds * rate_hz)
        self._buf_ts: deque[float] = deque(maxlen=int(window_sec * rate_hz))
        self._buf_val: deque[float] = deque(maxlen=int(window_sec * rate_hz))
        self._last: Optional[HeartRateEstimate] = None

    def push(self, ts: float, value: float) -> None:
        self._buf_ts.append(float(ts))
        self._buf_val.append(float(value))

    def estimate(self) -> Optional[HeartRateEstimate]:
        n = len(self._buf_val)
        if n < self._min_samples:
            return self._last
        # Use scipy.signal lazily so users without scipy can still import
        # this module for compute_bfi / BaselineNormalizer.
        try:
            from scipy.signal import welch
        except ImportError:
            return None
        arr = np.fromiter(self._buf_val, dtype=np.float64, count=n)
        arr = arr - float(arr.mean())
        nperseg = min(n, 128)
        freqs, psd = welch(arr, fs=self._rate, nperseg=nperseg)
        band = (freqs >= self._f_lo) & (freqs <= self._f_hi)
        if not band.any() or psd[band].size == 0:
            return self._last
        peak_idx_in_band = int(np.argmax(psd[band]))
        peak_hz = float(freqs[band][peak_idx_in_band])
        peak_power = float(psd[band][peak_idx_in_band])
        mean_power = float(np.mean(psd[band]) + 1e-12)
        snr = peak_power / mean_power
        ok = snr >= self._snr_threshold
        self._last = HeartRateEstimate(
            hz=peak_hz,
            bpm=peak_hz * 60.0,
            snr=snr,
            ok=ok,
        )
        return self._last
