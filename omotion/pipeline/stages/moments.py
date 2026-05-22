"""MomentsStage — vectorized first/second moment + std + contrast.

See docs/SciencePipeline.md §6.
"""

from __future__ import annotations

import numpy as np

from ..batch import FrameBatch


class MomentsStage:
    name = "moments"

    _BIN_VALUES    = np.arange(1024, dtype=np.float64)
    _BIN_VALUES_SQ = _BIN_VALUES ** 2

    def process(self, batch: FrameBatch) -> FrameBatch:
        h = batch.raw_histograms
        counts = h.sum(axis=-1)
        safe_counts = np.where(counts > 0, counts, 1).astype(np.float64)

        u1 = np.einsum('nsci,i->nsc', h, self._BIN_VALUES)    / safe_counts
        u2 = np.einsum('nsci,i->nsc', h, self._BIN_VALUES_SQ) / safe_counts

        var = np.maximum(u2 - u1 ** 2, 0.0)
        std = np.sqrt(var)

        mean = np.where(counts > 0, u1, np.nan)

        with np.errstate(divide='ignore', invalid='ignore'):
            contrast = np.where(mean > 0, std / mean, np.nan)

        batch.mean_raw     = mean.astype(np.float32)
        batch.std_raw      = std.astype(np.float32)
        batch.contrast_raw = contrast.astype(np.float32)
        return batch

    def reset(self) -> None:
        pass
