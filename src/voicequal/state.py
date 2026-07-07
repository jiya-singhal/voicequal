"""Stateful audio analysis primitives for voicequal.

Some audio quality metrics require observing a signal over time
(e.g. temporal variance of noise floor, minimum-statistics background
level). This module provides a RollingStats helper that maintains
the rolling buffers those metrics need.
"""

from collections import deque

import numpy as np


class RollingStats:
    """Maintains rolling buffers of noise floor and RMS values.

    Two metrics are exposed:
      - temporal_variance: std-dev of the last <=20 noise-floor readings.
        Returns 0 until at least 10 readings have been observed.
      - background_db: median of the last <=30 RMS readings, converted
        to dB with a +94 offset. Uses the current RMS until 8 readings
        have been observed.

    HISTORY_LENGTH=20 for temporal, RMS_HISTORY_SIZE=30 for background —
    chosen to match ~2s and ~3s of frame history at ~10Hz analysis rate.
    """

    HISTORY_LENGTH: int = 20
    RMS_HISTORY_SIZE: int = 30
    MIN_TEMPORAL_SAMPLES: int = 10
    MIN_BACKGROUND_SAMPLES: int = 8

    def __init__(self) -> None:
        self._noise_floor_history: deque[float] = deque(maxlen=self.HISTORY_LENGTH)
        self._rms_history: deque[float] = deque(maxlen=self.RMS_HISTORY_SIZE)

    def update(self, current_noise_floor: float, current_rms: float) -> None:
        """Push one frame's (noise_floor_dB, rms) into the buffers."""
        self._noise_floor_history.append(float(current_noise_floor))
        self._rms_history.append(float(current_rms))

    def temporal_variance(self) -> float:
        """Std-dev of the noise-floor history. Returns 0 until MIN_TEMPORAL_SAMPLES."""
        if len(self._noise_floor_history) < self.MIN_TEMPORAL_SAMPLES:
            return 0.0
        return float(np.std(self._noise_floor_history))

    def background_db(self, current_rms: float, db_offset: float = 94.0) -> float:
        """Background loudness in dBA-like scale (median of RMS history + offset).

        If we have fewer than MIN_BACKGROUND_SAMPLES readings, use
        current_rms directly instead of the historical median.
        """
        if len(self._rms_history) >= self.MIN_BACKGROUND_SAMPLES:
            # Use the median (50th percentile) so the reading reflects how loud
            # the room *typically* is over the window, not the quietest gap.
            # A low percentile ducks under intermittent loud noise (the room
            # never reaches POOR); the median rises with sustained loudness,
            # rejects single-frame spikes, and decays as loud frames age out.
            background_rms = float(np.percentile(list(self._rms_history), 50))
        else:
            background_rms = current_rms
        return float(20 * np.log10(max(background_rms, 1e-10)) + db_offset)

    def reset(self) -> None:
        """Clear both rolling buffers."""
        self._noise_floor_history.clear()
        self._rms_history.clear()
