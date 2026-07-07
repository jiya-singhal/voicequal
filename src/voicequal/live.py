"""Real-time streaming quality detection for voicequal.

Where pipeline.assess() analyzes a complete audio file, LiveDetector
processes audio chunks as they arrive (from a mic, network stream, etc.)
and fires a callback whenever the quality tier changes.

Uses the same 2048-sample frame, 1600-sample hop, and steady-state
aggregation as pipeline.assess(). Adds tier-stability hysteresis so
the reported quality doesn't flicker on borderline frames.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from voicequal.assessment import assess_quality
from voicequal.metrics import noise_floor, rms, snr, spectral_flatness
from voicequal.pipeline import FRAME_SIZE, HOP_SIZE, STEADY_STATE_TAIL
from voicequal.state import RollingStats

# Number of consecutive frames a new tier must appear in before it
# replaces the current stable tier. Prevents single-frame flicker.
DEFAULT_STABILITY_FRAMES: int = 3


@dataclass(frozen=True)
class LiveAssessment:
    """Snapshot of the current live quality state.

    Fields mirror FileAssessment but describe the current rolling window,
    not a completed file.
    """

    quality: str
    reason: str
    background_db: float
    snr: float
    spectral_flatness: float
    temporal_variance: float
    frames_analyzed: int


class LiveDetector:
    """Streaming quality detector.

    Usage:
        detector = LiveDetector(sample_rate=16000)
        detector.on_change(lambda result: print(result.quality))

        while streaming:
            chunk = get_audio_chunk()   # 1D float32 numpy array
            detector.push(chunk)

    The push() call is non-blocking: it appends samples to an internal
    buffer, extracts as many complete frames as possible, updates
    internal state, and fires the on_change callback if the tier
    transitions to a NEW stable tier.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        stability_frames: int = DEFAULT_STABILITY_FRAMES,
        threshold_offset_db: float = 0.0,
        db_offset: float = 94.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.stability_frames = stability_frames
        self.threshold_offset_db = threshold_offset_db
        self.db_offset = db_offset

        # Ring buffer of raw samples not yet framed.
        self._buffer: np.ndarray = np.zeros(0, dtype=np.float32)

        # Rolling stats matching pipeline.assess().
        self._stats = RollingStats()

        # Steady-state metric history.
        self._snr_history: list[float] = []
        self._flatness_history: list[float] = []
        self._background_db_history: list[float] = []
        self._last_temporal_variance: float = 0.0

        # Hysteresis state.
        self._current_tier: Optional[str] = None      # last stable tier
        self._candidate_tier: Optional[str] = None    # tier currently being validated
        self._candidate_streak: int = 0
        self._latest_assessment: Optional[LiveAssessment] = None

        self._on_change_callback: Optional[Callable[[LiveAssessment], None]] = None
        self._frames_analyzed: int = 0

    def on_change(self, callback: Callable[[LiveAssessment], None]) -> None:
        """Register a callback to fire when the STABLE tier changes.

        The callback receives a LiveAssessment snapshot. If a callback
        was previously registered, it is replaced.
        """
        self._on_change_callback = callback

    def push(self, samples: np.ndarray) -> None:
        """Push a chunk of audio samples into the detector.

        Args:
            samples: 1D float32 numpy array. Any length.
        """
        if samples.size == 0:
            return

        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self._buffer = np.concatenate([self._buffer, samples])

        # Extract as many complete frames as we can, hop by hop.
        while self._buffer.size >= FRAME_SIZE:
            frame = self._buffer[:FRAME_SIZE]
            self._process_frame(frame)
            # Advance the buffer by HOP_SIZE (not FRAME_SIZE — we overlap).
            self._buffer = self._buffer[HOP_SIZE:]

    def _process_frame(self, frame: np.ndarray) -> None:
        """Compute metrics for one frame and update all state."""
        frame_rms = rms(frame)
        frame_snr = snr(frame)
        frame_flatness = spectral_flatness(frame)
        frame_noise_floor = noise_floor(frame)

        self._stats.update(current_noise_floor=frame_noise_floor, current_rms=frame_rms)

        frame_background_db = self._stats.background_db(
            current_rms=frame_rms, db_offset=self.db_offset,
        )
        frame_temporal_variance = self._stats.temporal_variance()

        self._snr_history.append(frame_snr)
        self._flatness_history.append(frame_flatness)
        self._background_db_history.append(frame_background_db)
        self._last_temporal_variance = frame_temporal_variance

        # Cap history at STEADY_STATE_TAIL (avoid unbounded growth).
        if len(self._snr_history) > STEADY_STATE_TAIL:
            self._snr_history = self._snr_history[-STEADY_STATE_TAIL:]
            self._flatness_history = self._flatness_history[-STEADY_STATE_TAIL:]
            self._background_db_history = self._background_db_history[-STEADY_STATE_TAIL:]

        self._frames_analyzed += 1

        # RollingStats needs MIN_BACKGROUND_SAMPLES RMS values before
        # background_db is reliable. Skip tier decisions during warmup.
        if self._frames_analyzed < self._stats.MIN_BACKGROUND_SAMPLES:
            return

        # Aggregate steady-state.
        agg_snr = float(np.mean(self._snr_history))
        agg_flatness = float(np.mean(self._flatness_history))
        agg_background_db = float(np.mean(self._background_db_history))
        agg_temporal_variance = self._last_temporal_variance

        verdict = assess_quality(
            background_db=agg_background_db,
            spectral_flatness=agg_flatness,
            snr=agg_snr,
            temporal_variance=agg_temporal_variance,
            threshold_offset_db=self.threshold_offset_db,
        )

        self._latest_assessment = LiveAssessment(
            quality=verdict.quality,
            reason=verdict.reason,
            background_db=agg_background_db,
            snr=agg_snr,
            spectral_flatness=agg_flatness,
            temporal_variance=agg_temporal_variance,
            frames_analyzed=self._frames_analyzed,
        )

        self._update_stability(verdict.quality)

    def _update_stability(self, new_tier: str) -> None:
        """Track how many consecutive frames the new_tier has appeared.

        If the streak reaches stability_frames and differs from
        _current_tier, promote it and fire the callback.
        """
        if new_tier == self._candidate_tier:
            self._candidate_streak += 1
        else:
            self._candidate_tier = new_tier
            self._candidate_streak = 1

        # First-ever assessment: skip streak check, promote immediately.
        if self._current_tier is None:
            if self._candidate_streak >= self.stability_frames:
                self._current_tier = new_tier
                self._fire_callback()
            return

        # Subsequent transitions: require the streak.
        if (
            self._candidate_streak >= self.stability_frames
            and new_tier != self._current_tier
        ):
            self._current_tier = new_tier
            self._fire_callback()

    def _fire_callback(self) -> None:
        if self._on_change_callback is not None and self._latest_assessment is not None:
            self._on_change_callback(self._latest_assessment)

    def get_current(self) -> Optional[LiveAssessment]:
        """Return the most recent assessment, or None before any frame processed."""
        return self._latest_assessment

    def reset(self) -> None:
        """Clear all state (buffer, history, hysteresis, callback stays)."""
        self._buffer = np.zeros(0, dtype=np.float32)
        self._stats.reset()
        self._snr_history.clear()
        self._flatness_history.clear()
        self._background_db_history.clear()
        self._last_temporal_variance = 0.0
        self._current_tier = None
        self._candidate_tier = None
        self._candidate_streak = 0
        self._latest_assessment = None
        self._frames_analyzed = 0
