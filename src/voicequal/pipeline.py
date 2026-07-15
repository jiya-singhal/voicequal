"""Top-level assess() pipeline for voicequal.

Ties together audio loading, per-frame metric computation,
RollingStats-based background/variance tracking, and the SNR-gated
quality assessment into one call.
"""

from dataclasses import dataclass
from typing import Union
from pathlib import Path

import numpy as np

from voicequal.assessment import (
    QualityAssessment,
    assess_quality,
)
from voicequal.io import load_audio
from voicequal.metrics import (
    noise_floor,
    rms,
    snr,
    spectral_concentration,
    spectral_flatness,
)
from voicequal.state import RollingStats

# 2048-sample frame at 16kHz = 128ms; 1600-sample hop = 100ms
# (~10Hz analysis rate).
FRAME_SIZE: int = 2048
HOP_SIZE: int = 1600
STEADY_STATE_TAIL: int = 10  # Use last N frames for aggregation.


@dataclass(frozen=True)
class FileAssessment:
    """End-to-end audio quality result for a full file.

    Fields:
        quality: The tier (excellent | good | fair | poor).
        reason: Which branch of assess_quality fired.
        background_db: Aggregated dBA-like background loudness.
        snr: Aggregated signal-to-noise ratio in dB.
        spectral_flatness: Aggregated 0..1 flatness.
        spectral_concentration: Aggregated 0..1 top-N energy concentration.
        temporal_variance: The final temporal-variance reading.
        primary_score: Composite score primary component (0 if fast path).
        secondary_score: Composite score secondary component.
        total_score: primary + secondary.
        duration_seconds: Length of the analyzed audio.
        sample_rate: Sample rate the audio was analyzed at.
        num_frames: Number of frames the audio was chopped into.
    """

    quality: str
    reason: str
    background_db: float
    snr: float
    spectral_flatness: float
    spectral_concentration: float
    temporal_variance: float
    primary_score: float
    secondary_score: float
    total_score: float
    duration_seconds: float
    sample_rate: int
    num_frames: int


def _iter_frames(samples: np.ndarray, frame_size: int, hop_size: int):
    """Yield overlapping frames of length frame_size, stepping by hop_size.

    The last (partial) frame is dropped rather than zero-padded.
    """
    n = samples.size
    if n < frame_size:
        return
    for start in range(0, n - frame_size + 1, hop_size):
        yield samples[start : start + frame_size]


def assess(
    path: Union[str, Path],
    target_sample_rate: int = 16000,
    threshold_offset_db: float = 0.0,
) -> FileAssessment:
    """Assess the audio quality of a file end-to-end.

    Pipeline:
      1. Load and mono/resample the audio.
      2. Iterate 2048-sample frames with a 1600-sample hop.
      3. Compute per-frame metrics; update RollingStats.
      4. Aggregate the last STEADY_STATE_TAIL frames (steady state).
      5. Return a FileAssessment with the tier + all metrics.

    Args:
        path: Path to a WAV/FLAC/OGG file.
        target_sample_rate: Sample rate to analyze at (default 16000).

    Returns:
        A FileAssessment.

    Raises:
        ValueError: If the audio is too short to produce a single frame.
    """
    samples, sample_rate = load_audio(str(path), target_sample_rate=target_sample_rate)
    duration_seconds = float(samples.size / sample_rate)

    stats = RollingStats()

    # Per-frame metric buffers.
    rms_values: list[float] = []
    snr_values: list[float] = []
    flatness_values: list[float] = []
    concentration_values: list[float] = []
    noise_floor_values: list[float] = []
    background_db_values: list[float] = []
    temporal_variance_values: list[float] = []

    for frame in _iter_frames(samples, FRAME_SIZE, HOP_SIZE):
        frame_rms = rms(frame)
        frame_snr = snr(frame)
        frame_flatness = spectral_flatness(frame)
        frame_concentration = spectral_concentration(frame)
        frame_noise_floor = noise_floor(frame)

        stats.update(current_noise_floor=frame_noise_floor, current_rms=frame_rms)

        frame_background_db = stats.background_db(current_rms=frame_rms)
        frame_temporal_variance = stats.temporal_variance()

        rms_values.append(frame_rms)
        snr_values.append(frame_snr)
        flatness_values.append(frame_flatness)
        concentration_values.append(frame_concentration)
        noise_floor_values.append(frame_noise_floor)
        background_db_values.append(frame_background_db)
        temporal_variance_values.append(frame_temporal_variance)

    num_frames = len(rms_values)
    if num_frames == 0:
        raise ValueError(
            f"Audio too short: need at least {FRAME_SIZE} samples "
            f"({FRAME_SIZE / target_sample_rate:.2f}s), got {samples.size}."
        )

    # Aggregate steady-state (last N frames).
    tail = min(STEADY_STATE_TAIL, num_frames)
    agg_background_db = float(np.mean(background_db_values[-tail:]))
    agg_snr = float(np.mean(snr_values[-tail:]))
    agg_flatness = float(np.mean(flatness_values[-tail:]))
    agg_concentration = float(np.mean(concentration_values[-tail:]))
    # Temporal variance is already a rolling statistic; use its last reading.
    agg_temporal_variance = float(temporal_variance_values[-1])

    quality: QualityAssessment = assess_quality(
        background_db=agg_background_db,
        spectral_flatness=agg_flatness,
        snr=agg_snr,
        temporal_variance=agg_temporal_variance,
        spectral_concentration=agg_concentration,
        threshold_offset_db=threshold_offset_db,
    )

    return FileAssessment(
        quality=quality.quality,
        reason=quality.reason,
        background_db=agg_background_db,
        snr=agg_snr,
        spectral_flatness=agg_flatness,
        spectral_concentration=agg_concentration,
        temporal_variance=agg_temporal_variance,
        primary_score=quality.primary_score,
        secondary_score=quality.secondary_score,
        total_score=quality.total_score,
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        num_frames=num_frames,
    )
