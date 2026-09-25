"""Quality tier assessment for voicequal.

Implements SNR-gated tier logic: use SNR as a "voice dominance" gate
before applying room-loudness penalties. When voice dominates (high
SNR), the room can be loud and it's still OK. When voice is weak (low
SNR), a composite score of four metrics decides the tier.
"""

from dataclasses import dataclass
from typing import Literal

Quality = Literal["excellent", "good", "fair", "poor"]


@dataclass(frozen=True)
class QualityAssessment:
    """Structured result of a quality assessment call.

    Fields:
        quality: One of "excellent" | "good" | "fair" | "poor".
        primary_score: The backgroundDB-tier score (0-7). Only
            populated when the low-SNR composite path is taken;
            0 in the high-SNR fast paths.
        secondary_score: The spectral/SNR/variance score (0-3).
            Same rule as primary_score.
        total_score: primary_score + secondary_score.
        reason: A short human-readable string describing which
            branch of the assessment logic fired.
    """

    quality: Quality
    primary_score: float
    secondary_score: float
    total_score: float
    reason: str


def assess_quality(
    background_db: float,
    spectral_flatness: float,
    snr: float,
    temporal_variance: float,
    spectral_concentration: float = 1.0,  # NEW; default preserves old behavior
    threshold_offset_db: float = 0.0,
) -> QualityAssessment:
    """Assess overall audio quality from four metrics.

    Uses SNR-gated tiered assessment. High SNR means voice dominates
    background noise; low SNR falls through to a composite score.

    Args:
        background_db: Estimated background loudness in a dBA-like scale
            (see RollingStats.background_db). Typical range 20-90.
        spectral_flatness: 0..1 flatness of the audio spectrum.
            See metrics.spectral_flatness.
        snr: Signal-to-noise ratio in dB. See metrics.snr.
        temporal_variance: Std-dev of the noise-floor history in dB.
            See RollingStats.temporal_variance.
        threshold_offset_db: Subtracted from each background_db comparison
            threshold. A positive offset makes the algorithm stricter
            (fires on quieter rooms). Default 0.0.

    Returns:
        A QualityAssessment describing the verdict.
    """
    # SNR-gated fast paths (the "voice dominates" shortcut):

    # CASE 1: Very high SNR (>50) -> excellent, no matter what.
    if snr > 50 and spectral_concentration > 0.4:
        return QualityAssessment(
            quality="excellent",
            primary_score=0.0,
            secondary_score=0.0,
            total_score=0.0,
            reason="snr>50 with concentration>0.4: voice dominates completely",
        )

    # CASE 2: High SNR (35-50).
    if snr > 35 and spectral_concentration > 0.3:
        if background_db > (72 - threshold_offset_db):
            return QualityAssessment(
                quality="good",
                primary_score=0.0,
                secondary_score=0.0,
                total_score=0.0,
                reason="35<snr<=50 with concentration>0.3 but bgDB>72: possible noise mixed with voice",
            )
        return QualityAssessment(
            quality="excellent",
            primary_score=0.0,
            secondary_score=0.0,
            total_score=0.0,
            reason="35<snr<=50 with concentration>0.3, clean room: excellent",
        )

    # CASE 3: Moderate SNR (25-35).
    if snr > 25 and spectral_concentration > 0.2:
        if background_db > (70 - threshold_offset_db):
            return QualityAssessment(
                quality="fair",
                primary_score=0.0,
                secondary_score=0.0,
                total_score=0.0,
                reason="25<snr<=35 and bgDB>70: noisy environment",
            )
        if background_db > (60 - threshold_offset_db):
            return QualityAssessment(
                quality="good",
                primary_score=0.0,
                secondary_score=0.0,
                total_score=0.0,
                reason="25<snr<=35 with moderate room: good",
            )
        return QualityAssessment(
            quality="excellent",
            primary_score=0.0,
            secondary_score=0.0,
            total_score=0.0,
            reason="25<snr<=35 with quiet room: excellent",
        )

    # CASE 4: Low SNR (<=25). Apply the composite score.

    # Primary: backgroundDB tiers (max 7 points).
    if background_db > (72 - threshold_offset_db):
        primary = 7.0
    elif background_db > (67 - threshold_offset_db):
        primary = 5.0
    elif background_db > (60 - threshold_offset_db):
        primary = 3.0
    elif background_db > (50 - threshold_offset_db):
        primary = 1.0
    else:
        primary = 0.0

    # Secondary: three validators (max 3 points total).
    secondary = 0.0

    # Flatness: high flatness suggests noise.
    if spectral_flatness > 0.8:
        secondary += 1.0
    elif spectral_flatness > 0.6:
        secondary += 0.5

    # SNR: extremely low SNR is a strong "poor" signal.
    if snr < 10:
        secondary += 1.0
    elif snr < 15:
        secondary += 0.5

    # Temporal variance: low variance in a loud room = sustained noise.
    if temporal_variance < 3 and background_db > (67 - threshold_offset_db):
        secondary += 1.0
    elif temporal_variance < 5 and background_db > (60 - threshold_offset_db):
        secondary += 0.5

    total = primary + secondary

    # Total score -> tier.
    if total >= 7:
        quality: Quality = "poor"
    elif total >= 4:
        quality = "fair"
    elif total >= 2:
        quality = "good"
    else:
        quality = "excellent"

    return QualityAssessment(
        quality=quality,
        primary_score=primary,
        secondary_score=secondary,
        total_score=total,
        reason=f"low-snr composite: primary={primary}, secondary={secondary}",
    )
