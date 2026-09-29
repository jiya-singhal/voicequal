"""Quality tier assessment for voicequal.

Implements SNR-gated tier logic: use SNR as a "voice dominance" gate
before applying room-loudness penalties. When voice dominates (high
SNR), the room can be loud and it's still OK. When voice is weak (low
SNR), a composite score of four metrics decides the tier.
"""

from dataclasses import dataclass
from typing import Literal

Quality = Literal["excellent", "good", "fair", "poor"]

# SNR-estimate ladder (v0.3.0). Fitted jointly on VoiceBank-DEMAND (speech,
# 824 clips) and the VocalSet+MUSAN set (singing, 200 clips); see the README
# benchmark section. The estimate is max(hnr, energy_snr).
QUIET_ROOM_DB: float = 60.0
SNR_EXCELLENT_DB: float = 18.5
SNR_GOOD_DB: float = 13.5
SNR_FAIR_DB: float = 10.0

# v0.2.0 HNR-only ladder, used when only ``hnr`` is passed.
HNR_EXCELLENT_DB: float = 14.5
HNR_GOOD_DB: float = 11.0
HNR_FAIR_DB: float = 7.0


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
    spectral_concentration: float = 1.0,
    threshold_offset_db: float = 0.0,
    hnr: float | None = None,
    snr_estimate: float | None = None,
) -> QualityAssessment:
    """Assess overall audio quality from the aggregated metrics.

    Three decision paths, most recent first:

    * **SNR-estimate ladder (v0.3.0, used when ``snr_estimate`` is given).**
      A quiet room is excellent regardless. Otherwise the mixing-SNR
      estimate, max(HNR, energy SNR), is read against the 18.5 / 13.5 / 10
      dB ladder. Works for both speech (pauses give the energy SNR) and
      sustained singing (voicing gives the HNR).
    * **HNR ladder (v0.2.0, used when only ``hnr`` is given).** Same shape
      with the 14.5 / 11 / 7 dB thresholds tuned on singing alone.
    * **Spectral-SNR-gated (v0.1.x, used when neither is given).** Kept for
      backward compatibility. High spectral SNR means voice dominates;
      low SNR falls through to a composite score.

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
        hnr: Aggregated harmonic-to-noise ratio in dB (see metrics.hnr).
            Selects the v0.2.0 HNR ladder when ``snr_estimate`` is absent.
        snr_estimate: Aggregated mixing-SNR estimate in dB, normally
            max(hnr, energy_snr). Selects the v0.3.0 ladder.

    Returns:
        A QualityAssessment describing the verdict.
    """
    if snr_estimate is not None:
        return _assess_ladder(
            background_db=background_db,
            value=snr_estimate,
            thresholds=(SNR_EXCELLENT_DB, SNR_GOOD_DB, SNR_FAIR_DB),
            label="snr",
            threshold_offset_db=threshold_offset_db,
        )
    if hnr is not None:
        return _assess_ladder(
            background_db=background_db,
            value=hnr,
            thresholds=(HNR_EXCELLENT_DB, HNR_GOOD_DB, HNR_FAIR_DB),
            label="hnr",
            threshold_offset_db=threshold_offset_db,
        )
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


def _assess_ladder(
    background_db: float,
    value: float,
    thresholds: tuple[float, float, float],
    label: str,
    threshold_offset_db: float,
) -> QualityAssessment:
    """Quiet-room gate, then a three-step ladder on ``value`` (dB)."""
    excellent_db, good_db, fair_db = thresholds

    def _verdict(quality: Quality, reason: str) -> QualityAssessment:
        return QualityAssessment(
            quality=quality,
            primary_score=0.0,
            secondary_score=0.0,
            total_score=0.0,
            reason=reason,
        )

    quiet_room = QUIET_ROOM_DB - threshold_offset_db
    if background_db < quiet_room:
        return _verdict("excellent", f"bgDB<{quiet_room:g}: quiet room")
    if value >= excellent_db:
        return _verdict("excellent", f"{label}>={excellent_db:g}: voice dominates noise")
    if value >= good_db:
        return _verdict("good", f"{good_db:g}<={label}<{excellent_db:g}: mild noise under voice")
    if value >= fair_db:
        return _verdict("fair", f"{fair_db:g}<={label}<{good_db:g}: noise competes with voice")
    return _verdict("poor", f"{label}<{fair_db:g} in a non-quiet room: noise dominates")
