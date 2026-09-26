"""Tests for voicequal.assessment.assess_quality."""

import dataclasses

import pytest

from voicequal.assessment import assess_quality


class TestSNRFastPaths:
    def test_very_high_snr_is_excellent(self):
        # snr>50 should short-circuit to excellent regardless of room.
        result = assess_quality(
            background_db=80.0,  # very loud room
            spectral_flatness=0.9,  # noise-like
            snr=60.0,
            temporal_variance=1.0,
        )
        assert result.quality == "excellent"
        assert result.total_score == 0.0

    def test_high_snr_with_clean_room_is_excellent(self):
        result = assess_quality(
            background_db=40.0,
            spectral_flatness=0.4,
            snr=40.0,
            temporal_variance=6.0,
        )
        assert result.quality == "excellent"

    def test_high_snr_with_very_loud_room_downgrades_to_good(self):
        result = assess_quality(
            background_db=75.0,  # >72
            spectral_flatness=0.5,
            snr=40.0,  # 35<snr<=50
            temporal_variance=6.0,
        )
        assert result.quality == "good"

    def test_moderate_snr_with_loud_room_is_fair(self):
        result = assess_quality(
            background_db=71.0,  # >70
            spectral_flatness=0.5,
            snr=30.0,  # 25<snr<=35
            temporal_variance=6.0,
        )
        assert result.quality == "fair"

    def test_moderate_snr_with_moderate_room_is_good(self):
        result = assess_quality(
            background_db=65.0,  # 60<bg<=70
            spectral_flatness=0.5,
            snr=30.0,
            temporal_variance=6.0,
        )
        assert result.quality == "good"

    def test_moderate_snr_with_quiet_room_is_excellent(self):
        result = assess_quality(
            background_db=45.0,
            spectral_flatness=0.5,
            snr=30.0,
            temporal_variance=6.0,
        )
        assert result.quality == "excellent"


class TestCompositeScore:
    def test_low_snr_quiet_room_is_excellent(self):
        # Low SNR (20) but very quiet room, low flatness, healthy variance
        # -> primary=0, secondary=0, total=0 -> excellent.
        result = assess_quality(
            background_db=40.0,
            spectral_flatness=0.3,
            snr=20.0,
            temporal_variance=8.0,
        )
        assert result.quality == "excellent"
        assert result.primary_score == 0.0
        assert result.secondary_score == 0.0

    def test_low_snr_moderate_room_is_good(self):
        # primary=3 (bg>60), secondary=0 -> total=3 -> good (>=2).
        result = assess_quality(
            background_db=65.0,
            spectral_flatness=0.3,
            snr=20.0,
            temporal_variance=8.0,
        )
        assert result.quality == "good"
        assert result.primary_score == 3.0
        assert result.total_score == 3.0

    def test_low_snr_loud_room_is_fair(self):
        # primary=5 (bg>67), secondary=0 -> total=5 -> fair (>=4).
        result = assess_quality(
            background_db=68.0,
            spectral_flatness=0.3,
            snr=20.0,
            temporal_variance=8.0,
        )
        assert result.quality == "fair"
        assert result.primary_score == 5.0

    def test_very_loud_room_forces_poor(self):
        # primary=7 (bg>72) -> total>=7 -> poor even with no secondary.
        result = assess_quality(
            background_db=75.0,
            spectral_flatness=0.3,
            snr=20.0,
            temporal_variance=8.0,
        )
        assert result.quality == "poor"
        assert result.primary_score == 7.0

    def test_secondary_flatness_penalty(self):
        # bg=65 -> primary=3, flatness=0.85 -> secondary +=1.0, total=4 -> fair.
        result = assess_quality(
            background_db=65.0,
            spectral_flatness=0.85,
            snr=20.0,
            temporal_variance=8.0,
        )
        assert result.quality == "fair"
        assert result.secondary_score >= 1.0

    def test_secondary_snr_penalty(self):
        # bg=65 -> primary=3, snr=8 -> secondary +=1.0, total=4 -> fair.
        result = assess_quality(
            background_db=65.0,
            spectral_flatness=0.3,
            snr=8.0,
            temporal_variance=8.0,
        )
        assert result.quality == "fair"
        assert result.secondary_score >= 1.0

    def test_secondary_temporal_variance_penalty(self):
        # bg=68 -> primary=5, temp_var=2 with bg>67 -> secondary+=1.0
        # -> total=6 -> fair. (Not poor because total is 6, poor needs >=7.)
        result = assess_quality(
            background_db=68.0,
            spectral_flatness=0.3,
            snr=20.0,
            temporal_variance=2.0,
        )
        assert result.quality == "fair"
        assert result.secondary_score >= 1.0

    def test_stacked_secondaries_push_to_poor(self):
        # bg=68 -> primary=5, flatness=0.85 -> +1, snr=8 -> +1
        # -> secondary=2, total=7 -> poor.
        result = assess_quality(
            background_db=68.0,
            spectral_flatness=0.85,
            snr=8.0,
            temporal_variance=8.0,
        )
        assert result.quality == "poor"


class TestResultFields:
    def test_assessment_is_frozen(self):
        result = assess_quality(
            background_db=40.0, spectral_flatness=0.3, snr=60.0, temporal_variance=6.0
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            # frozen dataclass -> cannot mutate
            result.quality = "poor"  # type: ignore[misc]

    def test_reason_is_populated(self):
        result = assess_quality(
            background_db=40.0, spectral_flatness=0.3, snr=60.0, temporal_variance=6.0
        )
        assert len(result.reason) > 0


class TestThresholdOffset:
    def test_offset_makes_algorithm_stricter(self):
        # Room at 62 dBA with SNR=20:
        # Default: primary=3 (bg>60) -> good
        # With offset=8: bg>60-8=52 -> primary=3, but bg>67-8=59 -> primary=5 -> fair
        default_result = assess_quality(
            background_db=62.0,
            spectral_flatness=0.3,
            snr=20.0,
            temporal_variance=8.0,
        )
        sensitive_result = assess_quality(
            background_db=62.0,
            spectral_flatness=0.3,
            snr=20.0,
            temporal_variance=8.0,
            threshold_offset_db=8.0,
        )
        assert default_result.quality in {"good", "excellent"}
        assert sensitive_result.total_score > default_result.total_score

    def test_zero_offset_is_default_behavior(self):
        # Explicit offset=0 must equal default.
        a = assess_quality(65.0, 0.3, 20.0, 8.0)
        b = assess_quality(65.0, 0.3, 20.0, 8.0, threshold_offset_db=0.0)
        assert a.quality == b.quality
        assert a.total_score == b.total_score


class TestConcentrationGate:
    def test_high_snr_with_low_concentration_no_longer_excellent(self):
        # Old behavior: snr=60 -> excellent unconditionally
        # New behavior: snr=60 + concentration=0.1 (noise-like) -> falls through
        # the gated fast-paths into the composite; a loud room then scores poor.
        result = assess_quality(
            background_db=75.0,  # loud room
            spectral_flatness=0.5,
            snr=60.0,
            temporal_variance=8.0,
            spectral_concentration=0.1,  # LOW: noise-like
        )
        assert result.quality != "excellent", (
            f"Expected fall-through, got excellent. Reason: {result.reason}"
        )

    def test_high_snr_with_high_concentration_still_excellent(self):
        result = assess_quality(
            background_db=55.0,
            spectral_flatness=0.5,
            snr=60.0,
            temporal_variance=8.0,
            spectral_concentration=0.6,  # HIGH: voice-like
        )
        assert result.quality == "excellent"

    def test_default_concentration_preserves_old_behavior(self):
        # Existing callers don't pass concentration -> default 1.0 -> old behavior
        result = assess_quality(
            background_db=55.0,
            spectral_flatness=0.5,
            snr=60.0,
            temporal_variance=8.0,
        )
        assert result.quality == "excellent"


class TestHNRGatedPath:
    """v0.2.0 decision path, selected whenever hnr is provided."""

    def _run(self, background_db: float, hnr: float, **kw):
        return assess_quality(
            background_db=background_db,
            spectral_flatness=0.3,
            snr=40.0,
            temporal_variance=5.0,
            spectral_concentration=0.5,
            hnr=hnr,
            **kw,
        )

    def test_quiet_room_is_excellent_regardless_of_hnr(self):
        result = self._run(background_db=45.0, hnr=-20.0)
        assert result.quality == "excellent"
        assert "quiet room" in result.reason

    def test_high_hnr_is_excellent(self):
        assert self._run(background_db=73.0, hnr=20.0).quality == "excellent"

    def test_mid_hnr_is_good(self):
        assert self._run(background_db=73.0, hnr=12.0).quality == "good"

    def test_low_hnr_is_fair(self):
        assert self._run(background_db=73.0, hnr=8.0).quality == "fair"

    def test_very_low_hnr_in_loud_room_is_poor(self):
        assert self._run(background_db=73.0, hnr=2.0).quality == "poor"

    def test_thresholds_are_inclusive_at_boundaries(self):
        from voicequal.assessment import HNR_EXCELLENT_DB, HNR_FAIR_DB, HNR_GOOD_DB

        assert self._run(73.0, HNR_EXCELLENT_DB).quality == "excellent"
        assert self._run(73.0, HNR_GOOD_DB).quality == "good"
        assert self._run(73.0, HNR_FAIR_DB).quality == "fair"

    def test_sensitive_offset_narrows_quiet_room_gate(self):
        # bgDB 55 is a quiet room by default (<60) but not with a 10 dB offset (<50).
        assert self._run(55.0, hnr=2.0).quality == "excellent"
        assert self._run(55.0, hnr=2.0, threshold_offset_db=10.0).quality == "poor"

    def test_spectral_snr_is_ignored_on_this_path(self):
        # The motivating bug: a buried vowel with high spectral SNR must not
        # ride to excellent when its HNR says noise dominates.
        result = assess_quality(
            background_db=74.0,
            spectral_flatness=0.1,
            snr=45.0,
            temporal_variance=5.0,
            spectral_concentration=0.5,
            hnr=4.0,
        )
        assert result.quality == "poor"

    def test_omitting_hnr_uses_legacy_path(self):
        legacy = assess_quality(
            background_db=74.0,
            spectral_flatness=0.1,
            snr=45.0,
            temporal_variance=5.0,
            spectral_concentration=0.5,
        )
        assert legacy.reason.startswith("35<snr<=50")
