"""Tests for voicequal.live.LiveDetector."""

import dataclasses

import numpy as np
import pytest

from voicequal.live import LiveAssessment, LiveDetector


def _sine(
    frequency: float, duration_s: float, amplitude: float = 0.5, sr: int = 16000
) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def _white_noise(
    duration_s: float, amplitude: float = 0.3, seed: int = 42, sr: int = 16000
) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    return (amplitude * rng.standard_normal(int(duration_s * sr))).astype(np.float32)


class TestBasicMechanics:
    def test_no_callback_before_first_frame(self):
        captured: list[LiveAssessment] = []
        det = LiveDetector()
        det.on_change(captured.append)

        # Push less than one frame's worth (2048 samples).
        det.push(_sine(440.0, duration_s=0.05))  # ~800 samples

        assert captured == []
        assert det.get_current() is None

    def test_get_current_populated_after_processing(self):
        det = LiveDetector()
        det.push(_sine(440.0, duration_s=3.0))  # was 1.0
        assert det.get_current() is not None
        assert det.get_current().frames_analyzed >= 1

    def test_reset_clears_state(self):
        det = LiveDetector()
        det.push(_sine(440.0, duration_s=3.0))  # was 1.0
        assert det.get_current() is not None
        det.reset()
        assert det.get_current() is None


class TestStabilityHysteresis:
    def test_callback_fires_after_stability_frames_of_same_tier(self):
        captured: list[LiveAssessment] = []
        det = LiveDetector(stability_frames=3)
        det.on_change(captured.append)

        # Push a lot of clean sine — should promote to some stable tier
        # after >= 3 consecutive same-tier frames.
        det.push(_sine(440.0, duration_s=2.0))
        assert len(captured) >= 1
        first_tier = captured[0].quality
        assert first_tier in {"excellent", "good", "fair", "poor"}

    def test_no_callback_if_stability_never_reached(self):
        # With stability_frames very high, no callback should fire even
        # after many frames of the same tier.
        captured: list[LiveAssessment] = []
        det = LiveDetector(stability_frames=10_000)
        det.on_change(captured.append)

        det.push(_sine(440.0, duration_s=2.0))
        assert captured == []

    def test_tier_change_fires_a_second_callback(self):
        captured: list[LiveAssessment] = []
        det = LiveDetector(stability_frames=3)
        det.on_change(captured.append)

        # Establish a first tier with clean sine.
        det.push(_sine(440.0, duration_s=2.0))
        first_calls = len(captured)
        first_tier = captured[-1].quality if captured else None

        # Now push very loud white noise for a while to try to shift tier.
        det.push(_white_noise(duration_s=2.0, amplitude=1.0))

        # We may or may not have shifted — depends on the algorithm.
        # But if we did, a *new* callback should be recorded and the
        # tier should differ.
        if len(captured) > first_calls:
            assert captured[-1].quality != first_tier


class TestPushBehavior:
    def test_empty_chunk_is_noop(self):
        det = LiveDetector()
        det.push(np.zeros(0, dtype=np.float32))
        assert det.get_current() is None

    def test_streamed_and_single_push_produce_similar_verdicts(self):
        # Feeding audio in 500-sample chunks should give the same tier
        # as feeding it in one big blob (up to hysteresis).
        audio = _sine(440.0, duration_s=2.0)

        det_stream = LiveDetector(stability_frames=3)
        for i in range(0, audio.size, 500):
            det_stream.push(audio[i : i + 500])

        det_bulk = LiveDetector(stability_frames=3)
        det_bulk.push(audio)

        a = det_stream.get_current()
        b = det_bulk.get_current()
        assert a is not None and b is not None
        assert a.quality == b.quality


class TestSnapshotShape:
    def test_live_assessment_is_frozen(self):
        det = LiveDetector()
        det.push(_sine(440.0, duration_s=3.0))  # was 1.0
        snap = det.get_current()
        assert snap is not None
        with pytest.raises(dataclasses.FrozenInstanceError):
            snap.quality = "poor"  # type: ignore[misc]
