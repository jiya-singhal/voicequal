"""Tests for voicequal.pipeline.assess."""

import dataclasses
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from voicequal.pipeline import FileAssessment, assess


def _write_wav(samples: np.ndarray, sample_rate: int) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)  # noqa: SIM115
    f.close()
    sf.write(f.name, samples, sample_rate)
    return f.name


class TestAssessOnSyntheticAudio:
    def test_pure_sine_is_excellent(self):
        # 3 seconds of clean 440Hz sine at 16kHz. Should get high SNR
        # -> assess_quality falls into the >50 fast path -> excellent.
        sample_rate = 16000
        t = np.linspace(0, 3, 3 * sample_rate, endpoint=False)
        samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        path = _write_wav(samples, sample_rate)
        try:
            result = assess(path)
            assert isinstance(result, FileAssessment)
            assert result.quality == "excellent"
            assert result.snr > 40.0
            assert result.duration_seconds == pytest.approx(3.0, abs=0.01)
            assert result.sample_rate == sample_rate
            assert result.num_frames > 20  # ~30 frames expected
        finally:
            os.unlink(path)

    def test_loud_white_noise_is_not_excellent(self):
        # 3 seconds of loud white noise. Should have low SNR and high
        # spectral flatness -> tier should be worse than "excellent".
        sample_rate = 16000
        rng = np.random.default_rng(seed=42)
        samples = (0.3 * rng.standard_normal(3 * sample_rate)).astype(np.float32)

        path = _write_wav(samples, sample_rate)
        try:
            result = assess(path)
            # White noise should NOT be graded excellent.
            assert result.quality != "excellent"
            # And spectral flatness should be genuinely high.
            assert result.spectral_flatness > 0.3
        finally:
            os.unlink(path)

    def test_resamples_to_target_rate(self):
        # Write a 48kHz file, ask for 16kHz analysis. Report should show 16kHz.
        sample_rate_in = 48000
        t = np.linspace(0, 2, 2 * sample_rate_in, endpoint=False)
        samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        path = _write_wav(samples, sample_rate_in)
        try:
            result = assess(path, target_sample_rate=16000)
            assert result.sample_rate == 16000
            assert result.duration_seconds == pytest.approx(2.0, abs=0.05)
        finally:
            os.unlink(path)


class TestAssessErrors:
    def test_audio_shorter_than_frame_raises(self):
        # A frame is 2048 samples at 16kHz. Give it 500 samples -> too short.
        sample_rate = 16000
        samples = np.zeros(500, dtype=np.float32)
        path = _write_wav(samples, sample_rate)
        try:
            with pytest.raises(ValueError, match="Audio too short"):
                assess(path)
        finally:
            os.unlink(path)


class TestFileAssessmentShape:
    def test_result_is_frozen(self):
        sample_rate = 16000
        t = np.linspace(0, 2, 2 * sample_rate, endpoint=False)
        samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        path = _write_wav(samples, sample_rate)
        try:
            result = assess(path)
            with pytest.raises(dataclasses.FrozenInstanceError):
                result.quality = "poor"  # type: ignore[misc]
        finally:
            os.unlink(path)

    def test_all_scores_are_finite(self):
        sample_rate = 16000
        t = np.linspace(0, 2, 2 * sample_rate, endpoint=False)
        samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        path = _write_wav(samples, sample_rate)
        try:
            result = assess(path)
            for field in [
                result.background_db,
                result.snr,
                result.spectral_flatness,
                result.temporal_variance,
                result.primary_score,
                result.secondary_score,
                result.total_score,
                result.duration_seconds,
            ]:
                assert np.isfinite(field), f"non-finite value: {field}"
        finally:
            os.unlink(path)
