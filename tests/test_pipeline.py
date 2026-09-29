"""Tests for voicequal.pipeline.assess."""

import dataclasses
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from voicequal.pipeline import FileAssessment, assess, assess_samples


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


class TestHNRFields:
    def test_pure_sine_has_high_hnr_and_no_clipping(self):
        sample_rate = 16000
        t = np.linspace(0, 2, 2 * sample_rate, endpoint=False)
        samples = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        path = _write_wav(samples, sample_rate)
        try:
            result = assess(path)
            assert result.hnr > 25.0
            assert result.clipping_ratio == 0.0
            assert result.quality == "excellent"
        finally:
            os.unlink(path)

    def test_tone_buried_in_loud_noise_is_poor(self):
        # 5 dB mixing SNR, loud enough not to be a quiet room. v0.1.1 called
        # this kind of clip excellent because the spectral peak still wins.
        sample_rate = 16000
        rng = np.random.default_rng(0)
        t = np.linspace(0, 3, 3 * sample_rate, endpoint=False)
        tone = 0.3 * np.sin(2 * np.pi * 220 * t)
        noise = rng.standard_normal(t.size)
        noise *= (np.sqrt(np.mean(tone**2)) / 10 ** (5 / 20)) / np.sqrt(np.mean(noise**2))
        samples = (tone + noise).astype(np.float32)
        path = _write_wav(samples, sample_rate)
        try:
            result = assess(path)
            assert result.hnr < 8.0
            assert result.quality == "poor"
        finally:
            os.unlink(path)

    def test_clipped_audio_reports_clipping(self):
        sample_rate = 16000
        t = np.linspace(0, 2, 2 * sample_rate, endpoint=False)
        samples = np.clip(2.0 * np.sin(2 * np.pi * 220 * t), -1.0, 1.0).astype(np.float32)
        path = _write_wav(samples, sample_rate)
        try:
            assert assess(path).clipping_ratio > 0.3
        finally:
            os.unlink(path)


class TestAssessSamples:
    def test_matches_file_path_result(self):
        sample_rate = 16000
        rng = np.random.default_rng(5)
        t = np.linspace(0, 2, 2 * sample_rate, endpoint=False)
        samples = (0.4 * np.sin(2 * np.pi * 330 * t) + 0.02 * rng.standard_normal(t.size)).astype(
            np.float32
        )
        path = _write_wav(samples, sample_rate)
        try:
            via_file = assess(path)
        finally:
            os.unlink(path)
        via_array = assess_samples(samples, sample_rate)
        assert via_array.quality == via_file.quality
        assert via_array.hnr == pytest.approx(via_file.hnr, abs=1e-3)
        assert via_array.background_db == pytest.approx(via_file.background_db, abs=1e-3)

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            assess_samples(np.zeros(100, dtype=np.float32), 16000)


class TestSNREstimate:
    def _write(self, samples):
        return _write_wav(samples.astype(np.float32), 16000)

    def test_speech_like_bursts_in_noise_use_energy_snr(self):
        # Gated tone (pauses) at 20 dB burst SNR: HNR under-reads because
        # half the frames are noise-only, energy SNR reads the mix correctly,
        # and the estimate takes the larger.
        rng = np.random.default_rng(0)
        t = np.arange(3 * 16000) / 16000
        gate = ((t % 0.5) < 0.25).astype(np.float64)
        tone = 0.3 * np.sin(2 * np.pi * 220 * t) * gate
        noise = rng.standard_normal(t.size)
        noise *= np.sqrt(np.mean(tone[gate > 0] ** 2) / 100) / np.sqrt(np.mean(noise**2))
        path = self._write(tone + noise)
        try:
            r = assess(path)
        finally:
            os.unlink(path)
        assert r.energy_snr == pytest.approx(20.0, abs=3.0)
        assert r.snr_estimate == max(r.hnr, r.energy_snr)
        assert r.quality == "excellent"

    def test_sustained_tone_in_noise_uses_hnr(self):
        rng = np.random.default_rng(1)
        t = np.arange(3 * 16000) / 16000
        tone = 0.3 * np.sin(2 * np.pi * 220 * t)
        noise = rng.standard_normal(t.size)
        noise *= (np.sqrt(np.mean(tone**2)) / 10 ** (25 / 20)) / np.sqrt(np.mean(noise**2))
        path = self._write(tone + noise)
        try:
            r = assess(path)
        finally:
            os.unlink(path)
        assert r.hnr > r.energy_snr  # no pauses: energy SNR collapses
        assert r.snr_estimate == r.hnr
        assert r.quality == "excellent"
