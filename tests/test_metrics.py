"""Tests for voicequal.metrics."""

import numpy as np
import pytest

from voicequal.metrics import (
    is_voice_active,
    noise_floor,
    rms,
    snr,
    spectral_concentration,
    spectral_flatness,
    voice_activity,
)


class TestRMS:
    def test_silence_is_zero(self):
        frame = np.zeros(1024, dtype=np.float32)
        assert rms(frame) == 0.0

    def test_constant_signal(self):
        # A constant signal of value 0.5 has RMS = 0.5.
        frame = np.full(1024, 0.5, dtype=np.float32)
        assert rms(frame) == pytest.approx(0.5)

    def test_sine_wave(self):
        # A sine wave with amplitude A has RMS = A / sqrt(2).
        # Generate a 440Hz sine at 16kHz sample rate with amplitude 1.0.
        t = np.linspace(0, 1, 16000, endpoint=False)
        frame = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        expected = 1.0 / np.sqrt(2)
        assert rms(frame) == pytest.approx(expected, abs=1e-3)

    def test_empty_frame(self):
        assert rms(np.array([], dtype=np.float32)) == 0.0


class TestSpectralFlatness:
    def test_silence_returns_zero(self):
        frame = np.zeros(1024, dtype=np.float32)
        assert spectral_flatness(frame) == 0.0

    def test_pure_sine_wave_is_peaked(self):
        # A pure sine wave has energy concentrated in one bin.
        # Its spectral flatness should be very LOW (near 0).
        t = np.linspace(0, 1, 16000, endpoint=False)
        frame = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        flatness = spectral_flatness(frame)
        assert flatness < 0.05, f"expected low flatness, got {flatness}"

    def test_white_noise_is_flat(self):
        # White noise has a roughly flat spectrum.
        # Its spectral flatness should be HIGH (well above 0.3).
        rng = np.random.default_rng(seed=42)
        frame = rng.standard_normal(16000).astype(np.float32)
        flatness = spectral_flatness(frame)
        assert flatness > 0.3, f"expected high flatness, got {flatness}"

    def test_bounded_between_0_and_1(self):
        # Flatness must always be in [0, 1].
        rng = np.random.default_rng(seed=7)
        for _ in range(5):
            frame = rng.standard_normal(1024).astype(np.float32)
            f = spectral_flatness(frame)
            assert 0.0 <= f <= 1.0


class TestNoiseFloor:
    def test_silence_returns_minus_100(self):
        frame = np.zeros(2048, dtype=np.float32)
        assert noise_floor(frame) == -100.0

    def test_empty_returns_minus_100(self):
        assert noise_floor(np.array([], dtype=np.float32)) == -100.0

    def test_pure_sine_has_low_noise_floor(self):
        # A pure sine wave has one loud bin and many quiet bins.
        # The 75th percentile should be quiet (well below -20 dB).
        t = np.linspace(0, 1, 16000, endpoint=False)
        frame = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        nf = noise_floor(frame)
        assert nf < -20.0, f"expected low noise floor, got {nf}"

    def test_white_noise_has_higher_noise_floor_than_sine(self):
        # White noise has energy in every bin, so its noise floor
        # should be HIGHER than a pure sine wave's.
        rng = np.random.default_rng(seed=1)
        noise = rng.standard_normal(16000).astype(np.float32) * 0.1
        t = np.linspace(0, 1, 16000, endpoint=False)
        sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        assert noise_floor(noise) > noise_floor(sine)


class TestSNR:
    def test_silence_returns_zero(self):
        frame = np.zeros(2048, dtype=np.float32)
        assert snr(frame) == 0.0

    def test_pure_sine_has_high_snr(self):
        # A pure sine wave: one loud bin, everything else quiet.
        # SNR should be high (>40 dB).
        t = np.linspace(0, 1, 16000, endpoint=False)
        frame = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        assert snr(frame) > 40.0

    def test_white_noise_has_lower_snr_than_sine(self):
        # White noise has similar energy everywhere, so peak - percentile
        # is small.
        rng = np.random.default_rng(seed=2)
        noise = rng.standard_normal(16000).astype(np.float32)
        t = np.linspace(0, 1, 16000, endpoint=False)
        sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        assert snr(noise) < snr(sine)


class TestSpectralConcentration:
    def test_silence_returns_zero(self):
        frame = np.zeros(2048, dtype=np.float32)
        assert spectral_concentration(frame) == 0.0

    def test_pure_sine_is_highly_concentrated(self):
        # A pure sine wave has all its energy in one bin.
        # Concentration should be very high (>0.9).
        t = np.linspace(0, 1, 16000, endpoint=False)
        frame = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        conc = spectral_concentration(frame)
        assert conc > 0.9

    def test_white_noise_is_spread(self):
        # White noise has energy in every bin.
        # Top 3 out of ~1000 bins should be a tiny fraction of total.
        rng = np.random.default_rng(seed=42)
        frame = rng.standard_normal(16000).astype(np.float32)
        conc = spectral_concentration(frame)
        assert conc < 0.1

    def test_bounded_between_0_and_1(self):
        rng = np.random.default_rng(seed=7)
        for _ in range(5):
            frame = rng.standard_normal(1024).astype(np.float32)
            c = spectral_concentration(frame)
            assert 0.0 <= c <= 1.0


class TestVoiceActivity:
    def test_silence_scores_zero(self):
        frame = np.zeros(2048, dtype=np.float32)
        assert voice_activity(frame) == 0.0

    def test_empty_scores_zero(self):
        frame = np.zeros(0, dtype=np.float32)
        assert voice_activity(frame) == 0.0

    def test_tonal_vowel_scores_high(self):
        # A pitched tone (stand-in for a voiced vowel) is tonal and
        # energetic -> should score clearly voice-active.
        t = np.linspace(0, 1, 16000, endpoint=False)
        frame = np.sin(2 * np.pi * 220 * t).astype(np.float32)
        assert voice_activity(frame) > 0.5

    def test_white_noise_scores_zero(self):
        # Broadband noise clears the energy gate but is spread across
        # bins, so concentration is below threshold -> score 0.0.
        rng = np.random.default_rng(seed=42)
        frame = rng.standard_normal(16000).astype(np.float32)
        assert voice_activity(frame) == 0.0

    def test_quiet_frame_below_energy_floor_scores_zero(self):
        # A tone scaled well below the RMS floor is treated as silence.
        t = np.linspace(0, 1, 16000, endpoint=False)
        frame = (1e-5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        assert voice_activity(frame) == 0.0

    def test_bounded_between_0_and_1(self):
        rng = np.random.default_rng(seed=11)
        t = np.linspace(0, 1, 2048, endpoint=False)
        frames = [
            np.zeros(2048, dtype=np.float32),
            np.sin(2 * np.pi * 300 * t).astype(np.float32),
            rng.standard_normal(2048).astype(np.float32),
        ]
        for frame in frames:
            assert 0.0 <= voice_activity(frame) <= 1.0

    def test_is_voice_active_agrees_with_score(self):
        t = np.linspace(0, 1, 16000, endpoint=False)
        vowel = np.sin(2 * np.pi * 220 * t).astype(np.float32)
        silence = np.zeros(16000, dtype=np.float32)
        assert is_voice_active(vowel) is True
        assert is_voice_active(silence) is False
