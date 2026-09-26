"""Tests for voicequal.metrics."""

import numpy as np
import pytest

from voicequal.metrics import (
    HNR_FLOOR_DB,
    clipping_ratio,
    harmonic_ratio,
    hnr,
    noise_floor,
    rms,
    snr,
    spectral_concentration,
    spectral_flatness,
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


def _sine(freq: float, n: int = 2048, sr: int = 16000, amp: float = 0.5) -> np.ndarray:
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    sig_rms = np.sqrt(np.mean(signal**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    gain = (sig_rms / (10 ** (snr_db / 20))) / noise_rms
    return (signal + gain * noise).astype(np.float32)


class TestHarmonicRatio:
    def test_silence_returns_zero(self):
        assert harmonic_ratio(np.zeros(2048, dtype=np.float32)) == 0.0

    def test_empty_returns_zero(self):
        assert harmonic_ratio(np.array([], dtype=np.float32)) == 0.0

    def test_pure_sine_is_near_one(self):
        assert harmonic_ratio(_sine(220.0)) > 0.95

    def test_white_noise_is_low(self):
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(2048).astype(np.float32)
        assert harmonic_ratio(noise) < 0.3

    def test_bounded_below_one(self):
        assert harmonic_ratio(_sine(440.0)) < 1.0


class TestHNR:
    def test_silence_returns_floor(self):
        assert hnr(np.zeros(2048, dtype=np.float32)) == HNR_FLOOR_DB

    def test_pure_sine_is_high(self):
        assert hnr(_sine(220.0)) > 25.0

    def test_white_noise_is_below_zero(self):
        rng = np.random.default_rng(1)
        noise = rng.standard_normal(2048).astype(np.float32)
        assert hnr(noise) < 0.0

    @pytest.mark.parametrize("target_snr", [20.0, 10.0, 5.0])
    def test_tracks_mixing_snr_of_tone_plus_noise(self, target_snr):
        # For a periodic signal plus white noise, HNR should approximate
        # the mixing SNR to within a few dB (r_max ~= Ps / (Ps + Pn)).
        rng = np.random.default_rng(2)
        noise = rng.standard_normal(2048).astype(np.float32)
        mixed = _mix_at_snr(_sine(200.0), noise, target_snr)
        assert hnr(mixed) == pytest.approx(target_snr, abs=3.0)

    def test_monotonic_in_snr(self):
        rng = np.random.default_rng(3)
        noise = rng.standard_normal(2048).astype(np.float32)
        tone = _sine(200.0)
        values = [hnr(_mix_at_snr(tone, noise, s)) for s in (30.0, 20.0, 10.0, 0.0)]
        assert values == sorted(values, reverse=True)

    def test_unlike_spectral_snr_it_drops_when_tone_is_buried(self):
        # The motivating case: spectral snr() stays high for a tone in
        # heavy noise because the peak bin still dominates; hnr() drops.
        rng = np.random.default_rng(4)
        noise = rng.standard_normal(2048).astype(np.float32)
        clean = _sine(200.0)
        buried = _mix_at_snr(clean, noise, 5.0)
        assert hnr(clean) - hnr(buried) > 15.0
        assert snr(buried) > 20.0  # spectral SNR still reads "clean-ish"


class TestClippingRatio:
    def test_empty_is_zero(self):
        assert clipping_ratio(np.array([], dtype=np.float32)) == 0.0

    def test_clean_signal_is_zero(self):
        assert clipping_ratio(_sine(440.0, amp=0.5)) == 0.0

    def test_hard_clipped_square_wave_is_one(self):
        frame = np.sign(_sine(440.0)).astype(np.float32)
        assert clipping_ratio(frame) == pytest.approx(1.0, abs=1e-3)

    def test_partial_clipping_counts_fraction(self):
        frame = np.clip(_sine(440.0, amp=2.0), -1.0, 1.0)
        ratio = clipping_ratio(frame)
        assert 0.3 < ratio < 0.8
