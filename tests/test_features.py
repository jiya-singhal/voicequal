"""Tests for voicequal.features (clip feature vector for the quality model)."""

import numpy as np
import pytest

from voicequal.features import (
    FEATURE_NAMES,
    N_FEATURES,
    N_MELS,
    clip_features,
    log_mel_spectrogram,
    mel_filterbank,
)


def _tone(freq=220.0, seconds=2.0, sr=16000, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestMelFilterbank:
    def test_shape_and_coverage(self):
        bank = mel_filterbank()
        assert bank.shape == (N_MELS, 257)
        assert np.all(bank.sum(axis=1) > 0)  # every band has support
        assert np.all(bank >= 0)

    def test_band_centres_increase(self):
        bank = mel_filterbank()
        centres = bank.argmax(axis=1)
        assert np.all(np.diff(centres) >= 0)


class TestLogMel:
    def test_shape_for_two_seconds(self):
        mel = log_mel_spectrogram(_tone())
        # 1 + (32000 - 400) // 160 frames
        assert mel.shape == (1 + (32000 - 400) // 160, N_MELS)

    def test_short_clip_gives_one_frame(self):
        assert log_mel_spectrogram(np.zeros(100, dtype=np.float32)).shape == (1, N_MELS)

    def test_tone_energy_lands_in_the_right_band(self):
        mel = log_mel_spectrogram(_tone(freq=1000.0)).mean(axis=0)
        bank = mel_filterbank()
        freqs = np.linspace(0, 8000, 257)
        peak_band = int(mel.argmax())
        band_centre = freqs[bank[peak_band].argmax()]
        assert abs(band_centre - 1000.0) < 250.0

    def test_floor_is_minus_100(self):
        mel = log_mel_spectrogram(np.zeros(16000, dtype=np.float32))
        assert np.all(mel == -100.0)


class TestClipFeatures:
    def test_length_names_and_finite(self):
        f = clip_features(_tone())
        assert f.shape == (N_FEATURES,)
        assert len(FEATURE_NAMES) == N_FEATURES == 9 + 4 * N_MELS
        assert f.dtype == np.float32
        assert np.all(np.isfinite(f))

    def test_first_features_match_assess_samples(self):
        from voicequal import assess_samples

        clip = _tone()
        f = clip_features(clip)
        r = assess_samples(clip, 16000)
        assert f[0] == pytest.approx(r.snr_estimate, rel=1e-5)
        assert f[1] == pytest.approx(r.hnr, rel=1e-5)
        assert f[3] == pytest.approx(r.background_db, rel=1e-5)

    def test_noise_raises_mel_std_less_than_bursty_signal(self):
        rng = np.random.default_rng(0)
        steady = rng.standard_normal(32000).astype(np.float32) * 0.1
        t = np.arange(32000) / 16000
        bursty = (steady * ((t % 0.5) < 0.25)).astype(np.float32)
        std_idx = [i for i, n in enumerate(FEATURE_NAMES) if n.endswith("_std")]
        assert clip_features(bursty)[std_idx].mean() > clip_features(steady)[std_idx].mean()

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            clip_features(np.zeros(100, dtype=np.float32))
