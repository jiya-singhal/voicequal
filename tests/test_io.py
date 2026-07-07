"""Tests for voicequal.io.load_audio."""

import os
import tempfile

import numpy as np
import soundfile as sf

from voicequal.io import load_audio


def test_load_mono_at_target_rate():
    """A mono 16kHz sine wave loads back as 1D audio at 16000 with the same length."""
    sample_rate = 16000
    duration = 1.0
    n_samples = int(sample_rate * duration)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        sf.write(tmp.name, sine, sample_rate)
        samples, sr = load_audio(tmp.name, target_sample_rate=16000)

        assert samples.ndim == 1
        assert sr == 16000
        assert len(samples) == n_samples
    finally:
        os.remove(tmp.name)


def test_load_stereo_averages_to_mono():
    """A stereo file with left=+0.5 and right=-0.5 averages to a mono signal near 0."""
    sample_rate = 16000
    n_samples = sample_rate  # 1 second
    left = np.full(n_samples, 0.5, dtype=np.float32)
    right = np.full(n_samples, -0.5, dtype=np.float32)
    stereo = np.stack([left, right], axis=1)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        sf.write(tmp.name, stereo, sample_rate)
        samples, sr = load_audio(tmp.name, target_sample_rate=16000)

        assert samples.ndim == 1
        assert sr == 16000
        assert np.allclose(samples, 0.0, atol=1e-4)
    finally:
        os.remove(tmp.name)


def test_resample_downsamples_correctly():
    """A 48kHz sine wave downsampled to 16kHz has output rate 16000 and ~1/3 the length."""
    source_rate = 48000
    duration = 1.0
    n_samples = int(source_rate * duration)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        sf.write(tmp.name, sine, source_rate)
        samples, sr = load_audio(tmp.name, target_sample_rate=16000)

        assert sr == 16000
        expected_length = n_samples // 3
        assert abs(len(samples) - expected_length) <= 1
    finally:
        os.remove(tmp.name)
