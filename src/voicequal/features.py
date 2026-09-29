"""Clip-level feature vector for the distilled quality model.

The vector concatenates two views of a clip:

1. **voicequal's own aggregates** (9 values): the same numbers
   :func:`voicequal.assess_samples` reports. They carry the mixing-SNR
   estimate and room loudness that the tier decision uses.
2. **Log-mel statistics** (4 x 40 values): mean, standard deviation, 10th
   and 90th percentile over time of a 40-band log-mel spectrogram
   (25 ms window, 10 ms hop, 16 kHz). This gives the model the spectral
   shape that the scalar metrics throw away, which is what a P.835
   predictor needs to separate "hiss" from "hum" from "babble".

Everything is numpy. No librosa, no torch.
"""

from __future__ import annotations

import numpy as np

from voicequal.pipeline import assess_samples

SAMPLE_RATE: int = 16000
N_FFT: int = 512  # 32 ms window at 16 kHz; 25 ms of it is used
WIN_LENGTH: int = 400
HOP_LENGTH: int = 160
N_MELS: int = 40
FMIN: float = 20.0
FMAX: float = 8000.0

DSP_FEATURE_NAMES: tuple[str, ...] = (
    "snr_estimate",
    "hnr",
    "energy_snr",
    "background_db",
    "spectral_snr",
    "spectral_flatness",
    "spectral_concentration",
    "temporal_variance",
    "clipping_ratio",
)
MEL_STATS: tuple[str, ...] = ("mean", "std", "p10", "p90")
FEATURE_NAMES: tuple[str, ...] = DSP_FEATURE_NAMES + tuple(
    f"mel{b:02d}_{stat}" for stat in MEL_STATS for b in range(N_MELS)
)
N_FEATURES: int = len(FEATURE_NAMES)  # 9 + 160 = 169

_MEL_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS
) -> np.ndarray:
    """Triangular mel filterbank, shape (n_mels, n_fft // 2 + 1). Cached."""
    key = (sample_rate, n_fft, n_mels)
    if key in _MEL_CACHE:
        return _MEL_CACHE[key]
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_bins)
    mel_points = np.linspace(_hz_to_mel(FMIN), _hz_to_mel(min(FMAX, sample_rate / 2.0)), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bank = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        lo, centre, hi = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        up = (fft_freqs - lo) / max(centre - lo, 1e-9)
        down = (hi - fft_freqs) / max(hi - centre, 1e-9)
        bank[m] = np.clip(np.minimum(up, down), 0.0, None)
    # Slaney-style area normalisation so bands are comparable.
    enorm = 2.0 / (hz_points[2:] - hz_points[:-2])
    bank *= enorm[:, np.newaxis]
    _MEL_CACHE[key] = bank
    return bank


def log_mel_spectrogram(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Log-mel spectrogram in dB, shape (n_frames, N_MELS).

    25 ms Hann window, 10 ms hop, zero-padded to N_FFT. Values are
    10*log10(power) floored at -100 dB. A clip shorter than one window
    yields a single frame.
    """
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if x.size < WIN_LENGTH:
        x = np.pad(x, (0, WIN_LENGTH - x.size))
    n_frames = 1 + (x.size - WIN_LENGTH) // HOP_LENGTH
    idx = np.arange(WIN_LENGTH)[np.newaxis, :] + HOP_LENGTH * np.arange(n_frames)[:, np.newaxis]
    frames = x[idx] * np.hanning(WIN_LENGTH)[np.newaxis, :]
    spec = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2
    mel = spec @ mel_filterbank(sample_rate).T
    return 10.0 * np.log10(np.maximum(mel, 1e-10))


def clip_features(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """The N_FEATURES-long float32 feature vector for one mono clip.

    Order matches :data:`FEATURE_NAMES`. Raises ValueError (from
    assess_samples) if the clip is shorter than one analysis frame.
    """
    result = assess_samples(samples, sample_rate)
    dsp = np.array(
        [
            result.snr_estimate,
            result.hnr,
            result.energy_snr,
            result.background_db,
            result.snr,
            result.spectral_flatness,
            result.spectral_concentration,
            result.temporal_variance,
            result.clipping_ratio,
        ],
        dtype=np.float64,
    )
    mel = log_mel_spectrogram(samples, sample_rate)
    stats = np.concatenate(
        [
            mel.mean(axis=0),
            mel.std(axis=0),
            np.percentile(mel, 10, axis=0),
            np.percentile(mel, 90, axis=0),
        ]
    )
    return np.concatenate([dsp, stats]).astype(np.float32)
