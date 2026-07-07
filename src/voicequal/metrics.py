"""Frame-level audio metrics for voicequal.

A "frame" is a chunk of audio samples (e.g. 2048 samples at 16kHz = 128ms).
Each metric function takes a frame and returns one scalar float.
"""

import numpy as np


def rms(frame: np.ndarray) -> float:
    """Root Mean Square of an audio frame.

    A time-domain measure of loudness. Returns 0 for a silent frame.

    Args:
        frame: 1D numpy array of audio samples, typically float32 in [-1, 1].

    Returns:
        A non-negative float. 0 means silence, ~0.1 is normal speech,
        values near 1.0 indicate clipping.
    """
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))


def spectral_flatness(frame: np.ndarray) -> float:
    """Spectral flatness of an audio frame.

    A frequency-domain measure of how "noise-like" the audio is.
    Ratio of geometric mean to arithmetic mean of the power spectrum.
    Result is bounded in [0, 1]:
      - Near 0: peaked spectrum (tonal, e.g. pure sine wave)
      - Near 1: flat spectrum (noise-like, e.g. white noise)

    Uses numpy's real FFT (rfft) since audio is real-valued. Computes
    the power spectrum (|FFT|^2). Skips the DC bin (bin 0). Applies a
    small floor (1e-10) to avoid log(0).

    Args:
        frame: 1D numpy array of audio samples.

    Returns:
        A float in [0.0, 1.0]. Returns 0.0 for silent or empty frames.
    """
    if frame.size == 0 or not np.any(frame):
        return 0.0
    power = np.clip(np.abs(np.fft.rfft(frame)) ** 2, 1e-10, None)[1:]
    geometric_mean = np.exp(np.mean(np.log(power)))
    return float(np.clip(geometric_mean / np.mean(power), 0.0, 1.0))


_HANN_CACHE: dict[int, np.ndarray] = {}


def _hann_window(n: int) -> np.ndarray:
    """Return a cached Hann window of length n."""
    if n not in _HANN_CACHE:
        _HANN_CACHE[n] = np.hanning(n).astype(np.float32)
    return _HANN_CACHE[n]


def _power_spectrum_db(frame: np.ndarray, apply_window: bool = True) -> np.ndarray:
    """Return the power spectrum of a frame in dB, skipping the DC bin.

    Args:
        frame: 1D numpy array of audio samples.
        apply_window: If True (default), apply a Hann window before FFT
            to reduce spectral leakage.

    Returns:
        1D numpy array of dB values, length = N/2 (bin 0 dropped).
        Values are clipped at a floor of -100 dB.
    """
    if frame.size == 0:
        return np.array([], dtype=np.float64)
    if apply_window:
        frame = frame * _hann_window(frame.size)
    power = np.abs(np.fft.rfft(frame)) ** 2
    db = 10 * np.log10(np.maximum(power, 1e-10))
    return np.maximum(db, -100.0)[1:]


def noise_floor(frame: np.ndarray, percentile: float = 75.0) -> float:
    """Estimate the noise floor of an audio frame, in dB.

    Uses the Nth percentile of the power spectrum in dB (default 75th).
    Digital silence bins (below -100 dB) are excluded before the
    percentile calculation.

    Args:
        frame: 1D numpy array of audio samples.
        percentile: Percentile in [0, 100]. Default 75.

    Returns:
        A float in dB (typically between -60 and -20 for real audio).
        Returns -100.0 for empty or all-silent frames.
    """
    db = _power_spectrum_db(frame)
    filtered = db[db > -100.0]
    if filtered.size == 0:
        return -100.0
    return float(np.percentile(filtered, percentile))


def snr(frame: np.ndarray) -> float:
    """Signal-to-Noise Ratio in dB.

    SNR = peak_bin_dB - noise_floor_dB. Both computed from the same
    windowed power spectrum. Returns 0.0 for empty or silent frames.

    Args:
        frame: 1D numpy array of audio samples.

    Returns:
        Non-negative float in dB. Typical values: 40+ = clean voice,
        20-40 = moderate noise, <20 = very noisy.
    """
    db = _power_spectrum_db(frame)
    filtered = db[db > -100.0]
    if filtered.size == 0:
        return 0.0
    peak = float(np.max(db))
    noise = float(np.percentile(filtered, 75.0))
    return max(peak - noise, 0.0)
