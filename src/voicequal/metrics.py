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


def spectral_concentration(frame: np.ndarray, top_n: int = 3) -> float:
    """Ratio of energy in the top-N loudest bins to total energy.

    A measure of how "concentrated" the spectrum is:
      - Near 1.0: energy is concentrated in a few bins (pure tones, clean voice)
      - Near 0.0: energy is spread evenly across bins (broadband noise)

    Voice typically produces concentration 0.4-0.7 (fundamental + harmonics).
    Broadband noise typically produces concentration <0.1.
    Voice mixed with heavy noise: 0.15-0.35.

    Args:
        frame: 1D numpy array of audio samples.
        top_n: Number of top bins to include (default 3 = fundamental
            + first two harmonics for typical voice signals).

    Returns:
        A float in [0.0, 1.0]. Returns 0.0 for silent or empty frames.
    """
    if frame.size == 0 or not np.any(frame):
        return 0.0
    power = np.abs(np.fft.rfft(frame)) ** 2
    # Drop DC bin.
    power = power[1:]
    total_energy = float(np.sum(power))
    if total_energy < 1e-10:
        return 0.0
    # Sum of top-N bins.
    top_energy = float(np.sum(np.sort(power)[-top_n:]))
    return float(np.clip(top_energy / total_energy, 0.0, 1.0))


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


# ---------------------------------------------------------------------------
# Harmonic-to-noise ratio (HNR)
#
# Boersma (1993), "Accurate short-term analysis of the fundamental frequency
# and the harmonics-to-noise ratio of a sampled sound". For a periodic
# signal plus additive noise, the normalised autocorrelation peak r_max in
# the pitch-lag range approximates P_harmonic / (P_harmonic + P_noise), so
# HNR = 10*log10(r_max / (1 - r_max)) tracks the *mixing* SNR of voice
# against noise. This is the signal voicequal's spectral snr() cannot see:
# a sung vowel buried in noise keeps a dominant spectral peak but loses
# autocorrelation coherence.

HNR_FLOOR_DB: float = -60.0
HNR_CEIL_DB: float = 60.0
_R_EPS: float = 1e-6

_AUTOCORR_CACHE: dict[int, tuple[int, np.ndarray]] = {}


def _autocorr_setup(n: int) -> tuple[int, np.ndarray]:
    """Return (fft_size, normalised window autocorrelation) for frame length n."""
    if n not in _AUTOCORR_CACHE:
        window = np.hanning(n)
        n_fft = 1 << (2 * n - 1).bit_length()
        r_w = np.fft.irfft(np.abs(np.fft.rfft(window, n_fft)) ** 2)[:n]
        _AUTOCORR_CACHE[n] = (n_fft, r_w / r_w[0])
    return _AUTOCORR_CACHE[n]


def harmonic_ratio(
    frame: np.ndarray,
    sample_rate: int = 16000,
    fmin: float = 60.0,
    fmax: float = 800.0,
) -> float:
    """Normalised autocorrelation peak within the pitch-lag range.

    Estimates the fraction of frame energy that is periodic. The frame is
    mean-removed, Hann-windowed, autocorrelated via FFT, normalised by lag
    zero, and divided by the window's own autocorrelation (Boersma's
    correction) so long lags are not penalised. The peak is searched over
    lags corresponding to fmax..fmin Hz.

    Args:
        frame: 1D numpy array of audio samples.
        sample_rate: Sample rate of the frame in Hz.
        fmin: Lowest fundamental frequency to consider, Hz.
        fmax: Highest fundamental frequency to consider, Hz.

    Returns:
        A float in [0, 1). Near 1 for a pure tone or clean sung vowel,
        near 0 for broadband noise. Returns 0.0 for empty or silent frames.
    """
    n = frame.size
    if n == 0 or not np.any(frame):
        return 0.0
    n_fft, r_window = _autocorr_setup(n)
    x = frame.astype(np.float64)
    x = (x - x.mean()) * np.hanning(n)
    r = np.fft.irfft(np.abs(np.fft.rfft(x, n_fft)) ** 2)[:n]
    if r[0] <= 0.0:
        return 0.0
    r = r / r[0]
    lag_min = max(1, int(sample_rate / fmax))
    lag_max = min(n - 1, int(sample_rate / fmin))
    if lag_max <= lag_min:
        return 0.0
    segment = r[lag_min : lag_max + 1] / r_window[lag_min : lag_max + 1]
    return float(np.clip(np.max(segment), 0.0, 1.0 - _R_EPS))


def hnr(
    frame: np.ndarray,
    sample_rate: int = 16000,
    fmin: float = 60.0,
    fmax: float = 800.0,
) -> float:
    """Harmonic-to-noise ratio of an audio frame, in dB.

    HNR = 10*log10(r / (1 - r)) where r is harmonic_ratio(). For voice
    mixed with broadband noise this tracks the mixing SNR (voice RMS vs
    noise RMS), unlike snr(), which measures the spectral peak against
    the spectral floor and stays high for a loud vowel buried in noise.

    Args:
        frame: 1D numpy array of audio samples.
        sample_rate: Sample rate of the frame in Hz.
        fmin: Lowest fundamental frequency to consider, Hz.
        fmax: Highest fundamental frequency to consider, Hz.

    Returns:
        A float in dB, clipped to [HNR_FLOOR_DB, HNR_CEIL_DB]. Typical
        values: 15-25 dB clean sung vowel, ~10 dB voice at 10 dB SNR,
        below 0 dB for noise only. Returns HNR_FLOOR_DB for empty or
        silent frames.
    """
    r = harmonic_ratio(frame, sample_rate=sample_rate, fmin=fmin, fmax=fmax)
    if r <= 0.0:
        return HNR_FLOOR_DB
    value = 10.0 * np.log10(r / (1.0 - r))
    return float(np.clip(value, HNR_FLOOR_DB, HNR_CEIL_DB))


def clipping_ratio(frame: np.ndarray, threshold: float = 0.99) -> float:
    """Fraction of samples at or above the clipping threshold in magnitude.

    A cheap saturation detector. Audio that clips is distorted regardless
    of how quiet the room is, so this is reported alongside the noise
    metrics rather than folded into them.

    Args:
        frame: 1D numpy array of audio samples, expected in [-1, 1].
        threshold: Magnitude at or above which a sample counts as clipped.

    Returns:
        A float in [0, 1]. Returns 0.0 for an empty frame.
    """
    if frame.size == 0:
        return 0.0
    return float(np.mean(np.abs(frame) >= threshold))
