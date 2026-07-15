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


def voice_activity(
    frame: np.ndarray,
    rms_floor: float = 1e-3,
    concentration_threshold: float = 0.2,
) -> float:
    """Estimate how voice-like a frame is, in [0.0, 1.0].

    This is a lightweight voice-activity primitive, not a trained VAD.
    It combines two signals already used elsewhere in voicequal:

      - Energy (RMS): silence has no voice. Frames below ``rms_floor``
        score 0.0 outright.
      - Spectral concentration: voiced sound (a pitched vowel with
        harmonics) puts most of its energy in a few bins; broadband
        noise spreads energy across all bins. Concentration is what
        separates "loud vowel" from "loud noise" when both are
        energetic. See :func:`spectral_concentration`.

    The score is the frame's concentration once it clears the energy
    gate, rescaled so ``concentration_threshold`` maps to 0.0 and 1.0
    maps to 1.0. So a tonal vowel scores high, broadband noise scores
    near 0.0, and silence scores exactly 0.0.

    Deliberately does NOT try to detect unvoiced speech (fricatives,
    whispers), which are broadband and look noise-like to this test.
    It is a "voiced energy vs. silence/noise" gate — enough to pool
    voice-frame energy separately from noise-frame energy for an
    RMS-domain SNR estimate (the v0.2.0 goal).

    Args:
        frame: 1D numpy array of audio samples.
        rms_floor: RMS below which the frame is treated as silence.
            Default 1e-3 (roughly -60 dBFS).
        concentration_threshold: Concentration at or below which the
            frame is treated as fully non-voice. Default 0.2.

    Returns:
        A float in [0.0, 1.0]. 0.0 for empty, silent, or broadband
        frames; higher for tonal/voiced frames.
    """
    if frame.size == 0:
        return 0.0
    if rms(frame) < rms_floor:
        return 0.0
    concentration = spectral_concentration(frame)
    if concentration <= concentration_threshold:
        return 0.0
    # Rescale (threshold, 1.0] -> (0.0, 1.0].
    scaled = (concentration - concentration_threshold) / (1.0 - concentration_threshold)
    return float(np.clip(scaled, 0.0, 1.0))


def is_voice_active(frame: np.ndarray, threshold: float = 0.5) -> bool:
    """Boolean voice-activity decision for a frame.

    Thin wrapper over :func:`voice_activity`: True when the graded
    voice-activity score is at least ``threshold``.

    Args:
        frame: 1D numpy array of audio samples.
        threshold: Minimum voice_activity score to count as active.
            Default 0.5.

    Returns:
        True if the frame is voice-active, else False.
    """
    return voice_activity(frame) >= threshold
