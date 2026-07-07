"""Audio loading utilities for voicequal."""

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(path: str, target_sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    """Load an audio file and return (samples, sample_rate).

    Loads the file, converts to mono if stereo (by averaging channels),
    and resamples to target_sample_rate using scipy.signal.resample_poly.

    Returns samples as a float32 numpy array normalized to [-1, 1],
    and the actual sample rate returned.

    Uses soundfile to load. Uses scipy.signal.resample_poly for resampling
    (not scipy.signal.resample -- poly is better quality).
    """
    samples, sample_rate = sf.read(path, dtype="float32")

    # Convert to mono by averaging channels if the file is stereo/multi-channel.
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    # Resample only when the source rate differs from the target rate.
    if sample_rate != target_sample_rate:
        samples = resample_poly(samples, target_sample_rate, sample_rate)
        sample_rate = target_sample_rate

    return samples.astype(np.float32), sample_rate
