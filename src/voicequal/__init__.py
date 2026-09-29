from voicequal import calibration
from voicequal.advice import Advice, advise
from voicequal.assessment import QualityAssessment, assess_quality
from voicequal.io import load_audio
from voicequal.live import LiveAssessment, LiveDetector
from voicequal.metrics import (
    block_rms,
    clipping_ratio,
    energy_snr,
    harmonic_ratio,
    hnr,
    noise_floor,
    rms,
    snr,
    spectral_concentration,
    spectral_flatness,
)
from voicequal.pipeline import FileAssessment, assess, assess_samples
from voicequal.state import RollingStats


def _read_version() -> str:
    """Resolve the installed version from package metadata.

    Falls back to a placeholder when running from a source checkout that
    has not been installed (for example a bare ``PYTHONPATH=src``).
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("voicequal")
    except PackageNotFoundError:  # pragma: no cover - only on uninstalled trees
        return "0.0.0+unknown"


__version__ = _read_version()

__all__ = [
    "Advice",
    "FileAssessment",
    "LiveAssessment",
    "LiveDetector",
    "QualityAssessment",
    "RollingStats",
    "__version__",
    "advise",
    "assess",
    "assess_quality",
    "assess_samples",
    "block_rms",
    "calibration",
    "clipping_ratio",
    "energy_snr",
    "harmonic_ratio",
    "hnr",
    "load_audio",
    "noise_floor",
    "rms",
    "snr",
    "spectral_concentration",
    "spectral_flatness",
]
