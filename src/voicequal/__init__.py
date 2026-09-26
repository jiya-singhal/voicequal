from voicequal import calibration
from voicequal.assessment import QualityAssessment, assess_quality
from voicequal.io import load_audio
from voicequal.live import LiveAssessment, LiveDetector
from voicequal.metrics import (
    clipping_ratio,
    harmonic_ratio,
    hnr,
    noise_floor,
    rms,
    snr,
    spectral_concentration,
    spectral_flatness,
)
from voicequal.pipeline import FileAssessment, assess
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
    "FileAssessment",
    "LiveAssessment",
    "LiveDetector",
    "QualityAssessment",
    "RollingStats",
    "__version__",
    "assess",
    "assess_quality",
    "calibration",
    "clipping_ratio",
    "harmonic_ratio",
    "hnr",
    "load_audio",
    "noise_floor",
    "rms",
    "snr",
    "spectral_concentration",
    "spectral_flatness",
]
