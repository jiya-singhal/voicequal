"""DNSMOS backend: Microsoft's non-intrusive P.835 predictor, via ONNX Runtime.

DNSMOS (Reddy, Gopal, Cutler; ICASSP 2021/2022) predicts the three
ITU-T P.835 mean-opinion scores for a noisy speech clip without a clean
reference:

* ``sig``  speech signal quality (1..5)
* ``bak``  background intrusiveness (1..5, higher = less intrusive)
* ``ovrl`` overall quality (1..5)

This module wraps the public ``sig_bak_ovr.onnx`` model from the
microsoft/DNS-Challenge repository (MIT licence). The model file is
downloaded on first use to ``~/.voicequal/models/`` and verified against
a pinned SHA-256. Set ``VOICEQUAL_MODEL_DIR`` to change the cache
location, or pass ``model_path`` to use a file you already have.

Inference follows the reference ``dnsmos_local.py``: the clip is tiled
to at least 9.01 s, scored in 9.01 s windows hopping 1 s, and the raw
outputs are mapped through Microsoft's published polynomials before
averaging.

.. warning::
   DNSMOS is trained on *speech*. It does not transfer to singing: on
   voicequal's VocalSet benchmark it scores clean sung vowels around
   1.1 on every axis, the same as white noise. Use it on speech only.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE: int = 16000
INPUT_SECONDS: float = 9.01
INPUT_SAMPLES: int = int(INPUT_SECONDS * SAMPLE_RATE)  # 144160
HOP_SAMPLES: int = SAMPLE_RATE

MODEL_URL: str = (
    "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
)
MODEL_SHA256: str = "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"
MODEL_FILENAME: str = "dnsmos_sig_bak_ovr.onnx"

# Raw-output -> MOS polynomials from dnsmos_local.py (non-personalised).
_POLY_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
_POLY_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
_POLY_OVRL = np.poly1d([-0.06766283, 1.11546468, 0.04602535])


def dnsmos_available() -> bool:
    """True if onnxruntime can be imported (the ``neural`` extra is installed)."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def default_model_dir() -> Path:
    """Directory where downloaded models are cached."""
    env = os.environ.get("VOICEQUAL_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".voicequal" / "models"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(model_dir: Path | None = None) -> Path:
    """Return the path to a verified DNSMOS model, downloading it if needed.

    Raises:
        RuntimeError: If the downloaded file does not match the pinned hash.
    """
    directory = model_dir or default_model_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL_FILENAME
    if path.exists() and _sha256(path) == MODEL_SHA256:
        return path
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)  # noqa: S310 - pinned https URL
    digest = _sha256(tmp)
    if digest != MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"DNSMOS model hash mismatch: expected {MODEL_SHA256[:12]}..., got {digest[:12]}..."
        )
    tmp.replace(path)
    return path


@dataclass(frozen=True)
class DNSMOSScores:
    """P.835 predictions for one clip, averaged over 9.01 s windows.

    Fields:
        sig: Speech quality, 1..5.
        bak: Background intrusiveness, 1..5 (higher is better).
        ovrl: Overall quality, 1..5.
        sig_raw, bak_raw, ovrl_raw: Model outputs before the polynomial map.
        num_windows: How many 9.01 s windows were averaged.
    """

    sig: float
    bak: float
    ovrl: float
    sig_raw: float
    bak_raw: float
    ovrl_raw: float
    num_windows: int


class DNSMOS:
    """Callable DNSMOS scorer. Loads the ONNX session once.

    Usage::

        from voicequal.neural import DNSMOS
        scorer = DNSMOS()
        scores = scorer(samples, sample_rate=16000)
        print(scores.ovrl)
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        if not dnsmos_available():
            raise ImportError(
                "DNSMOS needs onnxruntime. Install with: pip install 'voicequal[neural]'"
            )
        import onnxruntime as ort

        path = Path(model_path) if model_path else ensure_model()
        self.model_path = path
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def __call__(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> DNSMOSScores:
        """Score a mono clip. Resamples to 16 kHz if needed.

        Args:
            samples: 1D float array in [-1, 1].
            sample_rate: Sample rate of ``samples``.

        Returns:
            DNSMOSScores.

        Raises:
            ValueError: If the clip is empty.
        """
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise ValueError("DNSMOS needs a non-empty clip.")
        if sample_rate != SAMPLE_RATE:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(SAMPLE_RATE, sample_rate)
            audio = resample_poly(audio, SAMPLE_RATE // g, sample_rate // g).astype(np.float32)

        # Tile short clips, exactly as the reference implementation does.
        while audio.size < INPUT_SAMPLES:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(audio.size / SAMPLE_RATE) - INPUT_SECONDS) + 1
        raws: list[np.ndarray] = []
        for idx in range(max(num_hops, 1)):
            start = idx * HOP_SAMPLES
            segment = audio[start : start + INPUT_SAMPLES]
            if segment.size < INPUT_SAMPLES:
                continue
            out = self._session.run(None, {self._input_name: segment[np.newaxis, :]})[0][0]
            raws.append(np.asarray(out, dtype=np.float64))
        if not raws:  # pragma: no cover - tiling guarantees at least one window
            raise ValueError("DNSMOS produced no windows.")

        raw = np.mean(np.stack(raws), axis=0)
        sig_raw, bak_raw, ovrl_raw = (float(v) for v in raw)
        mapped = np.mean(
            np.stack(
                [
                    [float(_POLY_SIG(r[0])), float(_POLY_BAK(r[1])), float(_POLY_OVRL(r[2]))]
                    for r in raws
                ]
            ),
            axis=0,
        )
        return DNSMOSScores(
            sig=float(mapped[0]),
            bak=float(mapped[1]),
            ovrl=float(mapped[2]),
            sig_raw=sig_raw,
            bak_raw=bak_raw,
            ovrl_raw=ovrl_raw,
            num_windows=len(raws),
        )

    def score_file(self, path: str | Path) -> DNSMOSScores:
        """Load an audio file with voicequal's loader and score it."""
        from voicequal.io import load_audio

        samples, sr = load_audio(str(path), target_sample_rate=SAMPLE_RATE)
        return self(samples, sample_rate=sr)
