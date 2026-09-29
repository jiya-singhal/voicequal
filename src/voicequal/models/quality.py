"""Distilled P.835 quality model: numpy inference over clip features.

Architecture: standardise -> Linear -> ReLU -> Linear -> ReLU -> Linear,
three outputs (sig, bak, ovrl), each clipped to the MOS range 1..5.
Weights live in ``quality_mlp.npz`` next to this file and are produced by
``benchmarks/distill/train.py``. The same weights are exported to ONNX
for use outside Python.

The model is a *student* of DNSMOS (Microsoft's neural P.835 predictor)
on VoiceBank-DEMAND speech. It inherits DNSMOS's blind spot: it has not
seen singing and should not be trusted on it. The README benchmark
section reports how closely it tracks its teacher.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voicequal.features import N_FEATURES, clip_features

WEIGHTS_PATH: Path = Path(__file__).with_name("quality_mlp.npz")
MOS_MIN: float = 1.0
MOS_MAX: float = 5.0


@dataclass(frozen=True)
class MOSEstimate:
    """P.835-style scores predicted by the distilled model.

    Fields:
        sig: Speech signal quality, 1..5.
        bak: Background intrusiveness, 1..5 (higher = less intrusive).
        ovrl: Overall quality, 1..5.
    """

    sig: float
    bak: float
    ovrl: float


class QualityModel:
    """Loads the shipped MLP weights and predicts (sig, bak, ovrl)."""

    def __init__(self, weights_path: str | Path | None = None) -> None:
        path = Path(weights_path) if weights_path else WEIGHTS_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Quality model weights not found at {path}. "
                "Run benchmarks/distill/train.py to produce them."
            )
        with np.load(path) as z:
            self.mean = z["mean"].astype(np.float64)
            self.scale = z["scale"].astype(np.float64)
            self.layers: list[tuple[np.ndarray, np.ndarray]] = []
            n_layers = int(z["n_layers"])
            for i in range(n_layers):
                self.layers.append((z[f"W{i}"].astype(np.float64), z[f"b{i}"].astype(np.float64)))
            self.meta = {
                "teacher": str(z["teacher"]) if "teacher" in z else "unknown",
                "trained_on": str(z["trained_on"]) if "trained_on" in z else "unknown",
                "version": str(z["version"]) if "version" in z else "unknown",
            }
        if self.mean.shape != (N_FEATURES,):
            raise ValueError(
                f"Weights expect {self.mean.shape[0]} features, code produces {N_FEATURES}."
            )

    def predict_features(self, features: np.ndarray) -> MOSEstimate:
        """Predict from a precomputed feature vector (see voicequal.features)."""
        x = (np.asarray(features, dtype=np.float64).reshape(-1) - self.mean) / self.scale
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0.0)
        sig, bak, ovrl = np.clip(x, MOS_MIN, MOS_MAX)
        return MOSEstimate(sig=float(sig), bak=float(bak), ovrl=float(ovrl))

    def __call__(self, samples: np.ndarray, sample_rate: int = 16000) -> MOSEstimate:
        """Predict for a mono clip. Raises ValueError if the clip is too short."""
        return self.predict_features(clip_features(samples, sample_rate))


_DEFAULT: QualityModel | None = None


def predict_mos(samples: np.ndarray, sample_rate: int = 16000) -> MOSEstimate:
    """Module-level convenience: predict P.835 scores with the shipped model."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = QualityModel()
    return _DEFAULT(samples, sample_rate)
