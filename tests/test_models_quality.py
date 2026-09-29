"""Tests for voicequal.models.quality.

Unit tests use a synthetic weights file so they never depend on the
trained artefact. One test runs the shipped weights when present.
"""

from __future__ import annotations

import numpy as np
import pytest

from voicequal.features import N_FEATURES
from voicequal.models.quality import (
    MOS_MAX,
    MOS_MIN,
    WEIGHTS_PATH,
    MOSEstimate,
    QualityModel,
)


def _write_weights(path, w0, b0, w1, b1, mean=None, scale=None):
    np.savez(
        path,
        mean=np.zeros(N_FEATURES, dtype=np.float32) if mean is None else mean,
        scale=np.ones(N_FEATURES, dtype=np.float32) if scale is None else scale,
        n_layers=np.array(2),
        W0=w0.astype(np.float32),
        b0=b0.astype(np.float32),
        W1=w1.astype(np.float32),
        b1=b1.astype(np.float32),
        teacher=np.array("test"),
        trained_on=np.array("test"),
        version=np.array("0"),
    )


@pytest.fixture
def identity_model(tmp_path):
    # Hidden layer copies feature 0 (relu'd) into 4 units; output layer maps
    # unit 0 -> sig, unit 1 -> bak, unit 2 -> ovrl with bias 1.0.
    w0 = np.zeros((N_FEATURES, 4))
    w0[0, :3] = 1.0
    b0 = np.zeros(4)
    w1 = np.zeros((4, 3))
    w1[0, 0] = w1[1, 1] = w1[2, 2] = 1.0
    b1 = np.ones(3)
    path = tmp_path / "w.npz"
    _write_weights(path, w0, b0, w1, b1)
    return QualityModel(path)


class TestQualityModel:
    def test_missing_weights_gives_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="train.py"):
            QualityModel(tmp_path / "nope.npz")

    def test_feature_count_mismatch_is_rejected(self, tmp_path):
        path = tmp_path / "bad.npz"
        _write_weights(
            path,
            np.zeros((5, 2)),
            np.zeros(2),
            np.zeros((2, 3)),
            np.zeros(3),
            mean=np.zeros(5, dtype=np.float32),
            scale=np.ones(5, dtype=np.float32),
        )
        with pytest.raises(ValueError, match="features"):
            QualityModel(path)

    def test_forward_pass_matches_hand_computation(self, identity_model):
        x = np.zeros(N_FEATURES, dtype=np.float32)
        x[0] = 2.5
        out = identity_model.predict_features(x)
        assert isinstance(out, MOSEstimate)
        assert (out.sig, out.bak, out.ovrl) == pytest.approx((3.5, 3.5, 3.5))

    def test_relu_blocks_negative_hidden(self, identity_model):
        x = np.zeros(N_FEATURES, dtype=np.float32)
        x[0] = -7.0
        out = identity_model.predict_features(x)
        assert (out.sig, out.bak, out.ovrl) == pytest.approx((1.0, 1.0, 1.0))

    def test_outputs_are_clipped_to_mos_range(self, identity_model):
        x = np.zeros(N_FEATURES, dtype=np.float32)
        x[0] = 100.0
        out = identity_model.predict_features(x)
        assert out.sig == MOS_MAX
        x[0] = -100.0
        assert identity_model.predict_features(x).sig == MOS_MIN

    def test_standardisation_is_applied(self, tmp_path):
        w0 = np.zeros((N_FEATURES, 1))
        w0[0, 0] = 1.0
        w1 = np.ones((1, 3))
        mean = np.zeros(N_FEATURES, dtype=np.float32)
        mean[0] = 10.0
        scale = np.ones(N_FEATURES, dtype=np.float32)
        scale[0] = 2.0
        path = tmp_path / "w.npz"
        _write_weights(path, w0, np.zeros(1), w1, np.full(3, 2.0), mean=mean, scale=scale)
        x = np.zeros(N_FEATURES, dtype=np.float32)
        x[0] = 14.0  # (14 - 10) / 2 = 2 -> +2 bias = 4
        assert QualityModel(path).predict_features(x).ovrl == pytest.approx(4.0)

    def test_call_runs_feature_extraction(self, identity_model):
        t = np.arange(32000) / 16000
        clip = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        out = identity_model(clip)
        assert MOS_MIN <= out.ovrl <= MOS_MAX


@pytest.mark.skipif(not WEIGHTS_PATH.exists(), reason="shipped weights not built yet")
class TestShippedWeights:
    def test_clean_tone_scores_higher_than_noisy_tone(self):
        rng = np.random.default_rng(0)
        t = np.arange(3 * 16000) / 16000
        gate = ((t % 0.5) < 0.25).astype(np.float64)
        clean = (0.3 * np.sin(2 * np.pi * 180 * t) * gate).astype(np.float32)
        noise = rng.standard_normal(t.size).astype(np.float32)
        noise *= np.sqrt(np.mean(clean**2)) / np.sqrt(np.mean(noise**2))  # 0 dB
        model = QualityModel()
        a, b = model(clean), model(clean + noise)
        assert a.bak > b.bak
        assert a.ovrl > b.ovrl
