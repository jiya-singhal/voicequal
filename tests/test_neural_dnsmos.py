"""Tests for voicequal.neural.dnsmos.

Unit tests run without onnxruntime or network by substituting a fake
session. The end-to-end test runs only when onnxruntime is installed and
the model is already cached (it never downloads in CI).
"""

from __future__ import annotations

import numpy as np
import pytest

from voicequal.neural import dnsmos as mod
from voicequal.neural.dnsmos import (
    DNSMOS,
    INPUT_SAMPLES,
    MODEL_FILENAME,
    MODEL_SHA256,
    DNSMOSScores,
    default_model_dir,
    dnsmos_available,
    ensure_model,
)


class _FakeInput:
    name = "input_1"


class _FakeSession:
    """Returns a fixed raw triple and records every input it saw."""

    def __init__(self, raw=(4.0, 4.0, 4.0)):
        self.raw = np.asarray(raw, dtype=np.float32)
        self.calls: list[np.ndarray] = []

    def get_inputs(self):
        return [_FakeInput()]

    def run(self, _outputs, feeds):
        x = feeds["input_1"]
        self.calls.append(x)
        return [self.raw[np.newaxis, :]]


def _fake_scorer(monkeypatch, raw=(4.0, 4.0, 4.0)) -> tuple[DNSMOS, _FakeSession]:
    session = _FakeSession(raw)
    monkeypatch.setattr(mod, "dnsmos_available", lambda: True)
    scorer = DNSMOS.__new__(DNSMOS)
    scorer.model_path = None  # type: ignore[assignment]
    scorer._session = session  # type: ignore[assignment]
    scorer._input_name = "input_1"
    return scorer, session


class TestPolynomials:
    def test_raw_four_maps_near_mos_four(self):
        # The published polynomials are close to identity in the 3..4.5 range.
        assert mod._POLY_SIG(4.0) == pytest.approx(3.55, abs=0.15)
        assert mod._POLY_BAK(4.0) == pytest.approx(3.93, abs=0.15)
        assert mod._POLY_OVRL(4.0) == pytest.approx(3.43, abs=0.15)

    def test_polynomials_are_monotonic_on_valid_range(self):
        xs = np.linspace(1.0, 5.0, 50)
        for poly in (mod._POLY_SIG, mod._POLY_BAK, mod._POLY_OVRL):
            ys = poly(xs)
            assert np.all(np.diff(ys) > 0)


class TestWindowing:
    def test_short_clip_is_tiled_like_the_reference(self, monkeypatch):
        # dnsmos_local.py doubles the clip until it is >= 9.01 s: a 1 s clip
        # becomes 16 s, which yields floor(16) - 9.01 + 1 = 7 windows.
        scorer, session = _fake_scorer(monkeypatch)
        result = scorer(np.zeros(16000, dtype=np.float32) + 0.01)
        assert result.num_windows == 7
        assert all(c.shape == (1, INPUT_SAMPLES) for c in session.calls)

    def test_long_clip_hops_one_second(self, monkeypatch):
        scorer, session = _fake_scorer(monkeypatch)
        twelve_seconds = np.zeros(12 * 16000, dtype=np.float32) + 0.01
        result = scorer(twelve_seconds)
        # floor(12) - 9.01 + 1 -> 3 windows starting at 0 s, 1 s, 2 s.
        assert result.num_windows == 3
        assert all(c.shape == (1, INPUT_SAMPLES) for c in session.calls)

    def test_resamples_non_16k_input(self, monkeypatch):
        scorer, session = _fake_scorer(monkeypatch)
        scorer(np.zeros(48000, dtype=np.float32) + 0.01, sample_rate=48000)
        assert session.calls[0].shape == (1, INPUT_SAMPLES)

    def test_empty_clip_raises(self, monkeypatch):
        scorer, _ = _fake_scorer(monkeypatch)
        with pytest.raises(ValueError):
            scorer(np.array([], dtype=np.float32))

    def test_scores_are_polynomial_of_raw(self, monkeypatch):
        scorer, _ = _fake_scorer(monkeypatch, raw=(4.0, 3.0, 2.0))
        result = scorer(np.zeros(16000, dtype=np.float32) + 0.01)
        assert isinstance(result, DNSMOSScores)
        assert result.sig_raw == pytest.approx(4.0)
        assert result.bak_raw == pytest.approx(3.0)
        assert result.ovrl_raw == pytest.approx(2.0)
        assert result.sig == pytest.approx(float(mod._POLY_SIG(4.0)))
        assert result.bak == pytest.approx(float(mod._POLY_BAK(3.0)))
        assert result.ovrl == pytest.approx(float(mod._POLY_OVRL(2.0)))


class TestModelCache:
    def test_env_var_overrides_model_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VOICEQUAL_MODEL_DIR", str(tmp_path))
        assert default_model_dir() == tmp_path

    def test_hash_mismatch_is_rejected_and_partial_removed(self, monkeypatch, tmp_path):
        def fake_retrieve(_url, dest):
            with open(dest, "wb") as f:
                f.write(b"not the model")

        monkeypatch.setattr(mod.urllib.request, "urlretrieve", fake_retrieve)
        with pytest.raises(RuntimeError, match="hash mismatch"):
            ensure_model(tmp_path)
        assert not (tmp_path / MODEL_FILENAME).exists()
        assert not list(tmp_path.glob("*.part"))

    def test_cached_verified_model_is_reused_without_download(self, monkeypatch, tmp_path):
        path = tmp_path / MODEL_FILENAME
        path.write_bytes(b"cached")
        monkeypatch.setattr(mod, "MODEL_SHA256", mod._sha256(path))

        def boom(*_a, **_k):
            raise AssertionError("should not download")

        monkeypatch.setattr(mod.urllib.request, "urlretrieve", boom)
        assert ensure_model(tmp_path) == path

    def test_pinned_hash_is_well_formed(self):
        assert len(MODEL_SHA256) == 64
        int(MODEL_SHA256, 16)


class TestImportGuard:
    def test_constructor_explains_missing_extra(self, monkeypatch):
        monkeypatch.setattr(mod, "dnsmos_available", lambda: False)
        with pytest.raises(ImportError, match="voicequal\\[neural\\]"):
            DNSMOS()


@pytest.mark.skipif(
    not dnsmos_available() or not (default_model_dir() / MODEL_FILENAME).exists(),
    reason="needs onnxruntime and a cached DNSMOS model",
)
class TestRealModel:
    def test_synthetic_speech_like_signal_scores_in_range(self):
        scorer = DNSMOS()
        rng = np.random.default_rng(0)
        # Harmonic buzz with amplitude modulation, loosely speech-like.
        t = np.arange(3 * 16000) / 16000
        f0 = 120 + 20 * np.sin(2 * np.pi * 0.7 * t)
        phase = 2 * np.pi * np.cumsum(f0) / 16000
        voice = sum(np.sin(k * phase) / k for k in range(1, 8))
        voice *= 0.3 * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) ** 2
        clean = voice.astype(np.float32)
        noisy = (clean + 0.3 * rng.standard_normal(clean.size)).astype(np.float32)
        a, b = scorer(clean), scorer(noisy)
        for s in (a, b):
            assert 1.0 <= s.sig <= 5.0 and 1.0 <= s.bak <= 5.0 and 1.0 <= s.ovrl <= 5.0
        assert a.bak > b.bak  # adding noise must make background worse
