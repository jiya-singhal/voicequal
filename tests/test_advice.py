"""Tests for voicequal.advice."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from voicequal import advise
from voicequal.advice import (
    DEFAULT_LLM_MODEL,
    Advice,
    advise_with_llm,
)


def _result(**overrides) -> SimpleNamespace:
    base = dict(
        quality="good",
        snr_estimate=15.0,
        hnr=15.0,
        energy_snr=10.0,
        background_db=65.0,
        clipping_ratio=0.0,
        spectral_flatness=0.3,
        temporal_variance=4.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestRules:
    def test_clipping_warn(self):
        a = advise(_result(quality="excellent", clipping_ratio=0.02))
        assert a.headline == "Recording is clipping"
        assert a.severity == "warn"
        assert a.actions == ("Lower the input gain or move back from the mic",)

    def test_clipping_bad_beats_every_other_rule(self):
        a = advise(_result(quality="excellent", clipping_ratio=0.1))
        assert a.severity == "bad"

    def test_excellent(self):
        a = advise(_result(quality="excellent"))
        assert a == Advice("Clean enough to process", (), "ok")

    def test_loud_room_drowning_voice(self):
        a = advise(_result(quality="poor", background_db=74.0, snr_estimate=8.0))
        assert a.headline == "Loud room is drowning the voice"
        assert len(a.actions) == 3
        assert a.severity == "bad"

    def test_loud_room_fair_is_warn(self):
        a = advise(_result(quality="fair", background_db=72.0, snr_estimate=12.0))
        assert a.headline == "Loud room is drowning the voice"
        assert a.severity == "warn"

    def test_loud_room_rule_needs_low_snr(self):
        # Loud room but voice dominant: not the loud-room diagnosis.
        a = advise(_result(quality="good", background_db=74.0, snr_estimate=16.0))
        assert a.headline == "Usable, with mild background noise"

    def test_steady_noise(self):
        a = advise(_result(quality="fair", background_db=65.0, temporal_variance=1.0))
        assert a.headline == "Steady background noise (fan, AC, hum)"
        assert a.severity == "warn"
        assert "fans" in a.actions[0]

    def test_steady_noise_poor_is_bad(self):
        a = advise(_result(quality="poor", background_db=65.0, temporal_variance=2.9))
        assert a.severity == "bad"

    def test_intermittent_noise(self):
        a = advise(_result(quality="poor", background_db=65.0, temporal_variance=7.0))
        assert a.headline == "Intermittent noise is cutting into the voice"
        assert a.actions == ("Wait for the noise to pass", "Re-record the affected seconds")

    def test_good(self):
        a = advise(_result(quality="good"))
        assert a == Advice(
            "Usable, with mild background noise", ("Move the mic a little closer",), "warn"
        )

    def test_fallback_fair(self):
        # variance between the steady and intermittent thresholds
        a = advise(_result(quality="fair", background_db=65.0, temporal_variance=4.0))
        assert a.headline == "Too noisy to process reliably"
        assert a.severity == "warn"

    def test_fallback_poor(self):
        a = advise(_result(quality="poor", background_db=65.0, temporal_variance=4.0))
        assert a.severity == "bad"

    def test_actions_never_exceed_three(self):
        for q in ("excellent", "good", "fair", "poor"):
            for tv in (1.0, 4.0, 7.0):
                for bg in (50.0, 74.0):
                    assert (
                        len(
                            advise(
                                _result(quality=q, temporal_variance=tv, background_db=bg)
                            ).actions
                        )
                        <= 3
                    )

    def test_works_on_real_assessment(self, tmp_path):
        import numpy as np
        import soundfile as sf

        from voicequal import assess

        t = np.arange(2 * 16000) / 16000
        path = tmp_path / "tone.wav"
        sf.write(path, (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 16000)
        a = advise(assess(path))
        assert a.severity == "ok"
        assert a.to_dict()["actions"] == []


class _Block:
    def __init__(self, text: str, type_: str = "text"):
        self.text = text
        self.type = type_


class _Response:
    def __init__(self, blocks, stop_reason="end_turn"):
        self.content = blocks
        self.stop_reason = stop_reason


class _FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _FakeClient:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


class TestLLMPolish:
    def test_headline_from_model_actions_kept(self):
        client = _FakeClient(_Response([_Block("Your fan is loud. Move the mic closer.")]))
        result = _result(quality="fair", temporal_variance=1.0)
        a = advise_with_llm(result, client=client)
        base = advise(result)
        assert a.headline == "Your fan is loud. Move the mic closer."
        assert a.actions == base.actions
        assert a.severity == base.severity

    def test_request_shape(self):
        client = _FakeClient(_Response([_Block("ok")]))
        advise_with_llm(_result(), client=client)
        call = client.messages.calls[0]
        assert call["model"] == DEFAULT_LLM_MODEL
        assert "only rephrase" in call["system"]
        assert call["messages"][0]["role"] == "user"
        assert '"diagnosis"' in call["messages"][0]["content"]
        assert '"metrics"' in call["messages"][0]["content"]

    def test_model_override(self):
        client = _FakeClient(_Response([_Block("ok")]))
        advise_with_llm(_result(), model="claude-opus-5", client=client)
        assert client.messages.calls[0]["model"] == "claude-opus-5"

    def test_empty_or_non_text_falls_back_to_deterministic(self):
        client = _FakeClient(_Response([_Block("", type_="thinking")]))
        a = advise_with_llm(_result(quality="good"), client=client)
        assert a == advise(_result(quality="good"))

    def test_refusal_falls_back_to_deterministic(self):
        client = _FakeClient(_Response([_Block("no")], stop_reason="refusal"))
        assert advise_with_llm(_result(), client=client) == advise(_result())

    def test_missing_sdk_raises_import_error_with_hint(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("no module")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="voicequal\\[llm\\]"):
            advise_with_llm(_result())

    def test_default_client_is_constructed_from_sdk(self, monkeypatch):
        anthropic = pytest.importorskip("anthropic")
        fake = _FakeClient(_Response([_Block("fine")]))
        monkeypatch.setattr(anthropic, "Anthropic", lambda: fake)
        assert advise_with_llm(_result()).headline == "fine"
