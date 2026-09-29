"""Turn a voicequal assessment into one line of advice a person can act on.

:func:`advise` is deterministic and dependency-free: a short priority
list of rules over the numbers voicequal already computed. It never
invents a cause the metrics do not support. :func:`advise_with_llm` is an
optional polish step (``pip install 'voicequal[llm]'``) that asks Claude
to rephrase and prioritise the deterministic advice; it is never called
unless you call it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

Severity = Literal["ok", "warn", "bad"]

# Clipping is a gain problem, not a room problem; above this fraction of
# saturated samples the recording is distorted regardless of the tier.
CLIPPING_WARN: float = 0.01
CLIPPING_BAD: float = 0.05
# Room loudness above which the room itself, not the mic distance, is the
# dominant cause of a low SNR estimate.
LOUD_ROOM_DB: float = 72.0
# Below the "good" ladder step the voice is no longer dominant.
SNR_NOT_DOMINANT_DB: float = 13.5
# Temporal variance (std-dev of the noise floor over ~2 s) separates steady
# noise sources (fans, AC, hum) from intermittent ones (traffic, voices).
STEADY_NOISE_VARIANCE: float = 3.0
INTERMITTENT_NOISE_VARIANCE: float = 5.0


class _AssessmentLike(Protocol):
    """The fields advise() reads. Read-only so frozen dataclasses satisfy it."""

    @property
    def quality(self) -> str: ...
    @property
    def snr_estimate(self) -> float: ...
    @property
    def hnr(self) -> float: ...
    @property
    def energy_snr(self) -> float: ...
    @property
    def background_db(self) -> float: ...
    @property
    def clipping_ratio(self) -> float: ...
    @property
    def spectral_flatness(self) -> float: ...
    @property
    def temporal_variance(self) -> float: ...


@dataclass(frozen=True)
class Advice:
    """One headline, up to three actions, and how urgent it is.

    Fields:
        headline: One short sentence describing the situation.
        actions: Zero to three short imperative sentences, most useful first.
        severity: ``"ok"`` (proceed), ``"warn"`` (usable, could be better),
            ``"bad"`` (fix before recording again).
    """

    headline: str
    actions: tuple[str, ...]
    severity: Severity

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["actions"] = list(self.actions)
        return d


def advise(result: _AssessmentLike) -> Advice:
    """Deterministic advice from an assessment's metrics.

    Rules are checked in priority order and the first match wins. Each rule
    keys only on numbers the result already carries, so the advice is
    explainable from the same report the user sees.

    Args:
        result: A ``FileAssessment`` or ``LiveAssessment`` (anything with the
            same field names works).
    """
    quality = result.quality
    noisy = quality in ("fair", "poor")

    # 1. Clipping first. A saturated recording is distorted no matter how
    #    quiet the room is, and the fix (gain) is different from every other
    #    rule's fix (noise).
    if result.clipping_ratio > CLIPPING_WARN:
        return Advice(
            headline="Recording is clipping",
            actions=("Lower the input gain or move back from the mic",),
            severity="bad" if result.clipping_ratio > CLIPPING_BAD else "warn",
        )

    # 2. Excellent means the tier decision found nothing to fix.
    if quality == "excellent":
        return Advice(headline="Clean enough to process", actions=(), severity="ok")

    # 3. Loud room with the voice not dominant: the room is the cause, so
    #    the advice is about the room and the mic distance.
    if result.background_db >= LOUD_ROOM_DB and result.snr_estimate < SNR_NOT_DOMINANT_DB:
        return Advice(
            headline="Loud room is drowning the voice",
            actions=(
                "Move away from the noise source",
                "Close windows or pause fans/AC",
                "Bring the mic closer to your mouth",
            ),
            severity="bad" if quality == "poor" else "warn",
        )

    # 4. Steady noise floor in a noisy tier points at a constant source.
    if noisy and result.temporal_variance < STEADY_NOISE_VARIANCE:
        return Advice(
            headline="Steady background noise (fan, AC, hum)",
            actions=("Turn off fans or AC if you can", "Move to a quieter room"),
            severity="bad" if quality == "poor" else "warn",
        )

    # 5. A jumpy noise floor in a noisy tier points at intermittent events.
    if noisy and result.temporal_variance >= INTERMITTENT_NOISE_VARIANCE:
        return Advice(
            headline="Intermittent noise is cutting into the voice",
            actions=("Wait for the noise to pass", "Re-record the affected seconds"),
            severity="bad" if quality == "poor" else "warn",
        )

    # 6. Good: usable, one cheap improvement.
    if quality == "good":
        return Advice(
            headline="Usable, with mild background noise",
            actions=("Move the mic a little closer",),
            severity="warn",
        )

    # 7. Fair or poor with no more specific diagnosis.
    return Advice(
        headline="Too noisy to process reliably",
        actions=("Bring the mic closer", "Find a quieter spot"),
        severity="bad" if quality == "poor" else "warn",
    )


# ---------------------------------------------------------------------------
# Optional LLM polish

DEFAULT_LLM_MODEL: str = "claude-sonnet-5"

_SYSTEM_PROMPT = (
    "You rewrite audio-quality advice for a person about to record their voice. "
    "You are given the measured metrics and a deterministic diagnosis produced from "
    "them. You may only rephrase and prioritise that diagnosis. Never invent a cause, "
    "a device, or a room detail that the numbers do not support. Do not mention the "
    "metric names or values. Reply with at most two short sentences and nothing else."
)


def _metrics_payload(result: _AssessmentLike, base: Advice) -> str:
    fields = (
        "quality",
        "snr_estimate",
        "hnr",
        "energy_snr",
        "background_db",
        "clipping_ratio",
        "spectral_flatness",
        "temporal_variance",
    )
    metrics = {name: getattr(result, name) for name in fields}
    return json.dumps({"metrics": metrics, "diagnosis": base.to_dict()}, indent=1)


def advise_with_llm(
    result: _AssessmentLike,
    model: str | None = None,
    client: Any | None = None,
) -> Advice:
    """Rephrase :func:`advise` output with Claude. Needs ``voicequal[llm]``.

    The deterministic advice is computed first and sent along with the
    metrics. Claude may only reword and reorder it; the returned
    :class:`Advice` keeps the deterministic actions and severity and takes
    its headline from the model. If the model returns nothing usable, the
    deterministic headline is kept.

    Args:
        result: A ``FileAssessment`` or ``LiveAssessment``.
        model: Claude model id. Defaults to :data:`DEFAULT_LLM_MODEL`.
        client: An ``anthropic.Anthropic`` instance to reuse. Created from the
            environment if omitted.

    Raises:
        ImportError: If the ``anthropic`` package is not installed.
    """
    base = advise(result)
    if client is None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(
                "advise_with_llm needs the anthropic SDK. "
                "Install with: pip install 'voicequal[llm]'"
            ) from exc
        client = anthropic.Anthropic()

    response = client.messages.create(
        model=model or DEFAULT_LLM_MODEL,
        max_tokens=256,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _metrics_payload(result, base)}],
    )
    if getattr(response, "stop_reason", None) == "refusal":
        return base
    text = " ".join(
        str(getattr(block, "text", "")).strip()
        for block in response.content
        if getattr(block, "type", "") == "text"
    ).strip()
    if not text:
        return base
    return Advice(headline=text, actions=base.actions, severity=base.severity)
