# Advice

A tier tells you *whether* a recording is clean enough. `advise()` tells
the person *what to do about it*, in one line, from the same numbers.

```python
from voicequal import assess, advise

result = assess("recording.wav")
tip = advise(result)
print(tip.headline)  # "Steady background noise (fan, AC, hum)"
print(tip.actions)  # ("Turn off fans or AC if you can", "Move to a quieter room")
print(tip.severity)  # "warn"
```

`advise()` works on a `FileAssessment` or a `LiveAssessment`. It is
deterministic, dependency-free, and never invents a cause the metrics do
not support. Rules are checked in this order; the first match wins.

| # | Condition | Headline | Severity |
|---|---|---|---|
| 1 | `clipping_ratio` > 1% | Recording is clipping | bad if > 5%, else warn |
| 2 | `quality` = excellent | Clean enough to process | ok |
| 3 | `background_db` ≥ 72 and `snr_estimate` < 13.5 | Loud room is drowning the voice | bad for poor, warn for fair |
| 4 | fair/poor and `temporal_variance` < 3 | Steady background noise (fan, AC, hum) | bad for poor, warn for fair |
| 5 | fair/poor and `temporal_variance` ≥ 5 | Intermittent noise is cutting into the voice | bad for poor, warn for fair |
| 6 | `quality` = good | Usable, with mild background noise | warn |
| 7 | otherwise (fair/poor) | Too noisy to process reliably | bad for poor, warn for fair |

Each rule has at most three short imperative actions. `Advice.to_dict()`
gives a JSON-friendly form; the [MCP server](mcp.md) includes it in every
tool result.

## Optional: rephrasing with Claude

`advise_with_llm()` sends the metrics **and** the deterministic advice to
Claude and asks it to rephrase and prioritise, with a system prompt that
forbids inventing causes the numbers do not support and caps the reply at
two sentences. The returned `Advice` keeps the deterministic actions and
severity; only the headline comes from the model. Nothing in voicequal
calls it unless you do.

```bash
pip install 'voicequal[llm]'
export ANTHROPIC_API_KEY=...
```

```python
from voicequal import assess
from voicequal.advice import advise_with_llm

tip = advise_with_llm(assess("recording.wav"))  # default: claude-sonnet-5
tip = advise_with_llm(result, model="claude-opus-5-5")  # or pick a model
```

If the model refuses or returns no text, the deterministic headline is
kept. Without the `anthropic` package installed the call raises
`ImportError` with the install hint.
