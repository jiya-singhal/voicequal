# Contributing to voicequal

Thanks for taking an interest. This page covers the dev setup, how we
measure changes, and what a good pull request looks like.

## Dev setup

voicequal targets Python 3.10+ and is developed on 3.12 via pyenv.

```bash
git clone https://github.com/jiya-singhal/voicequal
cd voicequal
pyenv install 3.12.10        # once
pyenv local 3.12.10
poetry install --extras mic
```

Everyday commands:

```bash
poetry run pytest             # run the test suite
poetry run ruff check .       # lint
poetry run ruff format .      # format
poetry run mypy src           # type-check the library
```

CI runs all four on every push and pull request. Please run them locally
before opening a PR.

## Running the benchmark

```bash
poetry run python benchmarks/run_benchmark.py
```

The 200-clip test set regenerates deterministically from `seed=42`. See
`benchmarks/README.md` for how the clips are built from VocalSet and
MUSAN, and how to regenerate them from the raw datasets.

## The benchmark rule

**Any change to `src/voicequal/assessment.py` must include before and
after benchmark numbers in the pull request.** At minimum: exact tier
accuracy, off-by-one accuracy, Spearman correlation, and the per-category
table. Regressions are not automatically rejected, but they must be
visible and explained.

## The "informational field first" discipline

New metrics go through three stages, in order:

1. **Implement as a pure per-frame function** in `metrics.py`, with tests.
   No wiring.
2. **Expose it as an informational field** on `FileAssessment` (and
   `LiveAssessment` if it applies). Run the benchmark and check whether
   the field actually separates the tiers it is meant to separate.
3. **Only then** wire it into the tier decision in `assessment.py`,
   with the before/after numbers above.

Stage 2 is a go/no-go gate. If the field does not carry signal on the
benchmark, stop and say so. `rms_snr` on the `v0.2.0-vad` branch is a
worked example of a no-go that was documented rather than forced.

## Commit style

Short imperative subject line, matching the existing history:

```text
v0.2.0 step 2: rms_snr informational field
Polish README and update keywords
Ignore .omc tool directory
```

Keep the body for the *why*. If a commit changes benchmark results, put
the numbers in the body.

## Pull requests

Fill in the PR template. Update `CHANGELOG.md` under `[Unreleased]` for
anything user-visible.

## Code of conduct

Be kind. This project follows the
[Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
