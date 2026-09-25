## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## Checklist

- [ ] `poetry run pytest` passes
- [ ] `poetry run ruff check .` and `poetry run ruff format --check .` pass
- [ ] `poetry run mypy src` passes
- [ ] `CHANGELOG.md` updated under `[Unreleased]` (if user-visible)
- [ ] If `assessment.py` changed: before/after benchmark numbers are below

## Benchmark (only if the tier decision changed)

| Metric | Before | After |
|---|---|---|
| Exact tier accuracy | | |
| Off-by-one accuracy | | |
| Spearman correlation | | |

Per-category exact accuracy:

| Category | Before | After |
|---|---|---|
| `quiet_noise` | | |
| `clean_vocal` | | |
| `moderate_snr` | | |
| `loud_snr` | | |
| `very_loud_snr` | | |
