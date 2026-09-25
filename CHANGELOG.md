# Changelog

All notable changes to voicequal are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI running tests, ruff, and mypy on Python 3.10, 3.11, and 3.12.
- `ruff` (lint + format) and `mypy` configuration.
- MkDocs Material documentation site, deployed to GitHub Pages.
- `CONTRIBUTING.md`, `CHANGELOG.md`, `ROADMAP.md`, issue templates, and a
  pull request template.

### Changed
- `__version__` is now derived from installed package metadata instead of
  a hand-edited string, so it cannot drift from `pyproject.toml`.

## [0.1.1] - 2026-07-16

### Added
- `spectral_concentration` metric: ratio of energy in the top-3 loudest
  FFT bins to total energy. Exposed on `FileAssessment` and
  `LiveAssessment`.
- Reproducible 200-clip benchmark under `benchmarks/` (generator, runner,
  manifest), built from VocalSet vocals mixed with MUSAN noise at
  controlled signal-to-noise ratios.
- Benchmark section in the README with per-category results.

### Changed
- SNR fast-paths in the tier decision are now gated by
  `spectral_concentration`, so noise-like audio cannot ride a high
  spectral-SNR reading straight to `excellent`.

### Known issues
- The concentration gate slightly reduced exact tier accuracy on the
  benchmark, from 47.0% to 46.0%. It was kept for its diagnostic value.
- The `very_loud_snr` category (voice mixed with noise at ~5 dB) scores
  5.0% exact accuracy. Root cause is documented in the README.

## [0.1.0] - 2026-07

### Added
- Initial release.
- `assess(path)` for one-shot file analysis returning a `FileAssessment`.
- `LiveDetector` for streaming analysis with a hysteresis buffer and
  `on_change` callbacks.
- CLI: `voicequal assess <path>` and `voicequal listen`.
- Mic calibration (`voicequal listen --calibrate`), saved to
  `~/.voicequal/`.
- Four metrics: SNR, spectral flatness, temporal variance, and
  background dBA.

[Unreleased]: https://github.com/jiya-singhal/voicequal/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/jiya-singhal/voicequal/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/jiya-singhal/voicequal/releases/tag/v0.1.0
