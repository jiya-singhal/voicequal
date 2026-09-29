# Changelog

All notable changes to voicequal are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-09-30

Speech support and the first comparison against a neural reference.

### Added
- `energy_snr()` and `block_rms()` metrics: mixing SNR from the loudest
  half of 25 ms blocks against the quietest 30%. Tracks true SNR on read
  speech (VoiceBank-DEMAND: Spearman +0.91) where HNR does not,
  and collapses safely on sustained singing where HNR takes over.
- `snr_estimate` field on `FileAssessment` and `LiveAssessment`:
  `max(hnr, energy_snr)`. This is now the value the tier is read from.
- `assess_samples(samples, sample_rate)`: the file pipeline for an
  in-memory array.
- `voicequal[neural]` extra with `voicequal.neural.DNSMOS`: Microsoft's
  P.835 predictor (SIG / BAK / OVRL) via ONNX Runtime. Model is
  downloaded once, SHA-256 pinned, cached under `~/.voicequal/models/`.
- Speech benchmark `benchmarks/speech/run_voicebank_demand.py`: 824
  public VoiceBank-DEMAND test clips with the true mixing SNR computed
  from each clean/noisy pair, scored by voicequal and DNSMOS, with
  latency. Results in the README.

### Changed
- **Tier ladder now reads `snr_estimate` at 18.5 / 13.5 / 10 dB**, fitted
  jointly on the speech and singing benchmarks. `assess_quality()` gains
  a `snr_estimate` argument; `hnr`-only calls keep the v0.2.0 ladder and
  calls with neither keep the v0.1.x logic.
- VocalSet benchmark (200 clips): exact 55.5% -> 58.0%, off-by-one
  89.5% -> 86.0%, Spearman +0.662 -> +0.645. `very_loud_snr` 32.5% -> 60.0%,
  `loud_snr` 57.5% -> 55.0%, `moderate_snr` 32.5% -> 45.0%,
  `clean_vocal` 55.0% -> 30.0% (breathy / lip-trill clips now read `good`).
- VoiceBank-DEMAND speech (824 clips, nominal-SNR tiers): exact
  80.0%, off-by-one 99.2%. The v0.2.0 HNR-only ladder scored
  27.9% / 63.3% on the same set.
- CLI shows `SNR estimate` with its two inputs.

### Found
- DNSMOS does not transfer to singing: clean VocalSet vowels score ~1.1
  on every axis, the same as white noise. The DSP-vs-DNSMOS comparison
  is therefore done on speech only.

## [0.2.0] - 2026-09-25

The mixing-SNR fix. See the README benchmark section for the full numbers.

### Added
- `hnr()` and `harmonic_ratio()` metrics: harmonic-to-noise ratio from the
  normalised autocorrelation peak in the pitch-lag range (Boersma 1993).
  For voice mixed with broadband noise this tracks the *mixing* SNR that
  the spectral `snr()` cannot see. Exposed as `hnr` on `FileAssessment`
  and `LiveAssessment`, and shown by the CLI.
- `clipping_ratio()` metric and matching `clipping_ratio` field: fraction
  of samples at or above the saturation threshold.
- Benchmark runner now records `hnr`, `spectral_concentration`, and
  `clipping_ratio` per clip, and reports MAE / bias / Spearman of each
  dB-valued estimator against the manifest's mixing SNR.

### Changed
- **Tier decision is now HNR-gated.** `assess_quality()` takes an optional
  `hnr` argument; when present it uses a quiet-room gate (background
  below 60 dBA is excellent) followed by an HNR ladder at 14.5 / 11 / 7 dB.
  The v0.1.x spectral-SNR-gated path is kept when `hnr` is omitted.
- Benchmark (200 clips): exact 46.0% -> 55.5%, off-by-one 82.0% -> 89.5%,
  Spearman +0.496 -> +0.662. `very_loud_snr` 5.0% -> 32.5%, `loud_snr`
  27.5% -> 57.5%, `quiet_noise` 97.5% -> 100%. `moderate_snr` regressed
  50.0% -> 32.5% (mostly to `excellent`, one tier off).

### Tried and rejected
- WADA-SNR (Kim & Stern 2008): assumes Gamma-distributed speech amplitudes;
  sung vowels do not fit. MAE 13.5 dB against mixing SNR. Not shipped.
- Percentile level SNR (loud frames vs quiet frames): needs gaps between
  phrases; sustained scales have none. MAE ~6 dB, no tier separation.
- Pre-emphasis before autocorrelation: whitens the harmonics too and
  lowers HNR separation. Not used.

### Phase 0 hygiene
- GitHub Actions CI running tests, ruff, and mypy on Python 3.10, 3.11, and 3.12.
- `ruff` (lint + format) and `mypy` configuration.
- MkDocs Material documentation site, deployed to GitHub Pages.
- `CONTRIBUTING.md`, `CHANGELOG.md`, `ROADMAP.md`, issue templates, and a
  pull request template.
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

[Unreleased]: https://github.com/jiya-singhal/voicequal/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/jiya-singhal/voicequal/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jiya-singhal/voicequal/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/jiya-singhal/voicequal/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/jiya-singhal/voicequal/releases/tag/v0.1.0
