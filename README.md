# voicequal

[![CI](https://github.com/jiya-singhal/voicequal/actions/workflows/ci.yml/badge.svg)](https://github.com/jiya-singhal/voicequal/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/voicequal.svg)](https://pypi.org/project/voicequal/)
[![Python](https://img.shields.io/pypi/pyversions/voicequal.svg)](https://pypi.org/project/voicequal/)
[![Docs](https://img.shields.io/badge/docs-github%20pages-blue)](https://jiya-singhal.github.io/voicequal/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Real-time audio quality assessment for voice apps.

Answers the question every voice app eventually has to answer:
**"is this recording clean enough to process?"**

voicequal analyzes audio with a handful of cheap acoustic metrics, led
by the harmonic-to-noise ratio, and returns a tier — `excellent`,
`good`, `fair`, or `poor` — plus the numbers behind the decision. Pure
numpy/scipy, CPU-only, no models to download.

```text
$ voicequal listen
[15:26:37]  EXCELLENT   room= 45.5 dBA   HNR=18.4dB
[15:26:53]  CHANGE  GOOD   room= 55.0 dBA
[15:26:59]  CHANGE  FAIR   room= 60.4 dBA
[15:27:09]  CHANGE  POOR   room= 71.4 dBA
[15:27:53]  CHANGE  EXCELLENT   room= 45.5 dBA
```

## Install

```bash
pip install voicequal              # core library
pip install 'voicequal[mic]'       # + live-mic support
```

## Quick start

### Analyze a file

```python
from voicequal import assess

result = assess("recording.wav")
print(result.quality)  # "good"
print(result.background_db)  # 52.3
print(result.snr)  # 24.1
print(result.reason)  # "25<snr<=35 with moderate room: good"
```

### Real-time streaming

```python
from voicequal import LiveDetector

detector = LiveDetector()
detector.on_change(lambda result: print(f"→ {result.quality}"))

while streaming:
    chunk = get_audio_chunk()  # any float32 numpy array
    detector.push(chunk)
```

### Command line

```bash
voicequal assess my_recording.wav      # one-shot file report
voicequal listen                       # live mic streaming
voicequal listen --calibrate           # first-time mic calibration
voicequal listen --sensitive           # stricter thresholds
```

First `listen` run auto-calibrates the mic (10 seconds — sit quiet
then make noise). Calibration is saved to `~/.voicequal/`.

## How it works

voicequal is built around an **HNR-gated tier decision**. The
intuition: a loud room only matters if your voice isn't dominant, and
the honest way to measure "dominant" is how much of the signal's energy
is periodic (voice) versus aperiodic (noise). That is the
**harmonic-to-noise ratio (HNR)**, estimated per frame from the
normalised autocorrelation peak in the pitch range (Boersma, 1993) and
aggregated as the median over the louder half of frames.

For voice mixed with broadband noise, HNR tracks the *mixing* SNR to
within about 4 dB. The spectral peak-vs-floor SNR that v0.1.x used is
off by nearly 30 dB on the same clips, because a sung vowel buried in
noise still has a dominant harmonic peak.

```text
room < 60 dBA        → excellent (quiet room, nothing to fix)
HNR ≥ 14.5 dB        → excellent (voice dominates noise)
HNR 11–14.5 dB       → good
HNR 7–11 dB          → fair
HNR < 7 dB           → poor (noise dominates)
```

The metrics reported on every result:

| Metric                 | What it captures                                                   | Role          |
|------------------------|--------------------------------------------------------------------|---------------|
| HNR                    | Periodic (voice) energy vs aperiodic (noise) energy, in dB         | Drives tier   |
| Background dBA         | Room loudness, median of recent RMS                                | Quiet-room gate |
| SNR (spectral)         | How much louder the peak bin is than the noise floor               | Informational |
| Spectral flatness      | How "noise-like" (chaotic) vs "tonal" (structured) it is           | Informational |
| Spectral concentration | Share of energy in the top-3 FFT bins                              | Informational |
| Temporal variance      | Is noise sustained (fan) or transient (a passing car)              | Informational |
| Clipping ratio         | Fraction of samples at or above full scale                         | Informational |

The v0.1.x spectral-SNR-gated decision is still available: call
`assess_quality()` without an `hnr` argument.

Streaming is stabilized with a **hysteresis buffer** — a new tier has
to persist for 3 frames before it's announced, so single-frame blips
don't cause flicker. Room loudness is the **median** of the last ~3 s of
RMS values (tracks sustained noise, ignores single-frame silences), and
HNR is aggregated over the same window.

## Benchmark

voicequal is benchmarked against a fixed, reproducible test set of 200
clips generated from two public research datasets — VocalSet (clean
vocals) and MUSAN (environmental noise) — mixed at controlled
signal-to-noise ratios. To regenerate and rerun:

```bash
python benchmarks/run_benchmark.py
```

Results on the 200-clip test set:

| Metric                | v0.1.1 | **v0.2.0** |
|-----------------------|--------|------------|
| Exact tier accuracy   | 46.0%  | **55.5%**  |
| Off-by-one accuracy   | 82.0%  | **89.5%**  |
| Spearman correlation  | +0.496 | **+0.662** |

### Per-category exact accuracy

| Category        | Expected tier | v0.1.1 | **v0.2.0** |
|-----------------|---------------|--------|------------|
| `quiet_noise`   | excellent     | 97.5%  | **100%**   |
| `clean_vocal`   | excellent     | 50.0%  | **55.0%**  |
| `moderate_snr`  | good          | 50.0%  | 32.5%      |
| `loud_snr`      | fair          | 27.5%  | **57.5%**  |
| `very_loud_snr` | poor          | 5.0%   | **32.5%**  |

### How well each estimator tracks the true mixing SNR

Measured on the 120 mixed clips, whose mixing SNR is known (20, 10, 5 dB):

| Estimator      | MAE     | Bias     | Spearman |
|----------------|---------|----------|----------|
| HNR            | 4.35 dB | −0.34 dB | +0.483   |
| Spectral SNR   | 28.1 dB | +28.1 dB | +0.218   |

Read this honestly. Exact accuracy is still modest. What changed is the
*shape* of the errors: v0.1.1 called 9 of the 40 heavily-noised clips
`excellent` and predicted `poor` only 5 times in 200; v0.2.0 calls 4 of
them `excellent` and predicts `poor` 17 times, 13 correctly. Gross
errors (two or more tiers off) fell from 36 to 21.

`moderate_snr` regressed. Those 20 dB mixes now mostly read `excellent`,
one tier off. Clean VocalSet recordings and 20 dB mixes overlap heavily
in HNR because breathy, lip-trill and vibrato techniques are themselves
aperiodic; no threshold separates them well.

### What changed in v0.2.0

One new metric, `hnr`, and a new decision path that uses it. WADA-SNR
and a percentile level SNR were also tried as mixing-SNR estimators and
rejected (see `CHANGELOG.md` for the numbers). The concentration-based
voice activity detector from the earlier `v0.2.0-vad` attempt was a
no-go and is not used.

### Where it still falls short

- **10 dB vs 5 dB mixes barely separate.** HNR compresses at low SNR
  because the autocorrelation peak of pure noise is not zero. A 10 dB
  fair/poor threshold lifts `very_loud_snr` to 67.5% but drops
  `loud_snr` to 12.5% and off-by-one to 87%. The shipped 7 dB favours
  fewer gross errors.
- **The thresholds were tuned on this test set.** A cross-validated
  random forest on all seven metrics reaches 61.5% exact / 93.5%
  off-by-one here, so the hand rule is within about six points of what
  these features allow on this data, but the numbers will move on real
  speech. Public speech datasets are the next step (see `ROADMAP.md`).

## Limits — read this before using in production

- **voicequal is calibrated for voice / recording quality.**
  Whether a recording is *clean enough to process*, not whether it
  sounds subjectively pleasing to a human.
- **Not a certified acoustic dB meter.** Background dBA is a
  calibrated proxy using rough dB conversion, not the IEC 61672
  A-weighted filter a real SPL meter uses.
- **Different mics deliver different signal levels.** Run
  `voicequal listen --calibrate` once per new machine or mic setup.
- **Assumes reasonable audio input.** No echo cancellation or noise
  suppression built in. If your OS pre-processes mic audio (macOS
  Voice Isolation, browser noise suppression), your calibration
  will account for it — but detection accuracy will vary.

## API reference

### `assess(path, target_sample_rate=16000, threshold_offset_db=0.0)`

Analyze an audio file. Returns a `FileAssessment` with:
`quality`, `reason`, `hnr`, `background_db`, `snr`, `spectral_flatness`,
`spectral_concentration`, `clipping_ratio`, `temporal_variance`, `primary_score`,
`secondary_score`, `total_score`, `duration_seconds`, `sample_rate`,
`num_frames`.

### `LiveDetector(sample_rate=16000, stability_frames=3, threshold_offset_db=0.0, db_offset=94.0)`

Streaming detector. Methods:
- `push(samples)` — append audio (any length, float32 numpy array)
- `on_change(callback)` — fire when the stable tier changes
- `get_current()` — snapshot the latest `LiveAssessment`
- `reset()` — clear buffers and history

### CLI

```text
voicequal --version
voicequal assess <path>
voicequal listen [--calibrate] [--reset-calibration]
                 [--sensitive] [--stability-frames N]
                 [--heartbeat SECONDS]
```

## Roadmap

See [ROADMAP.md](ROADMAP.md). Short version: fix the mixing-SNR blind
spot (v0.2.0), emit ITU-T P.835-style SIG/BAK/OVRL scores and benchmark
against DNSMOS (v0.3.0), then distil a ~1 MB ONNX quality model and add
noise-type classification (v0.4.0).

## Development

```bash
git clone https://github.com/jiya-singhal/voicequal
cd voicequal
poetry install --extras mic
poetry run pytest -v
poetry run ruff check . && poetry run ruff format --check .
poetry run mypy
```

Contributions welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) first,
especially the rule about benchmark numbers for any change to the tier
logic.

106 tests, all under `tests/`. CI runs them on Python 3.10, 3.11, and 3.12. The library has no runtime dependencies
beyond numpy, scipy, soundfile, and rich (CLI). `sounddevice` is
optional (for live-mic support).

## License

MIT. See `LICENSE`.
