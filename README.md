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
pip install 'voicequal[neural]'    # + DNSMOS reference backend (onnxruntime)
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

### Reference neural scores (optional)

```python
from voicequal.neural import DNSMOS

scores = DNSMOS().score_file("recording.wav")  # Microsoft DNSMOS, P.835
print(scores.sig, scores.bak, scores.ovrl)  # 1..5 each
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

voicequal estimates the **mixing SNR** (how much louder the voice is
than the noise) without a clean reference, then reads a tier off a short
ladder. Two cheap estimators cover the two kinds of voice audio, and the
pipeline takes the larger:

- **HNR**, the harmonic-to-noise ratio from the normalised
  autocorrelation peak in the pitch range (Boersma, 1993). Right for
  sustained singing; under-reads on speech, whose consonants and pauses
  are unvoiced.
- **Energy SNR**, loud 25 ms blocks versus the quietest 30% of blocks.
  Right for speech, which has pauses; collapses on sustained singing,
  which has none.

Each fails low, so `max(HNR, energy SNR)` picks the one whose assumption
held. The spectral peak-vs-floor SNR that v0.1.x used is off by nearly
30 dB on sung vowels in noise and is now informational only.

```text
room < 60 dBA           → excellent (quiet room, nothing to fix)
snr_estimate ≥ 18.5 dB  → excellent (voice dominates noise)
13.5–18.5 dB            → good
10–13.5 dB              → fair
< 10 dB                 → poor (noise dominates)
```

The metrics reported on every result:

| Metric                 | What it captures                                                   | Role          |
|------------------------|--------------------------------------------------------------------|---------------|
| SNR estimate           | max(HNR, energy SNR), the mixing-SNR estimate in dB                | Drives tier   |
| HNR                    | Periodic (voice) energy vs aperiodic (noise) energy, in dB         | Feeds estimate |
| Energy SNR             | Loud 25 ms blocks vs quiet blocks, in dB                           | Feeds estimate |
| Background dBA         | Room loudness, median of recent RMS                                | Quiet-room gate |
| SNR (spectral)         | How much louder the peak bin is than the noise floor               | Informational |
| Spectral flatness      | How "noise-like" (chaotic) vs "tonal" (structured) it is           | Informational |
| Spectral concentration | Share of energy in the top-3 FFT bins                              | Informational |
| Temporal variance      | Is noise sustained (fan) or transient (a passing car)              | Informational |
| Clipping ratio         | Fraction of samples at or above full scale                         | Informational |

Older decision paths remain available through `assess_quality()`: pass
only `hnr` for the v0.2.0 HNR ladder, or neither `hnr` nor
`snr_estimate` for the v0.1.x spectral-SNR logic.

Streaming is stabilized with a **hysteresis buffer** — a new tier has
to persist for 3 frames before it's announced, so single-frame blips
don't cause flicker. Room loudness is the **median** of the last ~3 s of
RMS values (tracks sustained noise, ignores single-frame silences), and
both SNR estimators are aggregated over the same window.

## Benchmark

Two public test sets, one for each kind of voice audio. Both regenerate
or download from a single script each; every number below comes from
those scripts.

| Set | What it is | Clips | Ground truth |
|---|---|---|---|
| **VoiceBank-DEMAND** (speech) | Read English sentences + real environment noise at 17.5 / 12.5 / 7.5 / 2.5 dB | 824 | Mixing SNR computed from each clean/noisy pair |
| **VocalSet + MUSAN** (singing) | Sung scales (many techniques) + environment noise at 20 / 10 / 5 dB, plus clean and noise-only clips | 200 | Manifest SNR from the mix |

```bash
python benchmarks/speech/run_voicebank_demand.py   # needs voicequal[neural] + pyarrow
python benchmarks/run_benchmark.py
```

### Speech: VoiceBank-DEMAND, 824 clips

Tiers are scored against the four nominal SNR levels (17.5 dB →
excellent … 2.5 dB → poor). That mapping is a convention, not human
ratings, so read the exact figure as "did it land in the right SNR
band".

| | v0.2.0 (HNR only) | **v0.3.0** |
|---|---|---|
| Exact band | 27.9% | **80.0%** |
| Within one band | 63.3% | **99.2%** |
| Tier rank vs true SNR (Spearman) | +0.12 | **+0.90** |

How each estimator tracks the true mixing SNR on the same clips
(Spearman rank correlation):

| Estimator | Correlation | Notes |
|---|---|---|
| **voicequal `snr_estimate`** | **+0.90** | max(HNR, energy SNR); reads ~6 dB high, the ladder absorbs it |
| voicequal energy SNR | +0.91 | the component doing the work on speech |
| DNSMOS OVRL | +0.57 | neural P.835 predictor, speech-trained |
| DNSMOS BAK | +0.52 | |
| voicequal HNR | +0.19 | under-reads: consonants and pauses are unvoiced |
| voicequal spectral SNR (v0.1.x) | −0.16 | |

DNSMOS predicts human opinion, not SNR, so a lower correlation with SNR
is not a defect. voicequal's estimate agrees with DNSMOS OVRL at +0.60.

| Path | Median latency, 2.3 s clip | Runtime |
|---|---|---|
| voicequal DSP | 15 ms | numpy + scipy |
| DNSMOS | 880 ms | onnxruntime, 1.1 MB model |

### Singing: VocalSet + MUSAN, 200 clips

| Metric | v0.1.1 | v0.2.0 | **v0.3.0** |
|---|---|---|---|
| Exact tier accuracy | 46.0% | 55.5% | **58.0%** |
| Off-by-one accuracy | 82.0% | 89.5% | 86.0% |
| Spearman correlation | +0.496 | +0.662 | +0.645 |

| Category | Expected | v0.1.1 | v0.2.0 | **v0.3.0** |
|---|---|---|---|---|
| `quiet_noise` | excellent | 97.5% | 100% | **100%** |
| `clean_vocal` | excellent | 50.0% | 55.0% | 30.0% |
| `moderate_snr` (20 dB) | good | 50.0% | 32.5% | **45.0%** |
| `loud_snr` (10 dB) | fair | 27.5% | 57.5% | 55.0% |
| `very_loud_snr` (5 dB) | poor | 5.0% | 32.5% | **60.0%** |

Estimators against the manifest mixing SNR (120 mixed clips):

| Estimator | MAE | Spearman |
|---|---|---|
| `snr_estimate` | 4.2 dB | +0.50 |
| HNR | 4.4 dB | +0.48 |
| Energy SNR | 6.0 dB | +0.19 |
| Spectral SNR (v0.1.x) | 28.1 dB | +0.22 |

### Read this honestly

- **Speech is now the strong case.** Read speech has pauses, the energy
  SNR uses them, and the ladder lands in the right SNR band 80% of the
  time with almost no gross errors.
- **Singing is harder and the numbers are modest.** HNR is the only
  estimator that works there, and it compresses at low SNR. The 5 dB
  category climbed from 5% to 60% across two releases, but clean vocals
  fell to 30%: VocalSet's breathy, lip-trill and vibrato techniques are
  aperiodic by nature and now read `good`. That is a real limitation of
  a periodicity-based estimator, not a threshold to tune away.
- **The ladder was fitted on these two sets.** A single set of
  thresholds serves both; the fit favoured speech accuracy and singing
  off-by-one over singing exact.
- **DNSMOS is blind to singing.** It scores clean sung vowels around 1.1
  out of 5, the same as white noise, so the neural comparison is
  speech-only. voicequal is, as far as we know, the only lightweight tool
  in this space that handles both.
- **No human opinion scores yet.** Both sets give SNR ground truth, not
  MOS. Correlation with listener ratings is the next thing to measure.

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
`quality`, `reason`, `snr_estimate`, `hnr`, `energy_snr`, `background_db`, `snr`,
`spectral_flatness`, `spectral_concentration`, `clipping_ratio`, `temporal_variance`, `primary_score`,
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

140 tests, all under `tests/`. CI runs them on Python 3.10, 3.11, and 3.12. The library has no runtime dependencies
beyond numpy, scipy, soundfile, and rich (CLI). `sounddevice` is
optional (for live-mic support).

## License

MIT. See `LICENSE`.
