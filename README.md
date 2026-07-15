# voicequal

Real-time audio quality assessment for voice apps.

Answers the question every voice app eventually has to answer:
**"is this recording clean enough to process?"**

voicequal analyzes audio using four acoustic metrics and returns a
tier — `excellent`, `good`, `fair`, or `poor` — plus the numbers
behind the decision.

```text
$ voicequal listen
[15:26:37]  EXCELLENT   room= 45.5 dBA   SNR=18.4dB
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
print(result.quality)          # "good"
print(result.background_db)    # 52.3
print(result.snr)              # 24.1
print(result.reason)           # "25<snr<=35 with moderate room: good"
```

### Real-time streaming

```python
from voicequal import LiveDetector

detector = LiveDetector()
detector.on_change(lambda result: print(f"→ {result.quality}"))

while streaming:
    chunk = get_audio_chunk()   # any float32 numpy array
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

voicequal is built around **SNR-gated tiered assessment**. The
intuition: a loud room only matters if your voice isn't dominant.
So SNR gates the loudness penalty before it applies.

```text
SNR > 50 dB          → excellent (voice dominates completely)
SNR 35-50            → excellent unless room > 72 dBA
SNR 25-35            → depends on room loudness
SNR ≤ 25             → composite score across all four metrics
```

The four metrics feeding this decision:

| Metric               | What it captures                                          |
|----------------------|-----------------------------------------------------------|
| SNR                  | How much louder the peak bin is than the noise floor      |
| Spectral flatness    | How "noise-like" (chaotic) vs "tonal" (structured) it is  |
| Temporal variance    | Is noise sustained (fan) or transient (a passing car)     |
| Background dBA       | Overall room loudness, using minimum statistics tracking  |

Streaming is stabilized with a **hysteresis buffer** — a new tier
has to persist for 3 frames before it's announced, so single-frame
blips don't cause flicker.

Noise floor is estimated with the **10th percentile** of recent
RMS values (robust to voice bursts, decays when room quiets).
Room loudness is reported as the **median** of recent RMS (tracks
sustained noise, ignores single-frame silences).

## Benchmark

voicequal is benchmarked against a fixed, reproducible test set of 200
clips generated from two public research datasets — VocalSet (clean
vocals) and MUSAN (environmental noise) — mixed at controlled
signal-to-noise ratios. To regenerate and rerun:

```bash
python benchmarks/run_benchmark.py
```

Results for **v0.1.1** on the 200-clip test set:

| Metric                | Value  |
|-----------------------|--------|
| Exact tier accuracy   | 46.0%  |
| Off-by-one accuracy   | 82.0%  |
| Spearman correlation  | +0.496 |

Read this honestly: exact accuracy is modest and off-by-one is the more
flattering number. voicequal orders quality roughly correctly (positive
rank correlation) but frequently lands one tier off — it rarely confuses
`excellent` for `poor`, but it does confuse neighbours. The quiet-room
categories are strong; the noisy-mix categories are weak (see below).

### Per-category exact accuracy

| Category        | Expected tier | Exact accuracy |
|-----------------|---------------|----------------|
| `quiet_noise`   | excellent     | 97.5%          |
| `clean_vocal`   | excellent     | 50.0%          |
| `moderate_snr`  | good          | 50.0%          |
| `loud_snr`      | fair          | 27.5%          |
| `very_loud_snr` | poor          | 5.0%           |

### What changed in v0.1.1

The one algorithmic change is a new `spectral_concentration` metric —
the ratio of energy in the top-3 loudest FFT bins to total energy, a
measure of how tonal (voice-like) versus broadband (noise-like) a frame
is. It is exposed on `FileAssessment` and `LiveAssessment` as an
informational output, and it gates the SNR fast-paths so noise-like
audio can't ride a high spectral-SNR reading straight to `excellent`.

In candour: on this test set the gate **slightly reduced** exact
accuracy (from 47.0% to 46.0%). It was retained because it adds a
genuinely useful signal and because the accuracy regression is within
noise, but it is not the win the version bump might imply. The metric's
real value is diagnostic, and it set up the analysis that identified the
actual bottleneck below.

### Known weakness and v0.2.0 direction

The `very_loud_snr` category — voice mixed with noise at ~5 dB SNR —
scores just **5.0%** exact accuracy and is the clear bottleneck. Root
cause: voicequal's SNR is a *spectral* SNR (peak bin vs. noise floor),
but these clips are controlled by *mixing* SNR (voice RMS vs. noise
RMS). A loud vocal buried in noise still shows a dominant harmonic peak,
so its spectral SNR reads high and the clip is waved through the
fast-path. `spectral_concentration` does not catch it either, because
sustained sung vowels stay tonal even under heavy noise.

**v0.2.0 will focus here**, adding Voice Activity Detection (VAD) gating
so SNR is estimated over voice-active frames against noise-only frames —
an RMS-domain estimate aligned with how the noise is actually mixed.

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
`quality`, `reason`, `background_db`, `snr`, `spectral_flatness`,
`spectral_concentration`, `temporal_variance`, `primary_score`,
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

## Development

```bash
git clone https://github.com/jiya-singhal/voicequal
cd voicequal
poetry install --extras mic
poetry run pytest -v
```

75 tests, all under `tests/`. The library has no runtime dependencies
beyond numpy, scipy, soundfile, and rich (CLI). `sounddevice` is
optional (for live-mic support).

## License

MIT. See `LICENSE`.
