# voicequal

Real-time audio quality assessment for voice apps.

voicequal answers a simple question: **"is this audio recording good enough to process?"**
It works on saved files or a live microphone stream, and returns one of four
tiers — `excellent`, `good`, `fair`, `poor` — using an SNR-gated multi-metric
algorithm.

---

## Install

```bash
pip install voicequal            # core library
pip install 'voicequal[mic]'     # + live microphone streaming
```

## Quick start

**File-based:**

```python
from voicequal import assess

result = assess("my_recording.wav")
print(result.quality)        # "excellent" | "good" | "fair" | "poor"
print(result.reason)         # short explanation of which branch fired
print(result.background_db)  # room loudness in a dBA-like scale
print(result.snr)            # signal-to-noise ratio in dB
```

**Streaming (any audio source):**

```python
from voicequal import LiveDetector

detector = LiveDetector()
detector.on_change(lambda r: print(f"→ {r.quality}   ({r.background_db:.1f} dBA)"))

# feed float32 audio chunks from anywhere (mic, network, file)
detector.push(audio_chunk)
```

**Command line:**

```bash
voicequal assess my_recording.wav

voicequal listen                  # live mic — first run triggers calibration
voicequal listen --sensitive      # stricter thresholds for desktop testing
voicequal listen --calibrate      # re-run calibration
```

## How the algorithm works

voicequal computes four metrics per audio frame:

- **SNR** — signal-to-noise ratio (dB)
- **Spectral flatness** — how "noise-like" vs "tonal" the spectrum is
- **Temporal variance** — is background noise sustained or transient
- **Room loudness** — dBA-scale loudness of the current environment

It then applies an **SNR-gated tier decision**:

- If SNR > 50 → `excellent` (voice dominates; room doesn't matter)
- If SNR > 35 → `excellent` unless the room is very loud (> 72 dBA) → `good`
- If SNR > 25 → depends on room loudness
- If SNR ≤ 25 → a composite score of all four metrics decides the tier

The SNR gate is what separates this from naive "just measure dBA" libraries.
A loud room with a dominant voice signal is different from a loud room with
only ambient noise, and the algorithm reflects that.

## Streaming details

`LiveDetector` uses a ring buffer to accept any chunk size, computes metrics
per 2048-sample frame with a 1600-sample hop (~10 Hz analysis rate), and
fires an `on_change` callback only when the quality tier changes and holds
for `stability_frames` consecutive frames (default 3). This prevents
single-frame flicker.

The room-loudness estimator uses the median of a rolling RMS window, so it
recovers cleanly when noise stops. The noise floor used for SNR uses the 10th
percentile of the same window — deliberately lower, to reflect the true
quiet floor even during voice bursts.

## Calibration

Different mics deliver different signal levels for the same real-world dBA.
On first `voicequal listen`, you'll be prompted to record 3 seconds of quiet
and 3 seconds of loud audio. voicequal computes a per-machine offset so the
dBA readings match your specific mic. The offset is saved to
`~/.voicequal/calibration.json` and reused.

Re-run with `voicequal listen --calibrate` to recalibrate.

## Limitations (worth knowing)

- **Not a certified acoustic dB meter.** The dBA scale is calibrated
  per-mic and gives useful relative numbers, not absolute lab-grade
  measurements.
- **Calibrated for voice/singing use cases.** The tier thresholds
  (`> 60` = fair, `> 72` = poor by default) assume a voice-app context.
  Use `voicequal listen --sensitive` for stricter thresholds on
  general-purpose testing.
- **Silence-in silence-out.** If your mic is muted or macOS suppresses
  the input (Voice Isolation), voicequal will correctly say
  "excellent" for what looks to it like a very quiet room.

## Development

```bash
git clone https://github.com/jiya-singhal/voicequal
cd voicequal
poetry install
poetry run pytest
```

70 tests currently pass across metric computation, rolling stats, tier
assessment, file pipeline, streaming, CLI, and calibration.

## License

MIT — see LICENSE.
