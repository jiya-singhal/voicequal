# Quick start

## Install

```bash
pip install voicequal              # core library
pip install 'voicequal[mic]'       # + live-mic support (sounddevice)
```

## Analyze a file

```python
from voicequal import assess

result = assess("recording.wav")
print(result.quality)  # "good"
print(result.background_db)  # 52.3
print(result.snr)  # 24.1
print(result.reason)  # "25<snr<=35 with moderate room: good"
```

`assess` accepts any format `soundfile` can read (WAV, FLAC, OGG) and
resamples to 16 kHz internally. See the [API reference](api.md) for
every field on the result.

## Real-time streaming

```python
from voicequal import LiveDetector

detector = LiveDetector()
detector.on_change(lambda result: print(f"→ {result.quality}"))

while streaming:
    chunk = get_audio_chunk()  # any float32 numpy array
    detector.push(chunk)
```

`push` is non-blocking. The `on_change` callback fires only when the
tier changes *and* stays changed for a few frames, so single-frame
blips do not cause flicker.

## Command line

```bash
voicequal assess my_recording.wav      # one-shot file report
voicequal listen                       # live mic streaming
voicequal listen --calibrate           # first-time mic calibration
voicequal listen --sensitive           # stricter thresholds
```

First `listen` run auto-calibrates the mic (10 seconds: sit quiet, then
make noise). Calibration is saved to `~/.voicequal/`.

!!! note "Calibrate once per machine"
    Different mics deliver different signal levels. Re-run
    `voicequal listen --calibrate` whenever you change mic or machine.

## Examples

Two runnable scripts live in `examples/`:

- `examples/mic_check.py`: a minimal mic quality check.
- `examples/live_mic_demo.py`: the streaming demo with tier-change
  logging.
