# API reference

Everything public is importable from the top-level package:

```python
from voicequal import assess, LiveDetector, FileAssessment, LiveAssessment
```

## `assess(path, target_sample_rate=16000, threshold_offset_db=0.0)`

Analyze an audio file and return a `FileAssessment`.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `path` | `str` or `Path` | | Any file `soundfile` can read (WAV, FLAC, OGG). |
| `target_sample_rate` | `int` | `16000` | Audio is resampled to this rate before analysis. |
| `threshold_offset_db` | `float` | `0.0` | Shifts the dBA tier thresholds. Negative is stricter (the CLI's `--sensitive` uses this). |

## `FileAssessment`

Frozen dataclass returned by `assess`.

| Field | Type | Meaning |
|---|---|---|
| `quality` | `str` | `excellent`, `good`, `fair`, or `poor`. |
| `reason` | `str` | Which branch of the tier logic fired, in words. |
| `background_db` | `float` | Aggregated dBA-like room loudness. |
| `snr` | `float` | Aggregated spectral SNR in dB. |
| `spectral_flatness` | `float` | 0..1, higher is more noise-like. |
| `spectral_concentration` | `float` | 0..1, share of energy in the top-3 FFT bins. |
| `temporal_variance` | `float` | Final temporal-variance reading. |
| `primary_score` | `float` | Composite score primary component (0 if a fast-path fired). |
| `secondary_score` | `float` | Composite score secondary component. |
| `total_score` | `float` | `primary_score + secondary_score`. |
| `duration_seconds` | `float` | Length of the analyzed audio. |
| `sample_rate` | `int` | Rate the audio was analyzed at. |
| `num_frames` | `int` | Number of frames the audio was cut into. |

## `LiveDetector(sample_rate=16000, stability_frames=3, threshold_offset_db=0.0, db_offset=94.0)`

Streaming detector.

| Parameter | Meaning |
|---|---|
| `sample_rate` | Rate of the samples you will `push`. |
| `stability_frames` | Frames a new tier must persist before it is announced. |
| `threshold_offset_db` | Same as in `assess`. |
| `db_offset` | Mic calibration offset. The CLI loads this from `~/.voicequal/calibration.json`. |

Methods:

| Method | Meaning |
|---|---|
| `push(samples)` | Append audio (any length, 1-D float32 numpy array). Non-blocking. |
| `on_change(callback)` | Register a callback fired with a `LiveAssessment` when the stable tier changes. |
| `get_current()` | Snapshot the latest `LiveAssessment`, or `None` before the first frame. |
| `reset()` | Clear buffers and history. |

## `LiveAssessment`

Frozen dataclass describing the current rolling window. Fields mirror
`FileAssessment` minus the file-level ones:

`quality`, `reason`, `background_db`, `snr`, `spectral_flatness`,
`spectral_concentration`, `temporal_variance`, `frames_analyzed`.

## Calibration

```python
from voicequal import calibration
```

| Name | Meaning |
|---|---|
| `Calibration` | Frozen dataclass: `db_offset`, `quiet_rms`, `loud_rms`, `created_at`. |
| `calibration.load()` | Read `~/.voicequal/calibration.json`, or `None` if missing or malformed. |
| `calibration.save(cal)` | Write it. |
| `calibration.compute_offset(quiet_rms, loud_rms)` | Solve for the `db_offset` that makes `quiet_rms` read as 40 dBA. `loud_rms` is a sanity check only. |

The CLI handles calibration for you via `voicequal listen --calibrate`;
these functions exist for apps that run their own calibration flow.

## Lower-level building blocks

Also exported, for people who want the pieces:

- `voicequal.metrics`: `rms`, `snr`, `noise_floor`, `spectral_flatness`,
  `spectral_concentration`. Pure per-frame functions.
- `voicequal.state.RollingStats`: rolling noise-floor and RMS history.
- `voicequal.assessment.assess_quality`: the tier decision, returning a
  `QualityAssessment`.
- `voicequal.io.load_audio`: load and resample a file.
