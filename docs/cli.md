# CLI

```text
voicequal --version
voicequal assess <path>
voicequal listen [--calibrate] [--reset-calibration]
                 [--sensitive] [--stability-frames N]
                 [--heartbeat SECONDS]
```

## `voicequal assess <path>`

One-shot report for a WAV, FLAC, or OGG file. Prints the tier, the
reason, and every metric on the `FileAssessment`.

```bash
voicequal assess my_recording.wav
```

## `voicequal listen`

Live mic streaming. Requires the `mic` extra
(`pip install 'voicequal[mic]'`). Prints a line on every tier change and
a periodic heartbeat with the current readings.

```text
$ voicequal listen
[15:26:37]  EXCELLENT   room= 45.5 dBA   SNR=18.4dB
[15:26:53]  CHANGE  GOOD   room= 55.0 dBA
[15:26:59]  CHANGE  FAIR   room= 60.4 dBA
```

| Flag | Meaning |
|---|---|
| `--calibrate` | Run the 10-second calibration (sit quiet, then make noise) and save it to `~/.voicequal/`. Runs automatically the first time if no calibration exists. |
| `--reset-calibration` | Delete the saved calibration. |
| `--sensitive` | Stricter thresholds (a negative `threshold_offset_db`). |
| `--stability-frames N` | Frames a new tier must persist before it is announced. Default 3. |
| `--heartbeat SECONDS` | How often to print the current readings when nothing changes. |

Stop with `Ctrl+C`.
