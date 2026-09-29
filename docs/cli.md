# CLI

```text
voicequal --version
voicequal assess <path> [--no-timeline]
voicequal listen [--calibrate] [--reset-calibration]
                 [--sensitive] [--stability-frames N]
                 [--heartbeat SECONDS]
```

## `voicequal assess <path>`

One-shot report for a WAV, FLAC, or OGG file. Prints the tier, the
reason, every metric on the `FileAssessment`, then a per-second timeline
of the SNR estimate and the weakest segments, so you can see *which
seconds* dragged the score down.

```bash
voicequal assess my_recording.wav
```

```text
Timeline (1 s segments, SNR estimate):
     0-1s  ████░░   19.8 dB  excellent
     1-2s  ████░░   18.9 dB  excellent
     2-3s  ██░░░░    9.6 dB  poor
     3-4s  ██░░░░    8.1 dB  poor
Weakest: 3-4s (8.1 dB, poor), 2-3s (9.6 dB, poor)
```

The bar is the SNR estimate on a 0 to 30 dB scale. Segments in a quiet
room are never listed as weakest. `--no-timeline` prints only the
metrics panel.

## `voicequal listen`

Live mic streaming. Requires the `mic` extra
(`pip install 'voicequal[mic]'`). Prints a line on every tier change and
a periodic heartbeat with the current readings.

```text
$ voicequal listen
[15:26:37]  EXCELLENT   room= 45.5 dBA   HNR=18.4dB
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
