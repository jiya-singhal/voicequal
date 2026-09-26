# voicequal

Real-time audio quality assessment for voice apps.

Answers the question every voice app eventually has to answer:
**"is this recording clean enough to process?"**

voicequal analyzes audio with a handful of acoustic metrics, led by the harmonic-to-noise ratio, and returns a
tier, `excellent`, `good`, `fair`, or `poor`, plus the numbers behind
the decision.

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

## Where to go next

- [Quick start](quickstart.md): file analysis, streaming, and the CLI in
  five minutes.
- [How it works](how-it-works.md): the HNR-gated tier logic and the
  metrics behind it.
- [Benchmark](benchmark.md): the honest numbers, including the weak
  category.
- [API reference](api.md) and [CLI](cli.md).
- [Roadmap](roadmap.md): where this is going.

## Limits, read this before using in production

- **voicequal is calibrated for voice / recording quality.** Whether a
  recording is *clean enough to process*, not whether it sounds
  subjectively pleasing to a human.
- **Not a certified acoustic dB meter.** Background dBA is a calibrated
  proxy using rough dB conversion, not the IEC 61672 A-weighted filter a
  real SPL meter uses.
- **Different mics deliver different signal levels.** Run
  `voicequal listen --calibrate` once per new machine or mic setup.
- **Assumes reasonable audio input.** No echo cancellation or noise
  suppression built in. If your OS pre-processes mic audio (macOS Voice
  Isolation, browser noise suppression), your calibration will account
  for it, but detection accuracy will vary.

## License

MIT.
