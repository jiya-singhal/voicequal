---
name: Bug report
about: Something is wrong or a recording is assessed incorrectly
title: ''
labels: bug
assignees: ''
---

## What happened

A clear description of the bug. If voicequal gave the wrong tier, say
what tier you expected and why.

## Sample audio

Attach a short WAV that reproduces it, or describe how to generate one
(for example, "clean speech at 16 kHz mixed with fan noise at ~10 dB
SNR"). Without audio, tier bugs are very hard to reproduce.

## Output

Paste the full output of `voicequal assess <file>` or the
`FileAssessment` / `LiveAssessment` fields, including `reason`.

```text

```

## Environment

- voicequal version (`voicequal --version`):
- Python version:
- OS and version:
- Mic / input device (for live mode):
- Was OS audio processing on? (macOS Voice Isolation, Windows noise
  suppression, browser noise suppression, etc.):
- Did you run `voicequal listen --calibrate` on this machine?

## Anything else

Logs, screenshots, or context.
