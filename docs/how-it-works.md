# How it works

voicequal is built around an **HNR-gated tier decision**. The intuition:
a loud room only matters if your voice isn't dominant, and the honest
way to measure "dominant" is how much of the signal's energy is periodic
(voice) versus aperiodic (noise).

## The harmonic-to-noise ratio

For each frame, voicequal removes the mean, applies a Hann window,
computes the autocorrelation via FFT, normalises it by lag zero, and
divides by the window's own autocorrelation (Boersma, 1993). The peak
`r` of that curve within the pitch-lag range (60–800 Hz) approximates the
fraction of energy that is periodic, and

```text
HNR = 10 · log10( r / (1 − r) )
```

For a periodic signal plus additive noise this is, to a good
approximation, the mixing SNR. On the benchmark's 120 mixed clips it
tracks the true mixing SNR with a mean absolute error of 4.35 dB. The
spectral peak-vs-floor SNR that v0.1.x used is off by 28 dB on the same
clips, because a sung vowel buried in noise still has a dominant
harmonic peak.

Per-file (and per rolling window in live mode), HNR is aggregated as the
**median over the louder half of frames**, so breaths and gaps do not
drag it down.

## The decision

```text
room < 60 dBA        → excellent (quiet room, nothing to fix)
HNR ≥ 14.5 dB        → excellent (voice dominates noise)
HNR 11–14.5 dB       → good
HNR 7–11 dB          → fair
HNR < 7 dB           → poor (noise dominates)
```

The quiet-room gate comes first because a room below 60 dBA is fine no
matter what the voice metrics say (and a noise-only clip has no voice to
measure). The thresholds were tuned on the 200-clip benchmark; see the
[benchmark](benchmark.md) page for the caveat.

The v0.1.x spectral-SNR-gated path is kept for compatibility. It runs
when `assess_quality()` is called without an `hnr` argument.

## The metrics

| Metric                 | What it captures                                             | Role            |
|------------------------|--------------------------------------------------------------|-----------------|
| HNR                    | Periodic (voice) energy vs aperiodic (noise) energy, in dB   | Drives tier     |
| Background dBA         | Room loudness, median of recent RMS                          | Quiet-room gate |
| SNR (spectral)         | How much louder the peak bin is than the noise floor         | Informational   |
| Spectral flatness      | How "noise-like" (chaotic) vs "tonal" (structured) it is     | Informational   |
| Spectral concentration | Share of energy in the top-3 FFT bins                        | Informational   |
| Temporal variance      | Is noise sustained (fan) or transient (a passing car)        | Informational   |
| Clipping ratio         | Fraction of samples at or above full scale                   | Informational   |

## Frame processing

Audio is resampled to 16 kHz and cut into 2048-sample frames with a
1600-sample hop (about 10 frames per second). Each frame produces the
per-frame metrics above. A rolling state object keeps short histories:

- **Room loudness** is the **median** of the last ~3 s of RMS. It tracks
  sustained noise and ignores single-frame silences.
- **Temporal variance** is the standard deviation of the last ~2 s of
  noise-floor readings.
- **HNR** is aggregated over the same ~3 s window in live mode, and over
  the whole file in file mode.

## Streaming stability

Streaming is stabilized with a **hysteresis buffer**. A new tier has to
persist for 3 frames (configurable via `stability_frames`) before it is
announced, so single-frame blips do not cause flicker.

## Calibration

Background dBA is `20·log10(rms) + db_offset`. The offset is chosen so a
quiet room reads about 40 dBA. `voicequal listen --calibrate` measures
your mic's quiet and loud RMS and solves for the offset, then saves it
to `~/.voicequal/calibration.json`.

!!! note "Why not a voice activity detector?"
    The first v0.2.0 attempt built an energy-domain SNR on top of a
    concentration-based voice activity detector. It failed: sustained
    sung vowels stay tonal under heavy noise, so the detector rated
    noisy mixes as *more* voice-like than clean vocals. HNR sidesteps
    the problem by measuring periodicity directly, so no separate VAD
    is needed for this fix.
