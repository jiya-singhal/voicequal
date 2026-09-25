# How it works

voicequal is built around **SNR-gated tiered assessment**. The
intuition: a loud room only matters if your voice isn't dominant. So SNR
gates the loudness penalty before it applies.

```text
SNR > 50 dB          → excellent (voice dominates completely)
SNR 35-50            → excellent unless room > 72 dBA
SNR 25-35            → depends on room loudness
SNR ≤ 25             → composite score across all four metrics
```

## The four metrics

| Metric               | What it captures                                          |
|----------------------|-----------------------------------------------------------|
| SNR                  | How much louder the peak bin is than the noise floor      |
| Spectral flatness    | How "noise-like" (chaotic) vs "tonal" (structured) it is  |
| Temporal variance    | Is noise sustained (fan) or transient (a passing car)     |
| Background dBA       | Overall room loudness, using minimum statistics tracking  |

A fifth, informational metric, **spectral concentration**, is the ratio
of energy in the top-3 loudest FFT bins to total energy. It measures how
tonal (voice-like) versus broadband (noise-like) a frame is, and it
gates the SNR fast-paths so noise-like audio cannot ride a high
spectral-SNR reading straight to `excellent`.

## Frame processing

Audio is resampled to 16 kHz and cut into overlapping frames. Each frame
produces per-frame RMS, spectral flatness, spectral concentration, SNR,
and noise floor. Those feed a rolling state object that keeps short
histories and produces the aggregated values the tier decision uses.

- **Noise floor** is the **10th percentile** of recent RMS values.
  Robust to voice bursts, and it decays when the room quiets.
- **Room loudness** is the **median** of recent RMS. It tracks sustained
  noise and ignores single-frame silences.

## Streaming stability

Streaming is stabilized with a **hysteresis buffer**. A new tier has to
persist for 3 frames (configurable via `stability_frames`) before it is
announced, so single-frame blips do not cause flicker.

## Calibration

Background dBA is `20·log10(rms) + db_offset`. The offset is chosen so a
quiet room reads about 40 dBA. `voicequal listen --calibrate` measures
your mic's quiet and loud RMS and solves for the offset, then saves it
to `~/.voicequal/calibration.json`.

!!! warning "Known weakness"
    The SNR here is a *spectral* SNR (peak bin vs. noise floor). A loud
    vocal buried in broadband noise still has a dominant harmonic peak,
    so its spectral SNR reads high even when the *mixing* SNR is poor.
    See the [benchmark](benchmark.md) for the numbers and the
    [roadmap](roadmap.md) for the fix.
