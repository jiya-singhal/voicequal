# Benchmark

voicequal is benchmarked against a fixed, reproducible test set of 200
clips generated from two public research datasets, VocalSet (clean
vocals) and MUSAN (environmental noise), mixed at controlled
signal-to-noise ratios. The test set regenerates deterministically from
`seed=42`.

```bash
poetry run python benchmarks/run_benchmark.py
```

See `benchmarks/README.md` for how to rebuild the clips from the raw
datasets.

## Results for v0.1.1

| Metric                | Value  |
|-----------------------|--------|
| Exact tier accuracy   | 46.0%  |
| Off-by-one accuracy   | 82.0%  |
| Spearman correlation  | +0.496 |

Read this honestly: exact accuracy is modest and off-by-one is the more
flattering number. voicequal orders quality roughly correctly (positive
rank correlation) but frequently lands one tier off. It rarely confuses
`excellent` for `poor`, but it does confuse neighbours. The quiet-room
categories are strong; the noisy-mix categories are weak.

### Per-category exact accuracy

| Category        | Expected tier | Exact accuracy |
|-----------------|---------------|----------------|
| `quiet_noise`   | excellent     | 97.5%          |
| `clean_vocal`   | excellent     | 50.0%          |
| `moderate_snr`  | good          | 50.0%          |
| `loud_snr`      | fair          | 27.5%          |
| `very_loud_snr` | poor          | 5.0%           |

### Confusion matrix (expected → predicted)

| Expected \ Predicted | excellent | good | fair | poor |
|---|---|---|---|---|
| excellent | 59 | 20 | 1 | 0 |
| good | 15 | 20 | 5 | 0 |
| fair | 10 | 16 | 11 | 3 |
| poor | 9 | 16 | 13 | 2 |

The model almost never predicts `poor`: 5 of 200 predictions.

## What changed in v0.1.1

The one algorithmic change is the `spectral_concentration` metric,
which gates the SNR fast-paths. In candour: on this test set the gate
**slightly reduced** exact accuracy (from 47.0% to 46.0%). It was
retained because it adds a genuinely useful diagnostic signal and the
regression is within noise, but it is not the win the version bump
might imply.

## Known weakness

The `very_loud_snr` category, voice mixed with noise at ~5 dB SNR,
scores just **5.0%** exact accuracy and is the clear bottleneck.

Root cause: voicequal's SNR is a *spectral* SNR (peak bin vs. noise
floor), but these clips are controlled by *mixing* SNR (voice RMS vs.
noise RMS). A loud vocal buried in noise still shows a dominant harmonic
peak, so its spectral SNR reads high (median ~38 dB across all three
noisy tiers) and the clip is waved through the fast-path.
`spectral_concentration` does not catch it either, because sustained
sung vowels stay tonal even under heavy noise.

A first attempt at a fix (an energy-domain SNR gated by a
concentration-based voice activity detector, on the `v0.2.0-vad`
branch) was a **no-go**: the detector rated noisy mixes as *more*
voice-like than clean vocals. The [roadmap](roadmap.md) describes what
comes next.
