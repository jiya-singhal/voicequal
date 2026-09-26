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

## Results

| Metric                | v0.1.1 | **v0.2.0** |
|-----------------------|--------|------------|
| Exact tier accuracy   | 46.0%  | **55.5%**  |
| Off-by-one accuracy   | 82.0%  | **89.5%**  |
| Spearman correlation  | +0.496 | **+0.662** |

### Per-category exact accuracy

| Category        | Expected tier | v0.1.1 | **v0.2.0** |
|-----------------|---------------|--------|------------|
| `quiet_noise`   | excellent     | 97.5%  | **100%**   |
| `clean_vocal`   | excellent     | 50.0%  | **55.0%**  |
| `moderate_snr`  | good          | 50.0%  | 32.5%      |
| `loud_snr`      | fair          | 27.5%  | **57.5%**  |
| `very_loud_snr` | poor          | 5.0%   | **32.5%**  |

### Confusion matrix, v0.2.0 (expected → predicted)

| Expected \ Predicted | excellent | good | fair | poor |
|---|---|---|---|---|
| excellent | 62 | 12 | 6 | 0 |
| good | 18 | 13 | 8 | 1 |
| fair | 4 | 10 | 23 | 3 |
| poor | 4 | 6 | 17 | 13 |

For comparison, v0.1.1 predicted `poor` 5 times in 200 (2 correctly) and
called 9 of the 40 heavily-noised clips `excellent`. Gross errors (two
or more tiers off) fell from 36 to 21.

### How well each estimator tracks the true mixing SNR

Measured on the 120 mixed clips, whose mixing SNR is known (20, 10, 5 dB).
The benchmark runner reports this table on every run.

| Estimator      | MAE     | Bias     | Spearman |
|----------------|---------|----------|----------|
| HNR            | 4.35 dB | −0.34 dB | +0.483   |
| Spectral SNR   | 28.1 dB | +28.1 dB | +0.218   |

## What changed in v0.2.0

One new metric, `hnr`, and a new decision path that uses it. See
[How it works](how-it-works.md).

Also tried, and rejected with numbers:

- **WADA-SNR** (Kim & Stern, 2008) assumes Gamma-distributed speech
  amplitudes. Sung vowels do not fit. MAE 13.5 dB against mixing SNR.
- **Percentile level SNR** (loud frames vs quiet frames) needs gaps
  between phrases. Sustained scales have none. MAE about 6 dB and no
  tier separation.
- **Pre-emphasis before autocorrelation** whitens the harmonics along
  with the noise and lowers HNR separation.
- **A concentration-based voice activity detector** (the earlier
  `v0.2.0-vad` branch) rated noisy mixes as *more* voice-like than clean
  vocals.

## Where it still falls short

**`moderate_snr` regressed** from 50% to 32.5%. Those 20 dB mixes now
mostly read `excellent`, one tier off. Clean VocalSet recordings and
20 dB mixes overlap heavily in HNR because breathy, lip-trill and
vibrato techniques are themselves aperiodic. No single threshold
separates them well.

**10 dB and 5 dB mixes barely separate.** HNR compresses at low SNR
because the autocorrelation peak of pure noise is not zero. Moving the
fair/poor threshold from 7 dB to 10 dB lifts `very_loud_snr` to 67.5%
but drops `loud_snr` to 12.5% and off-by-one to 87%. The shipped 7 dB
favours fewer gross errors.

**The thresholds were tuned on this test set.** As a ceiling check, a
cross-validated random forest on all seven metrics reaches 61.5% exact
and 93.5% off-by-one here, so the hand rule is within about six points
of what these features allow on this data. Without HNR the same forest
gets 50.0%. The numbers will move on real speech, which is why the
[roadmap](roadmap.md) moves to public speech datasets next.
