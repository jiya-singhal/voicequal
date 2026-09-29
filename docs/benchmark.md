# Benchmark

voicequal is measured on two public test sets, one per kind of voice
audio. Each has a single script; every number on this page comes from
running it.

| Set | What it is | Clips | Ground truth |
|---|---|---|---|
| **VoiceBank-DEMAND** (speech) | Read English sentences from two speakers, mixed with real DEMAND environment noise at 17.5 / 12.5 / 7.5 / 2.5 dB. CC BY 4.0. | 824 | Mixing SNR computed from each clean/noisy pair |
| **VocalSet + MUSAN** (singing) | Sung scales in many techniques mixed with MUSAN noise at 20 / 10 / 5 dB, plus clean vocals and noise-only clips. Regenerates from `seed=42`. | 200 | Manifest SNR |

```bash
pip install 'voicequal[neural]' pyarrow
poetry run python benchmarks/speech/run_voicebank_demand.py   # downloads 132 MB once
poetry run python benchmarks/run_benchmark.py
```

## Speech: VoiceBank-DEMAND

Tiers are scored against the four nominal SNR levels with the
convention 17.5 dB → excellent, 12.5 → good, 7.5 → fair, 2.5 → poor.
That is an SNR-band check, not a human rating.

| | v0.2.0 (HNR only) | **v0.3.0** |
|---|---|---|
| Exact band | 27.9% | **80.0%** |
| Within one band | 63.3% | **99.2%** |
| Tier rank vs true SNR (Spearman) | +0.12 | **+0.90** |

### Per nominal SNR, v0.3.0

| Nominal SNR | n | True SNR (median) | `snr_estimate` | HNR | Energy SNR | Tiers exc / good / fair / poor |
|---|---|---|---|---|---|---|
| 17.5 dB | 191 | 16.1 | 19.8 | 6.8 | 19.8 | **172** / 19 / 0 / 0 |
| 12.5 dB | 206 | 11.1 | 16.2 | 6.1 | 16.2 | 24 / **181** / 1 / 0 |
| 7.5 dB | 209 | 6.2 | 12.3 | 5.9 | 12.3 | 3 / 49 / **150** / 7 |
| 2.5 dB | 218 | 1.1 | 8.7 | 5.1 | 8.4 | 1 / 3 / 58 / **156** |

HNR barely moves across the four bands on speech. The energy SNR
separates them cleanly and over-reads by a roughly constant 4 to 6 dB
(the loudest half of blocks is louder than the average speech level),
which the ladder absorbs.

### Estimators against true mixing SNR

| Estimator | Spearman | Notes |
|---|---|---|
| **voicequal `snr_estimate`** | **+0.903** | MAE 5.85 dB, bias +5.85 dB |
| voicequal energy SNR | +0.912 | the component doing the work on speech |
| DNSMOS OVRL | +0.571 | neural P.835 predictor, speech-trained |
| DNSMOS BAK | +0.524 | |
| DNSMOS SIG | +0.439 | |
| voicequal HNR | +0.192 | MAE 5.12 dB, bias −2.21 dB |
| voicequal spectral SNR (v0.1.x) | −0.164 | |

DNSMOS predicts listener opinion, not SNR, so its lower correlation with
SNR is not a defect. voicequal's estimate agrees with DNSMOS OVRL at
+0.60 and with DNSMOS BAK at +0.61.

### Latency

| Path | Median per 2.3 s clip | Runtime |
|---|---|---|
| voicequal DSP | 15 ms | numpy + scipy |
| DNSMOS | 880 ms | onnxruntime, 1.1 MB model |

Measured on one laptop CPU, single thread, including the 9.01 s tiling
DNSMOS requires for short clips.

## Singing: VocalSet + MUSAN

| Metric | v0.1.1 | v0.2.0 | **v0.3.0** |
|---|---|---|---|
| Exact tier accuracy | 46.0% | 55.5% | **58.0%** |
| Off-by-one accuracy | 82.0% | 89.5% | 86.0% |
| Spearman correlation | +0.496 | +0.662 | +0.645 |

| Category | Expected | v0.1.1 | v0.2.0 | **v0.3.0** |
|---|---|---|---|---|
| `quiet_noise` | excellent | 97.5% | 100% | **100%** |
| `clean_vocal` | excellent | 50.0% | 55.0% | 30.0% |
| `moderate_snr` (20 dB) | good | 50.0% | 32.5% | **45.0%** |
| `loud_snr` (10 dB) | fair | 27.5% | 57.5% | 55.0% |
| `very_loud_snr` (5 dB) | poor | 5.0% | 32.5% | **60.0%** |

### Confusion matrix, v0.3.0 (expected → predicted)

| Expected \ Predicted | excellent | good | fair | poor |
|---|---|---|---|---|
| excellent | 52 | 14 | 10 | 4 |
| good | 10 | 18 | 9 | 3 |
| fair | 1 | 6 | 22 | 11 |
| poor | 1 | 9 | 6 | 24 |

### Estimators against manifest mixing SNR (120 mixed clips)

| Estimator | MAE | Bias | Spearman |
|---|---|---|---|
| `snr_estimate` | 4.24 dB | +0.87 dB | +0.501 |
| HNR | 4.35 dB | −0.34 dB | +0.483 |
| Energy SNR | 5.96 dB | −3.60 dB | +0.193 |
| Spectral SNR (v0.1.x) | 28.1 dB | +28.1 dB | +0.218 |

On sustained scales there are no pauses, so the energy SNR collapses and
HNR carries the estimate. That is the intended behaviour of
`max(hnr, energy_snr)`.

## What was tried and rejected

- **WADA-SNR** (Kim & Stern, 2008): assumes Gamma-distributed speech
  amplitudes; sung vowels do not fit. MAE 13.5 dB on singing.
- **Percentile level SNR on singing**: needs pauses; sustained scales
  have none. It is what became the energy SNR once speech was added.
- **Pre-emphasis before autocorrelation**: whitens harmonics with the
  noise and lowers HNR separation.
- **Concentration-based voice activity detector** (the `v0.2.0-vad`
  branch): rated noisy mixes as *more* voice-like than clean vocals.
- **DNSMOS on singing**: scores clean sung vowels ~1.1 on every axis,
  the same as white noise. Speech-only.

## Read this honestly

- **Speech is the strong case.** 80% in the right SNR band, 99% within
  one, with a 15 ms pure-numpy path.
- **Singing is harder and the numbers are modest.** HNR compresses at
  low SNR, and VocalSet's breathy, lip-trill and vibrato techniques are
  aperiodic by nature, so clean vocals now read `good` 55% of the time.
  That is a limitation of a periodicity estimator, not a threshold to
  tune away.
- **The ladder was fitted on these two sets.** One set of thresholds
  serves both; the joint fit favoured speech accuracy and singing
  off-by-one over singing exact. As a ceiling check on the singing set,
  a cross-validated random forest on all metrics reached 61.5% exact.
- **No human opinion scores yet.** Both sets provide SNR ground truth.
  Correlation with listener ratings (P.835 MOS) is the next thing to
  measure, and the [roadmap](roadmap.md) says where.
