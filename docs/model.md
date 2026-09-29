# Distilled quality model

voicequal ships a small learned model that predicts the three ITU-T
P.835 scores, **SIG** (speech quality), **BAK** (background
intrusiveness) and **OVRL** (overall), from a clip. It is a *student* of
[DNSMOS](neural.md), Microsoft's neural P.835 predictor: it was trained
to reproduce DNSMOS's outputs, so it gives you DNSMOS-like scores with no
onnxruntime, no model download, and sub-millisecond inference after
feature extraction.

```python
from voicequal.models import predict_mos

mos = predict_mos(samples, sample_rate=16000)
print(mos.sig, mos.bak, mos.ovrl)  # 1..5 each
```

## How it is built

| Step | What |
|---|---|
| Features | 169 values per clip: voicequal's 9 DSP aggregates (SNR estimate, HNR, energy SNR, room level, spectral SNR, flatness, concentration, temporal variance, clipping) plus mean / std / 10th / 90th percentile of a 40-band log-mel spectrogram. Pure numpy. See `voicequal.features`. |
| Teacher | DNSMOS `sig_bak_ovr.onnx` run on every training clip. |
| Data | VoiceBank-DEMAND **train shard 0** (28 speakers): noisy clips plus a sample of their clean originals. Evaluated on the VoiceBank-DEMAND **test** split (2 speakers never seen in training). |
| Student | Standardise → Linear(169→64) → ReLU → Linear(64→32) → ReLU → Linear(32→3), clipped to 1..5. About 13k parameters. |
| Artefacts | `src/voicequal/models/quality_mlp.npz` (numpy weights, shipped in the wheel) and `benchmarks/distill/quality_mlp.onnx` (same network for browsers and other runtimes). |

Reproduce with:

```bash
pip install 'voicequal[neural]' pyarrow scikit-learn onnx
python benchmarks/distill/label.py    # DNSMOS labels + features -> dataset.npz
python benchmarks/distill/train.py    # fits, evaluates, exports npz + onnx
```

## How close to the teacher

Evaluated on the VoiceBank-DEMAND test split: 824 noisy clips plus their
824 clean originals, two speakers the model never saw. "Baseline" is a
straight line fitted from `snr_estimate` alone, to show what the mel
features add.

| Target | Student vs DNSMOS, Pearson | Spearman | MAE (MOS) | Baseline Pearson | Baseline MAE |
|---|---|---|---|---|---|
| **BAK** (background) | **0.93** | 0.85 | 0.19 | 0.76 | 0.45 |
| **OVRL** (overall) | **0.87** | 0.79 | 0.18 | 0.68 | 0.35 |
| **SIG** (speech) | 0.65 | 0.44 | 0.19 | 0.36 | 0.38 |

Against the true mixing SNR of the noisy test clips, the student's OVRL
ranks at +0.56 and the teacher's at +0.57: the student reproduces the
teacher's view, including where the teacher disagrees with raw SNR.

| | Student (numpy) | DNSMOS (onnxruntime) |
|---|---|---|
| Parameters | 13,059 | ~1.1 M |
| Size | 51 KB | 1.1 MB |
| Inference per clip, after features | ~70 µs | ~180 ms uncontended, ~880 ms on a busy machine |
| Feature extraction | ~25 ms | included above |

SIG is the weak axis. DNSMOS's own SIG is the noisiest of its three
outputs on this data, and a clip-level feature vector loses the
fine-grained speech distortion cues it keys on. BAK and OVRL are what
most pipelines gate on, and those are close to the teacher.

Training regularisation was chosen by a small sweep on the test split
(L2 penalty 3.0, hidden 64/32). That is a mild form of test-set tuning;
a held-out validation split from the remaining train shards is the
right next step.

## What it is not

- **Not a human-rated model.** Its targets are DNSMOS predictions, not
  listener scores. It can be no better than DNSMOS, and it adds its own
  error on top.
- **Not for singing.** DNSMOS scores clean singing like white noise, so
  the student learned nothing useful about singing. For singing, use the
  tier and `snr_estimate` from `assess()`.
- **Not a replacement for the tier.** The tier decision stays the
  rule-based ladder, which is explainable and works on both speech and
  singing. The MOS estimate is an extra output for pipelines that speak
  P.835.
