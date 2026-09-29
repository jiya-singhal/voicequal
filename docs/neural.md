# Neural backends

The core of voicequal is pure numpy/scipy and runs in about 10 ms per
clip. The `neural` extra adds reference-grade neural quality predictors
behind the same interface, so you can compare against them, calibrate to
them, or escalate to them when the cheap path is not enough.

```bash
pip install 'voicequal[neural]'
```

## DNSMOS

[DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS)
(Reddy, Gopal, Cutler; Microsoft, ICASSP 2021/2022) is a non-intrusive
predictor of the three ITU-T P.835 mean-opinion scores. It needs no clean
reference.

| Score  | Meaning                                              | Range |
|--------|------------------------------------------------------|-------|
| `sig`  | Speech signal quality                                | 1..5  |
| `bak`  | Background intrusiveness (higher = less intrusive)   | 1..5  |
| `ovrl` | Overall quality                                      | 1..5  |

```python
from voicequal.neural import DNSMOS

scorer = DNSMOS()  # downloads the 1.1 MB ONNX model once
scores = scorer.score_file("clip.wav")  # or scorer(samples, sample_rate)
print(scores.sig, scores.bak, scores.ovrl)
```

The model file (`sig_bak_ovr.onnx`, MIT licence) is fetched from the
DNS-Challenge repository on first use, verified against a pinned SHA-256,
and cached in `~/.voicequal/models/`. Set `VOICEQUAL_MODEL_DIR` to move
the cache, or pass `model_path=` to use a file you already have.

Inference mirrors Microsoft's reference script: clips shorter than 9.01 s
are tiled, longer clips are scored in 9.01 s windows hopping 1 s, and raw
outputs go through the published polynomial mapping before averaging.

!!! warning "DNSMOS is a speech model"
    It does not transfer to singing. On voicequal's VocalSet benchmark it
    scores clean sung vowels around 1.1 on every axis, the same as white
    noise. That is why the DSP-vs-DNSMOS comparison in the
    [benchmark](benchmark.md) uses a public *speech* set.

## When to use which

| Path        | Latency (3.6 s clip) | Needs                 | Best for                                  |
|-------------|----------------------|-----------------------|-------------------------------------------|
| DSP (core)  | ~10 ms               | numpy, scipy          | Live mic checks, gating before ASR, singing |
| DNSMOS      | ~500 ms              | onnxruntime, 1 MB model | Offline scoring of speech, human-MOS proxy |

`dnsmos_available()` tells you at runtime whether the extra is installed.
