# voicequal roadmap

_Last updated: 2026-09-25. Current release: v0.1.1._

voicequal answers one question: **is this recording clean enough to
process?** Today it does that with a rule-based, dependency-free DSP
pipeline that reaches 46% exact tier accuracy on a 200-clip benchmark.
This roadmap turns it into the lightweight, honest, well-benchmarked
member of the non-intrusive speech-quality field.

Guiding rules, in order:

1. **Every number is public and reproducible.** Regressions are
   documented, not hidden. See the v0.1.1 README for the bar.
2. **The core stays CPU-only and dependency-light.** Neural models are
   optional extras, never required.
3. **Fix the science before adding AI.** A model trained on a broken
   feature pipeline learns the bug.
4. **Speak the industry's language.** ITU-T P.835 terms, public
   datasets, comparisons against DNSMOS.

---

## Where the field is

| Standard / tool | What it is | How voicequal relates |
|---|---|---|
| ITU-T P.835 | Rates speech as SIG (voice), BAK (background), OVRL (overall) | The tier is a crude OVRL. Emit all three. |
| DNSMOS (Microsoft) | Neural, non-intrusive P.835 predictor | The benchmark to compare against, then distill. |
| NISQA, torchaudio SQUIM, UTMOS | Other neural quality predictors | Optional backends. |
| WADA-SNR, NIST STNR | Classic blind SNR estimators | WADA-SNR measures mixing SNR, which is the axis voicequal currently gets wrong. |
| Silero VAD, WebRTC VAD | Proven voice activity detectors | Replaces the concentration-based VAD that failed in v0.2.0 step 2. |
| DNS Challenge, VoiceBank-DEMAND, LibriMix | Public noisy-speech datasets with ground truth | Benchmark on these, not only the homemade clip set. |

---

## Phase 0: hygiene

Goal: make the repo look and behave like a maintained project.

- [x] Move the working clone to a stable location, push the stalled `v0.2.0-vad` branch.
- [x] GitHub Actions CI: pytest on Python 3.10 / 3.11 / 3.12, ruff, mypy.
- [x] `ruff` and `mypy` configured in `pyproject.toml`, codebase passes both.
- [x] `__version__` derived from package metadata so it cannot drift from `pyproject.toml`.
- [ ] `CHANGELOG.md`, `CONTRIBUTING.md`, issue and PR templates.
- [ ] Docs site (MkDocs Material) deployed to GitHub Pages.
- [x] Repo topics and description set on GitHub.
- [x] Benchmark runner records `spectral_concentration` and any new informational fields.

**Done when:** CI is green on main, docs are live, a stranger can open a
well-formed PR without asking questions.

---

## Phase 1: fix the science (v0.2.0)

Goal: raise the `very_loud_snr` category from 5% exact accuracy to 50%
or better without regressing `quiet_noise` (97.5%) or off-by-one (82%).

Why it is broken: voicequal's SNR is *spectral* (peak bin vs. floor).
A sung vowel keeps a dominant harmonic peak even when buried in noise,
so its spectral SNR reads high. The v0.2.0 step 2 experiment showed a
concentration-based VAD cannot fix this. See `handoff.md` history.

- [ ] Add **WADA-SNR** (Kim & Stern, 2008) as a blind mixing-SNR estimator. Informational field first.
- [ ] Add **harmonic-to-noise ratio (HNR)** via autocorrelation as the pure-numpy voiced/unvoiced signal.
- [ ] Add **Silero VAD** as an optional extra (`voicequal[vad]`) for the accurate path.
- [ ] Add a **clipping detector** (sample saturation ratio).
- [ ] Add a **reverberation proxy** (C50 or decay-rate estimate).
- [ ] Extend the benchmark to report **SNR error in dB** (MAE), not only tier hits.
- [ ] Prove tier separation on the benchmark for each new field **before** wiring it into `assessment.py`.
- [ ] Rewire the tier decision on the winning signal. Mirror into `live.py`.
- [ ] Re-verify live mic on real audio. Update README numbers.

**Done when:** `very_loud_snr` >= 50% exact, no regression elsewhere,
README benchmark table updated with the real numbers.

---

## Phase 2: speak the standard (v0.3.0)

Goal: make voicequal a drop-in first check in a standard speech-quality
pipeline.

- [ ] Emit **P.835-style SIG / BAK / OVRL** scores alongside the tier.
- [ ] `voicequal[neural]` extra with **DNSMOS** and **NISQA** as pluggable backends behind one interface.
- [ ] Benchmark on **DNS Challenge** and **VoiceBank-DEMAND** with published ground truth.
- [ ] Publish one comparison table: DSP path vs DNSMOS on latency, model size, and correlation with human MOS.
- [ ] Position in README: fast CPU-only pre-check, escalate to neural when needed.

**Done when:** the comparison table is in the README and the numbers
are reproducible from a single script.

---

## Phase 3: AI (v0.4.0)

Goal: the parts that make the project interesting, built on a fixed
feature pipeline.

- [ ] **Distilled quality model.** Label DNS Challenge clips with DNSMOS, train a small model (GRU or gradient boosting) on DSP features plus log-mel, export to ONNX at ~1 MB. Target: 0.9 correlation with DNSMOS at 50x lower latency.
- [ ] **Noise-type classification.** Hum, hiss, wind, crowd, broadband. Train on MUSAN and ESC-50. Restores a feature the original TypeScript service had.
- [ ] **Per-segment explainability.** Which seconds dragged the score down, with a spectrogram overlay in the CLI and web demo.
- [ ] **LLM advice layer.** Turn metrics into one sentence of actionable advice ("move away from the fan", "lower the mic gain, it is clipping").
- [ ] **MCP server.** Expose `assess` as a tool so any agent can call voicequal.

**Done when:** the ONNX model ships in the wheel, the correlation
number is in the README, and the MCP server is documented.

---

## Phase 4: reach (ongoing)

- [ ] Browser build: ONNX model in-browser or the DSP core compiled to WebAssembly, published as a JS package.
- [ ] Integrate into the singing coach app as the first production user.
- [ ] Gradio demo on Hugging Face Spaces.
- [ ] Blog post: why spectral SNR lies about sung vowels.
- [ ] Listings on awesome-speech and awesome-audio lists.
- [ ] Launch posts with the benchmark table.

---

## Loose ends

- Decide whether to commit `benchmarks/test_set/*.wav` or keep seed-based regeneration as the reproducibility story.
- Earlier planning notes referenced a "240 clips / 72.5% exact" v0.1.1 that was never implemented. Do not reuse those numbers anywhere.
