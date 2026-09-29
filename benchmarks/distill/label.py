"""Build the distillation dataset: clip features + DNSMOS labels.

Uses the VoiceBank-DEMAND parquets under benchmarks/speech/data/:
  - train shard 0 (noisy + clean columns), speakers disjoint from test
  - test split (noisy + clean)
For every clip we store voicequal's 169-d feature vector, DNSMOS
(sig, bak, ovrl), true mixing SNR (noisy clips only; NaN for clean),
split name, and clip id. Output: benchmarks/distill/data/dataset.npz.

DNSMOS is the slow part (~0.3-0.9 s per clip). Noisy test-clip scores
are reused from benchmarks/speech/results/vbd_v*.json when present.

Usage:
    python benchmarks/distill/label.py [--max-train N] [--clean-train N]
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from voicequal.features import FEATURE_NAMES, clip_features
from voicequal.neural import DNSMOS

HERE = Path(__file__).resolve().parent
SPEECH = HERE.parent / "speech"
OUT = HERE / "data" / "dataset.npz"


def decode(blob: bytes) -> np.ndarray:
    a, sr = sf.read(io.BytesIO(blob), dtype="float32")
    assert sr == 16000
    return a if a.ndim == 1 else a.mean(axis=1).astype(np.float32)


def true_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    n = min(clean.size, noisy.size)
    noise = noisy[:n].astype(np.float64) - clean[:n]
    return float(10 * np.log10(np.mean(clean[:n] ** 2) / max(np.mean(noise**2), 1e-12)))


def cached_test_scores() -> dict[str, tuple[float, float, float]]:
    files = sorted(glob.glob(str(SPEECH / "results" / "vbd_v*.json")))
    if not files:
        return {}
    data = json.loads(Path(files[-1]).read_text())
    return {c["clip_id"]: (c["dns_sig"], c["dns_bak"], c["dns_ovrl"]) for c in data["clips"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--clean-train", type=int, default=600, help="clean train clips to include")
    args = ap.parse_args()

    scorer = DNSMOS()
    cache = cached_test_scores()
    rng = np.random.default_rng(0)

    X: list[np.ndarray] = []
    Y: list[tuple[float, float, float]] = []
    snr: list[float] = []
    split: list[str] = []
    ids: list[str] = []
    kind: list[str] = []

    def add(samples, label, clip_id, sp, k, s):
        X.append(clip_features(samples))
        Y.append(label)
        snr.append(s)
        split.append(sp)
        ids.append(clip_id)
        kind.append(k)

    t0 = time.perf_counter()
    for sp, fname in (
        ("train", "voicebank_demand_train0_16k.parquet"),
        ("test", "voicebank_demand_test_16k.parquet"),
    ):
        table = pq.read_table(SPEECH / "data" / fname)
        rows = table.to_pylist()
        if sp == "train" and args.max_train:
            rows = rows[: args.max_train]
        clean_pick = set(
            rng.choice(
                len(rows),
                size=min(args.clean_train if sp == "train" else len(rows), len(rows)),
                replace=False,
            ).tolist()
        )
        for i, rec in enumerate(rows):
            clean = decode(rec["clean"]["bytes"])
            noisy = decode(rec["noisy"]["bytes"])
            s = true_snr(clean, noisy)
            if sp == "test" and rec["id"] in cache:
                label = cache[rec["id"]]
            else:
                d = scorer(noisy)
                label = (d.sig, d.bak, d.ovrl)
            add(noisy, label, rec["id"], sp, "noisy", s)
            if i in clean_pick:
                d = scorer(clean)
                add(clean, (d.sig, d.bak, d.ovrl), rec["id"] + "_clean", sp, "clean", float("nan"))
            if (i + 1) % 100 == 0:
                print(f"  {sp} {i + 1}/{len(rows)}  ({time.perf_counter() - t0:.0f}s)", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        X=np.stack(X),
        Y=np.array(Y, dtype=np.float32),
        true_snr=np.array(snr, dtype=np.float32),
        split=np.array(split),
        ids=np.array(ids),
        kind=np.array(kind),
        feature_names=np.array(FEATURE_NAMES),
    )
    print(
        f"wrote {OUT}: {len(X)} clips, {len(FEATURE_NAMES)} features, {time.perf_counter() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
