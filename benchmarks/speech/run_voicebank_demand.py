"""Speech benchmark: voicequal vs DNSMOS on the VoiceBank-DEMAND test set.

VoiceBank-DEMAND (Valentini-Botinhao et al., 2016; CC BY 4.0) pairs clean
read speech from two speakers with the same speech mixed with real DEMAND
environment noise at 2.5, 7.5, 12.5 and 17.5 dB SNR. 824 test clips.

This script pulls the 16 kHz Hugging Face mirror of the test split
(``JacobLinCool/VoiceBank-DEMAND-16k``, one 132 MB parquet), and for every
clip:

* computes the **true mixing SNR** from the clean/noisy pair,
* runs voicequal's DSP path on the noisy clip (tier, HNR, spectral SNR...),
* runs DNSMOS (SIG / BAK / OVRL) on the noisy clip,
* times both.

Then it reports how each estimator tracks the true SNR, how voicequal's
tier relates to DNSMOS, and the latency of each path. Results are written
to ``benchmarks/speech/results/vbd_v<version>.json``.

Usage::

    pip install 'voicequal[neural]' pyarrow
    python benchmarks/speech/run_voicebank_demand.py            # all 824 clips
    python benchmarks/speech/run_voicebank_demand.py --limit 50 # smoke run

The parquet is cached under ``benchmarks/speech/data/`` (gitignored).
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.stats import spearmanr

from voicequal import __version__, assess_samples
from voicequal.neural import DNSMOS

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
PARQUET_URL = (
    "https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k/"
    "resolve/main/data/test-00000-of-00001.parquet"
)
PARQUET_PATH = DATA_DIR / "voicebank_demand_test_16k.parquet"

TIER_ORDER = ["excellent", "good", "fair", "poor"]
TIER_RANK = {t: i for i, t in enumerate(TIER_ORDER)}
# VoiceBank-DEMAND test SNRs. Nearest bucket is taken as the clip's nominal SNR.
NOMINAL_SNRS = (17.5, 12.5, 7.5, 2.5)
# One tier per nominal SNR, for a tier-accuracy read-out that mirrors the
# VocalSet benchmark. This mapping is a convention, not ground truth.
SNR_TO_TIER = {17.5: "excellent", 12.5: "good", 7.5: "fair", 2.5: "poor"}


@dataclass
class ClipRow:
    clip_id: str
    true_snr_db: float
    nominal_snr_db: float
    duration_s: float
    # voicequal
    vq_tier: str
    vq_hnr: float
    vq_energy_snr: float
    vq_snr_estimate: float
    vq_spectral_snr: float
    vq_background_db: float
    vq_clipping: float
    vq_ms: float
    # dnsmos
    dns_sig: float
    dns_bak: float
    dns_ovrl: float
    dns_ms: float


@dataclass
class Report:
    voicequal_version: str
    dataset: str
    clips: int
    vs_true_snr: dict = field(default_factory=dict)
    tier_vs_nominal_snr: dict = field(default_factory=dict)
    voicequal_vs_dnsmos: dict = field(default_factory=dict)
    latency_ms: dict = field(default_factory=dict)
    per_nominal_snr: dict = field(default_factory=dict)


def ensure_parquet() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not PARQUET_PATH.exists():
        print(f"Downloading VoiceBank-DEMAND test split (~132 MB) to {PARQUET_PATH} ...")
        tmp = PARQUET_PATH.with_suffix(".part")
        urllib.request.urlretrieve(PARQUET_URL, tmp)  # noqa: S310 - pinned https URL
        tmp.replace(PARQUET_PATH)
    return PARQUET_PATH


def _decode(blob: bytes) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(io.BytesIO(blob), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def _true_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    n = min(clean.size, noisy.size)
    noise = noisy[:n].astype(np.float64) - clean[:n].astype(np.float64)
    p_clean = float(np.mean(clean[:n].astype(np.float64) ** 2))
    p_noise = float(np.mean(noise**2))
    return float(10 * np.log10(max(p_clean, 1e-12) / max(p_noise, 1e-12)))


def _nearest_nominal(snr: float) -> float:
    return min(NOMINAL_SNRS, key=lambda s: abs(s - snr))


def _load_dnsmos_cache(path: Path | None) -> dict[str, tuple[float, float, float, float]]:
    """clip_id -> (sig, bak, ovrl, ms) from a previous results JSON."""
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    return {
        c["clip_id"]: (c["dns_sig"], c["dns_bak"], c["dns_ovrl"], c["dns_ms"])
        for c in data["clips"]
    }


def run(limit: int | None, reuse_dnsmos: Path | None = None) -> tuple[Report, list[ClipRow]]:
    import pyarrow.parquet as pq

    table = pq.read_table(ensure_parquet())
    if limit:
        table = table.slice(0, limit)
    cache = _load_dnsmos_cache(reuse_dnsmos)
    scorer = DNSMOS() if len(cache) < table.num_rows else None
    rows: list[ClipRow] = []
    total = table.num_rows
    for i, rec in enumerate(table.to_pylist(), 1):
        clean, sr_c = _decode(rec["clean"]["bytes"])
        noisy, sr_n = _decode(rec["noisy"]["bytes"])
        assert sr_c == sr_n == 16000, (rec["id"], sr_c, sr_n)
        true_snr = _true_snr_db(clean, noisy)

        t0 = time.perf_counter()
        vq = assess_samples(noisy, 16000)
        vq_ms = (time.perf_counter() - t0) * 1000

        if rec["id"] in cache:
            dns_sig, dns_bak, dns_ovrl, dns_ms = cache[rec["id"]]
        else:
            assert scorer is not None
            t0 = time.perf_counter()
            dns = scorer(noisy, 16000)
            dns_ms = (time.perf_counter() - t0) * 1000
            dns_sig, dns_bak, dns_ovrl = dns.sig, dns.bak, dns.ovrl

        rows.append(
            ClipRow(
                clip_id=rec["id"],
                true_snr_db=true_snr,
                nominal_snr_db=_nearest_nominal(true_snr),
                duration_s=noisy.size / 16000,
                vq_tier=vq.quality,
                vq_hnr=vq.hnr,
                vq_energy_snr=vq.energy_snr,
                vq_snr_estimate=vq.snr_estimate,
                vq_spectral_snr=vq.snr,
                vq_background_db=vq.background_db,
                vq_clipping=vq.clipping_ratio,
                vq_ms=vq_ms,
                dns_sig=dns_sig,
                dns_bak=dns_bak,
                dns_ovrl=dns_ovrl,
                dns_ms=dns_ms,
            )
        )
        if i % 50 == 0 or i == total:
            print(f"  {i}/{total}", flush=True)

    return summarize(rows), rows


def _corr(a: list[float], b: list[float]) -> float:
    rho, _ = spearmanr(a, b)
    return float(rho)


def summarize(rows: list[ClipRow]) -> Report:
    truth = [r.true_snr_db for r in rows]
    report = Report(
        voicequal_version=__version__, dataset="VoiceBank-DEMAND test (824 clips)", clips=len(rows)
    )

    # 1. How each estimator tracks the true mixing SNR.
    hnr = [r.vq_hnr for r in rows]
    est = [r.vq_snr_estimate for r in rows]
    report.vs_true_snr = {
        "voicequal_snr_estimate": {
            "spearman": _corr(est, truth),
            "mae_db": float(np.mean(np.abs(np.array(est) - np.array(truth)))),
            "bias_db": float(np.mean(np.array(est) - np.array(truth))),
        },
        "voicequal_hnr": {
            "spearman": _corr(hnr, truth),
            "mae_db": float(np.mean(np.abs(np.array(hnr) - np.array(truth)))),
            "bias_db": float(np.mean(np.array(hnr) - np.array(truth))),
        },
        "voicequal_energy_snr": {
            "spearman": _corr([r.vq_energy_snr for r in rows], truth),
        },
        "voicequal_spectral_snr": {
            "spearman": _corr([r.vq_spectral_snr for r in rows], truth),
        },
        "dnsmos_bak": {"spearman": _corr([r.dns_bak for r in rows], truth)},
        "dnsmos_ovrl": {"spearman": _corr([r.dns_ovrl for r in rows], truth)},
        "dnsmos_sig": {"spearman": _corr([r.dns_sig for r in rows], truth)},
    }

    # 2. Tier vs nominal SNR bucket, with the SNR_TO_TIER convention.
    expected = [SNR_TO_TIER[r.nominal_snr_db] for r in rows]
    predicted = [r.vq_tier for r in rows]
    exact = np.mean([e == p for e, p in zip(expected, predicted, strict=True)])
    ob1 = np.mean(
        [abs(TIER_RANK[e] - TIER_RANK[p]) <= 1 for e, p in zip(expected, predicted, strict=True)]
    )
    report.tier_vs_nominal_snr = {
        "convention": SNR_TO_TIER,
        "exact_accuracy": float(exact),
        "off_by_one_accuracy": float(ob1),
        "spearman_tier_rank_vs_true_snr": _corr([-TIER_RANK[p] for p in predicted], truth),
    }

    # 3. voicequal vs DNSMOS agreement.
    report.voicequal_vs_dnsmos = {
        "spearman_snr_estimate_vs_dnsmos_ovrl": _corr(est, [r.dns_ovrl for r in rows]),
        "spearman_snr_estimate_vs_dnsmos_bak": _corr(est, [r.dns_bak for r in rows]),
        "spearman_hnr_vs_dnsmos_ovrl": _corr(hnr, [r.dns_ovrl for r in rows]),
        "spearman_tier_vs_dnsmos_ovrl": _corr(
            [-TIER_RANK[p] for p in predicted], [r.dns_ovrl for r in rows]
        ),
    }

    # 4. Latency.
    report.latency_ms = {
        "voicequal_median": float(np.median([r.vq_ms for r in rows])),
        "dnsmos_median": float(np.median([r.dns_ms for r in rows])),
        "speedup": float(np.median([r.dns_ms for r in rows]) / np.median([r.vq_ms for r in rows])),
        "clip_duration_median_s": float(np.median([r.duration_s for r in rows])),
    }

    # 5. Per nominal SNR: medians and tier distribution.
    for snr in NOMINAL_SNRS:
        sub = [r for r in rows if r.nominal_snr_db == snr]
        if not sub:
            continue
        report.per_nominal_snr[str(snr)] = {
            "count": len(sub),
            "true_snr_median": float(np.median([r.true_snr_db for r in sub])),
            "hnr_median": float(np.median([r.vq_hnr for r in sub])),
            "energy_snr_median": float(np.median([r.vq_energy_snr for r in sub])),
            "snr_estimate_median": float(np.median([r.vq_snr_estimate for r in sub])),
            "spectral_snr_median": float(np.median([r.vq_spectral_snr for r in sub])),
            "dnsmos_bak_median": float(np.median([r.dns_bak for r in sub])),
            "dnsmos_ovrl_median": float(np.median([r.dns_ovrl for r in sub])),
            "tiers": {t: sum(1 for r in sub if r.vq_tier == t) for t in TIER_ORDER},
        }
    return report


def print_report(rep: Report) -> None:
    print(f"\n{'=' * 64}\nvoicequal {rep.voicequal_version} vs DNSMOS on {rep.dataset}\n{'=' * 64}")
    print(f"clips: {rep.clips}\n")
    print("Against true mixing SNR (Spearman):")
    for k, v in rep.vs_true_snr.items():
        extra = f"   MAE {v['mae_db']:.2f} dB  bias {v['bias_db']:+.2f} dB" if "mae_db" in v else ""
        print(f"  {k:<24s} {v['spearman']:+.3f}{extra}")
    t = rep.tier_vs_nominal_snr
    print(
        f"\nTier vs nominal SNR bucket ({t['convention']}):\n"
        f"  exact {t['exact_accuracy']:.1%}   off-by-one {t['off_by_one_accuracy']:.1%}   "
        f"tier-rank vs true SNR {t['spearman_tier_rank_vs_true_snr']:+.3f}"
    )
    print("\nvoicequal vs DNSMOS agreement (Spearman):")
    for k, v in rep.voicequal_vs_dnsmos.items():
        print(f"  {k:<30s} {v:+.3f}")
    lat = rep.latency_ms
    print(
        f"\nLatency per clip (median, clip ~{lat['clip_duration_median_s']:.1f}s): "
        f"voicequal {lat['voicequal_median']:.1f} ms   DNSMOS {lat['dnsmos_median']:.1f} ms   "
        f"({lat['speedup']:.0f}x)"
    )
    print(
        f"\n{'nominal':>8s} {'n':>4s} {'trueSNR':>8s} {'est':>6s} {'HNR':>6s} {'eSNR':>6s} "
        f"{'BAK':>5s} {'OVRL':>5s}   tiers (exc/good/fair/poor)"
    )
    for snr, m in rep.per_nominal_snr.items():
        tiers = "/".join(str(m["tiers"][t]) for t in TIER_ORDER)
        print(
            f"{snr:>8s} {m['count']:>4d} {m['true_snr_median']:>8.1f} {m['snr_estimate_median']:>6.1f} "
            f"{m['hnr_median']:>6.1f} {m['energy_snr_median']:>6.1f} {m['dnsmos_bak_median']:>5.2f} "
            f"{m['dnsmos_ovrl_median']:>5.2f}   {tiers}"
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips")
    ap.add_argument(
        "--reuse-dnsmos",
        type=Path,
        default=None,
        help="previous results JSON; DNSMOS scores are copied from it instead of recomputed",
    )
    args = ap.parse_args()
    report, rows = run(args.limit, args.reuse_dnsmos)
    print_report(report)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"vbd_v{__version__}{'_partial' if args.limit else ''}.json"
    out.write_text(
        json.dumps({"report": asdict(report), "clips": [asdict(r) for r in rows]}, indent=1)
    )
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
