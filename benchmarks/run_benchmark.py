"""Run voicequal on the benchmark test set and report accuracy metrics.

Reads benchmarks/manifest.csv, runs voicequal.assess() on each file
in benchmarks/test_set/, computes overall accuracy, per-category
accuracy, confusion matrix, off-by-one accuracy, and Spearman
correlation. Writes results to benchmarks/results/v{VERSION}.json.

Usage:
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --limit 20   # quick smoke run
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from scipy.stats import spearmanr

from voicequal import __version__, assess

TIER_ORDER = ["excellent", "good", "fair", "poor"]
TIER_RANK = {tier: i for i, tier in enumerate(TIER_ORDER)}

BENCHMARKS_DIR = Path(__file__).resolve().parent
TEST_SET_DIR = BENCHMARKS_DIR / "test_set"
MANIFEST_PATH = BENCHMARKS_DIR / "manifest.csv"
RESULTS_DIR = BENCHMARKS_DIR / "results"


@dataclass
class ClipResult:
    filename: str
    expected_tier: str
    predicted_tier: str
    category: str
    snr_db: float | None
    background_db: float
    snr_measured: float
    spectral_flatness: float
    spectral_concentration: float
    temporal_variance: float
    reason: str

    @property
    def correct(self) -> bool:
        return self.expected_tier == self.predicted_tier

    @property
    def off_by_one(self) -> bool:
        return abs(TIER_RANK[self.expected_tier] - TIER_RANK[self.predicted_tier]) <= 1


@dataclass
class BenchmarkReport:
    voicequal_version: str
    total_clips: int
    exact_accuracy: float
    off_by_one_accuracy: float
    spearman_correlation: float
    per_category: dict = field(default_factory=dict)
    confusion_matrix: dict = field(default_factory=dict)


def load_manifest(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run(limit: int | None = None) -> tuple[BenchmarkReport, list[ClipResult]]:
    rows = load_manifest(MANIFEST_PATH)
    if limit:
        rows = rows[:limit]

    results: list[ClipResult] = []
    for i, row in enumerate(rows, 1):
        fpath = TEST_SET_DIR / row["filename"]
        r = assess(str(fpath))
        snr_db = float(row["snr_db"]) if row["snr_db"] else None
        results.append(
            ClipResult(
                filename=row["filename"],
                expected_tier=row["expected_tier"],
                predicted_tier=r.quality,
                category=row["category"],
                snr_db=snr_db,
                background_db=r.background_db,
                snr_measured=r.snr,
                spectral_flatness=r.spectral_flatness,
                spectral_concentration=r.spectral_concentration,
                temporal_variance=r.temporal_variance,
                reason=r.reason,
            )
        )
        if i % 20 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} clips assessed...")

    # Overall metrics.
    correct = sum(1 for r in results if r.correct)
    off_by_one = sum(1 for r in results if r.off_by_one)

    # Spearman on tier ranks.
    expected_ranks = [TIER_RANK[r.expected_tier] for r in results]
    predicted_ranks = [TIER_RANK[r.predicted_tier] for r in results]
    rho, _ = spearmanr(expected_ranks, predicted_ranks)

    # Per-category accuracy.
    per_category: dict = {}
    for cat in sorted({r.category for r in results}):
        cat_results = [r for r in results if r.category == cat]
        per_category[cat] = {
            "count": len(cat_results),
            "exact_accuracy": sum(1 for r in cat_results if r.correct) / len(cat_results),
            "off_by_one_accuracy": sum(1 for r in cat_results if r.off_by_one) / len(cat_results),
        }

    # Confusion matrix: expected -> predicted -> count.
    confusion: dict = {t: {p: 0 for p in TIER_ORDER} for t in TIER_ORDER}
    for r in results:
        confusion[r.expected_tier][r.predicted_tier] += 1

    report = BenchmarkReport(
        voicequal_version=__version__,
        total_clips=len(results),
        exact_accuracy=correct / len(results),
        off_by_one_accuracy=off_by_one / len(results),
        spearman_correlation=float(rho),
        per_category=per_category,
        confusion_matrix=confusion,
    )
    return report, results


def print_report(report: BenchmarkReport) -> None:
    print(f"\n{'=' * 60}")
    print(f"voicequal {report.voicequal_version} benchmark results")
    print(f"{'=' * 60}\n")

    print(f"Test clips:              {report.total_clips}")
    print(f"Exact accuracy:          {report.exact_accuracy:.1%}")
    print(f"Off-by-one accuracy:     {report.off_by_one_accuracy:.1%}")
    print(f"Spearman correlation:    {report.spearman_correlation:+.3f}")

    print("\nPer-category accuracy:")
    print(f"{'category':<25s} {'count':>6s} {'exact':>8s} {'±1 tier':>10s}")
    for cat, m in sorted(report.per_category.items()):
        print(
            f"  {cat:<23s} {m['count']:>6d} "
            f"{m['exact_accuracy']:>7.1%} {m['off_by_one_accuracy']:>10.1%}"
        )

    print("\nConfusion matrix (rows = expected, cols = predicted):")
    header = "  " + " " * 12 + " ".join(f"{p:>10s}" for p in TIER_ORDER)
    print(header)
    for exp in TIER_ORDER:
        row = f"  {exp:>10s}: " + " ".join(
            f"{report.confusion_matrix[exp][p]:>10d}" for p in TIER_ORDER
        )
        print(row)
    print()


def save_results(report: BenchmarkReport, results: list[ClipResult]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"v{report.voicequal_version}.json"

    # Include per-clip records so future analysis is possible.
    payload = {
        "report": asdict(report),
        "clips": [asdict(r) for r in results],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None, help="Only run first N clips (for smoke tests)"
    )
    args = parser.parse_args()

    if not TEST_SET_DIR.exists() or not MANIFEST_PATH.exists():
        raise SystemExit("Missing benchmark data. Run benchmarks/generate_test_set.py first.")

    report, results = run(limit=args.limit)
    print_report(report)
    out = save_results(report, results)
    print(f"Full results written to {out}")


if __name__ == "__main__":
    main()
