"""Generate a reproducible test set for voicequal benchmarking.

Reads VocalSet + MUSAN files, mixes at controlled SNRs, writes
200 test WAVs + a manifest CSV. Deterministic (seed=42).

Usage:
    python benchmarks/generate_test_set.py \
        --vocalset /Volumes/UnitySSD/VocalSet1-2 \
        --musan /Volumes/UnitySSD/noise-testing/musan \
        --output benchmarks/test_set
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 16000
CLIP_DURATION_S = 3.0
NORMALIZE_RMS_DBFS = -20.0
SEED = 42

# Per-category counts.
N_CLEAN_VOCAL = 40
N_QUIET_NOISE = 40
N_MODERATE_SNR = 40  # target ~ 20 dB SNR
N_LOUD_SNR = 40  # target ~ 10 dB SNR
N_VERY_LOUD_SNR = 40  # target ~ 5 dB SNR

MODERATE_SNR_DB = 20.0
LOUD_SNR_DB = 10.0
VERY_LOUD_SNR_DB = 5.0


@dataclass
class TestClip:
    filename: str
    expected_tier: str
    category: str
    snr_db: float | None
    voice_source: str
    noise_source: str


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def load_and_prepare(path: Path, target_duration_s: float = CLIP_DURATION_S) -> np.ndarray | None:
    """Load a WAV, convert mono, resample to 16kHz, pad/truncate to target_duration.

    Returns None if the file is unreadable, too short, or silent.
    """
    try:
        samples, sr = sf.read(str(path), dtype="float32")
    except Exception:
        return None
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != TARGET_SAMPLE_RATE:
        samples = resample_poly(samples, TARGET_SAMPLE_RATE, sr).astype(np.float32)

    n_target = int(target_duration_s * TARGET_SAMPLE_RATE)

    if samples.size < n_target // 2:
        # Too short even to loop reasonably.
        return None

    if samples.size < n_target:
        # Loop-pad to target length.
        reps = int(np.ceil(n_target / samples.size))
        samples = np.tile(samples, reps)[:n_target]
    else:
        # Random crop to target length.
        start = random.randint(0, samples.size - n_target)
        samples = samples[start : start + n_target]

    # Reject silence.
    if rms(samples) < 1e-4:
        return None

    return samples.astype(np.float32)


def normalize_to_dbfs(samples: np.ndarray, target_dbfs: float) -> np.ndarray:
    current_rms = rms(samples)
    if current_rms < 1e-10:
        return samples
    current_dbfs = 20 * np.log10(current_rms)
    gain_db = target_dbfs - current_dbfs
    gain_linear = 10 ** (gain_db / 20.0)
    return (samples * gain_linear).astype(np.float32)


def mix_at_snr(voice: np.ndarray, noise: np.ndarray, target_snr_db: float) -> np.ndarray:
    """Mix voice + noise scaled to target SNR (voice as signal, noise as noise)."""
    voice_rms = rms(voice)
    noise_rms = rms(noise)
    if voice_rms < 1e-10 or noise_rms < 1e-10:
        return voice.copy()
    # SNR = 20 log10(voice_rms / noise_rms), solve for target noise gain.
    target_noise_rms = voice_rms / (10 ** (target_snr_db / 20.0))
    noise_gain = target_noise_rms / noise_rms
    mixed = voice + noise * noise_gain
    # Prevent clipping — normalize if peak > 0.95.
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.95:
        mixed = mixed * (0.95 / peak)
    return mixed.astype(np.float32)


def collect_vocal_files(vocalset_root: Path, max_files: int = 300) -> list[Path]:
    """Find all VocalSet 'scales' recordings across every technique.

    Structure: {root}/data_by_singer/{singer}/scales/**/*.wav
    The 'straight' technique alone yields only ~101 clips (20 singers x
    ~5 vowels), too few for the 160 distinct vocals the test set needs,
    so we take all techniques (vibrato, belt, etc.).
    """
    singers_root = vocalset_root / "data_by_singer"
    if not singers_root.exists():
        raise FileNotFoundError(f"VocalSet data_by_singer/ not found at {singers_root}")

    wavs: list[Path] = []
    for singer_dir in sorted(singers_root.iterdir()):
        if not singer_dir.is_dir():
            continue
        scales_dir = singer_dir / "scales"
        if not scales_dir.exists():
            continue
        wavs.extend(sorted(scales_dir.rglob("*.wav")))
        if len(wavs) >= max_files:
            break
    return wavs[:max_files]


def collect_noise_files(musan_root: Path, max_files: int = 400) -> list[Path]:
    """Find all MUSAN noise files (free-sound + sound-bible)."""
    noise_root = musan_root / "noise"
    if not noise_root.exists():
        raise FileNotFoundError(f"MUSAN noise/ not found at {noise_root}")
    wavs: list[Path] = []
    for subdir in ["free-sound", "sound-bible"]:
        wavs.extend(sorted((noise_root / subdir).glob("*.wav")))
        if len(wavs) >= max_files:
            break
    return wavs[:max_files]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocalset", required=True, type=Path)
    parser.add_argument("--musan", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/test_set"))
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)  # noqa: NPY002 -- legacy seeding keeps the test set bit-identical

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir.parent / "manifest.csv"

    print(f"Scanning VocalSet at {args.vocalset}...")
    vocal_files = collect_vocal_files(args.vocalset)
    print(f"  found {len(vocal_files)} vocal recordings")

    print(f"Scanning MUSAN at {args.musan}...")
    noise_files = collect_noise_files(args.musan)
    print(f"  found {len(noise_files)} noise recordings")

    # Shuffle deterministically.
    random.shuffle(vocal_files)
    random.shuffle(noise_files)

    manifest: list[TestClip] = []
    vocal_idx = 0
    noise_idx = 0

    def next_vocal() -> tuple[Path, np.ndarray]:
        nonlocal vocal_idx
        while vocal_idx < len(vocal_files):
            path = vocal_files[vocal_idx]
            vocal_idx += 1
            samples = load_and_prepare(path)
            if samples is not None:
                return path, samples
        raise RuntimeError("Ran out of usable VocalSet files")

    def next_noise() -> tuple[Path, np.ndarray]:
        nonlocal noise_idx
        while noise_idx < len(noise_files):
            path = noise_files[noise_idx]
            noise_idx += 1
            samples = load_and_prepare(path)
            if samples is not None:
                return path, samples
        raise RuntimeError("Ran out of usable MUSAN files")

    # --- Category 1: clean vocals -> excellent ---
    print(f"\nGenerating {N_CLEAN_VOCAL} clean vocal clips (expected: excellent)...")
    for i in range(N_CLEAN_VOCAL):
        vpath, voice = next_vocal()
        voice = normalize_to_dbfs(voice, NORMALIZE_RMS_DBFS)
        fname = f"excellent_clean_{i:03d}_{vpath.stem}.wav"
        sf.write(output_dir / fname, voice, TARGET_SAMPLE_RATE)
        manifest.append(TestClip(fname, "excellent", "clean_vocal", None, vpath.name, ""))

    # --- Category 2: quiet noise (attenuated) -> excellent ---
    print(f"Generating {N_QUIET_NOISE} quiet noise clips (expected: excellent)...")
    for i in range(N_QUIET_NOISE):
        npath, noise = next_noise()
        # Attenuate to very low RMS (~-40 dBFS = quiet room).
        noise = normalize_to_dbfs(noise, -40.0)
        fname = f"excellent_quiet_{i:03d}_{npath.stem}.wav"
        sf.write(output_dir / fname, noise, TARGET_SAMPLE_RATE)
        manifest.append(TestClip(fname, "excellent", "quiet_noise", None, "", npath.name))

    # --- Category 3: moderate SNR -> good ---
    print(
        f"Generating {N_MODERATE_SNR} moderate-SNR mixes (target {MODERATE_SNR_DB} dB, expected: good)..."
    )
    for i in range(N_MODERATE_SNR):
        vpath, voice = next_vocal()
        npath, noise = next_noise()
        voice = normalize_to_dbfs(voice, NORMALIZE_RMS_DBFS)
        noise = normalize_to_dbfs(noise, NORMALIZE_RMS_DBFS)
        mixed = mix_at_snr(voice, noise, MODERATE_SNR_DB)
        fname = f"good_snr{int(MODERATE_SNR_DB)}_{i:03d}.wav"
        sf.write(output_dir / fname, mixed, TARGET_SAMPLE_RATE)
        manifest.append(
            TestClip(fname, "good", "moderate_snr", MODERATE_SNR_DB, vpath.name, npath.name)
        )

    # --- Category 4: loud SNR -> fair ---
    print(f"Generating {N_LOUD_SNR} loud-SNR mixes (target {LOUD_SNR_DB} dB, expected: fair)...")
    for i in range(N_LOUD_SNR):
        vpath, voice = next_vocal()
        npath, noise = next_noise()
        voice = normalize_to_dbfs(voice, NORMALIZE_RMS_DBFS)
        noise = normalize_to_dbfs(noise, NORMALIZE_RMS_DBFS)
        mixed = mix_at_snr(voice, noise, LOUD_SNR_DB)
        fname = f"fair_snr{int(LOUD_SNR_DB)}_{i:03d}.wav"
        sf.write(output_dir / fname, mixed, TARGET_SAMPLE_RATE)
        manifest.append(TestClip(fname, "fair", "loud_snr", LOUD_SNR_DB, vpath.name, npath.name))

    # --- Category 5: very loud SNR -> poor ---
    print(
        f"Generating {N_VERY_LOUD_SNR} very-loud-SNR mixes (target {VERY_LOUD_SNR_DB} dB, expected: poor)..."
    )
    for i in range(N_VERY_LOUD_SNR):
        vpath, voice = next_vocal()
        npath, noise = next_noise()
        voice = normalize_to_dbfs(voice, NORMALIZE_RMS_DBFS)
        noise = normalize_to_dbfs(noise, NORMALIZE_RMS_DBFS)
        mixed = mix_at_snr(voice, noise, VERY_LOUD_SNR_DB)
        fname = f"poor_snr{int(VERY_LOUD_SNR_DB)}_{i:03d}.wav"
        sf.write(output_dir / fname, mixed, TARGET_SAMPLE_RATE)
        manifest.append(
            TestClip(fname, "poor", "very_loud_snr", VERY_LOUD_SNR_DB, vpath.name, npath.name)
        )

    # Write manifest.
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["filename", "expected_tier", "category", "snr_db", "voice_source", "noise_source"]
        )
        for clip in manifest:
            writer.writerow(
                [
                    clip.filename,
                    clip.expected_tier,
                    clip.category,
                    "" if clip.snr_db is None else f"{clip.snr_db:.1f}",
                    clip.voice_source,
                    clip.noise_source,
                ]
            )

    print(f"\nWrote {len(manifest)} test clips to {output_dir}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
