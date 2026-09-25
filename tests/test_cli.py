"""Smoke tests for the voicequal CLI."""

import os
import tempfile

import numpy as np
import soundfile as sf

from voicequal.cli import build_parser, main


def _write_wav(samples: np.ndarray, sample_rate: int) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)  # noqa: SIM115
    f.close()
    sf.write(f.name, samples, sample_rate)
    return f.name


def test_parser_supports_assess_and_listen():
    parser = build_parser()
    args = parser.parse_args(["assess", "some.wav"])
    assert args.command == "assess"
    assert args.path == "some.wav"

    args = parser.parse_args(["listen", "--stability-frames", "5"])
    assert args.command == "listen"
    assert args.stability_frames == 5


def test_assess_command_on_real_file(capsys):
    sr = 16000
    t = np.linspace(0, 2, 2 * sr, endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = _write_wav(samples, sr)
    try:
        code = main(["assess", path])
        captured = capsys.readouterr()
        assert code == 0
        # Should mention the quality tier somewhere in output.
        assert any(t in captured.out.upper() for t in ["EXCELLENT", "GOOD", "FAIR", "POOR"])
    finally:
        os.unlink(path)


def test_assess_command_missing_file(capsys):
    code = main(["assess", "/nonexistent/path/to/file.wav"])
    captured = capsys.readouterr()
    assert code == 1
    assert "not found" in captured.out.lower()
