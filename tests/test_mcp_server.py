"""Tests for voicequal.mcp_server. The tool functions are called directly."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import soundfile as sf

from voicequal.mcp_server import assess_base64_wav_tool, assess_file_tool


def _tone_wav_bytes(seconds: float = 2.0) -> bytes:
    t = np.arange(int(seconds * 16000)) / 16000
    buf = io.BytesIO()
    sf.write(buf, (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 16000, format="WAV")
    return buf.getvalue()


class TestToolFunctions:
    def test_assess_file_returns_metrics_and_advice(self, tmp_path):
        path = tmp_path / "tone.wav"
        path.write_bytes(_tone_wav_bytes())
        out = assess_file_tool(str(path))
        assert out["quality"] in ("excellent", "good", "fair", "poor")
        assert isinstance(out["snr_estimate"], float)
        assert out["advice"]["headline"]
        assert out["advice"]["severity"] in ("ok", "warn", "bad")
        assert out["path"] == str(path)

    def test_assess_file_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            assess_file_tool(str(tmp_path / "nope.wav"))

    def test_assess_base64_wav_roundtrip(self):
        data = base64.b64encode(_tone_wav_bytes()).decode()
        out = assess_base64_wav_tool(data)
        assert out["quality"] == "excellent"
        assert out["advice"]["headline"] == "Clean enough to process"
        assert "path" not in out

    def test_assess_base64_accepts_data_url(self):
        data = "data:audio/wav;base64," + base64.b64encode(_tone_wav_bytes()).decode()
        assert assess_base64_wav_tool(data)["quality"] == "excellent"

    def test_assess_base64_rejects_garbage(self):
        with pytest.raises(ValueError):
            assess_base64_wav_tool("not base64!!")
        with pytest.raises(ValueError):
            assess_base64_wav_tool("")

    def test_json_serialisable(self, tmp_path):
        import json

        path = tmp_path / "tone.wav"
        path.write_bytes(_tone_wav_bytes())
        json.dumps(assess_file_tool(str(path)))


class TestServer:
    def test_build_server_registers_both_tools(self):
        pytest.importorskip("mcp")
        import asyncio

        from voicequal.mcp_server import build_server

        server = build_server()
        tools = asyncio.run(server.list_tools())
        assert {t.name for t in tools} == {"assess_file", "assess_base64_wav"}

    def test_core_import_does_not_pull_in_mcp(self):
        import subprocess
        import sys

        code = "import sys, voicequal; sys.exit(1 if 'mcp' in sys.modules else 0)"
        assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
