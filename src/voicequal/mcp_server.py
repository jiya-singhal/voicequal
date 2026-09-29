"""MCP server exposing voicequal as tools any agent can call.

Needs the ``mcp`` extra::

    pip install 'voicequal[mcp]'
    voicequal-mcp            # stdio server, for Claude Desktop / Claude Code

Two tools are exposed:

* ``assess_file(path)`` - assess an audio file on disk.
* ``assess_base64_wav(data)`` - assess a base64-encoded WAV sent inline.

Both return the full ``FileAssessment`` as a JSON-able dict plus an
``advice`` object from :func:`voicequal.advice.advise`. The core
``voicequal`` package never imports this module; ``mcp`` is only imported
when the server is built.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from voicequal.advice import advise
from voicequal.pipeline import assess

SERVER_NAME = "voicequal"
SERVER_INSTRUCTIONS = (
    "voicequal answers 'is this recording clean enough to process?'. "
    "Call assess_file with a path, or assess_base64_wav with WAV bytes, and read "
    "`quality` (excellent | good | fair | poor), `snr_estimate` in dB, and `advice`."
)


def assess_file_tool(path: str) -> dict[str, Any]:
    """Assess an audio file (WAV/FLAC/OGG) and return metrics plus advice.

    Args:
        path: Path to the audio file on the machine running the server.

    Returns:
        Every ``FileAssessment`` field plus ``advice`` with ``headline``,
        ``actions`` and ``severity``.
    """
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"No such audio file: {file_path}")
    result = assess(file_path)
    payload = asdict(result)
    payload["path"] = str(file_path)
    payload["advice"] = advise(result).to_dict()
    return payload


def assess_base64_wav_tool(data: str) -> dict[str, Any]:
    """Assess a base64-encoded WAV file and return metrics plus advice.

    Args:
        data: The WAV file bytes, base64 encoded (a ``data:`` URL prefix is
            tolerated and stripped).

    Returns:
        Same shape as :func:`assess_file_tool`, without ``path``.
    """
    if "," in data and data.lstrip().startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("data is not valid base64") from exc
    if not raw:
        raise ValueError("data decoded to zero bytes")

    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
        result = assess(tmp_name)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
    payload = asdict(result)
    payload["advice"] = advise(result).to_dict()
    return payload


def build_server() -> Any:
    """Construct the FastMCP server. Imports ``mcp`` lazily.

    Raises:
        ImportError: If the ``mcp`` package is not installed.
    """
    server_cls: Any
    try:
        # mcp >= 2.0 renamed FastMCP to MCPServer.
        from mcp.server.mcpserver import MCPServer as server_cls
    except ImportError:
        try:  # mcp 1.x
            import importlib

            server_cls = importlib.import_module("mcp.server.fastmcp").FastMCP
        except (ImportError, AttributeError) as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "The voicequal MCP server needs the mcp package. "
                "Install with: pip install 'voicequal[mcp]'"
            ) from exc

    server = server_cls(SERVER_NAME, instructions=SERVER_INSTRUCTIONS)
    server.tool(name="assess_file")(assess_file_tool)
    server.tool(name="assess_base64_wav")(assess_base64_wav_tool)
    return server


def main() -> None:
    """Console entry point: run the stdio server."""
    build_server().run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
