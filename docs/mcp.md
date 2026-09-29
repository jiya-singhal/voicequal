# MCP server

voicequal ships a small [Model Context Protocol](https://modelcontextprotocol.io)
server so any MCP-capable agent (Claude Desktop, Claude Code, others) can
ask "is this recording clean enough?" as a tool call.

```bash
pip install 'voicequal[mcp]'
voicequal-mcp            # runs the stdio server
```

## Tools

| Tool | Input | Output |
|---|---|---|
| `assess_file` | `path`: audio file on the machine running the server | Every `FileAssessment` field, plus `path` and `advice` |
| `assess_base64_wav` | `data`: a WAV file, base64 encoded (a `data:` URL prefix is fine) | Every `FileAssessment` field, plus `advice` |

`advice` is the output of [`advise()`](advice.md): `headline`, `actions`,
`severity`. Both tools raise on a missing file or malformed base64 so the
agent sees an error rather than a made-up score.

The base64 tool writes the bytes to a temporary file, assesses it, and
deletes it before returning.

## Configuring a client

Both Claude Desktop and Claude Code use the same stdio config shape. Point
`command` at the `voicequal-mcp` executable in the environment where you
installed the extra.

**Claude Code** (`.mcp.json` in a project, or `~/.claude.json` for all projects):

```json
{
  "mcpServers": {
    "voicequal": {
      "command": "voicequal-mcp"
    }
  }
}
```

Or from the shell:

```bash
claude mcp add voicequal -- voicequal-mcp
```

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "voicequal": {
      "command": "/absolute/path/to/venv/bin/voicequal-mcp"
    }
  }
}
```

If the server runs inside a Poetry or virtualenv, use the absolute path to
that environment's `voicequal-mcp` (or `python -m voicequal.mcp_server`)
so the desktop app does not pick up a different Python.

## Notes

- Importing `voicequal` never imports `mcp`; the dependency is only
  loaded when the server is built.
- Works with `mcp` 1.x (`FastMCP`) and 2.x (`MCPServer`).
- The server is stdio only. No network listener is opened.
