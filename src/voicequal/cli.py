"""Command-line interface for voicequal.

Subcommands:
  voicequal assess <path>   Analyze an audio file and print a report.
  voicequal listen          Stream from the default input device and
                            print tier changes + heartbeat in real time.
                            Requires the [mic] extra (sounddevice).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from voicequal import __version__
from voicequal.pipeline import assess

if TYPE_CHECKING:
    from voicequal.live import LiveAssessment

console = Console()

QUALITY_COLOR = {
    "excellent": "bright_green",
    "good": "green",
    "fair": "yellow",
    "poor": "bright_red",
}


def _quality_text(quality: str) -> Text:
    color = QUALITY_COLOR.get(quality, "white")
    return Text(quality.upper(), style=f"bold {color}")


def _cmd_assess(path: str) -> int:
    if not os.path.isfile(path):
        console.print(f"[bold red]Error:[/] file not found: {path}")
        return 1
    try:
        result = assess(path)
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/] file not found: {path}")
        return 1
    except ValueError as e:
        console.print(f"[bold red]Error:[/] {e}")
        return 1

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("field", style="dim")
    table.add_column("value")

    table.add_row("Quality", _quality_text(result.quality))
    table.add_row("Reason", result.reason)
    table.add_row("", "")
    table.add_row("Background", f"{result.background_db:.1f} dBA")
    table.add_row("SNR", f"{result.snr:.1f} dB")
    table.add_row("Flatness", f"{result.spectral_flatness:.2f}")
    table.add_row("Temporal variance", f"{result.temporal_variance:.2f}")
    table.add_row("", "")
    table.add_row("Duration", f"{result.duration_seconds:.2f}s")
    table.add_row("Sample rate", f"{result.sample_rate} Hz")
    table.add_row("Frames analyzed", str(result.num_frames))

    console.print(Panel(table, title=f"voicequal · {path}", border_style="cyan"))
    return 0


def _run_calibration(sample_rate: int, chunk_size: int) -> "calibration.Calibration":
    import numpy as np
    import sounddevice as sd
    from voicequal import calibration

    def record(seconds: float) -> np.ndarray:
        frames = int(seconds * sample_rate)
        audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        return audio[:, 0] if audio.ndim > 1 else audio

    console.print(
        "\n[bold cyan]Calibration step 1/2:[/] Sit quietly for 3 seconds..."
    )
    for i in (3, 2, 1):
        console.print(f"  starting in {i}...")
        time.sleep(1)
    console.print("[dim]Recording quiet room...[/]")
    quiet = record(3.0)
    quiet_rms = float(np.sqrt(np.mean(quiet ** 2)))

    console.print(
        f"\n[bold cyan]Calibration step 2/2:[/] Now make LOUD noise "
        f"(play music, yell, clap) for 3 seconds..."
    )
    for i in (3, 2, 1):
        console.print(f"  starting in {i}...")
        time.sleep(1)
    console.print("[dim]Recording loud room...[/]")
    loud = record(3.0)
    loud_rms = float(np.sqrt(np.mean(loud ** 2)))

    offset = calibration.compute_offset(quiet_rms=quiet_rms, loud_rms=loud_rms)

    cal = calibration.Calibration(
        db_offset=offset,
        quiet_rms=quiet_rms,
        loud_rms=loud_rms,
        created_at=time.time(),
    )
    calibration.save(cal)

    console.print(
        f"\n[bold green]Calibration saved.[/]\n"
        f"  Quiet RMS: {quiet_rms:.4f}\n"
        f"  Loud RMS:  {loud_rms:.4f}\n"
        f"  dB offset: {offset:.1f}\n"
        f"  (saved to {calibration.CALIBRATION_PATH})\n"
    )
    return cal


def _cmd_listen(
    stability_frames: int,
    heartbeat_seconds: float,
    sensitive: bool,
    calibrate: bool,
    reset_calibration: bool,
) -> int:
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError:
        console.print(
            "[bold red]Error:[/] `voicequal listen` requires the mic extra.\n"
            "Install with:  [cyan]pip install 'voicequal[mic]'[/]"
        )
        return 1

    from voicequal import calibration
    from voicequal.live import LiveDetector

    sample_rate = 16000
    chunk_size = 1600  # 100 ms

    # Handle calibration reset.
    if reset_calibration and calibration.CALIBRATION_PATH.exists():
        calibration.CALIBRATION_PATH.unlink()
        console.print("[dim]Calibration reset.[/]")

    # Run calibration if requested OR if none exists yet.
    cal = calibration.load()
    if calibrate or cal is None:
        if cal is None and not calibrate:
            console.print(
                "[yellow]No calibration found. Running first-time calibration...[/]"
            )
        cal = _run_calibration(sample_rate, chunk_size)

    db_offset = cal.db_offset
    threshold_offset_db = 8.0 if sensitive else 0.0

    detector = LiveDetector(
        sample_rate=sample_rate,
        stability_frames=stability_frames,
        threshold_offset_db=threshold_offset_db,
        db_offset=db_offset,
    )

    def print_change(a: "LiveAssessment") -> None:
        ts = time.strftime("%H:%M:%S")
        qual = _quality_text(a.quality)
        console.print(
            f"[dim]{ts}[/]  [bold cyan]CHANGE[/]  ",
            qual,
            f"  SNR=[bold]{a.snr:5.1f}[/]dB  "
            f"bgDB=[bold]{a.background_db:5.1f}[/]  "
            f"flat=[bold]{a.spectral_flatness:.2f}[/]  "
            f"var=[bold]{a.temporal_variance:.1f}[/]",
            sep="",
        )
        console.print(f"[dim]           reason: {a.reason}[/]")

    detector.on_change(print_change)

    def audio_callback(indata, frames, time_info, status) -> None:
        if status:
            print(f"[stream status] {status}", file=sys.stderr)
        chunk = indata[:, 0] if indata.ndim > 1 else indata
        detector.push(chunk.astype(np.float32))

    poor_threshold = 72 - threshold_offset_db
    fair_threshold = 67 - threshold_offset_db
    good_threshold = 60 - threshold_offset_db

    mode_label = "sensitive (desktop testing)" if sensitive else "production (validated defaults)"

    console.print(
        Panel(
            f"[bold]voicequal · live[/]\n"
            f"Mode: [cyan]{mode_label}[/]   Sample rate: {sample_rate} Hz\n"
            f"Stability frames: {stability_frames}   Heartbeat: every {heartbeat_seconds:.1f}s\n"
            f"\n"
            f"[dim]dBA scale reference:[/]\n"
            f"  40 = quiet library   55 = normal conversation\n"
            f"  70 = busy cafe       85 = vacuum cleaner\n"
            f"\n"
            f"[dim]Calibration:[/] dB offset = {db_offset:.1f}\n"
            f"\n"
            f"[dim]Tier boundaries at this mode:[/]\n"
            f"  [bright_green]EXCELLENT[/]  bgDB < {good_threshold:.0f}\n"
            f"  [green]GOOD[/]       bgDB {good_threshold:.0f}-{fair_threshold:.0f}\n"
            f"  [yellow]FAIR[/]       bgDB {fair_threshold:.0f}-{poor_threshold:.0f}\n"
            f"  [bright_red]POOR[/]       bgDB > {poor_threshold:.0f}\n"
            f"\n[dim]Press Ctrl+C to stop.[/]",
            border_style="cyan",
        )
    )

    last_hb = time.time()
    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            blocksize=chunk_size,
            callback=audio_callback,
            dtype="float32",
        ):
            while True:
                time.sleep(0.1)
                now = time.time()
                if now - last_hb >= heartbeat_seconds:
                    current = detector.get_current()
                    ts = time.strftime("%H:%M:%S")
                    if current is None:
                        console.print(f"[dim]{ts}  (warming up...)[/]")
                    else:
                        qual = _quality_text(current.quality)
                        # Color-code dBA relative to tier thresholds.
                        bg = current.background_db
                        if bg > poor_threshold:
                            bg_color = "bright_red"
                        elif bg > fair_threshold:
                            bg_color = "yellow"
                        elif bg > good_threshold:
                            bg_color = "green"
                        else:
                            bg_color = "bright_green"
                        console.print(
                            f"[dim]{ts}[/]  ",
                            qual,
                            f"  room=[bold {bg_color}]{bg:5.1f} dBA[/]  ",
                            f"SNR={current.snr:5.1f}dB  flat={current.spectral_flatness:.2f}",
                            sep="",
                        )
                    last_hb = now
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voicequal",
        description="Audio quality assessment for voice apps.",
    )
    parser.add_argument("--version", action="version", version=f"voicequal {__version__}")

    subs = parser.add_subparsers(dest="command", required=True)

    p_assess = subs.add_parser("assess", help="Assess a single audio file.")
    p_assess.add_argument("path", help="Path to a WAV/FLAC/OGG file.")

    p_listen = subs.add_parser(
        "listen",
        help="Stream from the default microphone and print live tier changes.",
    )
    p_listen.add_argument(
        "--stability-frames",
        type=int,
        default=3,
        help="Frames a new tier must persist before firing a change event (default: 3).",
    )
    p_listen.add_argument(
        "--heartbeat",
        type=float,
        default=2.0,
        help="Seconds between heartbeat prints (default: 2.0).",
    )
    p_listen.add_argument(
        "--sensitive",
        action="store_true",
        help="Shift thresholds down by 8 dBA. Use for casual desktop testing.",
    )
    p_listen.add_argument(
        "--calibrate",
        action="store_true",
        help="Run mic calibration before listening. Saves to ~/.voicequal/",
    )
    p_listen.add_argument(
        "--reset-calibration",
        action="store_true",
        help="Delete saved calibration and use the default +94 dBA offset.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "assess":
        return _cmd_assess(args.path)
    if args.command == "listen":
        return _cmd_listen(
            stability_frames=args.stability_frames,
            heartbeat_seconds=args.heartbeat,
            sensitive=args.sensitive,
            calibrate=args.calibrate,
            reset_calibration=args.reset_calibration,
        )
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
