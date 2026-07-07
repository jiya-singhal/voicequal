"""Live microphone demo for voicequal.

Streams audio from the default input device, printing:
  - a periodic heartbeat every 2 seconds with the current state
  - a highlighted line whenever the stable quality tier changes

Ctrl+C to stop.

Requires: poetry add --group examples sounddevice
"""

import sys
import time

import numpy as np
import sounddevice as sd

from voicequal import LiveDetector

SAMPLE_RATE = 16000
CHUNK_SIZE = 1600  # 100ms chunks at 16kHz
HEARTBEAT_INTERVAL_S = 2.0


def _format_line(prefix: str, assessment) -> str:
    return (
        f"{prefix}  {assessment.quality.upper():>9s}   "
        f"SNR={assessment.snr:5.1f}dB   "
        f"bgDB={assessment.background_db:5.1f}   "
        f"flat={assessment.spectral_flatness:.2f}   "
        f"var={assessment.temporal_variance:.1f}   "
        f"frames={assessment.frames_analyzed}"
    )


def main() -> None:
    detector = LiveDetector(sample_rate=SAMPLE_RATE, stability_frames=3)

    def on_tier_change(assessment) -> None:
        ts = time.strftime("%H:%M:%S")
        print(_format_line(f"[{ts}]  >>> CHANGE >>>", assessment))
        print(f"                    reason: {assessment.reason}")

    detector.on_change(on_tier_change)

    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"[stream status] {status}", file=sys.stderr)
        chunk = indata[:, 0] if indata.ndim > 1 else indata
        detector.push(chunk.astype(np.float32))

    print(f"Listening at {SAMPLE_RATE} Hz.")
    print("Heartbeat every 2s; CHANGE lines when tier shifts.")
    print("Ctrl+C to stop.\n")

    last_heartbeat = time.time()
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
            dtype="float32",
        ):
            while True:
                time.sleep(0.1)
                now = time.time()
                if now - last_heartbeat >= HEARTBEAT_INTERVAL_S:
                    current = detector.get_current()
                    ts = time.strftime("%H:%M:%S")
                    if current is None:
                        print(f"[{ts}]  (warming up...)")
                    else:
                        print(_format_line(f"[{ts}]  heartbeat     ", current))
                    last_heartbeat = now
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
