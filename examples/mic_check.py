"""Quick mic sanity check — prints RMS of each incoming chunk.

If your mic is working, you'll see numbers like 0.001-0.1 (with speech
around 0.05+). If it's silent (permissions issue), you'll see 0.0000000.
"""

import sys
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHUNK_SIZE = 1600

print("Devices available:")
print(sd.query_devices())
print()
print(f"Default input device: {sd.query_devices(kind='input')['name']}")
print()
print("Listening for 15 seconds. Talk into the mic. Watch the RMS numbers.")
print()

start = time.time()


def cb(indata, frames, time_info, status):
    if status:
        print(f"[status] {status}", file=sys.stderr)
    chunk = indata[:, 0] if indata.ndim > 1 else indata
    rms = float(np.sqrt(np.mean(chunk**2)))
    peak = float(np.max(np.abs(chunk)))
    print(f"chunk: rms={rms:.6f}  peak={peak:.6f}  shape={indata.shape}")


with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    blocksize=CHUNK_SIZE,
    callback=cb,
    dtype="float32",
):
    while time.time() - start < 15:
        time.sleep(0.1)

print("\nDone.")
