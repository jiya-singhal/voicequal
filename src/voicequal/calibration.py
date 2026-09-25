"""Auto-calibration for voicequal.

Different mic hardware, drivers, and OS audio processing deliver
different signal levels for the same real-world dBA. Rather than
hardcoding a +94 offset (which assumes browser-based reference
calibration), we let users measure their own mic's response and
save an offset that maps their real quiet/loud room to the
algorithm's internal dBA scale.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

CALIBRATION_PATH = Path.home() / ".voicequal" / "calibration.json"

# Target dBA readings we WANT the algorithm to produce during calibration.
# A quiet room should read ~40 dBA. A loud room should read ~70 dBA.
TARGET_QUIET_DBA: float = 40.0
TARGET_LOUD_DBA: float = 70.0


@dataclass(frozen=True)
class Calibration:
    db_offset: float
    quiet_rms: float
    loud_rms: float
    created_at: float


def save(cal: Calibration) -> None:
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(asdict(cal), indent=2))


def load() -> Calibration | None:
    if not CALIBRATION_PATH.exists():
        return None
    try:
        data = json.loads(CALIBRATION_PATH.read_text())
        return Calibration(**data)
    except (json.JSONDecodeError, TypeError):
        return None


def compute_offset(quiet_rms: float, loud_rms: float) -> float:
    """Compute db_offset from measured quiet and loud RMS.

    We want the algorithm's dBA formula (20*log10(rms) + offset) to
    produce TARGET_QUIET_DBA at quiet_rms. Solve for offset:
        offset = TARGET_QUIET_DBA - 20*log10(quiet_rms)
    Loud_rms is used as a sanity check but not directly in the offset.
    """
    quiet_rms = max(quiet_rms, 1e-10)
    return TARGET_QUIET_DBA - 20.0 * float(np.log10(quiet_rms))
