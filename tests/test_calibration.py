"""Tests for voicequal.calibration."""

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from voicequal import calibration


class TestComputeOffset:
    def test_quiet_rms_of_001_gives_reasonable_offset(self):
        # 20*log10(0.01) = -40, so offset should be 40 - (-40) = 80
        offset = calibration.compute_offset(quiet_rms=0.01, loud_rms=0.1)
        assert offset == pytest.approx(80.0, abs=0.1)

    def test_quiet_rms_of_0001_gives_higher_offset(self):
        offset = calibration.compute_offset(quiet_rms=0.001, loud_rms=0.1)
        assert offset == pytest.approx(100.0, abs=0.1)

    def test_zero_rms_does_not_crash(self):
        offset = calibration.compute_offset(quiet_rms=0.0, loud_rms=0.1)
        assert math.isfinite(offset)


class TestSaveLoad:
    def test_roundtrip(self, tmp_path, monkeypatch):
        fake_path = tmp_path / "cal.json"
        monkeypatch.setattr(calibration, "CALIBRATION_PATH", fake_path)

        cal = calibration.Calibration(
            db_offset=80.5, quiet_rms=0.01, loud_rms=0.1, created_at=1234567.0
        )
        calibration.save(cal)
        loaded = calibration.load()
        assert loaded is not None
        assert loaded.db_offset == 80.5
        assert loaded.quiet_rms == pytest.approx(0.01)

    def test_load_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(calibration, "CALIBRATION_PATH", tmp_path / "nope.json")
        assert calibration.load() is None

    def test_load_corrupt_returns_none(self, tmp_path, monkeypatch):
        fake_path = tmp_path / "cal.json"
        fake_path.write_text("this is not json")
        monkeypatch.setattr(calibration, "CALIBRATION_PATH", fake_path)
        assert calibration.load() is None
