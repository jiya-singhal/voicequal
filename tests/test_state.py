"""Tests for voicequal.state.RollingStats."""

import math

import pytest

from voicequal.state import RollingStats


class TestTemporalVariance:
    def test_returns_zero_before_min_samples(self):
        s = RollingStats()
        for _ in range(9):
            s.update(current_noise_floor=-40.0, current_rms=0.01)
        assert s.temporal_variance() == 0.0

    def test_zero_for_constant_noise_floor(self):
        s = RollingStats()
        for _ in range(20):
            s.update(current_noise_floor=-40.0, current_rms=0.01)
        assert s.temporal_variance() == pytest.approx(0.0, abs=1e-9)

    def test_positive_for_varying_noise_floor(self):
        s = RollingStats()
        for i in range(20):
            # Alternate -40 / -50 -> nonzero std
            nf = -40.0 if i % 2 == 0 else -50.0
            s.update(current_noise_floor=nf, current_rms=0.01)
        assert s.temporal_variance() > 4.0

    def test_only_keeps_last_20(self):
        s = RollingStats()
        # First 20 wildly varying values...
        for i in range(20):
            s.update(current_noise_floor=(-30.0 if i % 2 else -70.0),
                     current_rms=0.01)
        # ...then 20 constant values. History should evict the old ones.
        for _ in range(20):
            s.update(current_noise_floor=-40.0, current_rms=0.01)
        assert s.temporal_variance() == pytest.approx(0.0, abs=1e-9)


class TestBackgroundDB:
    def test_uses_current_rms_before_min_samples(self):
        s = RollingStats()
        # No prior history: should use current_rms=0.001 directly.
        # 20*log10(0.001) + 94 = -60 + 94 = 34
        for _ in range(3):
            s.update(current_noise_floor=-50.0, current_rms=0.001)
        assert s.background_db(0.001) == pytest.approx(34.0, abs=1e-3)

    def test_uses_median_of_history_once_warm(self):
        s = RollingStats()
        # 9 loud values then a very quiet one. min would return 1e-5, but the
        # median of these 10 values is 0.01 (the single quiet outlier can't
        # drag the middle down). 20*log10(0.01) + 94 = -40 + 94 = 54.
        for _ in range(9):
            s.update(current_noise_floor=-40.0, current_rms=0.01)
        s.update(current_noise_floor=-40.0, current_rms=1e-5)
        # Median rejects the single quiet outlier, so this stays high.
        result = s.background_db(0.01)
        assert result > 0.0  # far above the strict-min case which was -6

    def test_silent_input_floors_low(self):
        s = RollingStats()
        for _ in range(30):
            s.update(current_noise_floor=-100.0, current_rms=0.0)
        # All zeros: median is 0 -> log10 clamped to 1e-10.
        # 20*log10(1e-10) + 94 = -200 + 94 = -106.
        assert s.background_db(0.0) == pytest.approx(-106.0, abs=1e-3)

    def test_reset_clears_history(self):
        s = RollingStats()
        for _ in range(30):
            s.update(current_noise_floor=-40.0, current_rms=0.5)
        s.reset()
        # After reset, no history -> uses current_rms directly.
        # 20*log10(0.001) + 94 = 34
        assert s.background_db(0.001) == pytest.approx(34.0, abs=1e-3)
