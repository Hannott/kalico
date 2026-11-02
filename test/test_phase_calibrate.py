# Tests for phase_calibrate.py: corrections around a calibration run.
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from klippy.extras import phase_calibrate  # noqa: E402


class FakePhaseStepping:
    def __init__(self, corrections, speeds=None):
        self.corrections = dict(corrections)
        self.speeds = speeds
        self.uploaded = []

    def get_harmonic(self, direction, harmonic):
        return self.corrections.get((direction, harmonic), 0j)

    def set_harmonic(self, direction, harmonic, value):
        self.corrections[(direction, harmonic)] = value

    def get_correction_speeds(self):
        return self.speeds

    def set_correction_speeds(self, speeds):
        self.speeds = speeds

    def apply_corrections(self):
        self.uploaded.append(dict(self.corrections))


EARLIER = {
    ("forward", 2): 0.01 + 0.02j,
    ("backward", 2): -0.01j,
    ("forward", 4): 0.005,
    ("backward", 1): 0.003j,
}


def _uncorrected(ps, harmonics, speeds=(50.0, 100.0)):
    return phase_calibrate.PhaseCalibrate._uncorrected(
        None, ps, harmonics, speeds
    )


def test_calibration_starts_without_correction():
    ps = FakePhaseStepping(EARLIER)
    with _uncorrected(ps, [2, 4]):
        assert ps.uploaded[-1][("forward", 2)] == 0j
        assert ps.uploaded[-1][("backward", 2)] == 0j
        assert ps.uploaded[-1][("forward", 4)] == 0j
        assert ps.uploaded[-1][("backward", 1)] == 0.003j
        assert ps.speeds == (50.0, 100.0)
        ps.set_harmonic("forward", 2, 0.04j)
    assert ps.corrections[("forward", 2)] == 0.04j
    assert ps.corrections[("backward", 2)] == 0j
    assert ps.speeds == (50.0, 100.0)


def test_failed_calibration_restores_earlier_corrections():
    ps = FakePhaseStepping(EARLIER, speeds=(30.0, 160.0))
    with pytest.raises(RuntimeError):
        with _uncorrected(ps, [2, 4]):
            ps.set_harmonic("forward", 2, 0.04j)
            raise RuntimeError("accelerometer lost")
    assert ps.uploaded[-1] == {
        **EARLIER,
        ("backward", 4): 0j,
    }
    assert ps.speeds == (30.0, 160.0)
