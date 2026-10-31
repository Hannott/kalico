# Tests for the host side helpers of phase_stepping.py
import math
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from klippy.extras import phase_stepping as ps  # noqa: E402


def _integrate(moves, phase=0):
    """Integer phase path exactly as the MCU's stepper_tick() computes it."""
    for count, vel, half in moves:
        for _ in range(count):
            phase += vel + half
            vel += 2 * half
    return phase, vel


def test_scaled_sine_matches_a_sine():
    for idx in range(0, 1024, 7):
        expected = 248 * math.sin(2 * math.pi * idx / 1024)
        assert ps.scaled_sine(idx, 248) == pytest.approx(expected, abs=1.0)


def test_coil_value_packs_9_bit_currents():
    # MSCNT 0: sine 0, cosine +248
    assert ps.coil_value(0, 248, False) == 248 << 16
    assert ps.coil_value(0, 248, True) == 248
    # MSCNT 768 (270 degrees): sine -248, cosine 0
    value = ps.coil_value(768 << ps.MSCNT_SHIFT, 248, False)
    assert value & 0x1FF == (-248) & 0x1FF
    assert value >> 16 == 0


def test_check_coil_mapping():
    assert ps.check_coil_mapping(0, 0, 247) is None
    assert ps.check_coil_mapping(256, 247, 0) is None
    assert ps.check_coil_mapping(128, 175, 175) is None
    assert ps.check_coil_mapping(256, 0, 247) is not None
    assert ps.check_coil_mapping(0, 0, 3) is not None


def test_trapezoid_covers_the_planned_distance_and_stops():
    moves, travel = ps.plan_trapezoid(5.4e10, 2.7e6, 1677.0)
    phase, vel = _integrate(moves)
    assert phase == travel
    assert vel == 0
    assert travel == pytest.approx(5.4e10, rel=0.001)
    assert len(moves) == 3


def test_trapezoid_without_cruise():
    moves, travel = ps.plan_trapezoid(1.0e6, 2.7e6, 1677.0)
    phase, vel = _integrate(moves)
    assert (phase, vel) == (travel, 0)
    assert travel <= 1.0e6
    assert len(moves) == 2


def test_reverse_returns_exactly_to_the_start():
    moves, _ = ps.plan_trapezoid(5.3e9, 3.1e6, 1201.0)
    full = moves + [(400, 0, 0)] + ps.reverse_moves(moves)
    phase, vel = _integrate(full, phase=123456789)
    assert (phase, vel) == (123456789, 0)


def test_trapezoid_rejects_impossible_moves():
    with pytest.raises(ValueError):
        ps.plan_trapezoid(10.0, 2.7e6, 1677.0)
    with pytest.raises(ValueError):
        ps.plan_trapezoid(1e15, 3e9, 1e6)
