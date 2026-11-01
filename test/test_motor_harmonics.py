# Tests for motor_harmonics.py: single-motor move directions and the
# harmonic analysis of constant-speed accelerometer windows.
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "klippy", "extras"))
import motor_harmonics as mh  # noqa: E402
import motor_kinematics as mk  # noqa: E402

SAMPLE_RATE = 1600.0
PERIOD = 0.8


def _synthetic_window(speed, disp_by_harmonic, duration=0.5, noise=0.0):
    rng = np.random.default_rng(1)
    times = 100.0 + np.arange(0.0, duration, 1.0 / SAMPLE_RATE)
    # Jitter the sample times like a real accelerometer clock does.
    times = times + rng.uniform(-5e-5, 5e-5, len(times))
    accel = rng.normal(0.0, noise, len(times))
    for harmonic, disp in disp_by_harmonic.items():
        omega = 2.0 * math.pi * harmonic * speed / PERIOD
        accel += disp * omega**2 * np.cos(omega * times + 0.7 * harmonic)
    return times, accel


def test_electrical_period_is_four_full_steps():
    assert mk.electrical_period(40.0, 200) == pytest.approx(0.8)
    assert mk.electrical_period(40.0, 400) == pytest.approx(0.4)


def test_plan_speeds_includes_both_ends():
    assert mh.plan_speeds(10.0, 160.0, 5.0)[-1] == pytest.approx(160.0)
    assert len(mh.plan_speeds(10.0, 160.0, 5.0)) == 31
    assert mh.plan_speeds(0.3, 0.9, 0.1)[-1] == pytest.approx(0.9)
    assert mh.plan_speeds(10.0, 12.0, 5.0) == [10.0]


def test_cartesian_direction_moves_one_axis():
    assert mk.single_motor_direction((1.0, 0.0), [(0.0, 1.0)]) == (1.0, 0.0)


def test_corexy_direction_keeps_the_other_motor_still():
    a, b = (1.0, 1.0), (1.0, -1.0)
    d = mk.single_motor_direction(a, [b, (0.0, 0.0)])
    assert d == pytest.approx((0.5, 0.5))
    assert a[0] * d[0] + a[1] * d[1] == pytest.approx(1.0)
    assert b[0] * d[0] + b[1] * d[1] == pytest.approx(0.0)


def test_parallel_motor_on_same_axis_is_ignored():
    d = mk.single_motor_direction((1.0, 0.0), [(1.0, 0.0), (0.0, 1.0)])
    assert d == (1.0, 0.0)


def test_conflicting_motors_have_no_direction():
    assert (
        mk.single_motor_direction((1.0, 0.0), [(0.0, 1.0), (1.0, 1.0)]) is None
    )
    assert mk.single_motor_direction((0.0, 0.0), [(0.0, 1.0)]) is None


def test_harmonic_amplitude_recovers_a_sinusoid():
    times = np.arange(0.0, 0.5, 1.0 / SAMPLE_RATE)
    values = 3.0 * np.cos(2.0 * math.pi * 137.0 * times + 0.4) + 5.0
    amp = mh.harmonic_amplitude(times, values, 137.0)
    assert abs(amp) == pytest.approx(3.0, rel=0.01)
    assert np.angle(amp) == pytest.approx(0.4, abs=0.01)


def test_harmonic_amplitude_survives_a_dropped_batch():
    times, accel = _synthetic_window(60.0, {2: 2e-3})
    keep = (times < 100.2) | (times > 100.25)
    full = mh.harmonic_amplitude(times, accel, 150.0)
    gapped = mh.harmonic_amplitude(times[keep], accel[keep], 150.0)
    assert abs(gapped) == pytest.approx(abs(full), rel=0.02)


def test_analyze_window_measures_each_harmonic():
    speed = 60.0
    times, accel = _synthetic_window(speed, {2: 2e-3, 4: 5e-4}, noise=50.0)
    results = mh.analyze_window(times, accel, speed, PERIOD, (1, 2, 3, 4))
    by_h = {r.harmonic: r for r in results}
    assert by_h[2].freq == pytest.approx(150.0)
    assert by_h[2].disp == pytest.approx(2e-3, rel=0.05)
    assert by_h[4].disp == pytest.approx(5e-4, rel=0.05)
    assert by_h[1].disp < by_h[1].noise_disp * 5.0
    assert by_h[3].disp < by_h[3].noise_disp * 5.0
    assert by_h[2].disp > 20.0 * by_h[2].noise_disp


def test_analyze_window_rejects_unresolvable_harmonics():
    # H4 at 160 mm/s is 800 Hz: at Nyquist for a 1600 Hz accelerometer.
    times, accel = _synthetic_window(160.0, {2: 1e-3})
    by_h = {
        r.harmonic: r
        for r in mh.analyze_window(times, accel, 160.0, PERIOD, (2, 4))
    }
    assert by_h[2].disp is not None
    assert by_h[4].disp is None
    # Too few cycles: H1 at 2 mm/s is 2.5 Hz, about 1 cycle in 0.5 s.
    times, accel = _synthetic_window(2.0, {1: 1e-3})
    assert mh.analyze_window(times, accel, 2.0, PERIOD, (1,))[0].disp is None
    empty = np.array([])
    assert mh.analyze_window(empty, empty, 60.0, PERIOD, (1,))[0].disp is None


def test_window_slice_selects_the_time_range():
    times = np.arange(0.0, 1.0, 0.1)
    assert list(times[mh.window_slice(times, 0.25, 0.55)]) == pytest.approx(
        [0.3, 0.4, 0.5]
    )


def test_project_samples_uses_the_move_direction():
    samples = [(0.0, 1.0, 1.0, 9.8), (0.1, 2.0, 0.0, 9.8)]
    s = math.sqrt(0.5)
    times, values = mh.project_samples(samples, s, s)
    assert list(times) == [0.0, 0.1]
    assert values == pytest.approx([2.0 * s, 2.0 * s])


def test_peak_per_harmonic_picks_largest_per_direction():
    def res(h, disp, noise=None):
        return mh.HarmonicResult(h, 100.0, 1.0, disp, noise)

    rows = [
        (20.0, 1, res(2, 1e-3)),
        (40.0, 1, res(2, 3e-3)),
        (40.0, -1, res(2, 2e-3)),
        (60.0, 1, res(2, None)),
        # Larger, but only 2x above its noise level.
        (10.0, 1, res(2, 5e-3, 2.5e-3)),
        (10.0, -1, res(4, 5e-3, 2.5e-3)),
    ]
    peaks = mh.peak_per_harmonic(rows)
    assert peaks[(2, 1)][0] == 40.0
    assert peaks[(2, -1)][0] == 40.0
    assert (4, 1) not in peaks
    assert (4, -1) not in peaks


def test_estimate_correction_finds_the_cancelling_correction():
    target = 0.031 * np.exp(1j * 2.2)
    gain = 4000.0 * np.exp(-1j * 0.7)

    def vibration(c):
        return gain * (c - target)

    c0 = 0.0 + 0.0j
    new, est_gain, nonlinearity = mh.estimate_correction(
        vibration(c0), vibration(c0 + 0.02), vibration(c0 + 0.02j), c0, 0.02
    )
    assert new == pytest.approx(target)
    assert est_gain == pytest.approx(gain)
    assert nonlinearity == pytest.approx(0.0, abs=1e-9)


def test_pick_calibration_speed_needs_both_directions():
    def res(disp, noise=0.01):
        return mh.HarmonicResult(2, 100.0, 1.0, disp, noise)

    rows = [
        (20.0, 1, res(5.0)),
        (20.0, -1, res(0.02)),  # backward barely above noise
        (40.0, 1, res(3.0)),
        (40.0, -1, res(2.8)),
        (60.0, 1, res(9.0)),  # backward missing
    ]
    assert mh.pick_calibration_speed(rows, 2) == 40.0
    assert mh.pick_calibration_speed(rows, 4) is None


def test_coherent_amplitude_keeps_phase_relative_to_reference():
    times = np.arange(10.0, 10.5, 1.0 / SAMPLE_RATE)
    values = 2.0 * np.cos(2.0 * math.pi * 80.0 * (times - 10.0) + 0.9)
    late = times[100:]
    amp = mh.harmonic_amplitude(late, values[100:], 80.0, t_ref=10.0)
    assert abs(amp) == pytest.approx(2.0, rel=0.02)
    assert np.angle(amp) == pytest.approx(0.9, abs=0.02)
