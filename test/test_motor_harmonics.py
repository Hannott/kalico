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
    assert mh.pick_calibration_speeds(rows, 2, 1) == [40.0]
    assert mh.pick_calibration_speeds(rows, 4, 1) == []


def test_coherent_amplitude_keeps_phase_relative_to_reference():
    times = np.arange(10.0, 10.5, 1.0 / SAMPLE_RATE)
    values = 2.0 * np.cos(2.0 * math.pi * 80.0 * (times - 10.0) + 0.9)
    late = times[100:]
    amp = mh.harmonic_amplitude(late, values[100:], 80.0, t_ref=10.0)
    assert abs(amp) == pytest.approx(2.0, rel=0.02)
    assert np.angle(amp) == pytest.approx(0.9, abs=0.02)


def _linear(gains, targets):
    """Vibration per speed for a correction c: gain * (c - target)."""
    return lambda c: {s: gains[s] * (c - targets[s]) for s in gains}


def test_multi_speed_correction_finds_a_shared_optimum():
    gains = {20.0: 3e-2 * np.exp(1j * 0.4), 60.0: 8e-3 * np.exp(-1j * 2.0)}
    target = 0.025 * np.exp(1j * 1.1)
    vib = _linear(gains, {s: target for s in gains})
    c0 = 0.0j
    new, est_gains = mh.multi_speed_correction(
        vib(c0), vib(c0 + 0.02), vib(c0 + 0.02j), c0, 0.02
    )
    assert new == pytest.approx(target)
    for s in gains:
        assert est_gains[s] == pytest.approx(gains[s])


def test_multi_speed_correction_minimises_the_worst_speed():
    # The best correction differs per speed, as when the load angle grows,
    # and one speed vibrates far more than the others
    gains = {s: 1e-2 * np.exp(1j * 0.3 * s / 10.0) for s in (10.0, 30.0, 50.0)}
    gains[50.0] *= 5.0
    targets = {
        10.0: 0.03 * np.exp(1j * 0.0),
        30.0: 0.03 * np.exp(1j * 0.6),
        50.0: 0.03 * np.exp(1j * 1.2),
    }
    vib = _linear(gains, targets)
    c0 = 0.0j
    d0 = vib(c0)
    new, est_gains = mh.multi_speed_correction(
        d0, vib(c0 + 0.02), vib(c0 + 0.02j), c0, 0.02
    )
    worst = mh.worst_vibration(vib(new))
    grid = np.linspace(-0.05, 0.05, 201)
    best_on_grid = min(
        mh.worst_vibration(vib(complex(re, im))) for re in grid for im in grid
    )
    assert worst <= best_on_grid * 1.001
    assert worst < 0.25 * mh.worst_vibration(d0)
    # A step from the optimum stays there
    assert mh.refine_correction(new, vib(new), est_gains) == (
        pytest.approx(new, abs=1e-6)
    )


def test_pick_calibration_speeds_covers_the_peaks():
    def res(disp):
        return mh.HarmonicResult(2, 100.0, 1.0, disp, 1e-5)

    # Two resonance peaks over a slope, quiet above 110 mm/s
    level = {s: 1e-3 * (1.0 - s / 200.0) for s in range(30, 161, 5)}
    level[60] = level[65] = 8e-3
    level[70] = 6e-3
    level[95] = 4e-3
    for s in range(115, 161, 5):
        level[s] = 1e-6
    rows = []
    for speed, disp in level.items():
        rows += [(float(speed), 1, res(disp)), (float(speed), -1, res(disp))]
    picked = mh.pick_calibration_speeds(rows, 2, 4)
    assert picked == [30.0, 60.0, 70.0, 95.0]
    assert mh.pick_calibration_speeds(rows, 2, 1) == [60.0]
    assert len(mh.pick_calibration_speeds(rows, 2, 20)) == 17
    assert mh.pick_calibration_speeds(rows, 4, 4) == []


def _synthetic_ramp(ramp, disp_by_harmonic, noise=0.0):
    rng = np.random.default_rng(2)
    times = ramp.start_time + np.arange(0.0, ramp.duration, 1.0 / SAMPLE_RATE)
    times = times + rng.uniform(-5e-5, 5e-5, len(times))
    t = times - ramp.start_time
    speed = ramp.start_speed + ramp.accel * t
    angle = 2.0 * math.pi * (ramp.start_speed * t + 0.5 * ramp.accel * t * t)
    angle /= PERIOD
    rate = 2.0 * math.pi * speed / PERIOD
    rate_change = 2.0 * math.pi * ramp.accel / PERIOD
    accel = rng.normal(0.0, noise, len(times))
    for harmonic, disp in disp_by_harmonic.items():
        phase = harmonic * angle + 0.7 * harmonic
        # Second derivative of disp * cos(phase) with a changing rate
        accel -= disp * (
            np.cos(phase) * (harmonic * rate) ** 2
            + np.sin(phase) * harmonic * rate_change
        )
    return times, accel


@pytest.mark.parametrize("direction", [1, -1])
def test_analyze_ramp_follows_the_changing_frequency(direction):
    up = direction > 0
    ramp = mh.Ramp(1, 100.0, 20.0 if up else 120.0, 50.0 * direction, 2.0)
    times, accel = _synthetic_ramp(ramp, {2: 2e-3, 4: 5e-4}, noise=50.0)
    speeds = [40.0, 70.0, 100.0]
    rows = mh.analyze_ramp(times, accel, ramp, PERIOD, (1, 2, 4), speeds, 2.5)
    by_key = {(s, r.harmonic): r for s, d, r in rows}
    assert {s for s, h in by_key} == set(speeds)
    for speed in speeds:
        assert by_key[speed, 2].disp == pytest.approx(2e-3, rel=0.05)
        assert by_key[speed, 4].disp == pytest.approx(5e-4, rel=0.05)
        assert by_key[speed, 2].disp > 20.0 * by_key[speed, 2].noise_disp
        assert by_key[speed, 1].disp < 5.0 * by_key[speed, 1].noise_disp


def test_analyze_ramp_skips_speeds_it_cannot_resolve():
    ramp = mh.Ramp(-1, 100.0, 20.0, 50.0, 2.0)
    times, accel = _synthetic_ramp(ramp, {2: 2e-3})
    rows = mh.analyze_ramp(
        times, accel, ramp, PERIOD, (1, 2), [10.0, 21.0, 60.0, 119.0], 2.5
    )
    by_key = {(s, r.harmonic): r for s, d, r in rows}
    # Outside the ramp, or a window reaching past either end
    assert (10.0, 2) not in by_key
    assert by_key[21.0, 2].disp is None and by_key[119.0, 2].disp is None
    assert by_key[60.0, 2].disp is not None
    assert all(d == -1 for s, d, r in rows)


def test_merge_rows_averages_ramps_up_and_down():
    a = mh.HarmonicResult(2, 75.0, 10.0, 2e-3, 1e-5)
    b = mh.HarmonicResult(2, 75.0, 20.0, 4e-3, 3e-5)
    none = mh.HarmonicResult(4, 150.0, None, None, None)
    unmeasured = mh.HarmonicResult(2, 75.0, None, None, None)
    rows = [
        (30.0, 1, a),
        (30.0, 1, b),
        (30.0, 1, unmeasured),
        (30.0, -1, a),
        (30.0, 1, none),
    ]
    merged = mh.merge_rows(rows)
    assert merged[0] == (30.0, 1, mh.HarmonicResult(2, 75.0, 15.0, 3e-3, 2e-5))
    assert merged[1] == (30.0, -1, a)
    assert merged[2][2].disp is None


def test_plan_ramps_covers_the_range_within_the_travel():
    segments = mh.plan_ramps(10.0, 160.0, PERIOD, (2000.0, 50.0), 150.0, 0.1)
    assert segments[0][0] < 10.0 and segments[-1][1] > 160.0
    for (v1, v2, ramp_accel), nxt in zip(segments, segments[1:] + [None]):
        assert mh.ramp_travel(v1, v2, ramp_accel, 2000.0) <= 150.0 + 1e-6
        assert ramp_accel <= 50.0
        # A window of RAMP_WINDOW_PERIODS spans at most a quarter of v1
        window = mh.RAMP_WINDOW_PERIODS * PERIOD / v1
        assert ramp_accel * window <= 0.25 * v1 * 1.0001
        if nxt is not None:
            assert nxt[0] < v2
    # Slow speeds ramp more gently than fast ones
    assert segments[0][2] < segments[-1][2]
    with pytest.raises(ValueError):
        mh.plan_ramps(10.0, 160.0, PERIOD, (2000.0, 50.0), 5.0, 0.1)
